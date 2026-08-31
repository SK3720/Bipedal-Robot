"""Last-moment flat L foot touchdown A/B experiment + unchanged R recovery.

A = exact trajectory from staged_forward_catch_test.py (through catch lerp).
B = identical hip/knee/timing; L ankle flattens only when foot nears ground.

Trigger: L foot clearance < CLEARANCE_TRIGGER_MM and no contact.
Evaluation only — does not modify robot.xml, biped_env, or other scripts.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np
from ctypes import wintypes

from biped_env import BipedalWalkEnv, DEFAULT_POSE

import staged_forward_catch_test as base
from staged_forward_catch_test import (
    CATCH_MAX_STEPS,
    IDX_L_ANKLE_P,
    IDX_L_HIP_PITCH,
    IDX_L_KNEE,
    IDX_R_HIP_PITCH,
    IDX_R_KNEE,
    L_ANKLE_CATCH,
    L_HIP_CATCH,
    L_KNEE_CATCH,
    Phase,
    QPOS_L_HIP,
    QPOS_L_KNEE,
    RunState,
    StepDiagnostics,
    _foot_contact,
    _foot_normal_force,
    _foot_pos,
    _forward_mm,
    _forward_vel,
    _is_unloaded,
    _lerp_ctrl,
    _print_phase,
    _reset,
    _sim_step,
    _smooth,
    _sync_viewer,
    forward_lean_pose,
    left_leg_pose,
    rapid_shift_pose,
)
from staged_forward_catch_continue_test import (
    PLANT_FORCE_N,
    PLANT_SUSTAIN_STEPS,
    _is_heel_touchdown,
    _r_load_fraction,
)
from staged_forward_r_recovery_test import (
    QPOS_R_HIP,
    QPOS_R_KNEE,
    R_HIP_RAMP_STEP,
    R_HIP_RECOVERY_MAX_STEPS,
    R_HIP_RECOVERY_TARGET,
    R_KNEE_CLEARANCE_TARGET,
    R_KNEE_LIFT_MAX_STEPS,
    R_KNEE_LIFT_STEP,
    R_KNEE_MIN_RAMP_STEPS,
    R_LOAD_DROP_THRESH,
    R_NORMAL_UNLOAD_N,
    WAIT_PLANT_MAX_STEPS,
    MIN_R_FWD_RECOVERY_MM,
    RecoveryPhase,
    _apply_ctrl,
    _r_foot_fwd_vel,
    _set_leg_ctrl,
)

# Positive L ankle pitch flattens sole (kinematic check at catch pose).
L_ANKLE_FLAT_TARGET = 0.38
ANKLE_FLAT_RAMP_STEP = 0.020
CLEARANCE_TRIGGER_MM = 10.0
FOOT_CONTACT_Z = 1.042
SOLE_DESCENT_VEL_M_S = 0.02
QPOS_L_ANKLE = 15

VIEWER_WIDTH = base.VIEWER_WIDTH
VIEWER_HEIGHT = base.VIEWER_HEIGHT
VIEWER_TITLE_PREFIX = base.VIEWER_TITLE_PREFIX
VIEWER_POSITION_TIMEOUT_S = base.VIEWER_POSITION_TIMEOUT_S


class TouchdownPhase(str, Enum):
    LAST_MOMENT_CATCH = "LAST-MOMENT FLAT FOOT CATCH"


@dataclass
class TrajCheck:
    max_hip_cmd_diff: float = 0.0
    max_knee_cmd_diff: float = 0.0
    max_hip_qpos_diff: float = 0.0
    max_knee_qpos_diff: float = 0.0
    steps_compared: int = 0
    ankle_trigger_step: int | None = None
    clearance_at_trigger_mm: float | None = None


@dataclass
class ExperimentResult:
    label: str
    flat_ankle: bool
    base: StepDiagnostics
    heel_touchdown_step: int | None = None
    l_foot_xyz_at_td: np.ndarray | None = None
    touchdown_ankle_cmd: float | None = None
    touchdown_ankle_qpos: float | None = None
    touchdown_foot_pitch_rad: float | None = None
    torso_tilt_before_td_rad: float | None = None
    torso_tilt_after_td_rad: float | None = None
    torso_fwd_vel_before_td_m_s: float | None = None
    torso_fwd_vel_after_td_m_s: float | None = None
    fwd_vel_kick_m_s: float | None = None
    peak_l_normal_after_td_n: float = 0.0
    l_plant_confirmed_step: int | None = None
    r_lift_off_step: int | None = None
    peak_r_fwd_mm: float = 0.0
    peak_r_rel_l_fwd_mm: float = 0.0
    l_foot_remains_planted: bool = False
    l_foot_max_drift_mm: float = 0.0
    traj_check: TrajCheck = field(default_factory=TrajCheck)


def _sole_clearance_mm(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Distance from L foot body Z to nominal contact height (project convention)."""
    return float((_foot_pos(model, data, "L")[2] - FOOT_CONTACT_Z) * 1000.0)


def _l_foot_vert_vel_m_s(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    foot_vel = np.zeros(6)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        model.body("L_foot").id,
        foot_vel,
        0,
    )
    return float(foot_vel[2])


def _foot_pitch_rad(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    rot = data.xmat[model.body("L_foot").id].reshape(3, 3)
    candidates = [
        rot @ np.array([0.0, 0.0, -1.0]),
        rot @ np.array([0.0, 0.0, 1.0]),
        rot @ np.array([0.0, -1.0, 0.0]),
        rot @ np.array([0.0, 1.0, 0.0]),
    ]
    sole = max(candidates, key=lambda v: v[2])
    return float(np.arctan2(np.linalg.norm(sole[:2]), sole[2]))


def _run_locked_prefix(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cr: np.ndarray,
    st: RunState,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
    *,
    verbose: bool = True,
) -> None:
    """Identical to staged_forward_catch_test through HIP SWING."""
    shifted = rapid_shift_pose()
    lean = forward_lean_pose()

    def _phase(name: str) -> None:
        if verbose:
            _print_phase(name)

    _phase(Phase.STAND.value)
    for s in range(base.STAND_STEPS):
        st.ctrl = _lerp_ctrl(st.ctrl, DEFAULT_POSE, _smooth((s + 1) / base.STAND_STEPS), cr)
        _sim_step(env, model, data, st, viewer, slow)
    st.stand_r_xy = _foot_pos(model, data, "R")[:2].copy()

    _phase(Phase.FORWARD_FALL.value)
    for s in range(base.FALL_RAMP_STEPS):
        alpha = (s + 1) / base.FALL_RAMP_STEPS
        st.ctrl = _lerp_ctrl(st.ctrl, lean, alpha, cr)
        _sim_step(env, model, data, st, viewer, slow)
    for _ in range(base.FALL_MOMENTUM_STEPS):
        st.ctrl = lean.copy()
        _sim_step(env, model, data, st, viewer, slow)

    _phase(Phase.RAPID_SHIFT.value)
    shift_start = st.ctrl.copy()
    for s in range(base.RAPID_SHIFT_STEPS):
        alpha = (s + 1) / base.RAPID_SHIFT_STEPS
        st.ctrl = _lerp_ctrl(shift_start, shifted, alpha, cr)
        st.knee_cmd = float(st.ctrl[IDX_L_KNEE])
        st.hip_cmd = float(st.ctrl[IDX_L_HIP_PITCH])
        _sim_step(env, model, data, st, viewer, slow)

    post_shift = 0
    while post_shift < base.POST_SHIFT_MAX_STEPS and st.phase == Phase.RAPID_SHIFT:
        st.ctrl = shifted.copy()
        st.knee_cmd = float(st.ctrl[IDX_L_KNEE])
        st.hip_cmd = float(st.ctrl[IDX_L_HIP_PITCH])
        _sim_step(env, model, data, st, viewer, slow)
        post_shift += 1

    _phase(Phase.KNEE_LIFT.value)
    st.phase = Phase.KNEE_LIFT
    while st.knee_lift_steps < base.KNEE_LIFT_MAX_STEPS and st.phase == Phase.KNEE_LIFT:
        if _is_unloaded(model, data):
            st.clearance_knee = st.knee_cmd
            st.diag.clearance_knee_cmd = st.knee_cmd
            st.phase = Phase.HIP_SWING
            st.diag.hip_swing_start_step = st.diag.global_step + 1
            break
        st.knee_cmd = max(st.knee_cmd - base.KNEE_LIFT_STEP, base.KNEE_LIFT_TARGET)
        st.ctrl = left_leg_pose(st.knee_cmd, st.hip_cmd)
        _sim_step(env, model, data, st, viewer, slow)
        st.knee_lift_steps += 1

    if st.phase == Phase.KNEE_LIFT:
        st.clearance_knee = st.knee_cmd
        st.diag.clearance_knee_cmd = st.knee_cmd
        st.phase = Phase.HIP_SWING
        st.diag.hip_swing_start_step = st.diag.global_step + 1

    _phase(Phase.HIP_SWING.value)
    while st.hip_swing_steps < base.HIP_SWING_MAX_STEPS and st.phase == Phase.HIP_SWING:
        if st.airborne_ref_y is None and not _foot_contact(model, data, "L"):
            st.airborne_ref_y = float(_foot_pos(model, data, "L")[1])

        knee_hold = st.clearance_knee if st.clearance_knee is not None else st.knee_cmd
        st.hip_cmd = max(st.hip_cmd - base.HIP_RAMP_PER_STEP, base.HIP_SWING_TARGET)
        st.ctrl = left_leg_pose(knee_hold, st.hip_cmd)
        _sim_step(env, model, data, st, viewer, slow)
        st.hip_swing_steps += 1

        fwd_air = 0.0
        if st.airborne_ref_y is not None:
            fwd_air = _forward_mm(_foot_pos(model, data, "L")[1], st.airborne_ref_y)

        hip_at_target = st.hip_cmd <= base.HIP_SWING_TARGET + 1e-6
        if fwd_air >= base.MIN_FWD_AIRBORNE_MM and not _foot_contact(model, data, "L"):
            st.phase = Phase.CATCH
            st.diag.catch_start_step = st.diag.global_step + 1
            break
        if hip_at_target and st.hip_swing_steps > 80 and fwd_air >= 15.0:
            st.phase = Phase.CATCH
            st.diag.catch_start_step = st.diag.global_step + 1
            break

    if st.phase == Phase.HIP_SWING:
        st.phase = Phase.CATCH
        st.diag.catch_start_step = st.diag.global_step + 1


def _run_catch_to_heel(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cr: np.ndarray,
    st: RunState,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
    *,
    flat_ankle: bool,
    ref_cmds: dict[int, tuple[float, float, float, float]] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, ExperimentResult]:
    """Catch phase: original full lerp (A) or hip/knee lerp + clearance-triggered ankle (B)."""
    if verbose:
        phase_label = (
            TouchdownPhase.LAST_MOMENT_CATCH.value if flat_ankle else Phase.CATCH.value
        )
        _print_phase(phase_label)

    result = ExperimentResult(
        label="FLAT-AT-LAST-MOMENT" if flat_ankle else "ORIGINAL",
        flat_ankle=flat_ankle,
        base=st.diag,
    )

    catch_start_ctrl = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l_contact = _foot_contact(model, data, "L")
    was_airborne_in_catch = not prev_l_contact

    ankle_cmd = float(catch_start_ctrl[IDX_L_ANKLE_P])
    ankle_flat_active = False
    pre_td_tilt: float | None = None
    pre_td_vel: float | None = None
    heel_ctrl = st.ctrl.copy()

    while st.catch_steps < CATCH_MAX_STEPS and result.heel_touchdown_step is None:
        if pre_td_vel is None and not _foot_contact(model, data, "L"):
            pre_td_vel = _forward_vel(data)
            pre_td_tilt = env._quat_tilt_rad()

        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        full_lerp = _lerp_ctrl(catch_start_ctrl, catch_target, alpha, cr)
        knee_val = float(full_lerp[IDX_L_KNEE])
        hip_val = float(full_lerp[IDX_L_HIP_PITCH])

        if ref_cmds is not None and flat_ankle and not ankle_flat_active:
            step = st.diag.global_step + 1
            if step in ref_cmds:
                ref_hip_c, ref_knee_c, ref_hip_q, ref_knee_q = ref_cmds[step]
                tc = result.traj_check
                tc.max_hip_cmd_diff = max(tc.max_hip_cmd_diff, abs(hip_val - ref_hip_c))
                tc.max_knee_cmd_diff = max(tc.max_knee_cmd_diff, abs(knee_val - ref_knee_c))
                tc.max_hip_qpos_diff = max(
                    tc.max_hip_qpos_diff, abs(float(data.qpos[QPOS_L_HIP]) - ref_hip_q)
                )
                tc.max_knee_qpos_diff = max(
                    tc.max_knee_qpos_diff, abs(float(data.qpos[QPOS_L_KNEE]) - ref_knee_q)
                )
                tc.steps_compared += 1

        if flat_ankle:
            clearance = _sole_clearance_mm(model, data)
            l_contact = _foot_contact(model, data, "L")
            if not l_contact:
                if not ankle_flat_active:
                    descending = _l_foot_vert_vel_m_s(model, data) < -SOLE_DESCENT_VEL_M_S
                    if clearance < CLEARANCE_TRIGGER_MM and descending:
                        ankle_flat_active = True
                        result.traj_check.ankle_trigger_step = st.diag.global_step
                        result.traj_check.clearance_at_trigger_mm = clearance
                if ankle_flat_active:
                    ankle_cmd = min(ankle_cmd + ANKLE_FLAT_RAMP_STEP, L_ANKLE_FLAT_TARGET)
            st.ctrl = left_leg_pose(knee_val, hip_val, ankle_cmd)
        else:
            st.ctrl = full_lerp.copy()

        _sim_step(env, model, data, st, viewer, slow)
        st.catch_steps += 1

        if _is_heel_touchdown(
            model,
            data,
            catch_started=True,
            was_airborne_in_catch=was_airborne_in_catch,
            prev_l_contact=prev_l_contact,
        ):
            l_pos = _foot_pos(model, data, "L")
            result.heel_touchdown_step = st.diag.global_step
            result.l_foot_xyz_at_td = l_pos.copy()
            result.touchdown_ankle_cmd = float(data.ctrl[IDX_L_ANKLE_P])
            result.touchdown_ankle_qpos = float(data.qpos[QPOS_L_ANKLE])
            result.touchdown_foot_pitch_rad = _foot_pitch_rad(model, data)
            result.torso_tilt_before_td_rad = pre_td_tilt
            result.torso_fwd_vel_before_td_m_s = pre_td_vel
            heel_ctrl = st.ctrl.copy()
            tag = "FLAT-AT-LAST-MOMENT" if flat_ankle else "ORIGINAL"
            if verbose:
                print(
                    f"\n  >> {tag} TOUCHDOWN step {result.heel_touchdown_step} "
                    f"ankle_cmd={result.touchdown_ankle_cmd:.3f} "
                    f"pitch={result.touchdown_foot_pitch_rad:.3f} rad "
                    f"clearance={_sole_clearance_mm(model, data):.1f} mm"
                )
            break

        prev_l_contact = _foot_contact(model, data, "L")
        if not prev_l_contact:
            was_airborne_in_catch = True

    if result.heel_touchdown_step is None:
        l_pos = _foot_pos(model, data, "L")
        result.heel_touchdown_step = st.diag.global_step
        result.l_foot_xyz_at_td = l_pos.copy()
        result.touchdown_ankle_cmd = float(data.ctrl[IDX_L_ANKLE_P])
        result.touchdown_ankle_qpos = float(data.qpos[QPOS_L_ANKLE])
        result.touchdown_foot_pitch_rad = _foot_pitch_rad(model, data)
        heel_ctrl = st.ctrl.copy()

    return heel_ctrl, result


def _record_catch_reference(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
) -> dict[int, tuple[float, float, float, float]]:
    """Run original once; record hip/knee cmd+qpos each catch step for A/B check."""
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    shifted = rapid_shift_pose()
    st = RunState(
        ctrl=DEFAULT_POSE.copy(),
        phase=Phase.STAND,
        stand_r_xy=_foot_pos(model, data, "R")[:2].copy(),
        diag=StepDiagnostics(),
        knee_cmd=float(shifted[IDX_L_KNEE]),
        hip_cmd=float(shifted[IDX_L_HIP_PITCH]),
    )
    _run_locked_prefix(env, model, data, cr, st, viewer=None, slow=False, verbose=False)

    catch_start_ctrl = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l_contact = _foot_contact(model, data, "L")
    was_airborne_in_catch = not prev_l_contact
    ref: dict[int, tuple[float, float, float, float]] = {}

    while st.catch_steps < CATCH_MAX_STEPS:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        st.ctrl = _lerp_ctrl(catch_start_ctrl, catch_target, alpha, cr)
        step = st.diag.global_step + 1
        ref[step] = (
            float(st.ctrl[IDX_L_HIP_PITCH]),
            float(st.ctrl[IDX_L_KNEE]),
            float(data.qpos[QPOS_L_HIP]),
            float(data.qpos[QPOS_L_KNEE]),
        )
        _sim_step(env, model, data, st, viewer=None, slow=False)
        st.catch_steps += 1
        if _is_heel_touchdown(
            model,
            data,
            catch_started=True,
            was_airborne_in_catch=was_airborne_in_catch,
            prev_l_contact=prev_l_contact,
        ):
            break
        prev_l_contact = _foot_contact(model, data, "L")
        if not prev_l_contact:
            was_airborne_in_catch = True

    return ref


def _run_r_recovery(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    st: RunState,
    heel_ctrl: np.ndarray,
    result: ExperimentResult,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
    *,
    verbose: bool = True,
) -> None:
    """Unchanged R recovery from staged_forward_r_recovery_test.py."""
    heel_l_xy = _foot_pos(model, data, "L")[:2].copy()
    r_recovery_ref_y = float(_foot_pos(model, data, "R")[1])

    stance_ctrl = heel_ctrl.copy()
    _apply_ctrl(env, model, data, st, stance_ctrl, viewer, slow)
    result.torso_fwd_vel_after_td_m_s = _forward_vel(data)
    result.torso_tilt_after_td_rad = env._quat_tilt_rad()
    if result.torso_fwd_vel_before_td_m_s is not None:
        result.fwd_vel_kick_m_s = (
            result.torso_fwd_vel_after_td_m_s - result.torso_fwd_vel_before_td_m_s
        )

    for _ in range(30):
        ln = _foot_normal_force(model, data, "L")
        result.peak_l_normal_after_td_n = max(result.peak_l_normal_after_td_n, ln)
        _apply_ctrl(env, model, data, st, stance_ctrl, viewer, slow)

    if verbose:
        _print_phase(RecoveryPhase.WAIT_L_PLANT.value)
    plant_streak = 0
    phase = RecoveryPhase.WAIT_L_PLANT
    r_knee_cmd = float(stance_ctrl[IDX_R_KNEE])
    clearance_r_knee = r_knee_cmd

    for _ in range(WAIT_PLANT_MAX_STEPS):
        l_nf = _foot_normal_force(model, data, "L")
        r_nf = _foot_normal_force(model, data, "R")
        r_frac = _r_load_fraction(model, data)
        l_pos = _foot_pos(model, data, "L")
        result.l_foot_max_drift_mm = max(
            result.l_foot_max_drift_mm,
            float(np.linalg.norm(l_pos[:2] - heel_l_xy) * 1000.0),
        )
        if _foot_contact(model, data, "L") and l_nf >= PLANT_FORCE_N:
            plant_streak += 1
        else:
            plant_streak = 0
        if (
            plant_streak >= PLANT_SUSTAIN_STEPS
            and l_nf >= PLANT_FORCE_N
            and (not np.isnan(r_frac) and r_frac <= R_LOAD_DROP_THRESH)
            and r_nf <= R_NORMAL_UNLOAD_N
        ):
            result.l_plant_confirmed_step = st.diag.global_step
            phase = RecoveryPhase.R_KNEE_LIFT
            break
        _apply_ctrl(env, model, data, st, stance_ctrl, viewer, slow)

    if phase == RecoveryPhase.WAIT_L_PLANT:
        result.l_plant_confirmed_step = st.diag.global_step
        phase = RecoveryPhase.R_KNEE_LIFT

    if verbose:
        _print_phase(RecoveryPhase.R_KNEE_LIFT.value)
    knee_steps = 0
    r_hip_cmd = float(stance_ctrl[IDX_R_HIP_PITCH])

    while knee_steps < R_KNEE_LIFT_MAX_STEPS and phase == RecoveryPhase.R_KNEE_LIFT:
        r_unloaded = (
            not _foot_contact(model, data, "R")
            or _foot_normal_force(model, data, "R") < R_NORMAL_UNLOAD_N
        )
        if r_unloaded and result.r_lift_off_step is None:
            result.r_lift_off_step = st.diag.global_step
        r_knee_cmd = max(r_knee_cmd - R_KNEE_LIFT_STEP, R_KNEE_CLEARANCE_TARGET)
        ctrl = _set_leg_ctrl(stance_ctrl, r_knee=r_knee_cmd)
        _apply_ctrl(env, model, data, st, ctrl, viewer, slow)
        knee_steps += 1
        if knee_steps >= R_KNEE_MIN_RAMP_STEPS and (
            r_unloaded or r_knee_cmd <= R_KNEE_CLEARANCE_TARGET + 1e-6
        ):
            clearance_r_knee = r_knee_cmd
            phase = RecoveryPhase.R_HIP_RECOVERY
            break

    if phase == RecoveryPhase.R_KNEE_LIFT:
        clearance_r_knee = r_knee_cmd
        phase = RecoveryPhase.R_HIP_RECOVERY

    if verbose:
        _print_phase(RecoveryPhase.R_HIP_RECOVERY.value)
    hip_steps = 0
    while hip_steps < R_HIP_RECOVERY_MAX_STEPS and phase == RecoveryPhase.R_HIP_RECOVERY:
        r_pos = _foot_pos(model, data, "R")
        l_pos = _foot_pos(model, data, "L")
        r_fwd = _forward_mm(r_pos[1], r_recovery_ref_y)
        r_rel = _forward_mm(r_pos[1], l_pos[1])
        result.peak_r_fwd_mm = max(result.peak_r_fwd_mm, r_fwd)
        result.peak_r_rel_l_fwd_mm = max(result.peak_r_rel_l_fwd_mm, r_rel)
        r_hip_cmd = min(r_hip_cmd + R_HIP_RAMP_STEP, R_HIP_RECOVERY_TARGET)
        ctrl = _set_leg_ctrl(stance_ctrl, r_knee=clearance_r_knee, r_hip=r_hip_cmd)
        _apply_ctrl(env, model, data, st, ctrl, viewer, slow)
        hip_steps += 1
        if r_fwd >= MIN_R_FWD_RECOVERY_MM and r_hip_cmd >= R_HIP_RECOVERY_TARGET - 1e-6:
            break

    result.l_foot_remains_planted = result.l_foot_max_drift_mm < 25.0

    if verbose:
        _print_phase(RecoveryPhase.STOP.value)
    for _ in range(60):
        _apply_ctrl(env, model, data, st, st.ctrl.copy(), viewer, slow)


def run_experiment(
    env: BipedalWalkEnv,
    *,
    flat_ankle: bool,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    ref_cmds: dict[int, tuple[float, float, float, float]] | None = None,
    quiet_prefix: bool = False,
) -> ExperimentResult:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    shifted = rapid_shift_pose()
    st = RunState(
        ctrl=DEFAULT_POSE.copy(),
        phase=Phase.STAND,
        stand_r_xy=_foot_pos(model, data, "R")[:2].copy(),
        diag=StepDiagnostics(),
        knee_cmd=float(shifted[IDX_L_KNEE]),
        hip_cmd=float(shifted[IDX_L_HIP_PITCH]),
    )

    _run_locked_prefix(
        env,
        model,
        data,
        cr,
        st,
        viewer,
        slow,
        verbose=not quiet_prefix,
    )

    heel_ctrl, result = _run_catch_to_heel(
        env,
        model,
        data,
        cr,
        st,
        viewer,
        slow,
        flat_ankle=flat_ankle,
        ref_cmds=ref_cmds if flat_ankle else None,
        verbose=not quiet_prefix,
    )
    _run_r_recovery(
        env,
        model,
        data,
        st,
        heel_ctrl,
        result,
        viewer,
        slow,
        verbose=not quiet_prefix,
    )
    return result


def _fmt_xyz(xyz: np.ndarray | None) -> str:
    if xyz is None:
        return "None"
    return f"[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]"


def _fmt(v: float | None, prec: int = 4) -> str:
    if v is None:
        return "None"
    return f"{v:.{prec}f}"


def print_ab_comparison(orig: ExperimentResult, flat: ExperimentResult) -> None:
    print("\n" + "=" * 72)
    print("A/B: ORIGINAL vs FLAT-AT-LAST-MOMENT (+ unchanged R recovery)")
    print("=" * 72)
    print(
        f"Ankle trigger: sole clearance < {CLEARANCE_TRIGGER_MM:.0f} mm, "
        f"descending, no contact (contact Z={FOOT_CONTACT_Z})"
    )
    print(f"Flat ankle target = +{L_ANKLE_FLAT_TARGET:.2f} rad, ramp = {ANKLE_FLAT_RAMP_STEP:.3f}/step")

    rows = [
        ("Touchdown step", orig.heel_touchdown_step, flat.heel_touchdown_step),
        ("L foot XYZ at TD", _fmt_xyz(orig.l_foot_xyz_at_td), _fmt_xyz(flat.l_foot_xyz_at_td)),
        ("L ankle cmd at TD", _fmt(orig.touchdown_ankle_cmd), _fmt(flat.touchdown_ankle_cmd)),
        ("L ankle qpos at TD", _fmt(orig.touchdown_ankle_qpos), _fmt(flat.touchdown_ankle_qpos)),
        ("L foot pitch at TD (rad)", _fmt(orig.touchdown_foot_pitch_rad, 3),
         _fmt(flat.touchdown_foot_pitch_rad, 3)),
        ("Torso pitch before TD (rad)", _fmt(orig.torso_tilt_before_td_rad, 3),
         _fmt(flat.torso_tilt_before_td_rad, 3)),
        ("Torso pitch after TD (rad)", _fmt(orig.torso_tilt_after_td_rad, 3),
         _fmt(flat.torso_tilt_after_td_rad, 3)),
        ("Torso fwd vel before TD (m/s)", _fmt(orig.torso_fwd_vel_before_td_m_s),
         _fmt(flat.torso_fwd_vel_before_td_m_s)),
        ("Torso fwd vel after TD (m/s)", _fmt(orig.torso_fwd_vel_after_td_m_s),
         _fmt(flat.torso_fwd_vel_after_td_m_s)),
        ("Fwd vel kick at TD (m/s)", _fmt(orig.fwd_vel_kick_m_s), _fmt(flat.fwd_vel_kick_m_s)),
        ("Peak L normal after TD (N)", _fmt(orig.peak_l_normal_after_td_n, 1),
         _fmt(flat.peak_l_normal_after_td_n, 1)),
        ("R lift-off step", orig.r_lift_off_step, flat.r_lift_off_step),
        ("R peak fwd displacement (mm)", _fmt(orig.peak_r_fwd_mm, 1),
         _fmt(flat.peak_r_fwd_mm, 1)),
        ("R peak rel L fwd (mm)", _fmt(orig.peak_r_rel_l_fwd_mm, 1),
         _fmt(flat.peak_r_rel_l_fwd_mm, 1)),
        ("L foot remains planted", orig.l_foot_remains_planted, flat.l_foot_remains_planted),
        ("L foot max drift (mm)", _fmt(orig.l_foot_max_drift_mm, 1),
         _fmt(flat.l_foot_max_drift_mm, 1)),
    ]

    print(f"\n{'Metric':<32} {'ORIGINAL':<20} {'FLAT-AT-LAST-MOMENT':<20}")
    print("-" * 72)
    for name, a, b in rows:
        print(f"{name:<32} {str(a):<20} {str(b):<20}")

    tc = flat.traj_check
    print("\n--- L HIP / L KNEE TRAJECTORY MATCH (until ankle trigger) ---")
    print(f"Ankle flatten trigger step = {tc.ankle_trigger_step}")
    print(f"Clearance at trigger (mm) = {tc.clearance_at_trigger_mm}")
    print(f"Steps compared vs original = {tc.steps_compared}")
    print(f"Max hip cmd diff (rad) = {tc.max_hip_cmd_diff:.6f}")
    print(f"Max knee cmd diff (rad) = {tc.max_knee_cmd_diff:.6f}")
    print(f"Max hip qpos diff (rad) = {tc.max_hip_qpos_diff:.6f}")
    print(f"Max knee qpos diff (rad) = {tc.max_knee_qpos_diff:.6f}")
    hip_knee_ok = (
        tc.max_hip_cmd_diff < 1e-4
        and tc.max_knee_cmd_diff < 1e-4
        and tc.ankle_trigger_step is not None
    )
    print(f"Hip/knee match before trigger: {'YES' if hip_knee_ok else 'CHECK'}")

    print("\nSingle fixed A/B experiment - no auto-tuning.")


def _find_mujoco_viewer_hwnd():
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    end = time.time() + VIEWER_POSITION_TIMEOUT_S
    while time.time() < end:
        found: list[int] = []

        def cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                n = user32.GetWindowTextLengthW(hwnd)
                if n > 0:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, buf, n + 1)
                    if buf.value.startswith(VIEWER_TITLE_PREFIX):
                        found.append(hwnd)
            return True

        fn = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(cb)
        user32.EnumWindows(fn, 0)
        if found:
            return found[0]
        time.sleep(0.05)
    return None


def _configure_viewer_window() -> None:
    if sys.platform != "win32":
        return
    hwnd = _find_mujoco_viewer_hwnd()
    if hwnd is None:
        return
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.wintypes.DWORD),
        ]

    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 1), ctypes.byref(info))
    wa = info.rcWork
    x = wa.left + (wa.right - wa.left - VIEWER_WIDTH) // 2
    y = wa.top + (wa.bottom - wa.top - VIEWER_HEIGHT) // 2
    user32.SetWindowPos(hwnd, 0, x, y, VIEWER_WIDTH, VIEWER_HEIGHT, 0x0004)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A/B: original catch vs last-moment flat L ankle + R recovery."
    )
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--original-only",
        action="store_true",
        help="Run only the original trajectory (no flat variant).",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Last-moment flat L foot touchdown A/B experiment")
    print(
        f"Trigger: sole clearance < {CLEARANCE_TRIGGER_MM:.0f} mm + descending, "
        f"contact=False (Z>{FOOT_CONTACT_Z})"
    )
    print(f"Flat target = +{L_ANKLE_FLAT_TARGET:.2f} rad (positive = flatter sole)")
    print("Viewer: robot only.\n")

    if args.headless:
        if args.original_only:
            orig = run_experiment(env, flat_ankle=False, viewer=None, quiet_prefix=True)
            flat = orig
        else:
            print("Recording original hip/knee reference for trajectory check...")
            ref_cmds = _record_catch_reference(env, viewer=None, slow=False)
            print(f"Running A: ORIGINAL ({len(ref_cmds)} catch steps recorded)...")
            orig = run_experiment(env, flat_ankle=False, viewer=None, quiet_prefix=True)
            print("Running B: FLAT-AT-LAST-MOMENT...")
            flat = run_experiment(
                env,
                flat_ankle=True,
                viewer=None,
                quiet_prefix=True,
                ref_cmds=ref_cmds,
            )
        print_ab_comparison(orig, flat)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        flat = run_experiment(env, flat_ankle=True, viewer=v, slow=args.slow)
    if not args.original_only:
        print("\n(Headless mode runs full A/B comparison. Viewer shows B only.)")
        print(f"B touchdown step = {flat.heel_touchdown_step}")
        print(f"B ankle trigger step = {flat.traj_check.ankle_trigger_step}")


if __name__ == "__main__":
    main(sys.argv[1:])
