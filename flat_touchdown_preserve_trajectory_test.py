"""Preserve golden staged_forward_catch trajectory; ankle-pitch only at last moment.

GOLDEN BASELINE: staged_forward_catch_test.py — prefix + catch lerp unchanged until
a late mesh-clearance trigger. Then ONLY L ankle PITCH is overridden (negative pitch
flattens sole per ankle_axis_diagnostic). No hip/knee/R-leg changes. No knee yield.

Evaluation only — does not modify robot.xml, biped_env, or staged_forward_catch_test.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np
from ctypes import wintypes

from biped_env import BipedalWalkEnv, DEFAULT_POSE

import staged_forward_catch_test as golden
from flat_foot_touchdown_test import (
    VIEWER_HEIGHT,
    VIEWER_POSITION_TIMEOUT_S,
    VIEWER_TITLE_PREFIX,
    VIEWER_WIDTH,
    _run_locked_prefix,
)
from foot_geometry import sole_metrics
from staged_forward_catch_continue_test import _is_heel_touchdown
from staged_forward_catch_test import (
    CATCH_MAX_STEPS,
    IDX_L_ANKLE_P,
    IDX_L_HIP_PITCH,
    IDX_L_KNEE,
    IDX_R_ANKLE_P,
    IDX_R_ANKLE_ROLL,
    IDX_R_HIP_PITCH,
    IDX_R_HIP_ROLL,
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
    _forward_vel,
    _lerp_ctrl,
    _print_phase,
    _reset,
    _sim_step,
    _smooth,
    _sync_viewer,
    left_leg_pose,
    rapid_shift_pose,
)

QPOS_L_ANKLE = 15

# Verified: negative ankle pitch flattens sole at catch pose (~-0.10 to -0.20).
FLAT_ANKLE_PITCH_TARGET = -0.15
ANKLE_PITCH_RAMP_STEP = 0.012
# Catch-phase mesh clearance stays low (~4–7 mm); trigger only in final approach.
MESH_CLEARANCE_TRIGGER_MM = 3.0
MIN_PEAK_MESH_CLEARANCE_MM = 5.0
POST_TD_STEPS = 120

R_LEG_IDX = (
    IDX_R_HIP_ROLL,
    IDX_R_HIP_PITCH,
    IDX_R_KNEE,
    IDX_R_ANKLE_P,
    IDX_R_ANKLE_ROLL,
)


@dataclass
class BaselineStep:
    ctrl: np.ndarray
    l_foot_xyz: np.ndarray
    l_hip_qpos: float
    l_knee_qpos: float
    l_ankle_qpos: float
    torso_pos: np.ndarray
    torso_quat: np.ndarray


@dataclass
class TrajLock:
    steps_compared: int = 0
    max_l_hip_ctrl_diff: float = 0.0
    max_l_knee_ctrl_diff: float = 0.0
    max_l_ankle_ctrl_diff: float = 0.0
    max_r_leg_ctrl_diff: float = 0.0
    max_l_foot_x_err_mm: float = 0.0
    max_l_foot_y_err_mm: float = 0.0
    max_l_foot_z_err_mm: float = 0.0
    max_l_hip_qpos_diff: float = 0.0
    max_l_knee_qpos_diff: float = 0.0
    max_l_ankle_qpos_diff: float = 0.0
    max_torso_pos_err_mm: float = 0.0
    max_torso_quat_err: float = 0.0
    ankle_trigger_step: int | None = None
    mesh_clearance_at_trigger_mm: float | None = None
    heel_toe_at_trigger_deg: float | None = None

    @property
    def max_pre_trigger_foot_trajectory_error_mm(self) -> float:
        return float(
            np.sqrt(
                self.max_l_foot_x_err_mm ** 2
                + self.max_l_foot_y_err_mm ** 2
                + self.max_l_foot_z_err_mm ** 2
            )
        )


@dataclass
class ExperimentMetrics:
    label: str
    diag: StepDiagnostics
    touchdown_step: int | None = None
    ankle_trigger_step: int | None = None
    peak_fwd_airborne_mm: float = 0.0
    heel_toe_pitch_at_td_deg: float | None = None
    sole_horiz_at_td_deg: float | None = None
    heel_clearance_at_td_mm: float | None = None
    toe_clearance_at_td_mm: float | None = None
    ankle_qpos_at_td: float | None = None
    hip_qpos_at_td: float | None = None
    knee_qpos_at_td: float | None = None
    torso_tilt_at_td_rad: float | None = None
    torso_fwd_vel_at_td: float | None = None
    peak_l_normal_after_td: float = 0.0
    l_foot_y_at_milestones: dict[str, float] = field(default_factory=dict)
    l_foot_z_at_milestones: dict[str, float] = field(default_factory=dict)
    traj_lock: TrajLock = field(default_factory=TrajLock)


def _snap_state(
    ref: dict[int, BaselineStep],
    step: int,
    st: RunState,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    ref[step] = BaselineStep(
        ctrl=st.ctrl.copy(),
        l_foot_xyz=_foot_pos(model, data, "L").copy(),
        l_hip_qpos=float(data.qpos[QPOS_L_HIP]),
        l_knee_qpos=float(data.qpos[QPOS_L_KNEE]),
        l_ankle_qpos=float(data.qpos[QPOS_L_ANKLE]),
        torso_pos=data.qpos[0:3].copy(),
        torso_quat=data.qpos[3:7].copy(),
    )


def _record_golden_reference(env: BipedalWalkEnv) -> tuple[dict[int, BaselineStep], StepDiagnostics, int | None]:
    """Replay full golden trajectory; record post-step state each step."""
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)
    lean = golden.forward_lean_pose()
    shifted = rapid_shift_pose()
    ctrl = DEFAULT_POSE.copy()
    stand_r_xy = _foot_pos(model, data, "R")[:2].copy()
    diag = StepDiagnostics()
    st = RunState(
        ctrl=ctrl,
        phase=Phase.STAND,
        stand_r_xy=stand_r_xy,
        diag=diag,
        knee_cmd=float(shifted[IDX_L_KNEE]),
        hip_cmd=float(shifted[IDX_L_HIP_PITCH]),
    )
    ref: dict[int, BaselineStep] = {}
    td_step: int | None = None

    for s in range(golden.STAND_STEPS):
        st.ctrl = _lerp_ctrl(st.ctrl, DEFAULT_POSE, _smooth((s + 1) / golden.STAND_STEPS), cr)
        _sim_step(env, model, data, st, None, False)
        _snap_state(ref, st.diag.global_step, st, model, data)

    for s in range(golden.FALL_RAMP_STEPS):
        alpha = (s + 1) / golden.FALL_RAMP_STEPS
        st.ctrl = _lerp_ctrl(st.ctrl, lean, alpha, cr)
        _sim_step(env, model, data, st, None, False)
        _snap_state(ref, st.diag.global_step, st, model, data)
    for _ in range(golden.FALL_MOMENTUM_STEPS):
        st.ctrl = lean.copy()
        _sim_step(env, model, data, st, None, False)
        _snap_state(ref, st.diag.global_step, st, model, data)

    shift_start = st.ctrl.copy()
    for s in range(golden.RAPID_SHIFT_STEPS):
        alpha = (s + 1) / golden.RAPID_SHIFT_STEPS
        st.ctrl = _lerp_ctrl(shift_start, shifted, alpha, cr)
        st.knee_cmd = float(st.ctrl[IDX_L_KNEE])
        st.hip_cmd = float(st.ctrl[IDX_L_HIP_PITCH])
        _sim_step(env, model, data, st, None, False)
        _snap_state(ref, st.diag.global_step, st, model, data)

    post = 0
    while post < golden.POST_SHIFT_MAX_STEPS and st.phase == Phase.RAPID_SHIFT:
        st.ctrl = shifted.copy()
        st.knee_cmd = float(st.ctrl[IDX_L_KNEE])
        st.hip_cmd = float(st.ctrl[IDX_L_HIP_PITCH])
        _sim_step(env, model, data, st, None, False)
        _snap_state(ref, st.diag.global_step, st, model, data)
        post += 1

    st.phase = Phase.KNEE_LIFT
    while st.knee_lift_steps < golden.KNEE_LIFT_MAX_STEPS and st.phase == Phase.KNEE_LIFT:
        if golden._is_unloaded(model, data):
            st.clearance_knee = st.knee_cmd
            st.diag.clearance_knee_cmd = st.knee_cmd
            st.phase = Phase.HIP_SWING
            st.diag.hip_swing_start_step = st.diag.global_step + 1
            break
        st.knee_cmd = max(st.knee_cmd - golden.KNEE_LIFT_STEP, golden.KNEE_LIFT_TARGET)
        st.ctrl = left_leg_pose(st.knee_cmd, st.hip_cmd)
        _sim_step(env, model, data, st, None, False)
        _snap_state(ref, st.diag.global_step, st, model, data)
        st.knee_lift_steps += 1

    if st.phase == Phase.KNEE_LIFT:
        st.clearance_knee = st.knee_cmd
        st.diag.clearance_knee_cmd = st.knee_cmd
        st.phase = Phase.HIP_SWING
        st.diag.hip_swing_start_step = st.diag.global_step + 1

    while st.hip_swing_steps < golden.HIP_SWING_MAX_STEPS and st.phase == Phase.HIP_SWING:
        if st.airborne_ref_y is None and not _foot_contact(model, data, "L"):
            st.airborne_ref_y = float(_foot_pos(model, data, "L")[1])
        knee_hold = st.clearance_knee if st.clearance_knee is not None else st.knee_cmd
        st.hip_cmd = max(st.hip_cmd - golden.HIP_RAMP_PER_STEP, golden.HIP_SWING_TARGET)
        st.ctrl = left_leg_pose(knee_hold, st.hip_cmd)
        _sim_step(env, model, data, st, None, False)
        _snap_state(ref, st.diag.global_step, st, model, data)
        st.hip_swing_steps += 1
        fwd_air = 0.0
        if st.airborne_ref_y is not None:
            fwd_air = golden._forward_mm(_foot_pos(model, data, "L")[1], st.airborne_ref_y)
        hip_at_target = st.hip_cmd <= golden.HIP_SWING_TARGET + 1e-6
        if fwd_air >= golden.MIN_FWD_AIRBORNE_MM and not _foot_contact(model, data, "L"):
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

    catch_start = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l = _foot_contact(model, data, "L")
    was_air = not prev_l

    while st.catch_steps < CATCH_MAX_STEPS:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        st.ctrl = _lerp_ctrl(catch_start, catch_target, alpha, cr)
        _sim_step(env, model, data, st, None, False)
        _snap_state(ref, st.diag.global_step, st, model, data)
        st.catch_steps += 1
        if _is_heel_touchdown(
            model, data,
            catch_started=True, was_airborne_in_catch=was_air, prev_l_contact=prev_l,
        ):
            td_step = st.diag.global_step
            break
        prev_l = _foot_contact(model, data, "L")
        if not prev_l:
            was_air = True

    return ref, diag, td_step


def _compare_pre_trigger(
    lock: TrajLock,
    step: int,
    ref: BaselineStep,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: np.ndarray,
) -> None:
    lock.steps_compared += 1
    lock.max_l_hip_ctrl_diff = max(
        lock.max_l_hip_ctrl_diff, abs(float(ctrl[IDX_L_HIP_PITCH]) - float(ref.ctrl[IDX_L_HIP_PITCH]))
    )
    lock.max_l_knee_ctrl_diff = max(
        lock.max_l_knee_ctrl_diff, abs(float(ctrl[IDX_L_KNEE]) - float(ref.ctrl[IDX_L_KNEE]))
    )
    lock.max_l_ankle_ctrl_diff = max(
        lock.max_l_ankle_ctrl_diff, abs(float(ctrl[IDX_L_ANKLE_P]) - float(ref.ctrl[IDX_L_ANKLE_P]))
    )
    for idx in R_LEG_IDX:
        lock.max_r_leg_ctrl_diff = max(
            lock.max_r_leg_ctrl_diff, abs(float(ctrl[idx]) - float(ref.ctrl[idx]))
        )

    l_pos = _foot_pos(model, data, "L")
    lock.max_l_foot_x_err_mm = max(lock.max_l_foot_x_err_mm, abs(l_pos[0] - ref.l_foot_xyz[0]) * 1000.0)
    lock.max_l_foot_y_err_mm = max(lock.max_l_foot_y_err_mm, abs(l_pos[1] - ref.l_foot_xyz[1]) * 1000.0)
    lock.max_l_foot_z_err_mm = max(lock.max_l_foot_z_err_mm, abs(l_pos[2] - ref.l_foot_xyz[2]) * 1000.0)

    lock.max_l_hip_qpos_diff = max(
        lock.max_l_hip_qpos_diff, abs(float(data.qpos[QPOS_L_HIP]) - ref.l_hip_qpos)
    )
    lock.max_l_knee_qpos_diff = max(
        lock.max_l_knee_qpos_diff, abs(float(data.qpos[QPOS_L_KNEE]) - ref.l_knee_qpos)
    )
    lock.max_l_ankle_qpos_diff = max(
        lock.max_l_ankle_qpos_diff, abs(float(data.qpos[QPOS_L_ANKLE]) - ref.l_ankle_qpos)
    )

    torso_pos = data.qpos[0:3]
    lock.max_torso_pos_err_mm = max(
        lock.max_torso_pos_err_mm, float(np.linalg.norm(torso_pos - ref.torso_pos) * 1000.0)
    )
    lock.max_torso_quat_err = max(
        lock.max_torso_quat_err, float(np.linalg.norm(data.qpos[3:7] - ref.torso_quat))
    )


def _mesh_min_clearance_mm(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    m = sole_metrics(model, data)
    return float(min(m["heel_clearance_mm"], m["toe_clearance_mm"]))


def _approach_trigger(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    peak_mesh_clearance_mm: float,
) -> bool:
    if _foot_contact(model, data, "L"):
        return False
    clr = _mesh_min_clearance_mm(model, data)
    return clr < MESH_CLEARANCE_TRIGGER_MM and peak_mesh_clearance_mm >= MIN_PEAK_MESH_CLEARANCE_MM


def run_preserve_trajectory_experiment(
    env: BipedalWalkEnv,
    ref: dict[int, BaselineStep],
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    verbose: bool = True,
) -> ExperimentMetrics:
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
    metrics = ExperimentMetrics(label="ANKLE-LAST-MOMENT", diag=st.diag)
    lock = metrics.traj_lock

    _run_locked_prefix(env, model, data, cr, st, viewer, slow, verbose=verbose)
    if verbose:
        _print_phase("CATCH — ANKLE PITCH ONLY (trajectory locked)")

    catch_start = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l = _foot_contact(model, data, "L")
    was_air = not prev_l

    ankle_active = False
    ankle_cmd = float(catch_start[IDX_L_ANKLE_P])
    peak_mesh_clr = 0.0

    while st.catch_steps < CATCH_MAX_STEPS:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        baseline_ctrl = _lerp_ctrl(catch_start, catch_target, alpha, cr)
        step_next = st.diag.global_step + 1

        if not _foot_contact(model, data, "L"):
            peak_mesh_clr = max(peak_mesh_clr, _mesh_min_clearance_mm(model, data))

        if not ankle_active:
            if _approach_trigger(model, data, peak_mesh_clr):
                ankle_active = True
                lock.ankle_trigger_step = step_next
                metrics.ankle_trigger_step = step_next
                m = sole_metrics(model, data)
                lock.mesh_clearance_at_trigger_mm = _mesh_min_clearance_mm(model, data)
                lock.heel_toe_at_trigger_deg = float(np.degrees(m["heel_toe_pitch_rad"]))
                if verbose:
                    print(
                        f"\n  >> ANKLE TRIGGER step {step_next} "
                        f"mesh_clr={lock.mesh_clearance_at_trigger_mm:.1f}mm "
                        f"heel-toe={lock.heel_toe_at_trigger_deg:.1f}° "
                        f"baseline_ankle_cmd={baseline_ctrl[IDX_L_ANKLE_P]:.3f}"
                    )
            ankle_cmd = float(baseline_ctrl[IDX_L_ANKLE_P])
        else:
            step_dir = -1.0 if FLAT_ANKLE_PITCH_TARGET < ankle_cmd else 1.0
            if abs(ankle_cmd - FLAT_ANKLE_PITCH_TARGET) > 1e-6:
                ankle_cmd += step_dir * ANKLE_PITCH_RAMP_STEP
                if step_dir < 0:
                    ankle_cmd = max(ankle_cmd, FLAT_ANKLE_PITCH_TARGET)
                else:
                    ankle_cmd = min(ankle_cmd, FLAT_ANKLE_PITCH_TARGET)

        st.ctrl = baseline_ctrl.copy()
        st.ctrl[IDX_L_ANKLE_P] = ankle_cmd

        _sim_step(env, model, data, st, viewer, slow)
        st.catch_steps += 1
        step_now = st.diag.global_step

        if step_now in ref:
            r = ref[step_now]
            if not ankle_active:
                _compare_pre_trigger(lock, step_now, r, model, data, st.ctrl)
            else:
                lock.max_l_hip_ctrl_diff = max(
                    lock.max_l_hip_ctrl_diff,
                    abs(float(st.ctrl[IDX_L_HIP_PITCH]) - float(r.ctrl[IDX_L_HIP_PITCH])),
                )
                lock.max_l_knee_ctrl_diff = max(
                    lock.max_l_knee_ctrl_diff,
                    abs(float(st.ctrl[IDX_L_KNEE]) - float(r.ctrl[IDX_L_KNEE])),
                )
                for idx in R_LEG_IDX:
                    lock.max_r_leg_ctrl_diff = max(
                        lock.max_r_leg_ctrl_diff, abs(float(st.ctrl[idx]) - float(r.ctrl[idx]))
                    )

        if _is_heel_touchdown(
            model, data,
            catch_started=True, was_airborne_in_catch=was_air, prev_l_contact=prev_l,
        ):
            metrics.touchdown_step = st.diag.global_step
            m = sole_metrics(model, data)
            metrics.heel_toe_pitch_at_td_deg = float(np.degrees(m["heel_toe_pitch_rad"]))
            metrics.sole_horiz_at_td_deg = float(np.degrees(m["sole_angle_from_horizontal_rad"]))
            metrics.heel_clearance_at_td_mm = float(m["heel_clearance_mm"])
            metrics.toe_clearance_at_td_mm = float(m["toe_clearance_mm"])
            metrics.ankle_qpos_at_td = float(data.qpos[QPOS_L_ANKLE])
            metrics.hip_qpos_at_td = float(data.qpos[QPOS_L_HIP])
            metrics.knee_qpos_at_td = float(data.qpos[QPOS_L_KNEE])
            metrics.torso_tilt_at_td_rad = env._quat_tilt_rad()
            metrics.torso_fwd_vel_at_td = _forward_vel(data)
            if verbose:
                print(
                    f"\n  >> TOUCHDOWN step {metrics.touchdown_step} "
                    f"heel-toe={metrics.heel_toe_pitch_at_td_deg:.1f}° "
                    f"sole={metrics.sole_horiz_at_td_deg:.1f}° "
                    f"heel={metrics.heel_clearance_at_td_mm:.1f}mm "
                    f"toe={metrics.toe_clearance_at_td_mm:.1f}mm"
                )
            hold = st.ctrl.copy()
            for _ in range(POST_TD_STEPS):
                st.ctrl = hold.copy()
                data.ctrl[:15] = st.ctrl
                mujoco.mj_step(model, data)
                metrics.peak_l_normal_after_td = max(
                    metrics.peak_l_normal_after_td, _foot_normal_force(model, data, "L")
                )
                _sync_viewer(viewer, slow)
            break

        prev_l = _foot_contact(model, data, "L")
        if not prev_l:
            was_air = True

    metrics.peak_fwd_airborne_mm = st.diag.peak_fwd_airborne_mm
    return metrics


def _golden_metrics(
    env: BipedalWalkEnv,
    golden_ref: dict[int, BaselineStep],
    golden_diag: StepDiagnostics,
    heel_td_step: int | None,
) -> ExperimentMetrics:
    m = ExperimentMetrics(label="GOLDEN BASELINE", diag=golden_diag)
    m.touchdown_step = heel_td_step
    m.peak_fwd_airborne_mm = golden_diag.peak_fwd_airborne_mm

    if heel_td_step is not None and heel_td_step in golden_ref:
        bs = golden_ref[heel_td_step]
        m.ankle_qpos_at_td = bs.l_ankle_qpos
        m.hip_qpos_at_td = bs.l_hip_qpos
        m.knee_qpos_at_td = bs.l_knee_qpos
        m.torso_tilt_at_td_rad = golden_diag.touchdown_torso_tilt_rad

    if heel_td_step is not None:
        _reset(env.model, env.data)
        shifted = rapid_shift_pose()
        st = RunState(
            ctrl=DEFAULT_POSE.copy(),
            phase=Phase.STAND,
            stand_r_xy=_foot_pos(env.model, env.data, "R")[:2].copy(),
            diag=StepDiagnostics(),
            knee_cmd=float(shifted[IDX_L_KNEE]),
            hip_cmd=float(shifted[IDX_L_HIP_PITCH]),
        )
        cr = env.model.actuator_ctrlrange[:15]
        _run_locked_prefix(env, env.model, env.data, cr, st, None, False, verbose=False)
        catch_start = st.ctrl.copy()
        catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
        st.phase = Phase.CATCH
        prev_l = _foot_contact(env.model, env.data, "L")
        was_air = not prev_l
        sm = sole_metrics(env.model, env.data)
        while st.catch_steps < CATCH_MAX_STEPS:
            alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
            st.ctrl = _lerp_ctrl(catch_start, catch_target, alpha, cr)
            _sim_step(env, env.model, env.data, st, None, False)
            st.catch_steps += 1
            if st.diag.global_step == heel_td_step:
                sm = sole_metrics(env.model, env.data)
                m.torso_fwd_vel_at_td = _forward_vel(env.data)
                m.torso_tilt_at_td_rad = env._quat_tilt_rad()
                hold = st.ctrl.copy()
                for _ in range(POST_TD_STEPS):
                    env.data.ctrl[:15] = hold
                    mujoco.mj_step(env.model, env.data)
                    m.peak_l_normal_after_td = max(
                        m.peak_l_normal_after_td,
                        _foot_normal_force(env.model, env.data, "L"),
                    )
                break
            if _is_heel_touchdown(
                env.model, env.data,
                catch_started=True, was_airborne_in_catch=was_air, prev_l_contact=prev_l,
            ):
                sm = sole_metrics(env.model, env.data)
                m.torso_fwd_vel_at_td = _forward_vel(env.data)
                m.torso_tilt_at_td_rad = env._quat_tilt_rad()
                hold = st.ctrl.copy()
                for _ in range(POST_TD_STEPS):
                    env.data.ctrl[:15] = hold
                    mujoco.mj_step(env.model, env.data)
                    m.peak_l_normal_after_td = max(
                        m.peak_l_normal_after_td,
                        _foot_normal_force(env.model, env.data, "L"),
                    )
                break
            prev_l = _foot_contact(env.model, env.data, "L")
            if not prev_l:
                was_air = True

        m.heel_toe_pitch_at_td_deg = float(np.degrees(sm["heel_toe_pitch_rad"]))
        m.sole_horiz_at_td_deg = float(np.degrees(sm["sole_angle_from_horizontal_rad"]))
        m.heel_clearance_at_td_mm = float(sm["heel_clearance_mm"])
        m.toe_clearance_at_td_mm = float(sm["toe_clearance_mm"])

    return m


def _foot_y_z_at_step(ref: dict[int, BaselineStep], step: int | None) -> tuple[float | None, float | None]:
    if step is None or step not in ref:
        return None, None
    p = ref[step].l_foot_xyz
    return float(p[1]), float(p[2])


def _milestone_foot_yz(
    diag: StepDiagnostics,
    ref: dict[int, BaselineStep],
    trigger_step: int | None = None,
    td_step: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    ys: dict[str, float] = {}
    zs: dict[str, float] = {}
    peak_step = None
    peak_fwd = -1.0
    for s in diag.swing_traj:
        if s.fwd_airborne_mm > peak_fwd:
            peak_fwd = s.fwd_airborne_mm
            peak_step = s.step
    milestones = {
        "contact_loss": diag.contact_loss_step,
        "hip_swing_start": diag.hip_swing_start_step,
        "catch_start": diag.catch_start_step,
        "peak_fwd_airborne": peak_step,
        "ankle_trigger": trigger_step,
        "touchdown": td_step,
    }
    for name, step in milestones.items():
        y, z = _foot_y_z_at_step(ref, step)
        if y is not None:
            ys[name] = y
        if z is not None:
            zs[name] = z
    return ys, zs


def print_ab_comparison(
    golden_m: ExperimentMetrics,
    mod_m: ExperimentMetrics,
    golden_ref: dict[int, BaselineStep],
) -> None:
    g_ys, g_zs = _milestone_foot_yz(
        golden_m.diag, golden_ref, td_step=golden_m.touchdown_step
    )
    m_ys, m_zs = _milestone_foot_yz(
        mod_m.diag, golden_ref,
        trigger_step=mod_m.ankle_trigger_step,
        td_step=mod_m.touchdown_step,
    )
    golden_m.l_foot_y_at_milestones = g_ys
    golden_m.l_foot_z_at_milestones = g_zs
    mod_m.l_foot_y_at_milestones = m_ys
    mod_m.l_foot_z_at_milestones = m_zs

    lock = mod_m.traj_lock
    print("\n" + "=" * 76)
    print("A/B: GOLDEN BASELINE vs ANKLE-LAST-MOMENT (trajectory preserved)")
    print("=" * 76)
    print(f"FLAT_ANKLE_PITCH_TARGET = {FLAT_ANKLE_PITCH_TARGET:.2f} rad")
    print(f"MESH_CLEARANCE_TRIGGER = {MESH_CLEARANCE_TRIGGER_MM:.1f} mm (heel/toe mesh)")

    rows = [
        ("Touchdown step", golden_m.touchdown_step, mod_m.touchdown_step),
        ("Ankle trigger step", None, mod_m.ankle_trigger_step),
        ("Peak airborne L-foot fwd (mm)", golden_m.peak_fwd_airborne_mm, mod_m.peak_fwd_airborne_mm),
        ("Heel-toe pitch at TD (deg)", golden_m.heel_toe_pitch_at_td_deg, mod_m.heel_toe_pitch_at_td_deg),
        ("Sole angle at TD (deg)", golden_m.sole_horiz_at_td_deg, mod_m.sole_horiz_at_td_deg),
        ("Heel clearance at TD (mm)", golden_m.heel_clearance_at_td_mm, mod_m.heel_clearance_at_td_mm),
        ("Toe clearance at TD (mm)", golden_m.toe_clearance_at_td_mm, mod_m.toe_clearance_at_td_mm),
        ("Ankle qpos at TD", golden_m.ankle_qpos_at_td, mod_m.ankle_qpos_at_td),
        ("Hip qpos at TD", golden_m.hip_qpos_at_td, mod_m.hip_qpos_at_td),
        ("Knee qpos at TD", golden_m.knee_qpos_at_td, mod_m.knee_qpos_at_td),
        ("Torso tilt at TD (rad)", golden_m.torso_tilt_at_td_rad, mod_m.torso_tilt_at_td_rad),
        ("Torso fwd vel at TD (m/s)", golden_m.torso_fwd_vel_at_td, mod_m.torso_fwd_vel_at_td),
        ("Peak L normal after TD (N)", golden_m.peak_l_normal_after_td, mod_m.peak_l_normal_after_td),
    ]
    print(f"\n{'Metric':<36} {'GOLDEN':<18} {'MODIFIED':<18}")
    print("-" * 72)
    for name, a, b in rows:
        print(f"{name:<36} {str(a):<18} {str(b):<18}")

    print("\n--- L-FOOT Y AT MILESTONES (world Y; forward = -Y) ---")
    for key in ("contact_loss", "hip_swing_start", "catch_start", "peak_fwd_airborne", "ankle_trigger", "touchdown"):
        gy = g_ys.get(key)
        my = m_ys.get(key)
        print(f"  {key:<22} golden={gy}  modified={my}")

    print("\n--- L-FOOT Z AT MILESTONES ---")
    for key in ("contact_loss", "hip_swing_start", "catch_start", "peak_fwd_airborne", "ankle_trigger", "touchdown"):
        gz = g_zs.get(key)
        mz = m_zs.get(key)
        print(f"  {key:<22} golden={gz}  modified={mz}")

    print("\n--- TRAJECTORY LOCK (pre ankle trigger) ---")
    print(f"steps_compared = {lock.steps_compared}")
    print(f"max L hip ctrl diff = {lock.max_l_hip_ctrl_diff:.6f}")
    print(f"max L knee ctrl diff = {lock.max_l_knee_ctrl_diff:.6f}")
    print(f"max L ankle ctrl diff = {lock.max_l_ankle_ctrl_diff:.6f}")
    print(f"max R leg ctrl diff = {lock.max_r_leg_ctrl_diff:.6f}")
    print(f"MAX_PRE_TRIGGER_FOOT_TRAJECTORY_ERROR (mm) = {lock.max_pre_trigger_foot_trajectory_error_mm:.4f}")
    print(f"  max L foot X err (mm) = {lock.max_l_foot_x_err_mm:.4f}")
    print(f"  max L foot Y err (mm) = {lock.max_l_foot_y_err_mm:.4f}")
    print(f"  max L foot Z err (mm) = {lock.max_l_foot_z_err_mm:.4f}")
    print(f"max L hip qpos diff = {lock.max_l_hip_qpos_diff:.6f}")
    print(f"max L knee qpos diff = {lock.max_l_knee_qpos_diff:.6f}")
    print(f"max L ankle qpos diff = {lock.max_l_ankle_qpos_diff:.6f}")
    print(f"max torso pos err (mm) = {lock.max_torso_pos_err_mm:.4f}")

    prefix_ok = (
        lock.max_l_hip_ctrl_diff < 1e-6
        and lock.max_l_knee_ctrl_diff < 1e-6
        and lock.max_l_ankle_ctrl_diff < 1e-6
        and lock.max_r_leg_ctrl_diff < 1e-6
        and lock.max_pre_trigger_foot_trajectory_error_mm < 0.5
    )
    print(f"\nPrefix genuinely locked: {'YES' if prefix_ok else 'NO — do not tune ankle yet'}")
    ht_improved = (
        golden_m.heel_toe_pitch_at_td_deg is not None
        and mod_m.heel_toe_pitch_at_td_deg is not None
        and abs(mod_m.heel_toe_pitch_at_td_deg) < abs(golden_m.heel_toe_pitch_at_td_deg)
    )
    print(f"Heel-toe leveling improved: {'YES' if ht_improved else 'NO'}")
    print("\nVisual: python flat_touchdown_preserve_trajectory_test.py --slow")


def _configure_viewer_window() -> None:
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.wintypes.DWORD), ("rcMonitor", RECT),
                    ("rcWork", RECT), ("dwFlags", ctypes.wintypes.DWORD)]

    hwnd = None
    end = time.time() + VIEWER_POSITION_TIMEOUT_S
    while time.time() < end:
        found: list[int] = []

        def cb(h, _):
            if user32.IsWindowVisible(h):
                n = user32.GetWindowTextLengthW(h)
                if n > 0:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(h, buf, n + 1)
                    if buf.value.startswith(VIEWER_TITLE_PREFIX):
                        found.append(h)
            return True

        fn = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(cb)
        user32.EnumWindows(fn, 0)
        if found:
            hwnd = found[0]
            break
        time.sleep(0.05)
    if hwnd is None:
        return
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 1), ctypes.byref(info))
    wa = info.rcWork
    x = wa.left + (wa.right - wa.left - VIEWER_WIDTH) // 2
    y = wa.top + (wa.bottom - wa.top - VIEWER_HEIGHT) // 2
    user32.SetWindowPos(hwnd, 0, x, y, VIEWER_WIDTH, VIEWER_HEIGHT, 0x0004)


def _hold_viewer(viewer: mujoco.viewer.Handle, slow: bool) -> None:
    if not viewer.is_running():
        return
    print("\nClose viewer window to exit.")
    while viewer.is_running():
        viewer.sync()
        time.sleep(golden.SLOW_SLEEP_S if slow else golden.NORMAL_SLEEP_S)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Golden trajectory + last-moment L ankle pitch only."
    )
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()
    print("Flat touchdown — preserve golden swing, ankle pitch at last moment")
    print("Golden baseline: staged_forward_catch_test.py\n")

    if args.headless:
        print("Recording golden reference trajectory...")
        golden_ref, golden_diag, golden_td = _record_golden_reference(env)
        print(f"  {len(golden_ref)} steps recorded, heel TD step = {golden_td}")

        print("Running A: GOLDEN BASELINE metrics...")
        golden_m = _golden_metrics(env, golden_ref, golden_diag, golden_td)

        print("Running B: ANKLE-LAST-MOMENT...")
        mod_m = run_preserve_trajectory_experiment(
            env, golden_ref, viewer=None, slow=False, verbose=False
        )
        print_ab_comparison(golden_m, mod_m, golden_ref)
        return

    print("Recording golden reference for trajectory lock...")
    golden_ref, _, _ = _record_golden_reference(env)
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        mod_m = run_preserve_trajectory_experiment(
            env, golden_ref, v, args.slow, verbose=True
        )
        _hold_viewer(v, args.slow)
    print(f"\nAnkle trigger step = {mod_m.ankle_trigger_step}")
    print(f"Touchdown step = {mod_m.touchdown_step}")
    print(f"MAX_PRE_TRIGGER_FOOT_TRAJECTORY_ERROR = "
          f"{mod_m.traj_lock.max_pre_trigger_foot_trajectory_error_mm:.4f} mm")


if __name__ == "__main__":
    main(sys.argv[1:])
