"""Simultaneous flat ankle pitch + knee yield before L touchdown.

LOCKED: staged_forward_catch_test prefix + catch hip/knee lerp until imminent contact.
NEW at sole clearance < 15 mm (descending, airborne):
  - rapid L ankle PITCH ramp toward +0.38 rad
  - simultaneous L knee yield ~-0.15 rad over 20 steps
  - L hip frozen at trigger; R leg unchanged.

Compare against l_knee_yield_touchdown_test (10 mm ankle, knee at contact).
Evaluation only — does not modify robot.xml, biped_env, or other scripts.
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

from flat_foot_touchdown_test import (
    SOLE_DESCENT_VEL_M_S,
    VIEWER_HEIGHT,
    VIEWER_POSITION_TIMEOUT_S,
    VIEWER_TITLE_PREFIX,
    VIEWER_WIDTH,
    _foot_pitch_rad,
    _l_foot_vert_vel_m_s,
    _run_locked_prefix,
    _sole_clearance_mm,
)
from l_knee_yield_touchdown_test import (
    YieldDiagnostics as BaselineDiagnostics,
    _record_flat_catch_reference,
    _torso_angvel_pitch_rad_s,
    run_knee_yield_experiment,
)
from staged_forward_catch_test import (
    CATCH_MAX_STEPS,
    IDX_L_ANKLE_P,
    IDX_L_HIP_PITCH,
    IDX_L_KNEE,
    L_ANKLE_CATCH,
    L_HIP_CATCH,
    L_KNEE_CATCH,
    Phase,
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
from staged_forward_catch_continue_test import _is_heel_touchdown, _r_load_fraction

QPOS_L_ANKLE = 15

# Imminent-contact trigger (this experiment only).
IMMINENT_CLEARANCE_MM = 15.0
L_ANKLE_FLAT_TARGET = 0.38
ANKLE_FLAT_RAMP_STEP = 0.025
KNEE_YIELD_DELTA_RAD = -0.15
KNEE_YIELD_STEPS = 20
POST_TOUCHDOWN_OBSERVE_STEPS = 100
L_PLANT_DRIFT_THRESH_MM = 25.0
FLAT_PITCH_IMPROVEMENT_RAD = 0.10


@dataclass
class ApproachSample:
    step: int
    clearance_mm: float
    foot_pitch_rad: float
    ankle_cmd: float
    ankle_qpos: float
    knee_cmd: float
    knee_qpos: float
    l_normal_n: float
    r_normal_n: float
    torso_tilt_rad: float
    torso_angvel_rad_s: float
    fwd_vel_m_s: float
    contact: bool


@dataclass
class TrajCheck:
    max_hip_cmd_diff: float = 0.0
    max_knee_cmd_diff: float = 0.0
    max_ankle_cmd_diff: float = 0.0
    steps_compared: int = 0
    trigger_step: int | None = None
    clearance_at_trigger_mm: float | None = None


@dataclass
class SimultaneousDiagnostics:
    base: StepDiagnostics
    traj_check: TrajCheck = field(default_factory=TrajCheck)
    approach_log: list[ApproachSample] = field(default_factory=list)
    trigger_step: int | None = None
    heel_touchdown_step: int | None = None
    foot_pitch_before_contact_rad: float | None = None
    foot_pitch_at_td_rad: float | None = None
    ankle_cmd_at_td: float | None = None
    ankle_qpos_at_td: float | None = None
    knee_qpos_before_td: float | None = None
    knee_qpos_after_prep_rad: float | None = None
    torso_fwd_vel_before_td_m_s: float | None = None
    torso_fwd_vel_after_td_m_s: float | None = None
    fwd_vel_kick_m_s: float | None = None
    peak_l_normal_n: float = 0.0
    peak_r_normal_n: float = 0.0
    l_foot_max_drift_mm: float = 0.0
    l_foot_remains_planted: bool = False
    r_foot_in_contact_at_td: bool = False
    r_foot_in_contact_at_end: bool = False
    foot_flattened_before_contact: bool = False


def _record_catch_lerp_reference(env: BipedalWalkEnv) -> dict[int, tuple[float, float, float]]:
    """Pure catch lerp reference (no ankle/knee prep) for trajectory verification."""
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
    ref: dict[int, tuple[float, float, float]] = {}

    while st.catch_steps < CATCH_MAX_STEPS:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        st.ctrl = _lerp_ctrl(catch_start_ctrl, catch_target, alpha, cr)
        step = st.diag.global_step + 1
        ref[step] = (
            float(st.ctrl[IDX_L_HIP_PITCH]),
            float(st.ctrl[IDX_L_KNEE]),
            float(st.ctrl[IDX_L_ANKLE_P]),
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


def _log_approach(
    diag: SimultaneousDiagnostics,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    env: BipedalWalkEnv,
    step: int,
) -> None:
    diag.approach_log.append(
        ApproachSample(
            step=step,
            clearance_mm=_sole_clearance_mm(model, data),
            foot_pitch_rad=_foot_pitch_rad(model, data),
            ankle_cmd=float(data.ctrl[IDX_L_ANKLE_P]),
            ankle_qpos=float(data.qpos[QPOS_L_ANKLE]),
            knee_cmd=float(data.ctrl[IDX_L_KNEE]),
            knee_qpos=float(data.qpos[QPOS_L_KNEE]),
            l_normal_n=_foot_normal_force(model, data, "L"),
            r_normal_n=_foot_normal_force(model, data, "R"),
            torso_tilt_rad=env._quat_tilt_rad(),
            torso_angvel_rad_s=_torso_angvel_pitch_rad_s(data),
            fwd_vel_m_s=_forward_vel(data),
            contact=_foot_contact(model, data, "L"),
        )
    )


def run_simultaneous_experiment(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    ref_cmds: dict[int, tuple[float, float, float]] | None = None,
    *,
    verbose: bool = True,
) -> SimultaneousDiagnostics:
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
    diag = SimultaneousDiagnostics(base=st.diag)

    _run_locked_prefix(env, model, data, cr, st, viewer, slow, verbose=verbose)

    if verbose:
        _print_phase("SIMULTANEOUS FLAT ANKLE + KNEE YIELD (IMMINENT CONTACT)")

    catch_start_ctrl = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l_contact = _foot_contact(model, data, "L")
    was_airborne_in_catch = not prev_l_contact

    ankle_cmd = float(catch_start_ctrl[IDX_L_ANKLE_P])
    prep_active = False
    hip_hold = 0.0
    knee_start = 0.0
    prep_steps = 0
    stance_ctrl: np.ndarray | None = None
    heel_l_xy: np.ndarray | None = None
    pre_td_vel: float | None = None
    pitch_at_trigger: float | None = None
    first_contact_logged = False

    while st.catch_steps < CATCH_MAX_STEPS:
        if pre_td_vel is None and not _foot_contact(model, data, "L"):
            pre_td_vel = _forward_vel(data)

        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        full_lerp = _lerp_ctrl(catch_start_ctrl, catch_target, alpha, cr)
        knee_val = float(full_lerp[IDX_L_KNEE])
        hip_val = float(full_lerp[IDX_L_HIP_PITCH])

        if ref_cmds is not None and not prep_active:
            step = st.diag.global_step + 1
            if step in ref_cmds:
                ref_hip, ref_knee, ref_ankle = ref_cmds[step]
                tc = diag.traj_check
                tc.max_hip_cmd_diff = max(tc.max_hip_cmd_diff, abs(hip_val - ref_hip))
                tc.max_knee_cmd_diff = max(tc.max_knee_cmd_diff, abs(knee_val - ref_knee))
                tc.max_ankle_cmd_diff = max(tc.max_ankle_cmd_diff, abs(ankle_cmd - ref_ankle))
                tc.steps_compared += 1

        if not prep_active:
            if not _foot_contact(model, data, "L"):
                clearance = _sole_clearance_mm(model, data)
                descending = _l_foot_vert_vel_m_s(model, data) < -SOLE_DESCENT_VEL_M_S
                if clearance < IMMINENT_CLEARANCE_MM and descending:
                    prep_active = True
                    diag.trigger_step = st.diag.global_step + 1
                    diag.traj_check.trigger_step = diag.trigger_step
                    diag.traj_check.clearance_at_trigger_mm = clearance
                    hip_hold = hip_val
                    knee_start = knee_val
                    stance_ctrl = st.ctrl.copy()
                    pitch_at_trigger = _foot_pitch_rad(model, data)
                    prep_steps = 0
                    if verbose:
                        print(
                            f"\n  >> PREP TRIGGER step {diag.trigger_step} "
                            f"clearance={clearance:.1f} mm pitch={pitch_at_trigger:.3f} rad"
                        )
            st.ctrl = left_leg_pose(knee_val, hip_val, ankle_cmd)
        else:
            prep_steps += 1
            t = _smooth(min(prep_steps, KNEE_YIELD_STEPS) / KNEE_YIELD_STEPS)
            knee_cmd = knee_start + t * KNEE_YIELD_DELTA_RAD
            ankle_cmd = min(ankle_cmd + ANKLE_FLAT_RAMP_STEP, L_ANKLE_FLAT_TARGET)
            st.ctrl = left_leg_pose(knee_cmd, hip_hold, ankle_cmd)
            if stance_ctrl is not None:
                for idx in range(15):
                    if idx not in (IDX_L_KNEE, IDX_L_HIP_PITCH, IDX_L_ANKLE_P):
                        st.ctrl[idx] = stance_ctrl[idx]

        _sim_step(env, model, data, st, viewer, slow)
        st.catch_steps += 1
        step_now = st.diag.global_step

        if prep_active or _sole_clearance_mm(model, data) <= IMMINENT_CLEARANCE_MM:
            _log_approach(diag, model, data, env, step_now)

        l_nf = _foot_normal_force(model, data, "L")
        r_nf = _foot_normal_force(model, data, "R")
        diag.peak_l_normal_n = max(diag.peak_l_normal_n, l_nf)
        diag.peak_r_normal_n = max(diag.peak_r_normal_n, r_nf)

        l_contact = _foot_contact(model, data, "L")
        if l_contact and not first_contact_logged:
            first_contact_logged = True
            if diag.approach_log:
                last_air = next(
                    (s for s in reversed(diag.approach_log) if not s.contact),
                    None,
                )
                if last_air is not None:
                    diag.foot_pitch_before_contact_rad = last_air.foot_pitch_rad

        if _is_heel_touchdown(
            model,
            data,
            catch_started=True,
            was_airborne_in_catch=was_airborne_in_catch,
            prev_l_contact=prev_l_contact,
        ):
            diag.heel_touchdown_step = step_now
            diag.foot_pitch_at_td_rad = _foot_pitch_rad(model, data)
            diag.ankle_cmd_at_td = float(data.ctrl[IDX_L_ANKLE_P])
            diag.ankle_qpos_at_td = float(data.qpos[QPOS_L_ANKLE])
            diag.knee_qpos_before_td = float(data.qpos[QPOS_L_KNEE])
            diag.knee_qpos_after_prep_rad = float(data.qpos[QPOS_L_KNEE])
            diag.torso_fwd_vel_before_td_m_s = pre_td_vel
            diag.torso_fwd_vel_after_td_m_s = _forward_vel(data)
            if pre_td_vel is not None:
                diag.fwd_vel_kick_m_s = diag.torso_fwd_vel_after_td_m_s - pre_td_vel
            diag.r_foot_in_contact_at_td = _foot_contact(model, data, "R")
            heel_l_xy = _foot_pos(model, data, "L")[:2].copy()
            if stance_ctrl is None:
                stance_ctrl = st.ctrl.copy()
            if verbose:
                print(
                    f"\n  >> TOUCHDOWN step {diag.heel_touchdown_step} "
                    f"pitch={diag.foot_pitch_at_td_rad:.3f} rad "
                    f"ankle_qpos={diag.ankle_qpos_at_td:.3f}"
                )
            break

        prev_l_contact = l_contact
        if not prev_l_contact:
            was_airborne_in_catch = True

    if pitch_at_trigger is not None and diag.foot_pitch_before_contact_rad is not None:
        improvement = pitch_at_trigger - diag.foot_pitch_before_contact_rad
        diag.foot_flattened_before_contact = improvement >= FLAT_PITCH_IMPROVEMENT_RAD
    elif pitch_at_trigger is not None and diag.foot_pitch_at_td_rad is not None:
        diag.foot_flattened_before_contact = (
            pitch_at_trigger - diag.foot_pitch_at_td_rad >= FLAT_PITCH_IMPROVEMENT_RAD
        )

    if stance_ctrl is None:
        stance_ctrl = st.ctrl.copy()
    if heel_l_xy is None:
        heel_l_xy = _foot_pos(model, data, "L")[:2].copy()

    hold_ctrl = st.ctrl.copy()
    if verbose:
        _print_phase("POST TOUCHDOWN OBSERVE")
    for _ in range(POST_TOUCHDOWN_OBSERVE_STEPS):
        l_nf = _foot_normal_force(model, data, "L")
        r_nf = _foot_normal_force(model, data, "R")
        diag.peak_l_normal_n = max(diag.peak_l_normal_n, l_nf)
        diag.peak_r_normal_n = max(diag.peak_r_normal_n, r_nf)
        l_pos = _foot_pos(model, data, "L")
        diag.l_foot_max_drift_mm = max(
            diag.l_foot_max_drift_mm,
            float(np.linalg.norm(l_pos[:2] - heel_l_xy) * 1000.0),
        )
        st.ctrl = hold_ctrl.copy()
        data.ctrl[:15] = st.ctrl
        mujoco.mj_step(model, data)
        st.diag.global_step += 1
        _sync_viewer(viewer, slow)

    diag.l_foot_remains_planted = diag.l_foot_max_drift_mm < L_PLANT_DRIFT_THRESH_MM
    diag.r_foot_in_contact_at_end = _foot_contact(model, data, "R")

    if verbose:
        _print_phase("STOP")
    return diag


def _fmt(v: float | None, prec: int = 4) -> str:
    if v is None:
        return "None"
    return f"{v:.{prec}f}"


def print_approach_log(log: list[ApproachSample]) -> None:
    print("\n--- FINAL APPROACH (step-by-step) ---")
    print(
        "  step  clr_mm  pitch   ank_c  ank_q  knee_c  knee_q  L_nf   R_nf  "
        "tilt   angvel  fwd_v  cnt"
    )
    for s in log:
        print(
            f"  {s.step:4d} {s.clearance_mm:7.1f} {s.foot_pitch_rad:6.3f} "
            f"{s.ankle_cmd:6.3f} {s.ankle_qpos:6.3f} {s.knee_cmd:6.3f} {s.knee_qpos:6.3f} "
            f"{s.l_normal_n:6.1f} {s.r_normal_n:6.1f} {s.torso_tilt_rad:6.3f} "
            f"{s.torso_angvel_rad_s:7.3f} {s.fwd_vel_m_s:6.3f} {int(s.contact):3d}"
        )


def print_summary(sim: SimultaneousDiagnostics, baseline: BaselineDiagnostics | None) -> None:
    print("\n" + "=" * 72)
    print("SIMULTANEOUS FLAT ANKLE PITCH + KNEE YIELD AT IMMINENT CONTACT")
    print("=" * 72)
    print(f"Trigger: sole clearance < {IMMINENT_CLEARANCE_MM:.0f} mm, descending, no contact")
    print(
        f"Ankle pitch ramp +{ANKLE_FLAT_RAMP_STEP:.3f}/step to {L_ANKLE_FLAT_TARGET:.2f} rad; "
        f"knee yield {KNEE_YIELD_DELTA_RAD:.2f} rad / {KNEE_YIELD_STEPS} steps"
    )

    tc = sim.traj_check
    print("\n--- TRAJECTORY VERIFICATION (before prep trigger) ---")
    print(f"PREP_TRIGGER_STEP = {tc.trigger_step}")
    print(f"CLEARANCE_AT_TRIGGER_MM = {tc.clearance_at_trigger_mm}")
    print(f"STEPS_COMPARED = {tc.steps_compared}")
    print(f"MAX_HIP_CMD_DIFF_RAD = {tc.max_hip_cmd_diff:.6f}")
    print(f"MAX_KNEE_CMD_DIFF_RAD = {tc.max_knee_cmd_diff:.6f}")
    print(f"MAX_ANKLE_CMD_DIFF_RAD = {tc.max_ankle_cmd_diff:.6f}")
    match_ok = (
        tc.max_hip_cmd_diff < 1e-4
        and tc.max_knee_cmd_diff < 1e-4
        and tc.max_ankle_cmd_diff < 1e-4
    )
    print(f"Pre-trigger match vs catch lerp: {'YES' if match_ok else 'CHECK'}")

    print_approach_log(sim.approach_log)

    print("\n--- TOUCHDOWN SUMMARY ---")
    print(f"TOUCHDOWN_STEP = {sim.heel_touchdown_step}")
    print(f"FOOT_PITCH_BEFORE_CONTACT_RAD = {_fmt(sim.foot_pitch_before_contact_rad, 3)}")
    print(f"FOOT_PITCH_AT_TD_RAD = {_fmt(sim.foot_pitch_at_td_rad, 3)}")
    print(f"ANKLE_PITCH_CMD_AT_TD = {_fmt(sim.ankle_cmd_at_td)}")
    print(f"ANKLE_PITCH_QPOS_AT_TD = {_fmt(sim.ankle_qpos_at_td)}")
    print(f"KNEE_QPOS_BEFORE_TD = {_fmt(sim.knee_qpos_before_td)}")
    print(f"KNEE_QPOS_AFTER_PREP = {_fmt(sim.knee_qpos_after_prep_rad)}")
    print(f"PEAK_L_NORMAL_N = {sim.peak_l_normal_n:.1f}")
    print(f"PEAK_R_NORMAL_N = {sim.peak_r_normal_n:.1f}")
    print(f"TORSO_FWD_VEL_BEFORE_TD = {_fmt(sim.torso_fwd_vel_before_td_m_s)}")
    print(f"TORSO_FWD_VEL_AFTER_TD = {_fmt(sim.torso_fwd_vel_after_td_m_s)}")
    print(f"FWD_VEL_KICK_AT_TD = {_fmt(sim.fwd_vel_kick_m_s)}")
    print(f"L_FOOT_MAX_DRIFT_MM = {sim.l_foot_max_drift_mm:.1f}")
    print(f"L_FOOT_REMAINS_PLANTED = {sim.l_foot_remains_planted}")
    print(f"R_FOOT_IN_CONTACT_AT_TD = {sim.r_foot_in_contact_at_td}")
    print(f"R_FOOT_IN_CONTACT_AT_END = {sim.r_foot_in_contact_at_end}")
    print(
        f"\nFOOT_SUBSTANTIALLY_FLATTER_BEFORE_CONTACT "
        f"(pitch improved >= {FLAT_PITCH_IMPROVEMENT_RAD:.2f} rad while airborne): "
        f"{'YES' if sim.foot_flattened_before_contact else 'NO'}"
    )

    if baseline is not None:
        print("\n--- COMPARISON vs l_knee_yield_touchdown_test ---")
        rows = [
            ("Touchdown step", baseline.heel_touchdown_step, sim.heel_touchdown_step),
            ("Foot pitch at TD (rad)", _fmt(baseline.foot_pitch_at_td_rad, 3),
             _fmt(sim.foot_pitch_at_td_rad, 3)),
            ("Ankle qpos at TD", _fmt(baseline.ankle_qpos_at_td), _fmt(sim.ankle_qpos_at_td)),
            ("Knee qpos before TD", _fmt(baseline.knee_qpos_before_td), _fmt(sim.knee_qpos_before_td)),
            ("Peak L normal (N)", _fmt(baseline.peak_l_normal_n, 1), _fmt(sim.peak_l_normal_n, 1)),
            ("Fwd vel kick (m/s)", _fmt(baseline.fwd_vel_kick_at_td_m_s), _fmt(sim.fwd_vel_kick_m_s)),
            ("L foot drift (mm)", _fmt(baseline.l_foot_max_drift_mm, 1), _fmt(sim.l_foot_max_drift_mm, 1)),
            ("L foot planted", baseline.l_foot_remains_planted, sim.l_foot_remains_planted),
        ]
        print(f"\n{'Metric':<28} {'KNEE-YIELD-BASE':<18} {'SIMULTANEOUS':<18}")
        print("-" * 64)
        for name, a, b in rows:
            print(f"{name:<28} {str(a):<18} {str(b):<18}")

    print("\nSingle fixed experiment - no auto-tuning.")


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
        description="Simultaneous flat ankle pitch + knee yield before touchdown."
    )
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-baseline", action="store_true", help="Skip baseline comparison run.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Simultaneous flat ankle pitch + knee yield at imminent contact")
    print(f"Trigger: clearance < {IMMINENT_CLEARANCE_MM:.0f} mm, descending, airborne")
    print("Viewer: robot only.\n")

    ref_cmds = _record_catch_lerp_reference(env)
    baseline: BaselineDiagnostics | None = None

    if args.headless and not args.no_baseline:
        print("Running baseline: l_knee_yield_touchdown_test...")
        flat_ref = _record_flat_catch_reference(env)
        baseline = run_knee_yield_experiment(
            env, viewer=None, slow=False, ref_cmds=flat_ref, verbose=False
        )

    if args.headless:
        sim = run_simultaneous_experiment(
            env, viewer=None, slow=False, ref_cmds=ref_cmds, verbose=False
        )
        print_summary(sim, baseline)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        sim = run_simultaneous_experiment(
            env, viewer=v, slow=args.slow, ref_cmds=ref_cmds, verbose=True
        )
    print_summary(sim, None)
    print("\n(Headless mode includes baseline comparison vs l_knee_yield_touchdown_test.)")


if __name__ == "__main__":
    main(sys.argv[1:])
