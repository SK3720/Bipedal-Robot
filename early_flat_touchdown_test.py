"""Early flat ankle pitch + knee yield before L touchdown.

LOCKED: staged_forward_catch_test trajectory through swing and catch lerp.
NEW at sole clearance < 28 mm (descending, airborne):
  - rapid L ankle PITCH ramp toward +0.38 rad
  - simultaneous L knee yield -0.15 rad over 20 steps
  - L hip frozen at trigger; R leg unchanged.

Focus: physical foot orientation and ankle qpos BEFORE contact, not command alone.
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
    FOOT_CONTACT_Z,
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
from staged_forward_catch_continue_test import _is_heel_touchdown

QPOS_L_ANKLE = 15

# Single fixed early-trigger experiment (25-30 mm range).
PREP_CLEARANCE_TRIGGER_MM = 28.0
L_ANKLE_FLAT_TARGET = 0.38
ANKLE_PITCH_RAMP_STEP = 0.032
KNEE_YIELD_DELTA_RAD = -0.15
KNEE_YIELD_STEPS = 20
POST_TOUCHDOWN_HOLD_STEPS = 120
FLAT_PITCH_IMPROVEMENT_RAD = 0.10


def _torso_angvel_pitch_rad_s(data: mujoco.MjData) -> float:
    return float(data.qvel[4])


@dataclass
class TrajCheck:
    max_hip_cmd_diff: float = 0.0
    max_knee_cmd_diff: float = 0.0
    max_ankle_cmd_diff: float = 0.0
    steps_compared: int = 0


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
    torso_angvel_rad_s: float
    fwd_vel_m_s: float
    contact: bool


@dataclass
class EarlyFlatDiagnostics:
    base: StepDiagnostics
    traj_check: TrajCheck = field(default_factory=TrajCheck)
    approach_log: list[ApproachSample] = field(default_factory=list)
    prep_trigger_step: int | None = None
    prep_clearance_at_trigger_mm: float | None = None
    foot_pitch_at_trigger_rad: float | None = None
    ankle_cmd_at_trigger: float | None = None
    ankle_qpos_at_trigger: float | None = None
    heel_touchdown_step: int | None = None
    foot_pitch_before_contact_rad: float | None = None
    foot_pitch_at_td_rad: float | None = None
    ankle_cmd_at_td: float | None = None
    ankle_qpos_at_td: float | None = None
    knee_cmd_at_td: float | None = None
    knee_qpos_at_td: float | None = None
    l_normal_at_td_n: float | None = None
    peak_l_normal_n: float = 0.0
    torso_angvel_at_td_rad_s: float | None = None
    torso_fwd_vel_at_td_m_s: float | None = None
    l_foot_max_drift_mm: float = 0.0
    foot_flattened_before_contact: bool = False
    pitch_improvement_before_contact_rad: float | None = None


def _record_catch_lerp_reference(env: BipedalWalkEnv) -> dict[int, tuple[float, float, float]]:
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
    diag: EarlyFlatDiagnostics,
    model: mujoco.MjModel,
    data: mujoco.MjData,
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
            torso_angvel_rad_s=_torso_angvel_pitch_rad_s(data),
            fwd_vel_m_s=_forward_vel(data),
            contact=_foot_contact(model, data, "L"),
        )
    )


def run_early_flat_experiment(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    ref_cmds: dict[int, tuple[float, float, float]] | None = None,
    *,
    verbose: bool = True,
) -> EarlyFlatDiagnostics:
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
    diag = EarlyFlatDiagnostics(base=st.diag)

    _run_locked_prefix(env, model, data, cr, st, viewer, slow, verbose=verbose)

    if verbose:
        _print_phase("EARLY FLAT ANKLE PITCH + KNEE YIELD")

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
    hold_ctrl: np.ndarray | None = None
    peak_clearance_in_catch_mm = 0.0

    while st.catch_steps < CATCH_MAX_STEPS:
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
            l_contact = _foot_contact(model, data, "L")
            clearance = _sole_clearance_mm(model, data)
            if not l_contact:
                peak_clearance_in_catch_mm = max(peak_clearance_in_catch_mm, clearance)
                descending = _l_foot_vert_vel_m_s(model, data) < -SOLE_DESCENT_VEL_M_S
                # FOOT_CONTACT_Z clearance during late catch is typically 4-15 mm while
                # airborne (25-30 mm is not reached). Earliest prep: first catch-phase
                # step below threshold; fallback: final descent from local peak.
                early_trigger = clearance < PREP_CLEARANCE_TRIGGER_MM
                late_trigger = (
                    peak_clearance_in_catch_mm >= 8.0
                    and clearance <= peak_clearance_in_catch_mm - 3.0
                    and descending
                )
                if early_trigger or late_trigger:
                    prep_active = True
                    diag.prep_trigger_step = st.diag.global_step + 1
                    diag.prep_clearance_at_trigger_mm = clearance
                    diag.foot_pitch_at_trigger_rad = _foot_pitch_rad(model, data)
                    diag.ankle_cmd_at_trigger = ankle_cmd
                    diag.ankle_qpos_at_trigger = float(data.qpos[QPOS_L_ANKLE])
                    hip_hold = hip_val
                    knee_start = knee_val
                    stance_ctrl = st.ctrl.copy()
                    prep_steps = 0
                    if verbose:
                        print(
                            f"\n  >> PREP TRIGGER step {diag.prep_trigger_step} "
                            f"clearance={clearance:.1f} mm "
                            f"pitch={diag.foot_pitch_at_trigger_rad:.3f} rad "
                            f"ankle_qpos={diag.ankle_qpos_at_trigger:.3f}"
                        )
            st.ctrl = left_leg_pose(knee_val, hip_val, ankle_cmd)
        else:
            prep_steps += 1
            t = _smooth(min(prep_steps, KNEE_YIELD_STEPS) / KNEE_YIELD_STEPS)
            knee_cmd = knee_start + t * KNEE_YIELD_DELTA_RAD
            ankle_cmd = min(ankle_cmd + ANKLE_PITCH_RAMP_STEP, L_ANKLE_FLAT_TARGET)
            st.ctrl = left_leg_pose(knee_cmd, hip_hold, ankle_cmd)
            if stance_ctrl is not None:
                for idx in range(15):
                    if idx not in (IDX_L_KNEE, IDX_L_HIP_PITCH, IDX_L_ANKLE_P):
                        st.ctrl[idx] = stance_ctrl[idx]

        _sim_step(env, model, data, st, viewer, slow)
        st.catch_steps += 1
        step_now = st.diag.global_step

        if prep_active:
            _log_approach(diag, model, data, step_now)

        l_nf = _foot_normal_force(model, data, "L")
        diag.peak_l_normal_n = max(diag.peak_l_normal_n, l_nf)

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
            diag.knee_cmd_at_td = float(data.ctrl[IDX_L_KNEE])
            diag.knee_qpos_at_td = float(data.qpos[QPOS_L_KNEE])
            diag.l_normal_at_td_n = l_nf
            diag.torso_angvel_at_td_rad_s = _torso_angvel_pitch_rad_s(data)
            diag.torso_fwd_vel_at_td_m_s = _forward_vel(data)
            heel_l_xy = _foot_pos(model, data, "L")[:2].copy()
            hold_ctrl = st.ctrl.copy()
            if diag.approach_log:
                last_air = next((s for s in reversed(diag.approach_log) if not s.contact), None)
                if last_air is not None:
                    diag.foot_pitch_before_contact_rad = last_air.foot_pitch_rad
            if verbose:
                print(
                    f"\n  >> TOUCHDOWN step {diag.heel_touchdown_step} "
                    f"pitch={diag.foot_pitch_at_td_rad:.3f} rad "
                    f"ankle_qpos={diag.ankle_qpos_at_td:.3f} "
                    f"L_nf={diag.l_normal_at_td_n:.1f} N"
                )
            break

        prev_l_contact = _foot_contact(model, data, "L")
        if not prev_l_contact:
            was_airborne_in_catch = True

    if diag.foot_pitch_at_trigger_rad is not None and diag.foot_pitch_before_contact_rad is not None:
        diag.pitch_improvement_before_contact_rad = (
            diag.foot_pitch_at_trigger_rad - diag.foot_pitch_before_contact_rad
        )
        diag.foot_flattened_before_contact = (
            diag.pitch_improvement_before_contact_rad >= FLAT_PITCH_IMPROVEMENT_RAD
        )

    if hold_ctrl is None:
        hold_ctrl = st.ctrl.copy()
    if heel_l_xy is None:
        heel_l_xy = _foot_pos(model, data, "L")[:2].copy()

    if verbose:
        _print_phase("HOLD POST TOUCHDOWN")
    for _ in range(POST_TOUCHDOWN_HOLD_STEPS):
        l_nf = _foot_normal_force(model, data, "L")
        diag.peak_l_normal_n = max(diag.peak_l_normal_n, l_nf)
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

    if verbose:
        _print_phase("STOP")
    return diag


def print_summary(d: EarlyFlatDiagnostics) -> None:
    print("\n" + "=" * 72)
    print("EARLY FLAT ANKLE PITCH + KNEE YIELD TOUCHDOWN")
    print("=" * 72)
    print(
        f"Trigger: clearance < {PREP_CLEARANCE_TRIGGER_MM:.0f} mm in catch "
        f"(FOOT_CONTACT_Z={FOOT_CONTACT_Z}; late-catch clearance is typically 4-15 mm)"
    )
    print(
        f"Ankle pitch ramp +{ANKLE_PITCH_RAMP_STEP:.3f}/step to {L_ANKLE_FLAT_TARGET:.2f} rad; "
        f"knee yield {KNEE_YIELD_DELTA_RAD:.2f} rad / {KNEE_YIELD_STEPS} steps"
    )

    tc = d.traj_check
    print("\n--- TRAJECTORY VERIFICATION (before prep trigger) ---")
    print(f"STEPS_COMPARED = {tc.steps_compared}")
    print(f"MAX_HIP_CMD_DIFF_RAD = {tc.max_hip_cmd_diff:.6f}")
    print(f"MAX_KNEE_CMD_DIFF_RAD = {tc.max_knee_cmd_diff:.6f}")
    print(f"MAX_ANKLE_CMD_DIFF_RAD = {tc.max_ankle_cmd_diff:.6f}")
    match_ok = (
        tc.max_hip_cmd_diff < 1e-4
        and tc.max_knee_cmd_diff < 1e-4
        and tc.max_ankle_cmd_diff < 1e-4
    )
    print(f"Swing unchanged before trigger: {'YES' if match_ok else 'CHECK'}")

    print("\n--- TRIGGER ---")
    print(f"PREP_TRIGGER_STEP = {d.prep_trigger_step}")
    print(f"PREP_CLEARANCE_AT_TRIGGER_MM = {d.prep_clearance_at_trigger_mm}")
    print(f"FOOT_PITCH_AT_TRIGGER_RAD = {d.foot_pitch_at_trigger_rad}")
    print(f"ANKLE_PITCH_CMD_AT_TRIGGER = {d.ankle_cmd_at_trigger}")
    print(f"ANKLE_PITCH_QPOS_AT_TRIGGER = {d.ankle_qpos_at_trigger}")

    if d.approach_log:
        print("\n--- FINAL APPROACH (from trigger to contact) ---")
        print("  step  clr_mm  pitch   ank_c  ank_q  knee_c  knee_q  L_nf  angvel  fwd_v  cnt")
        for s in d.approach_log:
            print(
                f"  {s.step:4d} {s.clearance_mm:7.1f} {s.foot_pitch_rad:6.3f} "
                f"{s.ankle_cmd:6.3f} {s.ankle_qpos:6.3f} {s.knee_cmd:6.3f} {s.knee_qpos:6.3f} "
                f"{s.l_normal_n:6.1f} {s.torso_angvel_rad_s:7.3f} {s.fwd_vel_m_s:6.3f} "
                f"{int(s.contact):3d}"
            )

    print("\n--- TOUCHDOWN ---")
    print(f"TOUCHDOWN_STEP = {d.heel_touchdown_step}")
    print(f"FOOT_PITCH_BEFORE_CONTACT_RAD = {d.foot_pitch_before_contact_rad}")
    print(f"FOOT_PITCH_AT_TD_RAD = {d.foot_pitch_at_td_rad}")
    print(f"ANKLE_PITCH_CMD_AT_TD = {d.ankle_cmd_at_td}")
    print(f"ANKLE_PITCH_QPOS_AT_TD = {d.ankle_qpos_at_td}")
    print(f"L_KNEE_CMD_AT_TD = {d.knee_cmd_at_td}")
    print(f"L_KNEE_QPOS_AT_TD = {d.knee_qpos_at_td}")
    print(f"L_NORMAL_AT_TD_N = {d.l_normal_at_td_n}")
    print(f"PEAK_L_NORMAL_N = {d.peak_l_normal_n:.1f}")
    print(f"TORSO_ANGVEL_AT_TD_RAD_S = {d.torso_angvel_at_td_rad_s}")
    print(f"TORSO_FWD_VEL_AT_TD_M_S = {d.torso_fwd_vel_at_td_m_s}")
    print(f"L_FOOT_MAX_DRIFT_MM = {d.l_foot_max_drift_mm:.1f}")
    print(f"PITCH_IMPROVEMENT_BEFORE_CONTACT_RAD = {d.pitch_improvement_before_contact_rad}")
    print(
        f"\nSOLE_SUBSTANTIALLY_FLATTER_BEFORE_CONTACT "
        f"(>= {FLAT_PITCH_IMPROVEMENT_RAD:.2f} rad): "
        f"{'YES' if d.foot_flattened_before_contact else 'NO'}"
    )
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
    p = argparse.ArgumentParser(description="Early flat ankle pitch + knee yield touchdown test.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Early flat ankle pitch + knee yield touchdown")
    print(
        f"Trigger: clearance < {PREP_CLEARANCE_TRIGGER_MM:.0f} mm in catch "
        f"(FOOT_CONTACT_Z={FOOT_CONTACT_Z})"
    )
    print("Locked swing from staged_forward_catch_test. Viewer: robot only.\n")

    ref_cmds = _record_catch_lerp_reference(env)

    if args.headless:
        diag = run_early_flat_experiment(
            env, viewer=None, slow=False, ref_cmds=ref_cmds, verbose=False
        )
        print_summary(diag)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        diag = run_early_flat_experiment(
            env, viewer=v, slow=args.slow, ref_cmds=ref_cmds, verbose=True
        )
    print_summary(diag)


if __name__ == "__main__":
    main(sys.argv[1:])
