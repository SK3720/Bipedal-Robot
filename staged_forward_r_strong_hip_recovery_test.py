"""Strong R hip recovery after left plant — single fixed experiment.

LOCKED prefix + L plant identical to staged_forward_r_recovery_test.py.
Only change: R hip target +0.55 (vs +0.38), extended recovery duration.

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
    FLOOR_Z,
    IDX_L_ANKLE_P,
    IDX_L_HIP_PITCH,
    IDX_L_KNEE,
    IDX_R_HIP_PITCH,
    IDX_R_KNEE,
    Phase,
    RunState,
    StepDiagnostics,
    _foot_contact,
    _foot_normal_force,
    _foot_pos,
    _forward_mm,
    _forward_vel,
    _reset,
    _smooth,
    _sync_viewer,
    rapid_shift_pose,
)
from staged_forward_catch_continue_test import (
    PLANT_FORCE_N,
    PLANT_SUSTAIN_STEPS,
    _r_load_fraction,
)
from staged_forward_r_recovery_test import (
    QPOS_R_HIP,
    QPOS_R_KNEE,
    R_KNEE_CLEARANCE_TARGET,
    R_KNEE_LIFT_MAX_STEPS,
    R_KNEE_LIFT_STEP,
    R_KNEE_MIN_RAMP_STEPS,
    R_LOAD_DROP_THRESH,
    R_NORMAL_UNLOAD_N,
    WAIT_PLANT_MAX_STEPS,
    RecoveryDiagnostics,
    RecoveryPhase,
    _apply_ctrl,
    _print_phase,
    _r_foot_fwd_vel,
    _run_locked_prefix_to_heel,
    _set_leg_ctrl,
)

R_HIP_RECOVERY_TARGET = 0.55
R_HIP_RAMP_STEPS = 36
R_HIP_HOLD_MAX_STEPS = 260
GROUND_APPROACH_CLEARANCE_MM = 42.0
STOP_STEPS = 60

VIEWER_WIDTH = base.VIEWER_WIDTH
VIEWER_HEIGHT = base.VIEWER_HEIGHT
VIEWER_TITLE_PREFIX = base.VIEWER_TITLE_PREFIX
VIEWER_POSITION_TIMEOUT_S = base.VIEWER_POSITION_TIMEOUT_S


class StopReason(str, Enum):
    R_AHEAD_OF_L = "R foot ahead of / under L foot"
    R_APPROACHING_GROUND = "R foot approaching ground contact"
    R_TOUCHDOWN = "R foot touchdown"
    MAX_DURATION = "maximum recovery duration reached"


@dataclass
class StrongHipDiagnostics:
    base: StepDiagnostics
    heel_touchdown_step: int | None = None
    heel_touchdown_l_rel_r_mm: float | None = None
    l_plant_confirmed_step: int | None = None
    r_lift_off_step: int | None = None
    r_knee_qpos_at_lift_end: float | None = None
    r_knee_cmd_held: float | None = None
    peak_r_clearance_mm: float = 0.0
    peak_r_fwd_from_lift_off_mm: float = 0.0
    peak_r_rel_l_fwd_mm: float = 0.0
    peak_r_foot_fwd_vel_m_s: float = 0.0
    peak_hip_cmd: float = 0.0
    peak_hip_qpos: float = 0.0
    r_touchdown_step: int | None = None
    r_touchdown_l_rel_mm: float | None = None
    r_touchdown_under_ahead_of_l: bool | None = None
    r_touchdown_xyz: np.ndarray | None = None
    l_foot_max_drift_mm: float = 0.0
    peak_torso_fwd_vel_m_s: float = 0.0
    final_torso_tilt_rad: float = 0.0
    final_forward_vel_m_s: float = 0.0
    final_r_rel_l_fwd_mm: float = 0.0
    stop_reason: str | None = None
    hip_recovery_steps: int = 0


def run_strong_hip_recovery(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> StrongHipDiagnostics:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    shifted = rapid_shift_pose()
    base_diag = StepDiagnostics()
    st = RunState(
        ctrl=DEFAULT_POSE.copy(),
        phase=Phase.STAND,
        stand_r_xy=_foot_pos(model, data, "R")[:2].copy(),
        diag=base_diag,
        knee_cmd=float(shifted[IDX_L_KNEE]),
        hip_cmd=float(shifted[IDX_L_HIP_PITCH]),
    )

    heel_rd = RecoveryDiagnostics(base=base_diag)
    diag = StrongHipDiagnostics(base=base_diag)

    heel_ctrl = _run_locked_prefix_to_heel(env, model, data, cr, st, heel_rd, viewer, slow)
    diag.heel_touchdown_step = heel_rd.heel_touchdown_step
    diag.heel_touchdown_l_rel_r_mm = heel_rd.heel_touchdown_l_rel_r_mm

    heel_l_xy = _foot_pos(model, data, "L")[:2].copy()
    r_lift_off_ref_y: float | None = None
    stance_ctrl = heel_ctrl.copy()
    r_knee_cmd = float(stance_ctrl[IDX_R_KNEE])
    r_hip_start = float(stance_ctrl[IDX_R_HIP_PITCH])
    clearance_r_knee = r_knee_cmd

    # --- WAIT L PLANT (unchanged) ---
    _print_phase(RecoveryPhase.WAIT_L_PLANT.value)
    plant_streak = 0
    phase = RecoveryPhase.WAIT_L_PLANT

    for _ in range(WAIT_PLANT_MAX_STEPS):
        l_nf = _foot_normal_force(model, data, "L")
        r_nf = _foot_normal_force(model, data, "R")
        r_frac = _r_load_fraction(model, data)
        l_pos = _foot_pos(model, data, "L")
        diag.l_foot_max_drift_mm = max(
            diag.l_foot_max_drift_mm,
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
            diag.l_plant_confirmed_step = st.diag.global_step
            print(
                f"\n  >> L PLANT confirmed step {diag.l_plant_confirmed_step} "
                f"L_nf={l_nf:.1f}N R_frac={r_frac:.3f}"
            )
            phase = RecoveryPhase.R_KNEE_LIFT
            break
        _apply_ctrl(env, model, data, st, stance_ctrl, viewer, slow)

    if phase == RecoveryPhase.WAIT_L_PLANT:
        diag.l_plant_confirmed_step = st.diag.global_step
        phase = RecoveryPhase.R_KNEE_LIFT

    # --- R KNEE LIFT (unchanged) ---
    _print_phase(RecoveryPhase.R_KNEE_LIFT.value)
    knee_steps = 0
    while knee_steps < R_KNEE_LIFT_MAX_STEPS and phase == RecoveryPhase.R_KNEE_LIFT:
        r_unloaded = (
            not _foot_contact(model, data, "R")
            or _foot_normal_force(model, data, "R") < R_NORMAL_UNLOAD_N
        )
        if r_unloaded and diag.r_lift_off_step is None:
            diag.r_lift_off_step = st.diag.global_step
            r_lift_off_ref_y = float(_foot_pos(model, data, "R")[1])
            print(f"\n  >> R LIFT-OFF step {diag.r_lift_off_step}")

        r_knee_cmd = max(r_knee_cmd - R_KNEE_LIFT_STEP, R_KNEE_CLEARANCE_TARGET)
        ctrl = _set_leg_ctrl(stance_ctrl, r_knee=r_knee_cmd)
        _apply_ctrl(env, model, data, st, ctrl, viewer, slow)
        knee_steps += 1

        if knee_steps >= R_KNEE_MIN_RAMP_STEPS and (r_unloaded or r_knee_cmd <= R_KNEE_CLEARANCE_TARGET + 1e-6):
            clearance_r_knee = r_knee_cmd
            diag.r_knee_cmd_held = clearance_r_knee
            diag.r_knee_qpos_at_lift_end = float(data.qpos[QPOS_R_KNEE])
            phase = RecoveryPhase.R_HIP_RECOVERY
            print(
                f"\n  >> R knee held at cmd={clearance_r_knee:.3f} "
                f"qpos={diag.r_knee_qpos_at_lift_end:.3f}"
            )
            break

    if phase == RecoveryPhase.R_KNEE_LIFT:
        clearance_r_knee = r_knee_cmd
        diag.r_knee_cmd_held = clearance_r_knee
        diag.r_knee_qpos_at_lift_end = float(data.qpos[QPOS_R_KNEE])
        phase = RecoveryPhase.R_HIP_RECOVERY

    if r_lift_off_ref_y is None:
        r_lift_off_ref_y = float(_foot_pos(model, data, "R")[1])

    # --- STRONG R HIP RECOVERY ---
    _print_phase(RecoveryPhase.R_HIP_RECOVERY.value)
    hip_steps = 0
    was_airborne = diag.r_lift_off_step is not None
    prev_r_contact = _foot_contact(model, data, "R")
    peak_clearance_seen = diag.peak_r_clearance_mm

    total_max = R_HIP_RAMP_STEPS + R_HIP_HOLD_MAX_STEPS
    stop_reason: StopReason | None = None

    while hip_steps < total_max and stop_reason is None:
        r_pos = _foot_pos(model, data, "R")
        l_pos = _foot_pos(model, data, "L")
        clearance = (r_pos[2] - FLOOR_Z) * 1000.0
        r_fwd_lift = _forward_mm(r_pos[1], r_lift_off_ref_y)
        r_rel_l = _forward_mm(r_pos[1], l_pos[1])
        r_contact = _foot_contact(model, data, "R")
        hip_cmd = float(data.ctrl[IDX_R_HIP_PITCH])
        hip_qpos = float(data.qpos[QPOS_R_HIP])

        diag.peak_r_clearance_mm = max(diag.peak_r_clearance_mm, clearance)
        diag.peak_r_fwd_from_lift_off_mm = max(diag.peak_r_fwd_from_lift_off_mm, r_fwd_lift)
        diag.peak_r_rel_l_fwd_mm = max(diag.peak_r_rel_l_fwd_mm, r_rel_l)
        diag.peak_r_foot_fwd_vel_m_s = max(diag.peak_r_foot_fwd_vel_m_s, _r_foot_fwd_vel(model, data))
        diag.peak_hip_cmd = max(diag.peak_hip_cmd, hip_cmd)
        diag.peak_hip_qpos = max(diag.peak_hip_qpos, hip_qpos)
        diag.peak_torso_fwd_vel_m_s = max(diag.peak_torso_fwd_vel_m_s, _forward_vel(data))

        if not r_contact:
            was_airborne = True
            peak_clearance_seen = max(peak_clearance_seen, clearance)

        if was_airborne and r_contact and not prev_r_contact and diag.r_touchdown_step is None:
            diag.r_touchdown_step = st.diag.global_step
            diag.r_touchdown_l_rel_mm = r_rel_l
            diag.r_touchdown_under_ahead_of_l = r_rel_l > 0.0
            diag.r_touchdown_xyz = r_pos.copy()
            stop_reason = StopReason.R_TOUCHDOWN
            print(
                f"\n  >> R TOUCHDOWN step {diag.r_touchdown_step} "
                f"L-R fwd={r_rel_l:.1f}mm under_ahead={diag.r_touchdown_under_ahead_of_l}"
            )

        if r_rel_l > 0.0 and hip_cmd >= R_HIP_RECOVERY_TARGET - 0.02:
            stop_reason = StopReason.R_AHEAD_OF_L
            print(f"\n  >> R foot ahead of L: rel_fwd={r_rel_l:.1f}mm at step {st.diag.global_step}")

        if (
            was_airborne
            and peak_clearance_seen > GROUND_APPROACH_CLEARANCE_MM + 8.0
            and clearance < GROUND_APPROACH_CLEARANCE_MM
            and hip_steps > R_HIP_RAMP_STEPS
        ):
            stop_reason = StopReason.R_APPROACHING_GROUND
            print(
                f"\n  >> R foot approaching ground: clearance={clearance:.1f}mm "
                f"step {st.diag.global_step}"
            )

        if hip_steps >= total_max - 1:
            stop_reason = StopReason.MAX_DURATION

        if stop_reason is None:
            if hip_steps < R_HIP_RAMP_STEPS:
                t = (hip_steps + 1) / R_HIP_RAMP_STEPS
                alpha = _smooth(t)
                target_hip = (1.0 - alpha) * r_hip_start + alpha * R_HIP_RECOVERY_TARGET
            else:
                target_hip = R_HIP_RECOVERY_TARGET
            ctrl = _set_leg_ctrl(stance_ctrl, r_knee=clearance_r_knee, r_hip=target_hip)
            _apply_ctrl(env, model, data, st, ctrl, viewer, slow)
            hip_steps += 1
            prev_r_contact = r_contact
        else:
            diag.hip_recovery_steps = hip_steps
            break

    diag.stop_reason = stop_reason.value if stop_reason else None
    diag.hip_recovery_steps = hip_steps

    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    diag.final_torso_tilt_rad = env._quat_tilt_rad()
    diag.final_forward_vel_m_s = _forward_vel(data)
    diag.final_r_rel_l_fwd_mm = _forward_mm(r_pos[1], l_pos[1])

    _print_phase(RecoveryPhase.STOP.value)
    for _ in range(STOP_STEPS):
        _apply_ctrl(env, model, data, st, st.ctrl.copy(), viewer, slow)

    return diag


def print_summary(d: StrongHipDiagnostics) -> None:
    b = d.base
    print("\n" + "=" * 60)
    print("STRONG R HIP RECOVERY AFTER LEFT PLANT")
    print("=" * 60)
    print(f"R_HIP_TARGET = {R_HIP_RECOVERY_TARGET:.2f} rad  R_KNEE_HELD = {R_KNEE_CLEARANCE_TARGET:.2f} rad")
    print("\n--- LOCKED PREFIX ---")
    print(f"HEEL_TOUCHDOWN_STEP = {d.heel_touchdown_step}")
    print(f"HEEL_L_REL_R_MM = {d.heel_touchdown_l_rel_r_mm}")
    print(f"HIP_SWING_START = {b.hip_swing_start_step}  CATCH_START = {b.catch_start_step}")
    print(f"L_PLANT_CONFIRMED_STEP = {d.l_plant_confirmed_step}")
    print(f"L_FOOT_MAX_DRIFT_MM = {d.l_foot_max_drift_mm:.1f}")

    print("\n--- R LEG RECOVERY ---")
    print(f"R_LIFT_OFF_STEP = {d.r_lift_off_step}")
    print(f"R_KNEE_QPOS_AT_LIFT_END = {d.r_knee_qpos_at_lift_end}")
    print(f"R_KNEE_CMD_HELD = {d.r_knee_cmd_held}")
    print(f"PEAK_R_CLEARANCE_MM = {d.peak_r_clearance_mm:.1f}")
    print(f"PEAK_R_HIP_CMD = {d.peak_hip_cmd:.3f}")
    print(f"PEAK_R_HIP_QPOS = {d.peak_hip_qpos:.3f}")
    print(f"PEAK_R_FWD_FROM_LIFT_OFF_MM = {d.peak_r_fwd_from_lift_off_mm:.1f}")
    print(f"PEAK_R_REL_L_FWD_MM = {d.peak_r_rel_l_fwd_mm:.1f}")
    print(f"PEAK_R_FOOT_FORWARD_VEL_M_S = {d.peak_r_foot_fwd_vel_m_s:.4f}")
    print(f"HIP_RECOVERY_STEPS = {d.hip_recovery_steps}")
    print(f"STOP_REASON = {d.stop_reason}")
    print(f"R_TOUCHDOWN_STEP = {d.r_touchdown_step}")
    print(f"R_TOUCHDOWN_L_REL_MM = {d.r_touchdown_l_rel_mm}")
    print(f"R_TOUCHDOWN_UNDER_AHEAD_OF_L = {d.r_touchdown_under_ahead_of_l}")
    if d.r_touchdown_xyz is not None:
        print(f"R_TOUCHDOWN_XYZ = {d.r_touchdown_xyz}")

    print("\n--- FINAL STATE ---")
    print(f"FINAL_R_REL_L_FWD_MM = {d.final_r_rel_l_fwd_mm:.1f}")
    print(f"PEAK_TORSO_FORWARD_VEL_M_S = {d.peak_torso_fwd_vel_m_s:.3f}")
    print(f"FINAL_TORSO_TILT_RAD = {d.final_torso_tilt_rad:.3f}")
    print(f"FINAL_FORWARD_VEL_M_S = {d.final_forward_vel_m_s:.3f}")

    if d.peak_r_rel_l_fwd_mm > 0 or d.r_touchdown_under_ahead_of_l:
        print("\nVERDICT: R foot recovered under/ahead of L stance foot.")
    elif d.peak_r_fwd_from_lift_off_mm > 35.0:
        print(f"\nVERDICT: Stronger hip improved forward travel ({d.peak_r_fwd_from_lift_off_mm:.0f} mm) but R not yet under L.")
    else:
        print(f"\nVERDICT: Limited forward recovery ({d.peak_r_fwd_from_lift_off_mm:.0f} mm from lift-off).")

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
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.wintypes.DWORD), ("rcMonitor", RECT),
                    ("rcWork", RECT), ("dwFlags", ctypes.wintypes.DWORD)]

    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 1), ctypes.byref(info))
    wa = info.rcWork
    x = wa.left + (wa.right - wa.left - VIEWER_WIDTH) // 2
    y = wa.top + (wa.bottom - wa.top - VIEWER_HEIGHT) // 2
    user32.SetWindowPos(hwnd, 0, x, y, VIEWER_WIDTH, VIEWER_HEIGHT, 0x0004)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strong R hip recovery after left plant.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Strong R hip recovery: locked prefix -> L plant -> R knee -0.35 -> R hip +0.55")
    print("Viewer: robot only.\n")

    if args.headless:
        diag = run_strong_hip_recovery(env, viewer=None, slow=False)
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
        diag = run_strong_hip_recovery(env, viewer=v, slow=args.slow)
    print_summary(diag)


if __name__ == "__main__":
    main(sys.argv[1:])
