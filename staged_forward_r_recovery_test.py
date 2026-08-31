"""Staged forward catch + right leg recovery after left plant.

LOCKED through left heel touchdown (identical to staged_forward_catch_test).
Then: wait for L sustained plant -> R knee lift -> R hip forward recovery.

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
    FLOOR_Z,
    IDX_L_ANKLE_P,
    IDX_L_HIP_PITCH,
    IDX_L_KNEE,
    IDX_R_HIP_PITCH,
    IDX_R_KNEE,
    L_ANKLE_CATCH,
    L_HIP_CATCH,
    L_KNEE_CATCH,
    Phase,
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

QPOS_R_HIP = 18
QPOS_R_KNEE = 19

WAIT_PLANT_MAX_STEPS = 250
R_LOAD_DROP_THRESH = 0.30
R_NORMAL_UNLOAD_N = 3.0

R_KNEE_LIFT_STEP = 0.012
R_KNEE_CLEARANCE_TARGET = -0.35
R_KNEE_LIFT_MAX_STEPS = 90
R_KNEE_MIN_RAMP_STEPS = 22

R_HIP_RAMP_STEP = 0.014
R_HIP_RECOVERY_TARGET = 0.38
R_HIP_RECOVERY_MAX_STEPS = 140
MIN_R_FWD_RECOVERY_MM = 35.0

STOP_STEPS = 80

VIEWER_WIDTH = base.VIEWER_WIDTH
VIEWER_HEIGHT = base.VIEWER_HEIGHT
VIEWER_TITLE_PREFIX = base.VIEWER_TITLE_PREFIX
VIEWER_POSITION_TIMEOUT_S = base.VIEWER_POSITION_TIMEOUT_S


class RecoveryPhase(str, Enum):
    WAIT_L_PLANT = "WAIT L PLANT"
    R_KNEE_LIFT = "R KNEE LIFT"
    R_HIP_RECOVERY = "R HIP FORWARD RECOVERY"
    STOP = "STOP"


@dataclass
class RecoveryDiagnostics:
    base: StepDiagnostics
    heel_touchdown_step: int | None = None
    heel_touchdown_l_rel_r_mm: float | None = None
    heel_touchdown_l_nf: float | None = None
    l_plant_confirmed_step: int | None = None
    r_recovery_start_step: int | None = None
    r_lift_off_step: int | None = None
    r_knee_cmd_at_lift_off: float | None = None
    r_knee_qpos_at_lift_off: float | None = None
    r_foot_airborne: bool = False
    peak_r_clearance_mm: float = 0.0
    peak_r_fwd_displacement_mm: float = 0.0
    peak_r_foot_fwd_vel_m_s: float = 0.0
    r_hip_cmd_end: float | None = None
    r_hip_qpos_end: float | None = None
    l_foot_max_drift_mm: float = 0.0
    peak_l_normal_post_heel: float = 0.0
    peak_r_normal_post_heel: float = 0.0
    final_torso_tilt_rad: float = 0.0
    final_forward_vel_m_s: float = 0.0
    r_under_body: bool = False
    timeline: list[str] = field(default_factory=list)


def _apply_ctrl(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    st: RunState,
    ctrl: np.ndarray,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
) -> None:
    st.ctrl = ctrl.copy()
    data.ctrl[:15] = st.ctrl
    mujoco.mj_step(model, data)
    st.diag.global_step += 1
    _sync_viewer(viewer, slow)


def _set_leg_ctrl(
    ctrl: np.ndarray,
    *,
    l_knee: float | None = None,
    l_hip: float | None = None,
    l_ankle: float | None = None,
    r_knee: float | None = None,
    r_hip: float | None = None,
) -> np.ndarray:
    out = ctrl.copy()
    if l_knee is not None:
        out[IDX_L_KNEE] = l_knee
    if l_hip is not None:
        out[IDX_L_HIP_PITCH] = l_hip
    if l_ankle is not None:
        out[IDX_L_ANKLE_P] = l_ankle
    if r_knee is not None:
        out[IDX_R_KNEE] = r_knee
    if r_hip is not None:
        out[IDX_R_HIP_PITCH] = r_hip
    return out


def _r_foot_fwd_vel(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    foot_vel = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, model.body("R_foot").id, foot_vel, 0,
    )
    return float(-foot_vel[1])


def _run_locked_prefix_to_heel(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cr: np.ndarray,
    st: RunState,
    rd: RecoveryDiagnostics,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
) -> np.ndarray:
    """Identical locked motion through catch-phase heel touchdown."""
    shifted = rapid_shift_pose()
    lean = forward_lean_pose()

    _print_phase(Phase.STAND.value)
    for s in range(base.STAND_STEPS):
        st.ctrl = _lerp_ctrl(st.ctrl, DEFAULT_POSE, _smooth((s + 1) / base.STAND_STEPS), cr)
        _sim_step(env, model, data, st, viewer, slow)
    st.stand_r_xy = _foot_pos(model, data, "R")[:2].copy()

    _print_phase(Phase.FORWARD_FALL.value)
    for s in range(base.FALL_RAMP_STEPS):
        alpha = (s + 1) / base.FALL_RAMP_STEPS
        st.ctrl = _lerp_ctrl(st.ctrl, lean, alpha, cr)
        _sim_step(env, model, data, st, viewer, slow)
    for _ in range(base.FALL_MOMENTUM_STEPS):
        st.ctrl = lean.copy()
        _sim_step(env, model, data, st, viewer, slow)

    _print_phase(Phase.RAPID_SHIFT.value)
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

    _print_phase(Phase.KNEE_LIFT.value)
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

    _print_phase(Phase.HIP_SWING.value)
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

    _print_phase(Phase.CATCH.value)
    catch_start_ctrl = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l_contact = _foot_contact(model, data, "L")
    was_airborne_in_catch = not prev_l_contact
    heel_ctrl = st.ctrl.copy()

    while st.catch_steps < CATCH_MAX_STEPS and rd.heel_touchdown_step is None:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        st.ctrl = _lerp_ctrl(catch_start_ctrl, catch_target, alpha, cr)
        _sim_step(env, model, data, st, viewer, slow)
        st.catch_steps += 1

        if _is_heel_touchdown(
            model, data,
            catch_started=True,
            was_airborne_in_catch=was_airborne_in_catch,
            prev_l_contact=prev_l_contact,
        ):
            l_pos = _foot_pos(model, data, "L")
            r_pos = _foot_pos(model, data, "R")
            rd.heel_touchdown_step = st.diag.global_step
            rd.heel_touchdown_l_rel_r_mm = _forward_mm(l_pos[1], r_pos[1])
            rd.heel_touchdown_l_nf = _foot_normal_force(model, data, "L")
            heel_ctrl = st.ctrl.copy()
            print(
                f"\n  >> HEEL TOUCHDOWN step {rd.heel_touchdown_step} "
                f"L-R fwd={rd.heel_touchdown_l_rel_r_mm:.1f}mm L_nf={rd.heel_touchdown_l_nf:.1f}N"
            )
            break

        prev_l_contact = _foot_contact(model, data, "L")
        if not prev_l_contact:
            was_airborne_in_catch = True

    if rd.heel_touchdown_step is None:
        rd.heel_touchdown_step = st.diag.global_step
        l_pos = _foot_pos(model, data, "L")
        r_pos = _foot_pos(model, data, "R")
        rd.heel_touchdown_l_rel_r_mm = _forward_mm(l_pos[1], r_pos[1])
        rd.heel_touchdown_l_nf = _foot_normal_force(model, data, "L")
        heel_ctrl = st.ctrl.copy()

    return heel_ctrl


def run_r_recovery_experiment(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> RecoveryDiagnostics:
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
    rd = RecoveryDiagnostics(base=base_diag)

    heel_ctrl = _run_locked_prefix_to_heel(env, model, data, cr, st, rd, viewer, slow)
    heel_l_xy = _foot_pos(model, data, "L")[:2].copy()
    r_recovery_ref_y = float(_foot_pos(model, data, "R")[1])

    stance_ctrl = heel_ctrl.copy()
    r_knee_cmd = float(stance_ctrl[IDX_R_KNEE])
    r_hip_cmd = float(stance_ctrl[IDX_R_HIP_PITCH])
    clearance_r_knee = r_knee_cmd

    # --- WAIT for L plant + R unload ---
    _print_phase(RecoveryPhase.WAIT_L_PLANT.value)
    plant_streak = 0
    phase = RecoveryPhase.WAIT_L_PLANT

    for _ in range(WAIT_PLANT_MAX_STEPS):
        l_nf = _foot_normal_force(model, data, "L")
        r_nf = _foot_normal_force(model, data, "R")
        r_frac = _r_load_fraction(model, data)
        rd.peak_l_normal_post_heel = max(rd.peak_l_normal_post_heel, l_nf)
        rd.peak_r_normal_post_heel = max(rd.peak_r_normal_post_heel, r_nf)

        l_pos = _foot_pos(model, data, "L")
        rd.l_foot_max_drift_mm = max(
            rd.l_foot_max_drift_mm,
            float(np.linalg.norm(l_pos[:2] - heel_l_xy) * 1000.0),
        )

        if _foot_contact(model, data, "L") and l_nf >= PLANT_FORCE_N:
            plant_streak += 1
        else:
            plant_streak = 0

        ready = (
            plant_streak >= PLANT_SUSTAIN_STEPS
            and l_nf >= PLANT_FORCE_N
            and (not np.isnan(r_frac) and r_frac <= R_LOAD_DROP_THRESH)
            and r_nf <= R_NORMAL_UNLOAD_N
        )
        if ready:
            rd.l_plant_confirmed_step = st.diag.global_step
            rd.r_recovery_start_step = st.diag.global_step + 1
            print(
                f"\n  >> L PLANT confirmed step {rd.l_plant_confirmed_step} "
                f"L_nf={l_nf:.1f}N R_frac={r_frac:.3f}"
            )
            phase = RecoveryPhase.R_KNEE_LIFT
            break

        _apply_ctrl(env, model, data, st, stance_ctrl, viewer, slow)

    if phase == RecoveryPhase.WAIT_L_PLANT:
        rd.l_plant_confirmed_step = st.diag.global_step
        rd.r_recovery_start_step = st.diag.global_step + 1
        phase = RecoveryPhase.R_KNEE_LIFT
        print("\n  >> WAIT timeout; starting R recovery anyway")

    # --- R KNEE LIFT (clearance only) ---
    _print_phase(RecoveryPhase.R_KNEE_LIFT.value)
    knee_steps = 0
    while knee_steps < R_KNEE_LIFT_MAX_STEPS and phase == RecoveryPhase.R_KNEE_LIFT:
        r_unloaded = (
            not _foot_contact(model, data, "R")
            or _foot_normal_force(model, data, "R") < R_NORMAL_UNLOAD_N
        )
        if r_unloaded and rd.r_lift_off_step is None:
            rd.r_lift_off_step = st.diag.global_step
            rd.r_knee_cmd_at_lift_off = r_knee_cmd
            rd.r_knee_qpos_at_lift_off = float(data.qpos[QPOS_R_KNEE])
            rd.r_foot_airborne = True
            print(
                f"\n  >> R LIFT-OFF step {rd.r_lift_off_step} "
                f"R_knee_cmd={r_knee_cmd:.3f}"
            )

        r_knee_cmd = max(r_knee_cmd - R_KNEE_LIFT_STEP, R_KNEE_CLEARANCE_TARGET)
        ctrl = _set_leg_ctrl(stance_ctrl, r_knee=r_knee_cmd)
        _apply_ctrl(env, model, data, st, ctrl, viewer, slow)
        knee_steps += 1

        knee_at_target = r_knee_cmd <= R_KNEE_CLEARANCE_TARGET + 1e-6
        if knee_steps >= R_KNEE_MIN_RAMP_STEPS and (r_unloaded or knee_at_target):
            clearance_r_knee = r_knee_cmd
            phase = RecoveryPhase.R_HIP_RECOVERY
            print(
                f"\n  >> R knee clearance done: cmd={clearance_r_knee:.3f} "
                f"steps={knee_steps}"
            )
            break

    if phase == RecoveryPhase.R_KNEE_LIFT:
        clearance_r_knee = r_knee_cmd
        phase = RecoveryPhase.R_HIP_RECOVERY
        print("\n  >> R knee lift max steps; proceeding to hip recovery")

    # --- R HIP FORWARD RECOVERY (knee held) ---
    _print_phase(RecoveryPhase.R_HIP_RECOVERY.value)
    hip_steps = 0
    while hip_steps < R_HIP_RECOVERY_MAX_STEPS and phase == RecoveryPhase.R_HIP_RECOVERY:
        r_pos = _foot_pos(model, data, "R")
        clearance = (r_pos[2] - FLOOR_Z) * 1000.0
        r_fwd = _forward_mm(r_pos[1], r_recovery_ref_y)
        rd.peak_r_clearance_mm = max(rd.peak_r_clearance_mm, clearance)
        rd.peak_r_fwd_displacement_mm = max(rd.peak_r_fwd_displacement_mm, r_fwd)
        rd.peak_r_foot_fwd_vel_m_s = max(rd.peak_r_foot_fwd_vel_m_s, _r_foot_fwd_vel(model, data))

        if not _foot_contact(model, data, "R"):
            rd.r_foot_airborne = True

        r_hip_cmd = min(r_hip_cmd + R_HIP_RAMP_STEP, R_HIP_RECOVERY_TARGET)
        ctrl = _set_leg_ctrl(stance_ctrl, r_knee=clearance_r_knee, r_hip=r_hip_cmd)
        _apply_ctrl(env, model, data, st, ctrl, viewer, slow)
        hip_steps += 1

        if r_fwd >= MIN_R_FWD_RECOVERY_MM and r_hip_cmd >= R_HIP_RECOVERY_TARGET - 1e-6:
            print(f"\n  >> R recovery target reached: fwd={r_fwd:.1f}mm hip={r_hip_cmd:.3f}")
            break

    rd.r_hip_cmd_end = float(data.ctrl[IDX_R_HIP_PITCH])
    rd.r_hip_qpos_end = float(data.qpos[QPOS_R_HIP])

    r_pos = _foot_pos(model, data, "R")
    l_pos = _foot_pos(model, data, "L")
    rd.r_under_body = _forward_mm(r_pos[1], l_pos[1]) > 0.0

    _print_phase(RecoveryPhase.STOP.value)
    final_ctrl = st.ctrl.copy()
    for _ in range(STOP_STEPS):
        _apply_ctrl(env, model, data, st, final_ctrl, viewer, slow)

    rd.final_torso_tilt_rad = env._quat_tilt_rad()
    rd.final_forward_vel_m_s = _forward_vel(data)

    return rd


def print_summary(rd: RecoveryDiagnostics) -> None:
    d = rd.base
    print("\n" + "=" * 60)
    print("RIGHT LEG RECOVERY AFTER LEFT PLANT")
    print("=" * 60)
    print("--- LOCKED PREFIX ---")
    print(f"HEEL_TOUCHDOWN_STEP = {rd.heel_touchdown_step}")
    print(f"HEEL_L_REL_R_MM = {rd.heel_touchdown_l_rel_r_mm}")
    print(f"HEEL_L_NORMAL_N = {rd.heel_touchdown_l_nf}")
    print(f"HIP_SWING_START = {d.hip_swing_start_step}  CATCH_START = {d.catch_start_step}")
    print(f"PEAK_FWD_AIRBORNE_MM = {d.peak_fwd_airborne_mm:.1f}")

    print("\n--- POST LEFT TOUCHDOWN ---")
    print(f"L_PLANT_CONFIRMED_STEP = {rd.l_plant_confirmed_step}")
    print(f"PEAK_L_NORMAL_N = {rd.peak_l_normal_post_heel:.1f}")
    print(f"PEAK_R_NORMAL_N = {rd.peak_r_normal_post_heel:.1f}")
    print(f"L_FOOT_MAX_DRIFT_MM = {rd.l_foot_max_drift_mm:.1f}")

    print("\n--- RIGHT LEG RECOVERY ---")
    print(f"R_RECOVERY_START_STEP = {rd.r_recovery_start_step}")
    print(f"R_LIFT_OFF_STEP = {rd.r_lift_off_step}")
    print(f"R_KNEE_CMD_AT_LIFT_OFF = {rd.r_knee_cmd_at_lift_off}")
    print(f"R_KNEE_QPOS_AT_LIFT_OFF = {rd.r_knee_qpos_at_lift_off}")
    print(f"R_FOOT_AIRBORNE = {rd.r_foot_airborne}")
    print(f"PEAK_R_CLEARANCE_MM = {rd.peak_r_clearance_mm:.1f}")
    print(f"PEAK_R_FWD_DISPLACEMENT_MM = {rd.peak_r_fwd_displacement_mm:.1f}")
    print(f"PEAK_R_FOOT_FORWARD_VEL_M_S = {rd.peak_r_foot_fwd_vel_m_s:.4f}")
    print(f"R_HIP_CMD_END = {rd.r_hip_cmd_end}")
    print(f"R_HIP_QPOS_END = {rd.r_hip_qpos_end}")
    print(f"R_FOOT_UNDER_AHEAD_OF_L = {rd.r_under_body}")

    print("\n--- FINAL STATE ---")
    print(f"FINAL_TORSO_TILT_RAD = {rd.final_torso_tilt_rad:.3f}")
    print(f"FINAL_FORWARD_VEL_M_S = {rd.final_forward_vel_m_s:.3f}")

    if rd.r_foot_airborne and rd.peak_r_fwd_displacement_mm >= MIN_R_FWD_RECOVERY_MM:
        print("\nVERDICT: R leg recovered forward underneath the body.")
    elif rd.r_foot_airborne:
        print("\nVERDICT: R foot airborne but limited forward recovery.")
    else:
        print("\nVERDICT: R foot did not achieve clear airborne recovery.")

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
    p = argparse.ArgumentParser(description="Right leg recovery after left heel plant.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Right leg recovery: locked prefix -> L plant -> R knee -> R hip forward")
    print(f"R knee target {R_KNEE_CLEARANCE_TARGET:.2f}, R hip target {R_HIP_RECOVERY_TARGET:.2f}")
    print("Viewer: robot only.\n")

    if args.headless:
        rd = run_r_recovery_experiment(env, viewer=None, slow=False)
        print_summary(rd)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        rd = run_r_recovery_experiment(env, viewer=v, slow=args.slow)
    print_summary(rd)


if __name__ == "__main__":
    main(sys.argv[1:])
