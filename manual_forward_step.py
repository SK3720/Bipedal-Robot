"""Manual forward-step stance-drift diagnostic — isolated UNLOAD/LIFT experiment.

Left leg swings; right leg is stance. Anatomical forward ~ world -Y.

Evaluation / visualization only — does not modify robot.xml, biped_env, or PPO artifacts.
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

from biped_env import (
    BipedalWalkEnv,
    CHEST_Z_CONTACT,
    DEFAULT_POSE,
    SETTLE_STEPS,
    STANDING_QUAT,
)

FLOOR_Z = 1.0
FOOT_CONTACT_Z = 1.042
MIN_AIRBORNE_CLEARANCE_M = 0.04
STANCE_DRIFT_SMALL_MM = 5.0

IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_R_HIP_ROLL = 10
IDX_R_HIP_PITCH = 11
IDX_R_KNEE = 12
IDX_R_ANKLE_P = 13

STAND_STEPS = 500
STAND_HOLD_STEPS = 350
UNLOAD_STEPS = 400
STANCE_LOCK_STEPS = 300
LIFT_SHALLOW_STEPS = 220
LIFT_DEEP_STEPS = 220
LIFT_FULL_STEPS = 280
SWING_STEPS = 550
PLACE_STEPS = 380
STABILIZE_STEPS = 500

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    STAND_HOLD = "STAND HOLD"
    UNLOAD_LEFT = "UNLOAD LEFT"
    STANCE_LOCK = "STANCE LOCK"
    LIFT_SHALLOW = "LIFT LEFT - SHALLOW"
    LIFT_DEEP = "LIFT LEFT - DEEP"
    LIFT_FULL = "LIFT LEFT - FULL"
    FORWARD_SWING = "FORWARD SWING"
    PLACE = "PLACE"
    STABILIZE = "STABILIZE"


@dataclass
class PhaseRecord:
    phase: Phase
    l_foot: np.ndarray
    r_foot: np.ndarray
    l_disp_mm: np.ndarray
    r_disp_mm: np.ndarray
    r_stance_drift_mm: float
    l_contact: int
    r_contact: int
    torso_tilt_rad: float
    torso_angvel_rad_s: np.ndarray


@dataclass
class StepDiagnostics:
    stand_l_foot: np.ndarray
    stand_r_foot: np.ndarray
    phase_records: list[PhaseRecord]
    did_left_foot_leave_ground: bool
    peak_clearance_mm: float
    peak_forward_while_airborne_mm: float
    anatomical_forward: np.ndarray
    lateral_axis: np.ndarray
    final_swing_pos: np.ndarray
    final_stance_pos: np.ndarray
    final_tilt_rad: float
    stance_lock_r_drift_mm: float
    lift_shallow_r_drift_mm: float
    lift_deep_r_drift_mm: float
    lift_full_r_drift_mm: float


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _foot_pos(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    return data.xpos[model.body(f"{side}_foot").id].copy()


def _foot_contacts(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> int:
    n = 0
    for ci in range(data.ncon):
        for gid in (data.contact[ci].geom1, data.contact[ci].geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if f"{side}_foot_collision" in name:
                n += 1
                break
    return n


def _horizontal_drift_mm(pos: np.ndarray, ref: np.ndarray) -> float:
    delta = pos - ref
    return float(np.linalg.norm(delta[:2]) * 1000.0)


def _torso_angvel(data: mujoco.MjData) -> np.ndarray:
    return data.qvel[3:6].copy()


def compute_axes(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    lateral = r_pos - l_pos
    lateral[2] = 0.0
    lat_norm = np.linalg.norm(lateral)
    if lat_norm < 1e-6:
        lateral = np.array([1.0, 0.0, 0.0])
    else:
        lateral /= lat_norm
    up = np.array([0.0, 0.0, 1.0])
    forward = np.cross(up, lateral)
    fn = np.linalg.norm(forward)
    if fn < 1e-6:
        forward = np.array([0.0, -1.0, 0.0])
    else:
        forward /= fn
    return forward, lateral


def _reset_pose_only(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = DEFAULT_POSE
    data.qvel[:] = 0.0
    data.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(model, data)


def _lerp_ctrl(ctrl: np.ndarray, target: np.ndarray, alpha: float, cr: np.ndarray) -> np.ndarray:
    return np.clip((1.0 - alpha) * ctrl + alpha * target, cr[:, 0], cr[:, 1])


def _legacy_swing_target() -> np.ndarray:
    """FORWARD SWING target preserved exactly from prior trajectory."""
    swing = DEFAULT_POSE.copy()
    swing[IDX_R_HIP_ROLL] = -0.06
    swing[IDX_R_HIP_PITCH] = 0.10
    swing[IDX_R_KNEE] = -0.06
    swing[IDX_L_HIP_PITCH] = -0.36
    swing[IDX_L_KNEE] = -0.42
    swing[IDX_L_ANKLE_P] = 0.16
    return swing


def _legacy_place_target() -> np.ndarray:
    """PLACE target preserved exactly from prior trajectory."""
    place = DEFAULT_POSE.copy()
    place[IDX_L_HIP_PITCH] = -0.28
    place[IDX_L_KNEE] = -0.28
    place[IDX_L_ANKLE_P] = 0.12
    return place


def build_manual_trajectory() -> list[tuple[Phase, np.ndarray, int]]:
    stand = DEFAULT_POSE.copy()

    unload = stand.copy()
    unload[IDX_R_HIP_PITCH] = 0.07

    lift_shallow = stand.copy()
    lift_shallow[IDX_L_KNEE] = -0.15

    lift_deep = stand.copy()
    lift_deep[IDX_L_KNEE] = -0.35

    lift_full = stand.copy()
    lift_full[IDX_L_HIP_PITCH] = -0.24
    lift_full[IDX_L_KNEE] = -0.50
    lift_full[IDX_L_ANKLE_P] = 0.24

    return [
        (Phase.STAND, stand, STAND_STEPS),
        (Phase.STAND_HOLD, stand, STAND_HOLD_STEPS),
        (Phase.UNLOAD_LEFT, unload, UNLOAD_STEPS),
        (Phase.STANCE_LOCK, unload, STANCE_LOCK_STEPS),
        (Phase.LIFT_SHALLOW, lift_shallow, LIFT_SHALLOW_STEPS),
        (Phase.LIFT_DEEP, lift_deep, LIFT_DEEP_STEPS),
        (Phase.LIFT_FULL, lift_full, LIFT_FULL_STEPS),
        (Phase.FORWARD_SWING, _legacy_swing_target(), SWING_STEPS),
        (Phase.PLACE, _legacy_place_target(), PLACE_STEPS),
        (Phase.STABILIZE, stand, STABILIZE_STEPS),
    ]


def _print_phase_status(
    phase: Phase,
    l_pos: np.ndarray,
    r_pos: np.ndarray,
    stand_l: np.ndarray,
    stand_r: np.ndarray,
    l_contact: int,
    r_contact: int,
    tilt_rad: float,
    angvel: np.ndarray,
) -> float:
    l_disp = (l_pos - stand_l) * 1000.0
    r_disp = (r_pos - stand_r) * 1000.0
    r_drift = _horizontal_drift_mm(r_pos, stand_r)
    print(f"\n--- {phase.value} ---")
    print(f"L foot XYZ: {l_pos}")
    print(f"R foot XYZ: {r_pos}")
    print(f"L foot displacement from STAND (mm): {l_disp}")
    print(f"R foot displacement from STAND (mm): {r_disp}")
    print(f"R_STANCE_DRIFT_MM = {r_drift:.2f}")
    print(f"L contact: {l_contact}")
    print(f"R contact: {r_contact}")
    print(f"torso tilt: {tilt_rad:.4f} rad")
    print(f"torso angular velocity (rad/s): {angvel}")
    if phase == Phase.STANCE_LOCK:
        small = r_drift < STANCE_DRIFT_SMALL_MM
        print(
            f"STANCE LOCK drift check: R_STANCE_DRIFT_MM {'<' if small else '>='} "
            f"{STANCE_DRIFT_SMALL_MM:.0f} mm -> {'SMALL (planted)' if small else 'NOT SMALL (skidding)'}"
        )
    return r_drift


def run_manual_forward_step(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> StepDiagnostics:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset_pose_only(model, data)

    segments = build_manual_trajectory()
    fwd, lat = compute_axes(model, data)
    ctrl = DEFAULT_POSE.copy()

    stand_l = _foot_pos(model, data, "L")
    stand_r = _foot_pos(model, data, "R")
    phase_records: list[PhaseRecord] = []

    lost_contact = False
    peak_clearance = 0.0
    peak_forward_airborne = 0.0
    swing_init = stand_l.copy()

    stance_lock_r_drift = lift_shallow_r_drift = lift_deep_r_drift = lift_full_r_drift = 0.0

    for phase, target, n_steps in segments:
        for s in range(n_steps):
            alpha = 1.0 if phase == Phase.STANCE_LOCK else _smooth((s + 1) / n_steps)
            ctrl = _lerp_ctrl(ctrl, target, alpha, cr)
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)

            l_pos = _foot_pos(model, data, "L")
            l_c = _foot_contacts(model, data, "L")
            clearance = l_pos[2] - FLOOR_Z
            peak_clearance = max(peak_clearance, clearance)

            if l_c == 0 and l_pos[2] > FOOT_CONTACT_Z:
                lost_contact = True
                fwd_mm = float(np.dot(l_pos - swing_init, fwd) * 1000)
                peak_forward_airborne = max(peak_forward_airborne, fwd_mm)

            if phase == Phase.STAND and s == n_steps - 1:
                stand_l = l_pos.copy()
                stand_r = _foot_pos(model, data, "R")
                swing_init = stand_l.copy()

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

        l_pos = _foot_pos(model, data, "L")
        r_pos = _foot_pos(model, data, "R")
        l_c = _foot_contacts(model, data, "L")
        r_c = _foot_contacts(model, data, "R")
        tilt = env._quat_tilt_rad()
        angvel = _torso_angvel(data)
        r_drift = _print_phase_status(
            phase, l_pos, r_pos, stand_l, stand_r, l_c, r_c, tilt, angvel,
        )

        phase_records.append(PhaseRecord(
            phase=phase,
            l_foot=l_pos.copy(),
            r_foot=r_pos.copy(),
            l_disp_mm=(l_pos - stand_l) * 1000.0,
            r_disp_mm=(r_pos - stand_r) * 1000.0,
            r_stance_drift_mm=r_drift,
            l_contact=l_c,
            r_contact=r_c,
            torso_tilt_rad=tilt,
            torso_angvel_rad_s=angvel.copy(),
        ))

        if phase == Phase.STANCE_LOCK:
            stance_lock_r_drift = r_drift
        elif phase == Phase.LIFT_SHALLOW:
            lift_shallow_r_drift = r_drift
        elif phase == Phase.LIFT_DEEP:
            lift_deep_r_drift = r_drift
        elif phase == Phase.LIFT_FULL:
            lift_full_r_drift = r_drift

    final_l = _foot_pos(model, data, "L")
    final_r = _foot_pos(model, data, "R")

    return StepDiagnostics(
        stand_l_foot=stand_l,
        stand_r_foot=stand_r,
        phase_records=phase_records,
        did_left_foot_leave_ground=lost_contact and peak_clearance >= MIN_AIRBORNE_CLEARANCE_M,
        peak_clearance_mm=peak_clearance * 1000,
        peak_forward_while_airborne_mm=peak_forward_airborne,
        anatomical_forward=fwd,
        lateral_axis=lat,
        final_swing_pos=final_l,
        final_stance_pos=final_r,
        final_tilt_rad=env._quat_tilt_rad(),
        stance_lock_r_drift_mm=stance_lock_r_drift,
        lift_shallow_r_drift_mm=lift_shallow_r_drift,
        lift_deep_r_drift_mm=lift_deep_r_drift,
        lift_full_r_drift_mm=lift_full_r_drift,
    )


def print_final_diagnostics(diag: StepDiagnostics) -> None:
    print("\n" + "=" * 60)
    print("STANCE-DRIFT DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"STAND reference R foot: {diag.stand_r_foot}")
    print(f"STAND reference L foot: {diag.stand_l_foot}")
    print()
    print("R_STANCE_DRIFT_MM at key phases:")
    print(f"  STANCE LOCK end:      {diag.stance_lock_r_drift_mm:.2f} mm")
    print(f"  LIFT SHALLOW end:     {diag.lift_shallow_r_drift_mm:.2f} mm")
    print(f"  LIFT DEEP end:        {diag.lift_deep_r_drift_mm:.2f} mm")
    print(f"  LIFT FULL end:        {diag.lift_full_r_drift_mm:.2f} mm")
    print()
    print(f"L foot left ground (informational only): {diag.did_left_foot_leave_ground}")
    print(f"PEAK_CLEARANCE (informational): {diag.peak_clearance_mm:.1f} mm")
    print(f"Final torso tilt: {diag.final_tilt_rad:.4f} rad")
    print()
    print(
        "This experiment does NOT count as a successful forward step. "
        "Success requires R foot to remain approximately planted through UNLOAD/LIFT."
    )
    planted_through_lift = diag.lift_full_r_drift_mm < STANCE_DRIFT_SMALL_MM * 4
    if diag.stance_lock_r_drift_mm < STANCE_DRIFT_SMALL_MM and planted_through_lift:
        print("RESULT: R stance foot remained reasonably planted through LIFT stages.")
    else:
        print("RESULT: R stance foot drifted significantly during UNLOAD/LIFT.")
        if diag.stance_lock_r_drift_mm >= STANCE_DRIFT_SMALL_MM:
            print(f"  - Drift already {diag.stance_lock_r_drift_mm:.1f} mm at STANCE LOCK end.")
        if diag.lift_full_r_drift_mm >= STANCE_DRIFT_SMALL_MM:
            print(f"  - Drift reached {diag.lift_full_r_drift_mm:.1f} mm by LIFT FULL end.")


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
    p = argparse.ArgumentParser(
        description="Stance-drift diagnostic: UNLOAD/LIFT with frozen R leg.",
    )
    p.add_argument("--slow", action="store_true", help="Slow-motion viewer playback")
    p.add_argument("--headless", action="store_true", help="Run without viewer")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Manual forward step - stance drift diagnostic")
    print("Inspect: STAND -> UNLOAD -> STANCE LOCK -> SHALLOW -> DEEP -> FULL LIFT")
    print("Console diagnostics only; viewer shows robot with no overlays.\n")

    if args.headless:
        diag = run_manual_forward_step(env, viewer=None, slow=False)
        print_final_diagnostics(diag)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset_pose_only(env.model, env.data)
        v.sync()
        diag = run_manual_forward_step(env, viewer=v, slow=args.slow)
    print_final_diagnostics(diag)


if __name__ == "__main__":
    main(sys.argv[1:])
