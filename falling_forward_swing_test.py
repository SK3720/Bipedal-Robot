"""Falling forward swing — visual motion diagnostic.

STAND -> forward fall -> rapid R weight shift ->
concurrent L knee lift + L hip forward swing -> touchdown -> stop.

Pure motion test: no stabilization, no overlays. The robot may fall.

Evaluation only — does not modify robot.xml, biped_env, or other scripts.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass
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
    STANDING_QUAT,
)

FLOOR_Z = 1.0
QPOS_L_HIP = 13
QPOS_L_KNEE = 14

IDX_L_HIP_ROLL = 5
IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_R_HIP_ROLL = 10
IDX_R_HIP_PITCH = 11
IDX_R_KNEE = 12
IDX_R_ANKLE_P = 13
IDX_R_ANKLE_ROLL = 14

STAND_STEPS = 500
FALL_RAMP_STEPS = 80
FALL_MOMENTUM_STEPS = 180
RAPID_SHIFT_STEPS = 40
POST_SHIFT_MAX_STEPS = 30
SWING_RAMP_STEPS = 55
SWING_HOLD_STEPS = 280
TOUCHDOWN_STEPS = 220
STOP_STEPS = 90

L_HIP_SWING = -0.48
L_KNEE_SWING = -0.62
L_ANKLE_SWING = 0.18

L_HIP_TOUCHDOWN = -0.36
L_KNEE_TOUCHDOWN = -0.16
L_ANKLE_TOUCHDOWN = 0.08

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    FORWARD_FALL = "FORWARD FALL"
    RAPID_SHIFT = "RAPID RIGHT SHIFT"
    SWING = "LEFT KNEE + HIP SWING"
    TOUCHDOWN = "TOUCHDOWN"
    STOP = "STOP"


@dataclass
class Diagnostics:
    global_step: int = 0
    peak_forward_vel_m_s: float = 0.0
    max_l_clearance_mm: float = 0.0
    left_foot_airborne: bool = False
    contact_loss_step: int | None = None
    contact_loss_phase: str | None = None
    swing_start_step: int | None = None
    left_touchdown_occurred: bool = False
    touchdown_step: int | None = None
    touchdown_l_rel_r_mm: float | None = None
    touchdown_tilt_rad: float | None = None
    touchdown_forward_vel_m_s: float | None = None
    airborne_seen: bool = False


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _forward_mm(y: float, y_ref: float) -> float:
    return float(-(y - y_ref) * 1000.0)


def _forward_vel(data: mujoco.MjData) -> float:
    return float(-data.qvel[1])


def _foot_pos(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    return data.xpos[model.body(f"{side}_foot").id].copy()


def _foot_contact(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> bool:
    for ci in range(data.ncon):
        for gid in (data.contact[ci].geom1, data.contact[ci].geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if f"{side}_foot_collision" in name:
                return True
    return False


def _reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = DEFAULT_POSE
    data.qvel[:] = 0.0
    data.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(model, data)


def _lerp_ctrl(ctrl: np.ndarray, target: np.ndarray, alpha: float, cr: np.ndarray) -> np.ndarray:
    return np.clip((1.0 - alpha) * ctrl + alpha * target, cr[:, 0], cr[:, 1])


def _apply_joints(base: np.ndarray, values: dict[int, float]) -> np.ndarray:
    pose = base.copy()
    for idx, val in values.items():
        pose[idx] = val
    return pose


def forward_lean_pose() -> np.ndarray:
    return _apply_joints(
        DEFAULT_POSE,
        {
            IDX_L_HIP_PITCH: -0.14,
            IDX_R_HIP_PITCH: 0.14,
            IDX_L_KNEE: -0.06,
            IDX_R_KNEE: -0.06,
            IDX_L_ANKLE_P: -0.12,
            IDX_R_ANKLE_P: -0.12,
        },
    )


def rapid_shift_pose() -> np.ndarray:
    return _apply_joints(
        forward_lean_pose(),
        {
            IDX_R_HIP_ROLL: -0.04,
            IDX_R_ANKLE_ROLL: 0.03,
            IDX_L_HIP_ROLL: -0.03,
        },
    )


def shift_base_pose() -> np.ndarray:
    return rapid_shift_pose()


def concurrent_swing_pose(knee: float, hip: float, ankle: float) -> np.ndarray:
    return _apply_joints(
        shift_base_pose(),
        {
            IDX_L_KNEE: knee,
            IDX_L_HIP_PITCH: hip,
            IDX_L_ANKLE_P: ankle,
        },
    )


def touchdown_pose() -> np.ndarray:
    return concurrent_swing_pose(L_KNEE_TOUCHDOWN, L_HIP_TOUCHDOWN, L_ANKLE_TOUCHDOWN)


def _print_boundary(phase: Phase) -> None:
    print(f"\n{'=' * 60}")
    print(f"PHASE: {phase.value}")
    print(f"{'=' * 60}")


def _print_status(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    label: str = "status",
) -> None:
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    clearance = (l_pos[2] - FLOOR_Z) * 1000.0
    airborne = not _foot_contact(model, data, "L")
    print(f"  [{label}]")
    print(f"    L knee cmd={data.ctrl[IDX_L_KNEE]:.3f}  actual={data.qpos[QPOS_L_KNEE]:.3f} rad")
    print(f"    L hip  cmd={data.ctrl[IDX_L_HIP_PITCH]:.3f}  actual={data.qpos[QPOS_L_HIP]:.3f} rad")
    print(f"    L foot XYZ = {l_pos}")
    print(f"    L clearance = {clearance:.1f} mm")
    print(f"    L-R forward = {_forward_mm(l_pos[1], r_pos[1]):.1f} mm")
    print(f"    torso tilt = {env._quat_tilt_rad():.3f} rad")
    print(f"    forward vel = {_forward_vel(data):.3f} m/s")
    print(f"    L foot airborne = {airborne}")


def _update_diag(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    diag: Diagnostics,
    phase: Phase,
) -> None:
    diag.global_step += 1
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    l_contact = _foot_contact(model, data, "L")
    clearance = (l_pos[2] - FLOOR_Z) * 1000.0
    fwd_vel = _forward_vel(data)

    diag.peak_forward_vel_m_s = max(diag.peak_forward_vel_m_s, fwd_vel)
    diag.max_l_clearance_mm = max(diag.max_l_clearance_mm, clearance)

    if not l_contact and not diag.left_foot_airborne:
        diag.left_foot_airborne = True
        diag.contact_loss_step = diag.global_step
        diag.contact_loss_phase = phase.value

    if diag.airborne_seen and l_contact and not diag.left_touchdown_occurred:
        diag.left_touchdown_occurred = True
        diag.touchdown_step = diag.global_step
        diag.touchdown_l_rel_r_mm = _forward_mm(l_pos[1], r_pos[1])
        diag.touchdown_tilt_rad = env._quat_tilt_rad()
        diag.touchdown_forward_vel_m_s = fwd_vel

    if not l_contact:
        diag.airborne_seen = True


def _sync_viewer(viewer: mujoco.viewer.Handle | None, slow: bool) -> bool:
    if viewer is None:
        return True
    if not viewer.is_running():
        return False
    viewer.sync()
    time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)
    return True


def _sim_step(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: np.ndarray,
    diag: Diagnostics,
    phase: Phase,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
) -> bool:
    data.ctrl[:15] = ctrl
    mujoco.mj_step(model, data)
    _update_diag(env, model, data, diag, phase)
    return _sync_viewer(viewer, slow)


def run_experiment(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> Diagnostics:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)
    diag = Diagnostics()
    ctrl = DEFAULT_POSE.copy()
    lean = forward_lean_pose()
    shifted = rapid_shift_pose()

    _print_boundary(Phase.STAND)
    for s in range(STAND_STEPS):
        ctrl = _lerp_ctrl(ctrl, DEFAULT_POSE, _smooth((s + 1) / STAND_STEPS), cr)
        if not _sim_step(env, model, data, ctrl, diag, Phase.STAND, viewer, slow):
            return diag
    _print_status(env, model, data, label="end STAND")

    _print_boundary(Phase.FORWARD_FALL)
    for s in range(FALL_RAMP_STEPS):
        alpha = (s + 1) / FALL_RAMP_STEPS
        ctrl = _lerp_ctrl(ctrl, lean, alpha, cr)
        if not _sim_step(env, model, data, ctrl, diag, Phase.FORWARD_FALL, viewer, slow):
            return diag
    for _ in range(FALL_MOMENTUM_STEPS):
        ctrl = lean.copy()
        if not _sim_step(env, model, data, ctrl, diag, Phase.FORWARD_FALL, viewer, slow):
            return diag
    _print_status(env, model, data, label="end FORWARD FALL")

    _print_boundary(Phase.RAPID_SHIFT)
    shift_start = ctrl.copy()
    swing_ready = False
    for s in range(RAPID_SHIFT_STEPS):
        alpha = (s + 1) / RAPID_SHIFT_STEPS
        ctrl = _lerp_ctrl(shift_start, shifted, alpha, cr)
        if not _sim_step(env, model, data, ctrl, diag, Phase.RAPID_SHIFT, viewer, slow):
            return diag
        if diag.left_foot_airborne:
            swing_ready = True
            break

    post = 0
    while not swing_ready and post < POST_SHIFT_MAX_STEPS:
        ctrl = shifted.copy()
        if not _sim_step(env, model, data, ctrl, diag, Phase.RAPID_SHIFT, viewer, slow):
            return diag
        post += 1
        if diag.left_foot_airborne:
            swing_ready = True

    _print_status(env, model, data, label="end RAPID RIGHT SHIFT")

    _print_boundary(Phase.SWING)
    diag.swing_start_step = diag.global_step
    swing_start_ctrl = ctrl.copy()
    knee0 = float(swing_start_ctrl[IDX_L_KNEE])
    hip0 = float(swing_start_ctrl[IDX_L_HIP_PITCH])
    ankle0 = float(swing_start_ctrl[IDX_L_ANKLE_P])

    total_swing = SWING_RAMP_STEPS + SWING_HOLD_STEPS
    for s in range(total_swing):
        if s < SWING_RAMP_STEPS:
            t = (s + 1) / SWING_RAMP_STEPS
            alpha = _smooth(t)
            knee = (1.0 - alpha) * knee0 + alpha * L_KNEE_SWING
            hip = (1.0 - alpha) * hip0 + alpha * L_HIP_SWING
            ankle = (1.0 - alpha) * ankle0 + alpha * L_ANKLE_SWING
        else:
            knee, hip, ankle = L_KNEE_SWING, L_HIP_SWING, L_ANKLE_SWING
        ctrl = concurrent_swing_pose(knee, hip, ankle)
        if not _sim_step(env, model, data, ctrl, diag, Phase.SWING, viewer, slow):
            return diag
    _print_status(env, model, data, label="end LEFT KNEE + HIP SWING")

    _print_boundary(Phase.TOUCHDOWN)
    td_start = ctrl.copy()
    td_target = touchdown_pose()
    for s in range(TOUCHDOWN_STEPS):
        alpha = _smooth((s + 1) / TOUCHDOWN_STEPS)
        ctrl = _lerp_ctrl(td_start if s == 0 else ctrl, td_target, alpha, cr)
        if not _sim_step(env, model, data, ctrl, diag, Phase.TOUCHDOWN, viewer, slow):
            return diag
    _print_status(env, model, data, label="end TOUCHDOWN")

    _print_boundary(Phase.STOP)
    hold = touchdown_pose()
    for s in range(STOP_STEPS):
        ctrl = hold.copy()
        if not _sim_step(env, model, data, ctrl, diag, Phase.STOP, viewer, slow):
            return diag
    _print_status(env, model, data, label="end STOP")

    return diag


def print_summary(diag: Diagnostics) -> None:
    print("\n" + "=" * 60)
    print("FALLING FORWARD SWING - SUMMARY")
    print("=" * 60)
    print(f"PEAK_FORWARD_VEL_M_S = {diag.peak_forward_vel_m_s:.4f}")
    print(f"MAX_L_CLEARANCE_MM = {diag.max_l_clearance_mm:.1f}")
    print(f"L_FOOT_AIRBORNE = {diag.left_foot_airborne}")
    print(f"L_CONTACT_LOSS_STEP = {diag.contact_loss_step}")
    print(f"L_CONTACT_LOSS_PHASE = {diag.contact_loss_phase}")
    print(f"SWING_START_STEP = {diag.swing_start_step}")
    print(f"L_TOUCHDOWN_OCCURRED = {diag.left_touchdown_occurred}")
    if diag.touchdown_l_rel_r_mm is not None:
        print(f"FIRST_TOUCHDOWN_L_REL_R_MM = {diag.touchdown_l_rel_r_mm:.1f}")
        print(f"TOUCHDOWN_TILT_RAD = {diag.touchdown_tilt_rad:.3f}")
        print(f"TOUCHDOWN_FORWARD_VEL_M_S = {diag.touchdown_forward_vel_m_s:.4f}")
    print()
    print("Phases: STAND -> FORWARD FALL -> RAPID RIGHT SHIFT ->")
    print("        LEFT KNEE + HIP SWING -> TOUCHDOWN -> STOP")
    print(f"Swing targets: L hip {L_HIP_SWING:.2f} rad, L knee {L_KNEE_SWING:.2f} rad (concurrent)")
    print("Forward = world -Y. Single fixed trajectory - no auto-tuning.")


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
    p = argparse.ArgumentParser(description="Falling forward swing visual diagnostic.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Falling forward swing: fall -> rapid shift -> concurrent L knee+hip swing")
    print(f"Concurrent swing: L hip {L_HIP_SWING:.2f} rad, L knee {L_KNEE_SWING:.2f} rad")
    print("Viewer: robot only.\n")

    if args.headless:
        diag = run_experiment(env, viewer=None, slow=False)
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
        diag = run_experiment(env, viewer=v, slow=args.slow)
    print_summary(diag)


if __name__ == "__main__":
    main(sys.argv[1:])
