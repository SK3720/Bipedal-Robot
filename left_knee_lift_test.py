"""Left knee lift diagnostic — can L knee alone lift L foot after R weight shift?

STAND -> lateral weight shift (hold) -> L knee ramp only -> hold.
No hip, ankle, R sagittal, push-off, or gait logic.

Evaluation only — does not modify robot.xml or other scripts.
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
FOOT_CONTACT_Z = 1.042
QPOS_L_KNEE = 14

IDX_L_HIP_ROLL = 5
IDX_L_KNEE = 7
IDX_R_HIP_ROLL = 10
IDX_R_ANKLE_ROLL = 14

STAND_STEPS = 500
SHIFT_RAMP_STEPS = 200
SHIFT_HOLD_STEPS = 1200
KNEE_RAMP_STEPS = 500
KNEE_HOLD_STEPS = 150

KNEE_START = -0.10
KNEE_END = -0.70

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    SHIFT_WEIGHT = "SHIFT WEIGHT RIGHT"
    KNEE_RAMP = "LIFT LEFT KNEE ONLY"
    KNEE_HOLD = "HOLD"


@dataclass
class PhaseSnapshot:
    phase: Phase
    l_knee_cmd: float
    l_knee_qpos: float
    l_foot_xyz: np.ndarray
    r_foot_xyz: np.ndarray
    l_normal_force: float
    r_normal_force: float
    r_load_fraction: float
    l_contact: bool
    torso_tilt_rad: float


@dataclass
class LiftMetrics:
    min_l_normal_force: float = float("inf")
    max_l_foot_height_mm: float = 0.0
    left_foot_left_ground: bool = False
    first_lift_knee_cmd: float | None = None
    first_lift_knee_qpos: float | None = None
    first_lift_clearance_mm: float | None = None


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _foot_pos(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    return data.xpos[model.body(f"{side}_foot").id].copy()


def _foot_contact(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> bool:
    for ci in range(data.ncon):
        for gid in (data.contact[ci].geom1, data.contact[ci].geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if f"{side}_foot_collision" in name:
                return True
    return False


def _foot_normal_force(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
    total = 0.0
    for ci in range(data.ncon):
        g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        if f"{side}_foot_collision" not in n1 and f"{side}_foot_collision" not in n2:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, wrench)
        total += max(0.0, float(wrench[0]))
    return total


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


def weight_shift_pose() -> np.ndarray:
    """Verified lateral shift — no R hip pitch or sagittal motion."""
    pose = DEFAULT_POSE.copy()
    pose[IDX_R_HIP_ROLL] = -0.04
    pose[IDX_R_ANKLE_ROLL] = 0.03
    pose[IDX_L_HIP_ROLL] = -0.03
    return pose


def knee_pose(knee_angle: float) -> np.ndarray:
    """Weight-shift pose with ONLY L knee changed."""
    pose = weight_shift_pose()
    pose[IDX_L_KNEE] = knee_angle
    return pose


def _snapshot(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: Phase,
) -> PhaseSnapshot:
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    total = l_nf + r_nf
    return PhaseSnapshot(
        phase=phase,
        l_knee_cmd=float(data.ctrl[IDX_L_KNEE]),
        l_knee_qpos=float(data.qpos[QPOS_L_KNEE]),
        l_foot_xyz=_foot_pos(model, data, "L"),
        r_foot_xyz=_foot_pos(model, data, "R"),
        l_normal_force=l_nf,
        r_normal_force=r_nf,
        r_load_fraction=(r_nf / total) if total > 1e-6 else float("nan"),
        l_contact=_foot_contact(model, data, "L"),
        torso_tilt_rad=env._quat_tilt_rad(),
    )


def _print_snapshot(snap: PhaseSnapshot) -> None:
    print(f"\n--- {snap.phase.value} ---")
    print(f"L knee commanded angle: {snap.l_knee_cmd:.4f} rad")
    print(f"L knee actual qpos: {snap.l_knee_qpos:.4f} rad")
    print(f"L foot XYZ: {snap.l_foot_xyz}")
    print(f"R foot XYZ: {snap.r_foot_xyz}")
    print(f"L normal force: {snap.l_normal_force:.2f} N")
    print(f"R normal force: {snap.r_normal_force:.2f} N")
    print(f"R load fraction: {snap.r_load_fraction:.3f}")
    print(f"L foot contact state: {snap.l_contact}")
    print(f"torso tilt: {snap.torso_tilt_rad:.4f} rad")


def _update_ramp_metrics(
    metrics: LiftMetrics,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    l_nf = _foot_normal_force(model, data, "L")
    l_pos = _foot_pos(model, data, "L")
    clearance_mm = (l_pos[2] - FLOOR_Z) * 1000.0
    metrics.min_l_normal_force = min(metrics.min_l_normal_force, l_nf)
    metrics.max_l_foot_height_mm = max(metrics.max_l_foot_height_mm, clearance_mm)

    airborne = not _foot_contact(model, data, "L")
    if airborne and not metrics.left_foot_left_ground:
        metrics.left_foot_left_ground = True
        metrics.first_lift_knee_cmd = float(data.ctrl[IDX_L_KNEE])
        metrics.first_lift_knee_qpos = float(data.qpos[QPOS_L_KNEE])
        metrics.first_lift_clearance_mm = clearance_mm


def run_left_knee_lift(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> tuple[list[PhaseSnapshot], LiftMetrics]:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    ctrl = DEFAULT_POSE.copy()
    snapshots: list[PhaseSnapshot] = []
    metrics = LiftMetrics()

    segments: list[tuple[Phase, np.ndarray, int, bool, bool]] = [
        (Phase.STAND, DEFAULT_POSE.copy(), STAND_STEPS, False, False),
        (Phase.SHIFT_WEIGHT, weight_shift_pose(), SHIFT_RAMP_STEPS, True, False),
        (Phase.SHIFT_WEIGHT, weight_shift_pose(), SHIFT_HOLD_STEPS, True, False),
        (Phase.KNEE_RAMP, knee_pose(KNEE_END), KNEE_RAMP_STEPS, False, True),
        (Phase.KNEE_HOLD, knee_pose(KNEE_END), KNEE_HOLD_STEPS, False, False),
    ]

    for phase, target, n_steps, linear, knee_ramp in segments:
        if knee_ramp:
            start_knee = KNEE_START
            ws = weight_shift_pose()
            for s in range(n_steps):
                t = (s + 1) / n_steps
                alpha = _smooth(t)
                knee_cmd = (1.0 - alpha) * start_knee + alpha * KNEE_END
                target_step = ws.copy()
                target_step[IDX_L_KNEE] = knee_cmd
                ctrl = _lerp_ctrl(ctrl, target_step, 1.0, cr)
                data.ctrl[:15] = ctrl
                mujoco.mj_step(model, data)
                _update_ramp_metrics(metrics, model, data)
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                    time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)
        else:
            for s in range(n_steps):
                t = (s + 1) / n_steps
                alpha = t if linear else _smooth(t)
                ctrl = _lerp_ctrl(ctrl, target, alpha, cr)
                data.ctrl[:15] = ctrl
                mujoco.mj_step(model, data)
                if phase == Phase.KNEE_HOLD:
                    _update_ramp_metrics(metrics, model, data)
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                    time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

        snap = _snapshot(env, model, data, phase)
        snapshots.append(snap)
        _print_snapshot(snap)

    return snapshots, metrics


def print_verdict(metrics: LiftMetrics) -> None:
    print("\n" + "=" * 60)
    print("LEFT KNEE LIFT TEST")
    print("=" * 60)
    print(f"min L normal force during knee ramp: {metrics.min_l_normal_force:.2f} N")
    print(f"max L foot height during knee ramp: {metrics.max_l_foot_height_mm:.1f} mm")
    print(f"L foot lost contact: {metrics.left_foot_left_ground}")
    print()
    if metrics.left_foot_left_ground:
        print("LEFT KNEE LIFT: YES")
        print(f"  first lift at L knee cmd = {metrics.first_lift_knee_cmd:.4f} rad")
        print(f"  first lift at L knee qpos = {metrics.first_lift_knee_qpos:.4f} rad")
        print(f"  foot clearance at first lift = {metrics.first_lift_clearance_mm:.1f} mm")
    else:
        print("LEFT KNEE LIFT: NO")
        print("  Mechanically: L knee flexed toward -0.70 rad but L foot never lost")
        print("  contact with the ground during the ramp.")
        if metrics.min_l_normal_force < float("inf"):
            print(f"  Minimum L normal force during ramp: {metrics.min_l_normal_force:.2f} N.")
        print(f"  Peak foot height above floor: {metrics.max_l_foot_height_mm:.1f} mm")
        print("  (standing clearance is ~32 mm above floor).")


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
    p = argparse.ArgumentParser(description="Left knee lift diagnostic only.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Left knee lift test — lateral weight shift, L knee only")
    print("No hip, ankle, R sagittal, or gait logic. Viewer: robot only.\n")

    if args.headless:
        _, metrics = run_left_knee_lift(env, viewer=None, slow=False)
        print_verdict(metrics)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        _, metrics = run_left_knee_lift(env, viewer=v, slow=args.slow)
    print_verdict(metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
