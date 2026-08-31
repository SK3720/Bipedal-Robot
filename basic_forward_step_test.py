"""Basic forward step — weight shift, simultaneous L lift+swing, place, stop.

Sequence: STAND -> SHIFT WEIGHT RIGHT (hold) -> LIFT+SWING (concurrent) ->
BRING LEFT FOOT DOWN -> STOP.

Verified joint signs:
  L hip pitch negative  = forward (key 2)
  L knee negative       = flex / raise foot (all prior lift scripts)
  L ankle pitch positive = dorsiflex / clearance assist (prior swing scripts)

Evaluation only — does not modify robot.xml, actuators, or other scripts.
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

IDX_L_HIP_ROLL = 5
IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_R_HIP_ROLL = 10
IDX_R_HIP_PITCH = 11
IDX_R_ANKLE_ROLL = 14

STAND_STEPS = 500
SHIFT_RAMP_STEPS = 200
SHIFT_HOLD_STEPS = 1200
SWING_STEPS = 450
PLACE_STEPS = 320
STOP_STEPS = 250

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    SHIFT_WEIGHT = "SHIFT WEIGHT RIGHT"
    LIFT_AND_SWING = "LIFT + FORWARD SWING LEFT"
    PLACE = "BRING LEFT FOOT DOWN"
    STOP = "STOP"


@dataclass
class StepMetrics:
    left_foot_left_ground: bool = False
    peak_l_clearance_mm: float = 0.0
    peak_l_forward_mm: float = 0.0
    final_l_forward_mm: float = 0.0
    peak_l_rel_r_forward_mm: float = 0.0
    final_l_rel_r_forward_mm: float = 0.0
    r_max_drift_mm: float = 0.0
    final_torso_tilt_rad: float = 0.0


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
    for i, v in values.items():
        pose[i] = v
    return pose


def weight_shift_pose() -> np.ndarray:
    """Verified small lateral + R pitch load from single_support_test."""
    return _apply_joints(
        DEFAULT_POSE,
        {
            IDX_R_HIP_PITCH: 0.07,
            IDX_R_HIP_ROLL: -0.04,
            IDX_R_ANKLE_ROLL: 0.03,
            IDX_L_HIP_ROLL: -0.03,
        },
    )


def lift_and_swing_pose() -> np.ndarray:
    """Knee up + hip forward together — single concurrent target."""
    return _apply_joints(
        weight_shift_pose(),
        {
            IDX_L_KNEE: -0.48,
            IDX_L_HIP_PITCH: -0.20,
            IDX_L_ANKLE_P: 0.15,
        },
    )


def place_pose() -> np.ndarray:
    """Lower L foot while keeping it ahead."""
    return _apply_joints(
        weight_shift_pose(),
        {
            IDX_L_HIP_PITCH: -0.16,
            IDX_L_KNEE: -0.12,
            IDX_L_ANKLE_P: 0.06,
        },
    )


def build_trajectory() -> list[tuple[Phase, np.ndarray, int, bool]]:
    ws = weight_shift_pose()
    swing = lift_and_swing_pose()
    place = place_pose()
    return [
        (Phase.STAND, DEFAULT_POSE.copy(), STAND_STEPS, False),
        (Phase.SHIFT_WEIGHT, ws, SHIFT_RAMP_STEPS, True),
        (Phase.SHIFT_WEIGHT, ws, SHIFT_HOLD_STEPS, True),
        (Phase.LIFT_AND_SWING, swing, SWING_STEPS, True),
        (Phase.PLACE, place, PLACE_STEPS, False),
        (Phase.STOP, place, STOP_STEPS, False),
    ]


def run_basic_forward_step(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> StepMetrics:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    ctrl = DEFAULT_POSE.copy()
    stand_l_y = _foot_pos(model, data, "L")[1]
    stand_r_xy = _foot_pos(model, data, "R")[:2].copy()
    metrics = StepMetrics()
    last_phase: Phase | None = None

    for phase, target, n_steps, linear in build_trajectory():
        if phase != last_phase:
            print(f"\n--- {phase.value} ---")
            last_phase = phase

        for s in range(n_steps):
            t = (s + 1) / n_steps
            alpha = t if linear else _smooth(t)
            ctrl = _lerp_ctrl(ctrl, target, alpha, cr)
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)

            if phase == Phase.STAND and s == n_steps - 1:
                stand_l_y = _foot_pos(model, data, "L")[1]
                stand_r_xy = _foot_pos(model, data, "R")[:2].copy()

            l_pos = _foot_pos(model, data, "L")
            r_pos = _foot_pos(model, data, "R")
            clearance = (l_pos[2] - FLOOR_Z) * 1000.0
            l_fwd = -(l_pos[1] - stand_l_y) * 1000.0
            l_rel_r = -(l_pos[1] - r_pos[1]) * 1000.0
            r_drift = float(np.linalg.norm(r_pos[:2] - stand_r_xy) * 1000.0)

            metrics.peak_l_clearance_mm = max(metrics.peak_l_clearance_mm, clearance)
            metrics.peak_l_forward_mm = max(metrics.peak_l_forward_mm, l_fwd)
            metrics.peak_l_rel_r_forward_mm = max(metrics.peak_l_rel_r_forward_mm, l_rel_r)
            metrics.r_max_drift_mm = max(metrics.r_max_drift_mm, r_drift)
            metrics.final_l_forward_mm = l_fwd
            metrics.final_l_rel_r_forward_mm = l_rel_r

            if not _foot_contact(model, data, "L") and l_pos[2] > FOOT_CONTACT_Z:
                metrics.left_foot_left_ground = True

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

        l_pos = _foot_pos(model, data, "L")
        r_pos = _foot_pos(model, data, "R")
        print(
            f"  end: L_fwd={-(l_pos[1]-stand_l_y)*1000:.1f}mm "
            f"L-R_fwd={-(l_pos[1]-r_pos[1])*1000:.1f}mm "
            f"clearance={(l_pos[2]-FLOOR_Z)*1000:.0f}mm "
            f"R_drift={np.linalg.norm(r_pos[:2]-stand_r_xy)*1000:.1f}mm "
            f"tilt={env._quat_tilt_rad():.3f}"
        )

    metrics.final_torso_tilt_rad = env._quat_tilt_rad()
    return metrics


def print_summary(m: StepMetrics) -> None:
    print("\n" + "=" * 60)
    print("BASIC FORWARD STEP SUMMARY")
    print("=" * 60)
    print(f"L_FOOT_LEFT_GROUND = {m.left_foot_left_ground}")
    print(f"PEAK_L_CLEARANCE_MM = {m.peak_l_clearance_mm:.1f}")
    print(f"PEAK_L_FORWARD_MM (world -Y from start) = {m.peak_l_forward_mm:.1f}")
    print(f"FINAL_L_FORWARD_MM = {m.final_l_forward_mm:.1f}")
    print(f"PEAK_L_REL_R_FORWARD_MM = {m.peak_l_rel_r_forward_mm:.1f}")
    print(f"FINAL_L_REL_R_FORWARD_MM = {m.final_l_rel_r_forward_mm:.1f}")
    print(f"R_FOOT_MAX_DRIFT_MM = {m.r_max_drift_mm:.1f}")
    print(f"FINAL_TORSO_TILT_RAD = {m.final_torso_tilt_rad:.3f}")
    print("\nTargets: weight shift R pitch +0.07, lateral roll shift")
    print("Swing (concurrent): L knee -0.48, L hip -0.20, L ankle +0.15")
    print("Place: L hip -0.16, L knee -0.12, L ankle +0.06")


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
    p = argparse.ArgumentParser(description="Basic L-foot forward step test.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Basic forward step: shift R -> concurrent L lift+swing -> place -> stop")
    print("Viewer: robot only.\n")

    if args.headless:
        metrics = run_basic_forward_step(env, viewer=None, slow=False)
        print_summary(metrics)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        metrics = run_basic_forward_step(env, viewer=v, slow=args.slow)
    print_summary(metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
