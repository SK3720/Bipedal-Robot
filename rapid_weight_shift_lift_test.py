"""Rapid weight-shift + left knee lift — momentum-assisted unloading test.

Compares to slow quasi-static shift in left_knee_lift_test.py.
STAND -> rapid lateral shift (~40 steps) -> brief hold (~35 steps) -> L knee ramp.

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
QPOS_L_KNEE = 14

IDX_L_HIP_ROLL = 5
IDX_L_KNEE = 7
IDX_R_HIP_ROLL = 10
IDX_R_ANKLE_ROLL = 14

STAND_STEPS = 500
RAPID_SHIFT_STEPS = 40
BRIEF_HOLD_STEPS = 35
KNEE_RAMP_STEPS = 500
KNEE_HOLD_STEPS = 150

KNEE_START = -0.10
KNEE_END = -0.70

# Reference results from left_knee_lift_test.py (slow shift).
SLOW_SHIFT_R_LOAD = 0.718
SLOW_FIRST_LIFT_KNEE_CMD = -0.40
SLOW_FIRST_LIFT_CLEARANCE_MM = 33.9
SLOW_MAX_CLEARANCE_MM = 38.1

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    RAPID_SHIFT = "RAPID WEIGHT SHIFT RIGHT"
    BRIEF_HOLD = "BRIEF HOLD"
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
    chest_lateral_vel_m_s: float
    r_drift_mm: float


@dataclass
class LiftMetrics:
    peak_r_load_during_shift: float = 0.0
    r_load_after_rapid_shift: float = 0.0
    r_load_after_brief_hold: float = 0.0
    min_l_normal_before_knee: float = float("inf")
    min_l_normal_during_knee: float = float("inf")
    max_l_foot_height_mm: float = 0.0
    left_foot_left_ground: bool = False
    first_lift_knee_cmd: float | None = None
    first_lift_knee_qpos: float | None = None
    first_lift_clearance_mm: float | None = None
    first_lift_phase: str | None = None
    peak_lateral_chest_vel_m_s: float = 0.0
    peak_torso_tilt_rad: float = 0.0
    r_max_drift_mm: float = 0.0


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


def _r_load_fraction(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    total = l_nf + r_nf
    return (r_nf / total) if total > 1e-6 else float("nan")


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
    pose = DEFAULT_POSE.copy()
    pose[IDX_R_HIP_ROLL] = -0.04
    pose[IDX_R_ANKLE_ROLL] = 0.03
    pose[IDX_L_HIP_ROLL] = -0.03
    return pose


def _snapshot(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: Phase,
    stand_r_xy: np.ndarray,
) -> PhaseSnapshot:
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    total = l_nf + r_nf
    r_pos = _foot_pos(model, data, "R")
    return PhaseSnapshot(
        phase=phase,
        l_knee_cmd=float(data.ctrl[IDX_L_KNEE]),
        l_knee_qpos=float(data.qpos[QPOS_L_KNEE]),
        l_foot_xyz=_foot_pos(model, data, "L"),
        r_foot_xyz=r_pos,
        l_normal_force=l_nf,
        r_normal_force=r_nf,
        r_load_fraction=(r_nf / total) if total > 1e-6 else float("nan"),
        l_contact=_foot_contact(model, data, "L"),
        torso_tilt_rad=env._quat_tilt_rad(),
        chest_lateral_vel_m_s=float(data.qvel[0]),
        r_drift_mm=float(np.linalg.norm(r_pos[:2] - stand_r_xy) * 1000.0),
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
    print(f"chest lateral velocity (world X, m/s): {snap.chest_lateral_vel_m_s:.4f}")
    print(f"R foot drift from STAND (mm): {snap.r_drift_mm:.2f}")


def _update_metrics(
    metrics: LiftMetrics,
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: Phase,
    stand_r_xy: np.ndarray,
    *,
    during_shift: bool = False,
    before_knee: bool = False,
    during_knee: bool = False,
) -> None:
    l_nf = _foot_normal_force(model, data, "L")
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    clearance_mm = (l_pos[2] - FLOOR_Z) * 1000.0
    r_load = _r_load_fraction(model, data)
    tilt = env._quat_tilt_rad()
    lat_vel = abs(float(data.qvel[0]))
    r_drift = float(np.linalg.norm(r_pos[:2] - stand_r_xy) * 1000.0)

    metrics.peak_torso_tilt_rad = max(metrics.peak_torso_tilt_rad, tilt)
    metrics.peak_lateral_chest_vel_m_s = max(metrics.peak_lateral_chest_vel_m_s, lat_vel)
    metrics.r_max_drift_mm = max(metrics.r_max_drift_mm, r_drift)
    metrics.max_l_foot_height_mm = max(metrics.max_l_foot_height_mm, clearance_mm)

    if during_shift and not np.isnan(r_load):
        metrics.peak_r_load_during_shift = max(metrics.peak_r_load_during_shift, r_load)

    if before_knee:
        metrics.min_l_normal_before_knee = min(metrics.min_l_normal_before_knee, l_nf)

    if during_knee:
        metrics.min_l_normal_during_knee = min(metrics.min_l_normal_during_knee, l_nf)

    if not _foot_contact(model, data, "L") and not metrics.left_foot_left_ground:
        metrics.left_foot_left_ground = True
        metrics.first_lift_knee_cmd = float(data.ctrl[IDX_L_KNEE])
        metrics.first_lift_knee_qpos = float(data.qpos[QPOS_L_KNEE])
        metrics.first_lift_clearance_mm = clearance_mm
        metrics.first_lift_phase = phase.value


def run_rapid_weight_shift_lift(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> tuple[list[PhaseSnapshot], LiftMetrics]:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    ctrl = DEFAULT_POSE.copy()
    stand_r_xy = _foot_pos(model, data, "R")[:2].copy()
    snapshots: list[PhaseSnapshot] = []
    metrics = LiftMetrics()

    segments: list[tuple[Phase, np.ndarray, int, bool, str]] = [
        (Phase.STAND, DEFAULT_POSE.copy(), STAND_STEPS, False, "none"),
        (Phase.RAPID_SHIFT, weight_shift_pose(), RAPID_SHIFT_STEPS, True, "shift"),
        (Phase.BRIEF_HOLD, weight_shift_pose(), BRIEF_HOLD_STEPS, True, "before_knee"),
        (Phase.KNEE_RAMP, weight_shift_pose(), KNEE_RAMP_STEPS, False, "knee"),
        (Phase.KNEE_HOLD, weight_shift_pose(), KNEE_HOLD_STEPS, False, "knee"),
    ]

    ws = weight_shift_pose()

    for phase, target, n_steps, linear, mode in segments:
        for s in range(n_steps):
            if mode == "knee":
                t = (s + 1) / n_steps
                alpha = _smooth(t)
                knee_cmd = (1.0 - alpha) * KNEE_START + alpha * KNEE_END
                step_target = ws.copy()
                step_target[IDX_L_KNEE] = knee_cmd
                ctrl = _lerp_ctrl(ctrl, step_target, 1.0, cr)
            else:
                t = (s + 1) / n_steps
                alpha = t if linear else _smooth(t)
                ctrl = _lerp_ctrl(ctrl, target, alpha, cr)

            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)

            if phase == Phase.STAND and s == n_steps - 1:
                stand_r_xy = _foot_pos(model, data, "R")[:2].copy()

            _update_metrics(
                metrics,
                env,
                model,
                data,
                phase,
                stand_r_xy,
                during_shift=(mode == "shift"),
                before_knee=(mode in ("shift", "before_knee")),
                during_knee=(mode == "knee"),
            )

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

        snap = _snapshot(env, model, data, phase, stand_r_xy)
        snapshots.append(snap)
        _print_snapshot(snap)

        if phase == Phase.RAPID_SHIFT:
            metrics.r_load_after_rapid_shift = snap.r_load_fraction
        elif phase == Phase.BRIEF_HOLD:
            metrics.r_load_after_brief_hold = snap.r_load_fraction

    return snapshots, metrics


def print_verdict(metrics: LiftMetrics) -> None:
    print("\n" + "=" * 60)
    print("RAPID WEIGHT SHIFT + LEFT KNEE LIFT")
    print("=" * 60)
    print(f"PEAK_R_LOAD_DURING_SHIFT = {metrics.peak_r_load_during_shift:.3f}")
    print(f"R_LOAD_AFTER_RAPID_SHIFT = {metrics.r_load_after_rapid_shift:.3f}")
    print(f"R_LOAD_AFTER_BRIEF_HOLD = {metrics.r_load_after_brief_hold:.3f}")
    print(f"MIN_L_NORMAL_BEFORE_KNEE = {metrics.min_l_normal_before_knee:.2f} N")
    print(f"MIN_L_NORMAL_DURING_KNEE = {metrics.min_l_normal_during_knee:.2f} N")
    print(f"MAX_L_FOOT_HEIGHT_MM = {metrics.max_l_foot_height_mm:.1f}")
    print(f"L_FOOT_LOST_CONTACT = {metrics.left_foot_left_ground}")
    print(f"PEAK_LATERAL_CHEST_VEL_M_S = {metrics.peak_lateral_chest_vel_m_s:.4f}")
    print(f"PEAK_TORSO_TILT_RAD = {metrics.peak_torso_tilt_rad:.3f}")
    print(f"R_MAX_DRIFT_MM = {metrics.r_max_drift_mm:.1f}")
    print()
    if metrics.left_foot_left_ground:
        print("LEFT KNEE LIFT: YES")
        print(f"  first lift during: {metrics.first_lift_phase}")
        print(f"  L knee cmd = {metrics.first_lift_knee_cmd:.4f} rad")
        print(f"  L knee qpos = {metrics.first_lift_knee_qpos:.4f} rad")
        print(f"  clearance at first lift = {metrics.first_lift_clearance_mm:.1f} mm")
    else:
        print("LEFT KNEE LIFT: NO")
    print()
    print("--- COMPARISON vs slow shift (left_knee_lift_test.py) ---")
    print(f"  slow R load after hold:        {SLOW_SHIFT_R_LOAD:.3f}")
    print(f"  rapid R load after brief hold: {metrics.r_load_after_brief_hold:.3f}")
    print(f"  slow first lift knee cmd:      {SLOW_FIRST_LIFT_KNEE_CMD:.2f} rad")
    if metrics.first_lift_knee_cmd is not None:
        print(f"  rapid first lift knee cmd:     {metrics.first_lift_knee_cmd:.2f} rad")
    print(f"  slow max clearance:            {SLOW_MAX_CLEARANCE_MM:.1f} mm")
    print(f"  rapid max clearance:           {metrics.max_l_foot_height_mm:.1f} mm")
    print()
    better_unload = (
        metrics.peak_r_load_during_shift > SLOW_SHIFT_R_LOAD + 0.05
        or metrics.min_l_normal_before_knee < 3.0
    )
    easier_lift = (
        metrics.left_foot_left_ground
        and metrics.first_lift_knee_cmd is not None
        and metrics.first_lift_knee_cmd > SLOW_FIRST_LIFT_KNEE_CMD
    )
    more_clearance = metrics.max_l_foot_height_mm > SLOW_MAX_CLEARANCE_MM + 5.0
    if better_unload or easier_lift or more_clearance:
        print("HYPOTHESIS: momentum-assisted shift helps unloading/lift.")
        if better_unload:
            print("  + higher transient R load or lower L load before knee")
        if easier_lift and metrics.first_lift_knee_cmd is not None:
            print(f"  + contact lost at less knee flex ({metrics.first_lift_knee_cmd:.2f} vs {SLOW_FIRST_LIFT_KNEE_CMD:.2f})")
        if more_clearance:
            print(f"  + more foot clearance ({metrics.max_l_foot_height_mm:.1f} vs {SLOW_MAX_CLEARANCE_MM:.1f} mm)")
    else:
        print("HYPOTHESIS: rapid shift did NOT clearly outperform slow shift.")
        print("  Momentum did not produce substantially better L-foot unloading.")


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
    p = argparse.ArgumentParser(description="Rapid weight shift + L knee lift test.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Rapid weight shift + left knee lift")
    print(f"Shift: {RAPID_SHIFT_STEPS} steps linear, hold {BRIEF_HOLD_STEPS} steps, then L knee ramp")
    print("Viewer: robot only.\n")

    if args.headless:
        _, metrics = run_rapid_weight_shift_lift(env, viewer=None, slow=False)
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
        _, metrics = run_rapid_weight_shift_lift(env, viewer=v, slow=args.slow)
    print_verdict(metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
