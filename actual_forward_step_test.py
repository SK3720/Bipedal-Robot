"""Actual forward step test — one dynamic L-foot step exploiting transient R-load window.

Uses SMALL LATERAL WEIGHT SHIFT for dynamic unloading, then immediate L lift and
negative L hip-pitch forward swing along world -Y. No parameter sweep.

Evaluation only — does not modify robot.xml, biped_env, or PPO artifacts.
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
    STANDING_QUAT,
)

FLOOR_Z = 1.0
FOOT_CONTACT_Z = 1.042
MIN_CLEARANCE_M = 0.04
MIN_FORWARD_MM = 30.0
MIN_FORWARD_RATIO = 0.60
MAX_R_DRIFT_SWING_MM = 15.0
MAX_CATASTROPHIC_TILT_RAD = 0.45

# Verified anatomical / world axes
WORLD_FORWARD = np.array([0.0, -1.0, 0.0])  # anatomical forward = world -Y
WORLD_LATERAL = np.array([1.0, 0.0, 0.0])  # lateral = world X

IDX_L_HIP_ROLL = 5
IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_R_HIP_ROLL = 10
IDX_R_HIP_PITCH = 11
IDX_R_ANKLE_ROLL = 14

STAND_STEPS = 500
WEIGHT_SHIFT_STEPS = 520
IMMEDIATE_LIFT_STEPS = 220
LIFT_STEPS = 300
FORWARD_SWING_STEPS = 500
PLACE_STEPS = 360
STABILIZE_STEPS = 450

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    WEIGHT_SHIFT = "WEIGHT SHIFT"
    IMMEDIATE_LIFT = "IMMEDIATE LIFT"
    LIFT = "LIFT"
    FORWARD_SWING = "FORWARD SWING"
    PLACE = "PLACE"
    STABILIZE = "STABILIZE"


@dataclass
class PhaseSnapshot:
    phase: Phase
    l_foot: np.ndarray
    r_foot: np.ndarray
    l_contact: int
    r_contact: int
    l_normal_force: float
    r_normal_force: float
    r_load_fraction: float
    r_stance_drift_mm: float
    l_forward_mm: float
    l_lateral_mm: float
    l_height_mm: float
    torso_tilt_rad: float
    torso_angvel_rad_s: np.ndarray


@dataclass
class StepMetrics:
    lost_contact: bool = False
    peak_clearance_mm: float = 0.0
    peak_forward_mm: float = 0.0
    peak_forward_airborne_mm: float = 0.0
    max_r_drift_swing_mm: float = 0.0
    final_forward_mm: float = 0.0
    final_lateral_mm: float = 0.0
    final_ahead_mm: float = 0.0
    final_tilt_rad: float = 0.0
    min_l_normal_force: float = field(default_factory=lambda: float("inf"))


@dataclass
class RunResult:
    stand_l_foot: np.ndarray
    stand_r_foot: np.ndarray
    snapshots: list[PhaseSnapshot]
    metrics: StepMetrics
    success: bool
    success_reasons: list[str]
    failure_reasons: list[str]


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _foot_pos(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    return data.xpos[model.body(f"{side}_foot").id].copy()


def _foot_contact_count(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> int:
    n = 0
    for ci in range(data.ncon):
        for gid in (data.contact[ci].geom1, data.contact[ci].geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if f"{side}_foot_collision" in name:
                n += 1
                break
    return n


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


def _horizontal_drift_mm(pos: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm((pos - ref)[:2]) * 1000.0)


def _torso_angvel(data: mujoco.MjData) -> np.ndarray:
    return data.qvel[3:6].copy()


def _foot_kinematics(l_pos: np.ndarray, stand_l: np.ndarray) -> tuple[float, float, float]:
    delta = l_pos - stand_l
    forward_mm = float(-delta[1] * 1000.0)
    lateral_mm = float(delta[0] * 1000.0)
    height_mm = float(delta[2] * 1000.0)
    return forward_mm, lateral_mm, height_mm


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


def weight_shift_pose() -> np.ndarray:
    """SMALL LATERAL WEIGHT SHIFT — transient ~80% R-load configuration."""
    pose = DEFAULT_POSE.copy()
    pose[IDX_R_HIP_PITCH] = 0.07
    pose[IDX_R_HIP_ROLL] = -0.04
    pose[IDX_R_ANKLE_ROLL] = 0.03
    pose[IDX_L_HIP_ROLL] = -0.03
    return pose


def immediate_lift_pose() -> np.ndarray:
    """Begin lift during dynamic window — conservative L knee/ankle only."""
    pose = weight_shift_pose()
    pose[IDX_L_KNEE] = -0.10
    pose[IDX_L_ANKLE_P] = 0.06
    return pose


def lift_pose() -> np.ndarray:
    """Conservative clearance — L hip pitch + knee + ankle; R leg unchanged."""
    pose = weight_shift_pose()
    pose[IDX_L_HIP_PITCH] = -0.14
    pose[IDX_L_KNEE] = -0.30
    pose[IDX_L_ANKLE_P] = 0.12
    return pose


def forward_swing_pose() -> np.ndarray:
    """Negative L hip pitch drives foot along world -Y; R leg at weight-shift."""
    pose = weight_shift_pose()
    pose[IDX_L_HIP_PITCH] = -0.28
    pose[IDX_L_KNEE] = -0.34
    pose[IDX_L_ANKLE_P] = 0.12
    return pose


def place_pose() -> np.ndarray:
    """Lower L foot ahead — from prior manual_forward_step demonstration."""
    pose = DEFAULT_POSE.copy()
    pose[IDX_L_HIP_PITCH] = -0.28
    pose[IDX_L_KNEE] = -0.28
    pose[IDX_L_ANKLE_P] = 0.12
    return pose


def build_trajectory() -> list[tuple[Phase, np.ndarray, int]]:
    return [
        (Phase.STAND, DEFAULT_POSE.copy(), STAND_STEPS),
        (Phase.WEIGHT_SHIFT, weight_shift_pose(), WEIGHT_SHIFT_STEPS),
        (Phase.IMMEDIATE_LIFT, immediate_lift_pose(), IMMEDIATE_LIFT_STEPS),
        (Phase.LIFT, lift_pose(), LIFT_STEPS),
        (Phase.FORWARD_SWING, forward_swing_pose(), FORWARD_SWING_STEPS),
        (Phase.PLACE, place_pose(), PLACE_STEPS),
        (Phase.STABILIZE, DEFAULT_POSE.copy(), STABILIZE_STEPS),
    ]


def _snapshot(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: Phase,
    stand_l: np.ndarray,
    stand_r: np.ndarray,
) -> PhaseSnapshot:
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    total = l_nf + r_nf
    fwd, lat, ht = _foot_kinematics(l_pos, stand_l)
    return PhaseSnapshot(
        phase=phase,
        l_foot=l_pos,
        r_foot=r_pos,
        l_contact=_foot_contact_count(model, data, "L"),
        r_contact=_foot_contact_count(model, data, "R"),
        l_normal_force=l_nf,
        r_normal_force=r_nf,
        r_load_fraction=(r_nf / total) if total > 1e-6 else float("nan"),
        r_stance_drift_mm=_horizontal_drift_mm(r_pos, stand_r),
        l_forward_mm=fwd,
        l_lateral_mm=lat,
        l_height_mm=ht,
        torso_tilt_rad=env._quat_tilt_rad(),
        torso_angvel_rad_s=_torso_angvel(data),
    )


def _print_snapshot(snap: PhaseSnapshot) -> None:
    print(f"\n--- {snap.phase.value} ---")
    print(f"L foot XYZ: {snap.l_foot}")
    print(f"R foot XYZ: {snap.r_foot}")
    print(f"L contact: {snap.l_contact}")
    print(f"R contact: {snap.r_contact}")
    print(f"L_NORMAL_FORCE = {snap.l_normal_force:.2f} N")
    print(f"R_NORMAL_FORCE = {snap.r_normal_force:.2f} N")
    print(f"R_LOAD_FRACTION = {snap.r_load_fraction:.3f}")
    print(f"R stance-foot displacement from STAND (mm) = {snap.r_stance_drift_mm:.2f}")
    print(f"torso tilt: {snap.torso_tilt_rad:.4f} rad")
    print(f"torso angular velocity (rad/s): {snap.torso_angvel_rad_s}")
    print(f"L-foot forward displacement along world -Y (mm) = {snap.l_forward_mm:.2f}")
    print(f"L-foot lateral displacement (world X, mm) = {snap.l_lateral_mm:.2f}")
    print(f"L-foot height above STAND (mm) = {snap.l_height_mm:.2f}")


def _update_metrics(
    metrics: StepMetrics,
    snap_phase: Phase,
    l_pos: np.ndarray,
    stand_l: np.ndarray,
    stand_r: np.ndarray,
    l_contact: int,
    r_pos: np.ndarray,
    l_nf: float,
    env: BipedalWalkEnv,
) -> None:
    clearance_mm = (l_pos[2] - FLOOR_Z) * 1000.0
    metrics.peak_clearance_mm = max(metrics.peak_clearance_mm, clearance_mm)
    metrics.min_l_normal_force = min(metrics.min_l_normal_force, l_nf)

    fwd, lat, _ = _foot_kinematics(l_pos, stand_l)
    metrics.peak_forward_mm = max(metrics.peak_forward_mm, fwd)
    metrics.final_forward_mm = fwd
    metrics.final_lateral_mm = lat

    airborne = l_contact == 0 and l_pos[2] > FOOT_CONTACT_Z
    if airborne:
        metrics.lost_contact = True
        metrics.peak_forward_airborne_mm = max(metrics.peak_forward_airborne_mm, fwd)

    if snap_phase in (Phase.FORWARD_SWING, Phase.PLACE):
        drift = _horizontal_drift_mm(r_pos, stand_r)
        metrics.max_r_drift_swing_mm = max(metrics.max_r_drift_swing_mm, drift)

    metrics.final_tilt_rad = env._quat_tilt_rad()


def _evaluate_success(
    metrics: StepMetrics,
    stand_l: np.ndarray,
    final_l: np.ndarray,
    final_r: np.ndarray,
) -> tuple[bool, list[str], list[str]]:
    reasons_ok: list[str] = []
    reasons_fail: list[str] = []

    if metrics.lost_contact:
        reasons_ok.append("L foot genuinely lost contact during swing")
    else:
        reasons_fail.append("L foot never lost contact")

    if metrics.peak_clearance_mm >= MIN_CLEARANCE_M * 1000.0:
        reasons_ok.append(f"peak clearance {metrics.peak_clearance_mm:.1f} mm >= 40 mm")
    else:
        reasons_fail.append(f"insufficient clearance ({metrics.peak_clearance_mm:.1f} mm)")

    horiz = np.hypot(metrics.final_forward_mm, metrics.final_lateral_mm)
    ratio = metrics.final_forward_mm / horiz if horiz > 1e-6 else 0.0
    if metrics.peak_forward_mm >= MIN_FORWARD_MM and ratio >= MIN_FORWARD_RATIO:
        reasons_ok.append(
            f"predominantly forward motion (peak {metrics.peak_forward_mm:.1f} mm along -Y, "
            f"ratio {ratio:.2f})"
        )
    else:
        reasons_fail.append(
            f"insufficient forward -Y motion (peak {metrics.peak_forward_mm:.1f} mm, ratio {ratio:.2f})"
        )

    ahead_mm = float(-(final_l[1] - final_r[1]) * 1000.0)
    metrics.final_ahead_mm = ahead_mm
    start_ahead = float(-(stand_l[1] - final_r[1]) * 1000.0)
    if metrics.final_forward_mm > 0 and ahead_mm > start_ahead:
        reasons_ok.append("L foot finished ahead of its starting -Y position")
    else:
        reasons_fail.append(
            f"L foot did not land meaningfully ahead (forward {metrics.final_forward_mm:.1f} mm)"
        )

    if metrics.max_r_drift_swing_mm <= MAX_R_DRIFT_SWING_MM:
        reasons_ok.append(f"R foot drift during swing {metrics.max_r_drift_swing_mm:.1f} mm")
    else:
        reasons_fail.append(
            f"R foot drifted {metrics.max_r_drift_swing_mm:.1f} mm during swing (limit {MAX_R_DRIFT_SWING_MM:.0f} mm)"
        )

    if metrics.final_tilt_rad < MAX_CATASTROPHIC_TILT_RAD:
        reasons_ok.append(f"torso tilt {metrics.final_tilt_rad:.3f} rad - no catastrophic fall")
    else:
        reasons_fail.append(f"catastrophic tilt ({metrics.final_tilt_rad:.3f} rad)")

    success = len(reasons_fail) == 0
    return success, reasons_ok, reasons_fail


def run_actual_forward_step(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> RunResult:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset_pose_only(model, data)

    ctrl = DEFAULT_POSE.copy()
    stand_l = _foot_pos(model, data, "L")
    stand_r = _foot_pos(model, data, "R")
    snapshots: list[PhaseSnapshot] = []
    metrics = StepMetrics()

    for phase, target, n_steps in build_trajectory():
        for s in range(n_steps):
            ctrl = _lerp_ctrl(ctrl, target, _smooth((s + 1) / n_steps), cr)
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)

            if phase == Phase.STAND and s == n_steps - 1:
                stand_l = _foot_pos(model, data, "L")
                stand_r = _foot_pos(model, data, "R")

            l_pos = _foot_pos(model, data, "L")
            r_pos = _foot_pos(model, data, "R")
            l_c = _foot_contact_count(model, data, "L")
            l_nf = _foot_normal_force(model, data, "L")
            _update_metrics(metrics, phase, l_pos, stand_l, stand_r, l_c, r_pos, l_nf, env)

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

        snap = _snapshot(env, model, data, phase, stand_l, stand_r)
        snapshots.append(snap)
        _print_snapshot(snap)

    final_l = _foot_pos(model, data, "L")
    final_r = _foot_pos(model, data, "R")
    success, ok_reasons, fail_reasons = _evaluate_success(metrics, stand_l, final_l, final_r)

    return RunResult(
        stand_l_foot=stand_l,
        stand_r_foot=stand_r,
        snapshots=snapshots,
        metrics=metrics,
        success=success,
        success_reasons=ok_reasons,
        failure_reasons=fail_reasons,
    )


def print_summary(result: RunResult) -> None:
    m = result.metrics
    print("\n" + "=" * 60)
    print("ACTUAL FORWARD STEP TEST - SUMMARY")
    print("=" * 60)
    print(f"STAND L foot: {result.stand_l_foot}")
    print(f"STAND R foot: {result.stand_r_foot}")
    print()
    print("Aggregate metrics:")
    print(f"  LOST_CONTACT = {m.lost_contact}")
    print(f"  PEAK_CLEARANCE_MM = {m.peak_clearance_mm:.1f}")
    print(f"  PEAK_FORWARD_MM (world -Y) = {m.peak_forward_mm:.1f}")
    print(f"  PEAK_FORWARD_WHILE_AIRBORNE_MM = {m.peak_forward_airborne_mm:.1f}")
    print(f"  MAX_R_DRIFT_DURING_SWING_MM = {m.max_r_drift_swing_mm:.1f}")
    print(f"  MIN_L_NORMAL_FORCE = {m.min_l_normal_force:.2f} N")
    print(f"  FINAL_FORWARD_MM = {m.final_forward_mm:.1f}")
    print(f"  FINAL_LATERAL_MM = {m.final_lateral_mm:.1f}")
    print(f"  FINAL_TILT_RAD = {m.final_tilt_rad:.3f}")
    print()
    if result.success:
        print("VERDICT: SUCCESS - genuine forward step criteria met.")
        for r in result.success_reasons:
            print(f"  + {r}")
    else:
        print("VERDICT: NOT A SUCCESSFUL FORWARD STEP.")
        for r in result.failure_reasons:
            print(f"  - {r}")
        if result.success_reasons:
            print("\nPartial positives:")
            for r in result.success_reasons:
                print(f"  + {r}")
    print("\nCoordinate convention: forward = world -Y, lateral = world X")
    print("Diagnostic only — single mechanical experiment, no parameter sweep.")


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
    p = argparse.ArgumentParser(description="One actual forward L-foot step test.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Actual forward step test — dynamic R-load window + L swing along world -Y")
    print("Weight shift: R pitch +0.07, R roll -0.04, R ankle roll +0.03, L roll -0.03")
    print("R leg fixed at weight-shift during lift/swing. No R crouch. Viewer: robot only.\n")

    if args.headless:
        result = run_actual_forward_step(env, viewer=None, slow=False)
        print_summary(result)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset_pose_only(env.model, env.data)
        v.sync()
        result = run_actual_forward_step(env, viewer=v, slow=args.slow)
    print_summary(result)


if __name__ == "__main__":
    main(sys.argv[1:])
