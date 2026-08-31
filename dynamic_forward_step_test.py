"""Dynamic forward step feasibility — one L-foot step along world -Y.

Controlled single-step experiment: rapid lateral unload, L lift, R push-off,
L forward swing, touchdown ahead of R, stabilize. No parameter sweep.

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
MIN_CLEARANCE_MM = 25.0
MIN_FORWARD_RATIO = 0.55
MAX_CATASTROPHIC_TILT_RAD = 0.50

# Actuator ctrl indices (verified interactive mapping)
IDX_L_HIP_ROLL = 5
IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_R_HIP_ROLL = 10
IDX_R_HIP_PITCH = 11
IDX_R_KNEE = 12
IDX_R_ANKLE_P = 13
IDX_R_ANKLE_ROLL = 14

# Verified sagittal signs (true_forward_step_test.py / interactive keys):
#   L forward: negative L hip pitch (key 2)
#   R forward: positive R hip pitch (key 7)
#   R knee negative / R ankle pitch negative: chest moves along world -Y under load

STAND_STEPS = 500
RAPID_UNLOAD_STEPS = 120
LEFT_LIFT_STEPS = 100
RIGHT_PUSH_STEPS = 60
LEFT_SWING_STEPS = 360
LEFT_TOUCHDOWN_STEPS = 300
STABILIZE_STEPS = 450

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    RAPID_UNLOAD = "RAPID LEFT UNLOAD"
    LEFT_LIFT = "LEFT FOOT LIFT"
    RIGHT_PUSH = "RIGHT PUSH-OFF"
    LEFT_SWING = "LEFT FORWARD SWING"
    LEFT_TOUCHDOWN = "LEFT TOUCHDOWN"
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
    torso_tilt_rad: float
    torso_angvel_rad_s: np.ndarray
    l_forward_mm: float
    l_height_mm: float
    r_drift_mm: float


@dataclass
class StepMetrics:
    left_foot_left_ground: bool = False
    left_touchdown_occurred: bool = False
    left_touchdown_ahead_of_right: bool = False
    peak_l_clearance_mm: float = 0.0
    peak_forward_displacement_mm: float = 0.0
    peak_forward_while_airborne_mm: float = 0.0
    peak_lateral_while_airborne_mm: float = 0.0
    r_foot_max_drift_mm: float = 0.0
    final_forward_displacement_mm: float = 0.0
    final_lateral_mm: float = 0.0
    final_torso_tilt_rad: float = 0.0
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


def _forward_mm(pos_y: float, y0: float) -> float:
    return float(-(pos_y - y0) * 1000.0)


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


def _apply_joint_values(base: np.ndarray, values: dict[int, float]) -> np.ndarray:
    pose = base.copy()
    for idx, val in values.items():
        pose[idx] = val
    return pose


def rapid_unload_pose() -> np.ndarray:
    """Verified lateral weight shift — no R hip pitch (roll/ankle only)."""
    return _apply_joint_values(
        DEFAULT_POSE,
        {
            IDX_R_HIP_ROLL: -0.04,
            IDX_R_ANKLE_ROLL: 0.03,
            IDX_L_HIP_ROLL: -0.03,
        },
    )


def left_lift_pose() -> np.ndarray:
    """Knee/ankle flexion first — priority is contact loss, not hip pitch yet."""
    return _apply_joint_values(
        rapid_unload_pose(),
        {
            IDX_L_KNEE: -0.62,
            IDX_L_ANKLE_P: 0.30,
        },
    )


def right_push_pose() -> np.ndarray:
    """Mild R sagittal extension while L remains lifted (verified forward signs)."""
    return _apply_joint_values(
        left_lift_pose(),
        {
            IDX_R_HIP_PITCH: 0.04,
            IDX_R_KNEE: -0.05,
            IDX_R_ANKLE_P: -0.04,
        },
    )


def left_forward_swing_pose() -> np.ndarray:
    """Larger swing than prior conservative attempt; R push continues."""
    return _apply_joint_values(
        rapid_unload_pose(),
        {
            IDX_L_HIP_PITCH: -0.50,
            IDX_L_KNEE: -0.62,
            IDX_L_ANKLE_P: 0.28,
            IDX_R_HIP_PITCH: 0.04,
            IDX_R_KNEE: -0.05,
            IDX_R_ANKLE_P: -0.04,
        },
    )


def left_touchdown_pose() -> np.ndarray:
    """Lower L foot while keeping it ahead of R along world -Y."""
    return _apply_joint_values(
        DEFAULT_POSE,
        {
            IDX_L_HIP_PITCH: -0.42,
            IDX_L_KNEE: -0.30,
            IDX_L_ANKLE_P: 0.16,
            IDX_R_HIP_PITCH: 0.02,
            IDX_R_HIP_ROLL: -0.04,
            IDX_R_ANKLE_ROLL: 0.03,
            IDX_L_HIP_ROLL: -0.03,
        },
    )


def stabilize_mid_pose() -> np.ndarray:
    """Partial recovery before returning to neutral stand."""
    return _apply_joint_values(
        DEFAULT_POSE,
        {
            IDX_L_HIP_PITCH: -0.25,
            IDX_L_KNEE: -0.12,
            IDX_L_ANKLE_P: 0.08,
        },
    )


def build_trajectory() -> list[tuple[Phase, np.ndarray, int, bool]]:
    """Return (phase, target, steps, use_linear_ramp)."""
    return [
        (Phase.STAND, DEFAULT_POSE.copy(), STAND_STEPS, False),
        (Phase.RAPID_UNLOAD, rapid_unload_pose(), RAPID_UNLOAD_STEPS, True),
        (Phase.LEFT_LIFT, left_lift_pose(), LEFT_LIFT_STEPS, True),
        (Phase.RIGHT_PUSH, right_push_pose(), RIGHT_PUSH_STEPS, True),
        (Phase.LEFT_SWING, left_forward_swing_pose(), LEFT_SWING_STEPS, False),
        (Phase.LEFT_TOUCHDOWN, left_touchdown_pose(), LEFT_TOUCHDOWN_STEPS, False),
        (Phase.STABILIZE, stabilize_mid_pose(), STABILIZE_STEPS // 2, False),
        (Phase.STABILIZE, DEFAULT_POSE.copy(), STABILIZE_STEPS - STABILIZE_STEPS // 2, False),
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
    return PhaseSnapshot(
        phase=phase,
        l_foot=l_pos,
        r_foot=r_pos,
        l_contact=_foot_contact_count(model, data, "L"),
        r_contact=_foot_contact_count(model, data, "R"),
        l_normal_force=l_nf,
        r_normal_force=r_nf,
        r_load_fraction=(r_nf / total) if total > 1e-6 else float("nan"),
        torso_tilt_rad=env._quat_tilt_rad(),
        torso_angvel_rad_s=_torso_angvel(data),
        l_forward_mm=_forward_mm(l_pos[1], stand_l[1]),
        l_height_mm=float((l_pos[2] - stand_l[2]) * 1000.0),
        r_drift_mm=_horizontal_drift_mm(r_pos, stand_r),
    )


def _print_snapshot(snap: PhaseSnapshot) -> None:
    print(f"\n--- {snap.phase.value} ---")
    print(f"L foot XYZ: {snap.l_foot}")
    print(f"R foot XYZ: {snap.r_foot}")
    print(f"L foot contact state: {snap.l_contact}")
    print(f"R foot contact state: {snap.r_contact}")
    print(f"L normal force: {snap.l_normal_force:.2f} N")
    print(f"R normal force: {snap.r_normal_force:.2f} N")
    print(f"R load fraction: {snap.r_load_fraction:.3f}")
    print(f"torso tilt: {snap.torso_tilt_rad:.4f} rad")
    print(f"torso angular velocity (rad/s): {snap.torso_angvel_rad_s}")
    print(f"L foot forward displacement along WORLD -Y (mm): {snap.l_forward_mm:.2f}")
    print(f"L foot height above STAND (mm): {snap.l_height_mm:.2f}")
    print(f"R foot displacement from initial position (mm): {snap.r_drift_mm:.2f}")


def _update_metrics(
    metrics: StepMetrics,
    phase: Phase,
    l_pos: np.ndarray,
    stand_l: np.ndarray,
    stand_r: np.ndarray,
    l_contact: int,
    r_pos: np.ndarray,
    l_nf: float,
) -> None:
    clearance_mm = (l_pos[2] - FLOOR_Z) * 1000.0
    metrics.peak_l_clearance_mm = max(metrics.peak_l_clearance_mm, clearance_mm)
    metrics.min_l_normal_force = min(metrics.min_l_normal_force, l_nf)

    fwd = _forward_mm(l_pos[1], stand_l[1])
    lat = float((l_pos[0] - stand_l[0]) * 1000.0)
    metrics.peak_forward_displacement_mm = max(metrics.peak_forward_displacement_mm, fwd)
    metrics.final_forward_displacement_mm = fwd
    metrics.final_lateral_mm = lat

    airborne = l_contact == 0 and l_pos[2] > FOOT_CONTACT_Z
    swing_phases = (Phase.LEFT_LIFT, Phase.RIGHT_PUSH, Phase.LEFT_SWING)
    if airborne and phase in swing_phases:
        metrics.left_foot_left_ground = True
        if fwd > metrics.peak_forward_while_airborne_mm:
            metrics.peak_forward_while_airborne_mm = fwd
            metrics.peak_lateral_while_airborne_mm = abs(lat)
        else:
            metrics.peak_forward_while_airborne_mm = max(
                metrics.peak_forward_while_airborne_mm, fwd,
            )

    metrics.r_foot_max_drift_mm = max(
        metrics.r_foot_max_drift_mm,
        _horizontal_drift_mm(r_pos, stand_r),
    )


def _evaluate_success(
    metrics: StepMetrics,
    final_l: np.ndarray,
    final_r: np.ndarray,
) -> tuple[bool, list[str], list[str]]:
    ok: list[str] = []
    fail: list[str] = []

    if metrics.left_foot_left_ground:
        ok.append("L foot lost ground contact during lift/swing")
    else:
        fail.append("L foot never lost ground contact")

    if metrics.left_touchdown_occurred:
        ok.append("L foot regained contact after airborne phase")
    else:
        fail.append("L foot did not regain contact after swing")

    if metrics.peak_l_clearance_mm >= MIN_CLEARANCE_MM:
        ok.append(f"peak L clearance {metrics.peak_l_clearance_mm:.1f} mm")
    else:
        fail.append(f"insufficient L clearance ({metrics.peak_l_clearance_mm:.1f} mm)")

    if metrics.left_touchdown_ahead_of_right:
        ok.append("L touchdown position ahead of R foot along world -Y")
    else:
        fail.append("L foot did not land ahead of R foot")

    air_fwd = metrics.peak_forward_while_airborne_mm
    air_lat = metrics.peak_lateral_while_airborne_mm
    air_horiz = np.hypot(air_fwd, air_lat)
    air_ratio = air_fwd / air_horiz if air_horiz > 1e-6 else 0.0
    if air_fwd > 5.0 and air_ratio >= MIN_FORWARD_RATIO:
        ok.append(
            f"forward motion primarily along world -Y while airborne "
            f"(peak {air_fwd:.1f} mm, ratio {air_ratio:.2f})"
        )
    else:
        fail.append(
            f"motion not primarily along world -Y while airborne "
            f"(peak fwd {air_fwd:.1f} mm, ratio {air_ratio:.2f})"
        )

    if metrics.left_foot_left_ground and metrics.peak_forward_while_airborne_mm > 5.0:
        ok.append(
            f"forward progress while airborne ({metrics.peak_forward_while_airborne_mm:.1f} mm)"
        )
    elif metrics.left_foot_left_ground:
        fail.append("foot left ground but showed little forward progress while airborne")
    else:
        fail.append("cannot assess airborne forward progress without contact loss")

    success = len(fail) == 0
    return success, ok, fail


def run_dynamic_forward_step(
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

    airborne_seen = False

    for phase, target, n_steps, linear_ramp in build_trajectory():
        for s in range(n_steps):
            t = (s + 1) / n_steps
            alpha = t if linear_ramp else _smooth(t)
            ctrl = _lerp_ctrl(ctrl, target, alpha, cr)
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)

            if phase == Phase.STAND and s == n_steps - 1:
                stand_l = _foot_pos(model, data, "L")
                stand_r = _foot_pos(model, data, "R")

            l_pos = _foot_pos(model, data, "L")
            r_pos = _foot_pos(model, data, "R")
            l_c = _foot_contact_count(model, data, "L")
            l_nf = _foot_normal_force(model, data, "L")
            _update_metrics(metrics, phase, l_pos, stand_l, stand_r, l_c, r_pos, l_nf)

            airborne = l_c == 0 and l_pos[2] > FOOT_CONTACT_Z
            if airborne:
                airborne_seen = True
            if (
                airborne_seen
                and l_c > 0
                and phase in (Phase.LEFT_SWING, Phase.LEFT_TOUCHDOWN, Phase.STABILIZE)
                and not metrics.left_touchdown_occurred
            ):
                metrics.left_touchdown_occurred = True
                ahead_mm = _forward_mm(l_pos[1], r_pos[1])
                metrics.left_touchdown_ahead_of_right = ahead_mm > 0.0

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

        snap = _snapshot(env, model, data, phase, stand_l, stand_r)
        snapshots.append(snap)
        _print_snapshot(snap)

    metrics.final_torso_tilt_rad = env._quat_tilt_rad()
    final_l = _foot_pos(model, data, "L")
    final_r = _foot_pos(model, data, "R")
    success, ok_reasons, fail_reasons = _evaluate_success(metrics, final_l, final_r)

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
    print("DYNAMIC FORWARD STEP SUMMARY")
    print("=" * 60)
    print(f"LEFT_FOOT_LEFT_GROUND = {m.left_foot_left_ground}")
    print(f"PEAK_L_CLEARANCE_MM = {m.peak_l_clearance_mm:.1f}")
    print(f"PEAK_FORWARD_DISPLACEMENT_MM = {m.peak_forward_displacement_mm:.1f}")
    print(f"PEAK_FORWARD_WHILE_AIRBORNE_MM = {m.peak_forward_while_airborne_mm:.1f}")
    print(f"R_FOOT_MAX_DRIFT_MM = {m.r_foot_max_drift_mm:.1f}")
    print(f"LEFT_TOUCHDOWN_OCCURRED = {m.left_touchdown_occurred}")
    print(f"LEFT_TOUCHDOWN_AHEAD_OF_RIGHT = {m.left_touchdown_ahead_of_right}")
    print(f"FINAL_FORWARD_DISPLACEMENT_MM = {m.final_forward_displacement_mm:.1f}")
    print(f"FINAL_TORSO_TILT = {m.final_torso_tilt_rad:.3f}")
    print()
    if result.success:
        print("VERDICT: SUCCESS - one genuine dynamic forward step.")
        for r in result.success_reasons:
            print(f"  + {r}")
    else:
        print("VERDICT: NOT A SUCCESSFUL DYNAMIC FORWARD STEP.")
        for r in result.failure_reasons:
            print(f"  - {r}")
        if result.success_reasons:
            print("\nPartial positives:")
            for r in result.success_reasons:
                print(f"  + {r}")
    print("\nCoordinate convention: forward = world -Y, lateral ~ world -X")
    print("R push signs: R hip pitch +, R knee -, R ankle pitch - (verified)")
    print("L swing sign: L hip pitch - (key 2, verified)")
    print("Diagnostic only — single controlled experiment, no parameter sweep.")


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
    p = argparse.ArgumentParser(description="One dynamic L-foot forward step along world -Y.")
    p.add_argument("--slow", action="store_true", help="Slow viewer playback.")
    p.add_argument("--headless", action="store_true", help="Run without viewer.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Dynamic forward step test — rapid unload, R push-off, L swing along world -Y")
    print("Unload: R hip roll -0.04, R ankle roll +0.03, L hip roll -0.03")
    print("R push: R hip pitch +0.04, R knee -0.05, R ankle pitch -0.04")
    print("L swing: hip -0.50, knee -0.62, ankle +0.28")
    print("Viewer: robot only (no debug overlays).\n")

    if args.headless:
        result = run_dynamic_forward_step(env, viewer=None, slow=False)
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
        result = run_dynamic_forward_step(env, viewer=v, slow=args.slow)
    print_summary(result)


if __name__ == "__main__":
    main(sys.argv[1:])
