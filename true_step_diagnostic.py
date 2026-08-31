"""Conservative true-step diagnostic — relative foot motion, not world displacement.

One deliberately mild trajectory to test whether L can move forward relative
to planted R, touch down ahead, and avoid catastrophic rotation.

Does NOT modify robot.xml, actuators, or existing trajectory scripts.
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

# Conservative success thresholds (first experiment — not full walking).
MIN_CLEARANCE_MM = 20.0
MIN_REL_FORWARD_MM = 40.0
MAX_R_DRIFT_SWING_MM = 20.0
MAX_TOUCHDOWN_TILT_RAD = 0.60

IDX_L_HIP_ROLL = 5
IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_R_HIP_ROLL = 10
IDX_R_HIP_PITCH = 11
IDX_R_ANKLE_ROLL = 14

QPOS_L_HIP = 13
QPOS_L_KNEE = 14
QPOS_L_ANKLE = 15

STAND_STEPS = 500
UNLOAD_STEPS = 160
LIFT_STEPS = 240
HOLD_STEPS = 350
MOVE_FORWARD_STEPS = 360
TOUCHDOWN_STEPS = 300
STOP_STEPS = 120

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    RAPID_UNLOAD = "RAPID LATERAL UNLOAD"
    LIFT = "LIFT LEFT FOOT"
    HOLD = "HOLD SINGLE SUPPORT"
    MOVE_FORWARD = "MOVE LEFT FOOT FORWARD RELATIVE TO R"
    TOUCHDOWN = "TOUCHDOWN"
    STOP = "STOP"


@dataclass
class PhaseSnapshot:
    phase: Phase
    pelvis_xyz: np.ndarray
    pelvis_pitch_rad: float
    l_foot_xyz: np.ndarray
    r_foot_xyz: np.ndarray
    l_rel_pelvis: np.ndarray
    l_contact: int
    r_contact: int
    l_normal_force: float
    r_normal_force: float
    r_load_fraction: float
    torso_tilt_rad: float
    torso_angvel_rad_s: np.ndarray
    l_rel_r_forward_mm: float
    l_rel_r_lateral_mm: float
    l_world_forward_mm: float
    l_clearance_mm: float
    r_drift_mm: float
    l_hip_cmd: float
    l_hip_qpos: float
    l_knee_cmd: float
    l_knee_qpos: float
    l_ankle_cmd: float
    l_ankle_qpos: float


@dataclass
class StepMetrics:
    left_foot_left_ground: bool = False
    r_stance_maintained: bool = True
    peak_l_clearance_mm: float = 0.0
    peak_l_rel_r_forward_mm: float = 0.0
    peak_l_rel_r_forward_airborne_mm: float = 0.0
    min_l_rel_r_forward_swing_mm: float = field(default_factory=lambda: float("inf"))
    r_max_drift_swing_mm: float = 0.0
    left_touchdown_occurred: bool = False
    left_touchdown_ahead_of_right: bool = False
    touchdown_tilt_rad: float | None = None
    touchdown_rel_forward_mm: float | None = None
    final_rel_forward_mm: float = 0.0
    final_torso_tilt_rad: float = 0.0


@dataclass
class RunResult:
    stand_r_foot_xy: np.ndarray
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
    for ci in range(data.ncon):
        for gid in (data.contact[ci].geom1, data.contact[ci].geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if f"{side}_foot_collision" in name:
                return 1
    return 0


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


def _pelvis_pitch_rad(model: mujoco.MjModel, data: mujoco.MjData, chest_id: int) -> float:
    rot = data.xmat[chest_id].reshape(3, 3)
    up_world = rot @ np.array([0.0, 1.0, 0.0])
    return float(np.arctan2(-up_world[1], up_world[2]))


def _rel_forward_mm(l_y: float, r_y: float) -> float:
    return float(-(l_y - r_y) * 1000.0)


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
    """Verified lateral shift + mild R pitch load (no separate push-off phase)."""
    return _apply_joint_values(
        DEFAULT_POSE,
        {
            IDX_R_HIP_ROLL: -0.04,
            IDX_R_ANKLE_ROLL: 0.03,
            IDX_L_HIP_ROLL: -0.03,
            IDX_R_HIP_PITCH: 0.05,
        },
    )


def lift_pose() -> np.ndarray:
    """Knee-first lift — moderate flexion, no hip pitch yet."""
    return _apply_joint_values(
        rapid_unload_pose(),
        {IDX_L_KNEE: -0.30, IDX_L_ANKLE_P: 0.10},
    )


def move_forward_pose() -> np.ndarray:
    """Small nominal-forward L hip increment while maintaining clearance."""
    return _apply_joint_values(
        rapid_unload_pose(),
        {
            IDX_L_HIP_PITCH: -0.12,
            IDX_L_KNEE: -0.35,
            IDX_L_ANKLE_P: 0.12,
        },
    )


def touchdown_pose() -> np.ndarray:
    return _apply_joint_values(
        rapid_unload_pose(),
        {
            IDX_L_HIP_PITCH: -0.08,
            IDX_L_KNEE: -0.18,
            IDX_L_ANKLE_P: 0.08,
        },
    )


def build_trajectory() -> list[tuple[Phase, np.ndarray, int, bool]]:
    hold = lift_pose()
    stop = touchdown_pose()
    return [
        (Phase.STAND, DEFAULT_POSE.copy(), STAND_STEPS, False),
        (Phase.RAPID_UNLOAD, rapid_unload_pose(), UNLOAD_STEPS, True),
        (Phase.LIFT, lift_pose(), LIFT_STEPS, False),
        (Phase.HOLD, hold, HOLD_STEPS, False),
        (Phase.MOVE_FORWARD, move_forward_pose(), MOVE_FORWARD_STEPS, False),
        (Phase.TOUCHDOWN, touchdown_pose(), TOUCHDOWN_STEPS, False),
        (Phase.STOP, stop, STOP_STEPS, False),
    ]


def _snapshot(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: Phase,
    stand_l_y: float,
    stand_r_xy: np.ndarray,
) -> PhaseSnapshot:
    chest_id = env.chest_body_id
    pelvis = data.xpos[chest_id].copy()
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    total = l_nf + r_nf
    return PhaseSnapshot(
        phase=phase,
        pelvis_xyz=pelvis,
        pelvis_pitch_rad=_pelvis_pitch_rad(model, data, chest_id),
        l_foot_xyz=l_pos,
        r_foot_xyz=r_pos,
        l_rel_pelvis=l_pos - pelvis,
        l_contact=_foot_contact_count(model, data, "L"),
        r_contact=_foot_contact_count(model, data, "R"),
        l_normal_force=l_nf,
        r_normal_force=r_nf,
        r_load_fraction=(r_nf / total) if total > 1e-6 else float("nan"),
        torso_tilt_rad=env._quat_tilt_rad(),
        torso_angvel_rad_s=data.qvel[3:6].copy(),
        l_rel_r_forward_mm=_rel_forward_mm(l_pos[1], r_pos[1]),
        l_rel_r_lateral_mm=float((l_pos[0] - r_pos[0]) * 1000.0),
        l_world_forward_mm=float(-(l_pos[1] - stand_l_y) * 1000.0),
        l_clearance_mm=float((l_pos[2] - FLOOR_Z) * 1000.0),
        r_drift_mm=float(np.linalg.norm(r_pos[:2] - stand_r_xy) * 1000.0),
        l_hip_cmd=float(data.ctrl[IDX_L_HIP_PITCH]),
        l_hip_qpos=float(data.qpos[QPOS_L_HIP]),
        l_knee_cmd=float(data.ctrl[IDX_L_KNEE]),
        l_knee_qpos=float(data.qpos[QPOS_L_KNEE]),
        l_ankle_cmd=float(data.ctrl[IDX_L_ANKLE_P]),
        l_ankle_qpos=float(data.qpos[QPOS_L_ANKLE]),
    )


def _print_snapshot(snap: PhaseSnapshot) -> None:
    print(f"\n--- {snap.phase.value} ---")
    print(f"pelvis XYZ: {snap.pelvis_xyz}")
    print(f"pelvis pitch (rad): {snap.pelvis_pitch_rad:.4f}")
    print(f"L foot XYZ: {snap.l_foot_xyz}")
    print(f"R foot XYZ: {snap.r_foot_xyz}")
    print(f"L foot relative to pelvis: {snap.l_rel_pelvis}")
    print(f"L foot contact state: {snap.l_contact}")
    print(f"R foot contact state: {snap.r_contact}")
    print(f"L normal force: {snap.l_normal_force:.2f} N")
    print(f"R normal force: {snap.r_normal_force:.2f} N")
    print(f"R load fraction: {snap.r_load_fraction:.3f}")
    print(f"torso tilt: {snap.torso_tilt_rad:.4f} rad")
    print(f"torso angular velocity (rad/s): {snap.torso_angvel_rad_s}")
    print(f"L relative-to-R forward (mm): {snap.l_rel_r_forward_mm:.2f}")
    print(f"L relative-to-R lateral (mm): {snap.l_rel_r_lateral_mm:.2f}")
    print(f"L foot world forward (mm, secondary): {snap.l_world_forward_mm:.2f}")
    print(f"L foot clearance (mm): {snap.l_clearance_mm:.2f}")
    print(f"R foot drift from STAND (mm): {snap.r_drift_mm:.2f}")
    print(
        f"L joints cmd/qpos: hip {snap.l_hip_cmd:+.3f}/{snap.l_hip_qpos:+.3f}, "
        f"knee {snap.l_knee_cmd:+.3f}/{snap.l_knee_qpos:+.3f}, "
        f"ankle {snap.l_ankle_cmd:+.3f}/{snap.l_ankle_qpos:+.3f}"
    )


def _evaluate_success(metrics: StepMetrics) -> tuple[bool, list[str], list[str]]:
    ok: list[str] = []
    fail: list[str] = []

    if metrics.left_foot_left_ground:
        ok.append("L foot left the ground")
    else:
        fail.append("L foot never left the ground")

    if metrics.r_stance_maintained:
        ok.append("R foot maintained contact before L touchdown")
    else:
        fail.append("R foot lost contact before L touchdown")

    if metrics.peak_l_clearance_mm >= MIN_CLEARANCE_MM:
        ok.append(f"peak L clearance {metrics.peak_l_clearance_mm:.1f} mm")
    else:
        fail.append(f"insufficient L clearance ({metrics.peak_l_clearance_mm:.1f} mm)")

    if metrics.peak_l_rel_r_forward_mm >= MIN_REL_FORWARD_MM:
        ok.append(
            f"L moved forward relative to R (peak {metrics.peak_l_rel_r_forward_mm:.1f} mm)"
        )
    else:
        fail.append(
            f"L did not reach {MIN_REL_FORWARD_MM:.0f} mm forward relative to R "
            f"(peak {metrics.peak_l_rel_r_forward_mm:.1f} mm)"
        )

    if metrics.r_max_drift_swing_mm <= MAX_R_DRIFT_SWING_MM:
        ok.append(f"R drift during swing {metrics.r_max_drift_swing_mm:.1f} mm")
    else:
        fail.append(
            f"R foot drifted {metrics.r_max_drift_swing_mm:.1f} mm during swing "
            f"(limit {MAX_R_DRIFT_SWING_MM:.0f} mm)"
        )

    if metrics.left_touchdown_occurred:
        ok.append("L foot regained contact")
    else:
        fail.append("L foot did not regain contact")

    if metrics.left_touchdown_ahead_of_right:
        ok.append("L touchdown ahead of R along world -Y")
    else:
        fail.append("L did not touch down ahead of R")

    tilt = metrics.touchdown_tilt_rad
    if tilt is not None and tilt < MAX_TOUCHDOWN_TILT_RAD:
        ok.append(f"torso tilt at touchdown {tilt:.3f} rad")
    else:
        fail.append(
            f"excessive tilt at touchdown ({tilt if tilt is not None else float('nan'):.3f} rad)"
        )

    if metrics.min_l_rel_r_forward_swing_mm < 0.0:
        fail.append(
            f"L moved backward relative to R during swing "
            f"(min {metrics.min_l_rel_r_forward_swing_mm:.1f} mm)"
        )

    return len(fail) == 0, ok, fail


def run_true_step_diagnostic(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> RunResult:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset_pose_only(model, data)

    ctrl = DEFAULT_POSE.copy()
    stand_l_y = _foot_pos(model, data, "L")[1]
    stand_r_xy = _foot_pos(model, data, "R")[:2].copy()
    snapshots: list[PhaseSnapshot] = []
    metrics = StepMetrics()

    swing_phases = {
        Phase.LIFT,
        Phase.HOLD,
        Phase.MOVE_FORWARD,
        Phase.TOUCHDOWN,
    }
    airborne_seen = False

    for phase, target, n_steps, linear in build_trajectory():
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
            l_c = _foot_contact_count(model, data, "L")
            r_c = _foot_contact_count(model, data, "R")
            rel_fwd = _rel_forward_mm(l_pos[1], r_pos[1])
            clearance = (l_pos[2] - FLOOR_Z) * 1000.0

            metrics.peak_l_clearance_mm = max(metrics.peak_l_clearance_mm, clearance)
            metrics.peak_l_rel_r_forward_mm = max(metrics.peak_l_rel_r_forward_mm, rel_fwd)

            if phase in swing_phases:
                metrics.min_l_rel_r_forward_swing_mm = min(
                    metrics.min_l_rel_r_forward_swing_mm, rel_fwd,
                )
                metrics.r_max_drift_swing_mm = max(
                    metrics.r_max_drift_swing_mm,
                    float(np.linalg.norm(r_pos[:2] - stand_r_xy) * 1000.0),
                )
                if r_c == 0 and not metrics.left_touchdown_occurred:
                    metrics.r_stance_maintained = False

            airborne = l_c == 0 and l_pos[2] > FOOT_CONTACT_Z
            if airborne and phase in swing_phases:
                metrics.left_foot_left_ground = True
                airborne_seen = True
                metrics.peak_l_rel_r_forward_airborne_mm = max(
                    metrics.peak_l_rel_r_forward_airborne_mm, rel_fwd,
                )

            if (
                airborne_seen
                and l_c > 0
                and not metrics.left_touchdown_occurred
                and phase in (Phase.MOVE_FORWARD, Phase.TOUCHDOWN, Phase.STOP)
            ):
                metrics.left_touchdown_occurred = True
                metrics.left_touchdown_ahead_of_right = rel_fwd > 0.0
                metrics.touchdown_tilt_rad = env._quat_tilt_rad()
                metrics.touchdown_rel_forward_mm = rel_fwd

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

        snap = _snapshot(env, model, data, phase, stand_l_y, stand_r_xy)
        snapshots.append(snap)
        _print_snapshot(snap)

    final_l = _foot_pos(model, data, "L")
    final_r = _foot_pos(model, data, "R")
    metrics.final_rel_forward_mm = _rel_forward_mm(final_l[1], final_r[1])
    metrics.final_torso_tilt_rad = env._quat_tilt_rad()

    success, ok_reasons, fail_reasons = _evaluate_success(metrics)
    return RunResult(
        stand_r_foot_xy=stand_r_xy,
        snapshots=snapshots,
        metrics=metrics,
        success=success,
        success_reasons=ok_reasons,
        failure_reasons=fail_reasons,
    )


def print_summary(result: RunResult) -> None:
    m = result.metrics
    print("\n" + "=" * 60)
    print("TRUE STEP DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"LEFT_FOOT_LEFT_GROUND = {m.left_foot_left_ground}")
    print(f"PEAK_L_CLEARANCE_MM = {m.peak_l_clearance_mm:.1f}")
    print(f"PEAK_L_REL_R_FORWARD_MM = {m.peak_l_rel_r_forward_mm:.1f}")
    print(f"PEAK_L_REL_R_FORWARD_AIRBORNE_MM = {m.peak_l_rel_r_forward_airborne_mm:.1f}")
    print(f"MIN_L_REL_R_FORWARD_SWING_MM = {m.min_l_rel_r_forward_swing_mm:.1f}")
    print(f"R_FOOT_MAX_DRIFT_SWING_MM = {m.r_max_drift_swing_mm:.1f}")
    print(f"LEFT_TOUCHDOWN_OCCURRED = {m.left_touchdown_occurred}")
    print(f"LEFT_TOUCHDOWN_AHEAD_OF_RIGHT = {m.left_touchdown_ahead_of_right}")
    print(f"TOUCHDOWN_TILT_RAD = {m.touchdown_tilt_rad}")
    print(f"TOUCHDOWN_REL_FORWARD_MM = {m.touchdown_rel_forward_mm}")
    print(f"FINAL_REL_FORWARD_MM = {m.final_rel_forward_mm:.1f}")
    print(f"FINAL_TORSO_TILT = {m.final_torso_tilt_rad:.3f}")
    print()
    if result.success:
        print("VERDICT: SUCCESS - relative forward step criteria met.")
        for r in result.success_reasons:
            print(f"  + {r}")
    else:
        print("VERDICT: NOT A TRUE STEP (relative criteria).")
        for r in result.failure_reasons:
            print(f"  - {r}")
        if result.success_reasons:
            print("\nPartial positives:")
            for r in result.success_reasons:
                print(f"  + {r}")
    print("\nPrimary metric: L_rel_R_forward = -(L_y - R_y)")
    print("World -Y displacement is reported but NOT used for success.")


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
    p = argparse.ArgumentParser(description="Conservative true-step diagnostic.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("True step diagnostic - conservative relative-motion experiment")
    print("Unload: lateral roll shift + mild R hip pitch +0.05 (no R push-off phase)")
    print("Lift: L knee -0.30 only; forward: L hip -0.12 (small); no stabilization phase")
    print("Viewer: robot only.\n")

    if args.headless:
        result = run_true_step_diagnostic(env, viewer=None, slow=False)
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
        result = run_true_step_diagnostic(env, viewer=v, slow=args.slow)
    print_summary(result)


if __name__ == "__main__":
    main(sys.argv[1:])
