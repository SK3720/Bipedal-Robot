"""Stance-locked L swing mini-experiment — relative motion with R planted.

Sequence: stand -> lateral unload -> knee-only lift -> hold -> small L hip
increments. Monitors R drift and pelvis tilt; primary metric is L foot forward
relative to R, not world displacement.

Evaluation only — does not modify robot.xml, actuators, or existing scripts.
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

# Stance-lock limits for this experiment (not full walking).
MAX_R_DRIFT_MM = 20.0
MAX_TILT_HOLD_RAD = 0.35
MAX_TILT_NUDGE_RAD = 0.45
MIN_CLEARANCE_MM = 20.0
MIN_REL_FORWARD_GAIN_MM = 5.0

IDX_L_HIP_ROLL = 5
IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_R_HIP_ROLL = 10
IDX_R_ANKLE_ROLL = 14

QPOS_L_HIP = 13
QPOS_L_KNEE = 14
QPOS_L_ANKLE = 15

STAND_STEPS = 500
UNLOAD_STEPS = 140
LIFT_STEPS = 280
HOLD_STEPS = 400
NUDGE_A_STEPS = 220
NUDGE_B_STEPS = 220
STOP_STEPS = 200

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    UNLOAD = "LATERAL UNLOAD"
    LIFT = "KNEE-ONLY LIFT"
    HOLD = "HOLD SINGLE SUPPORT"
    NUDGE_A = "L HIP NUDGE A (-0.04)"
    NUDGE_B = "L HIP NUDGE B (-0.07)"
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
    stance_ok: bool
    limit_note: str


@dataclass
class StepMetrics:
    left_foot_left_ground: bool = False
    r_stance_locked_through_hold: bool = True
    r_stance_locked_through_nudges: bool = True
    peak_l_clearance_mm: float = 0.0
    l_rel_r_after_lift_mm: float = 0.0
    l_rel_r_after_hold_mm: float = 0.0
    l_rel_r_after_nudge_a_mm: float = 0.0
    l_rel_r_after_nudge_b_mm: float = 0.0
    rel_forward_gain_nudge_a_mm: float = 0.0
    rel_forward_gain_nudge_b_mm: float = 0.0
    peak_l_rel_r_forward_mm: float = 0.0
    min_l_rel_r_forward_mm: float = field(default_factory=lambda: float("inf"))
    r_max_drift_mm: float = 0.0
    r_drift_at_hold_end_mm: float = 0.0
    tilt_at_hold_end_rad: float = 0.0
    tilt_at_nudge_b_rad: float = 0.0
    first_r_drift_breach_mm: float | None = None
    first_r_drift_breach_phase: str | None = None
    final_rel_forward_mm: float = 0.0
    final_torso_tilt_rad: float = 0.0


@dataclass
class RunResult:
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


def lateral_unload_pose() -> np.ndarray:
    """Verified small lateral shift only — no R sagittal push-off."""
    return _apply_joint_values(
        DEFAULT_POSE,
        {
            IDX_R_HIP_ROLL: -0.04,
            IDX_R_ANKLE_ROLL: 0.03,
            IDX_L_HIP_ROLL: -0.03,
        },
    )


def lift_pose() -> np.ndarray:
    """Mild knee flexion only; hip pitch stays at zero."""
    return _apply_joint_values(
        lateral_unload_pose(),
        {IDX_L_KNEE: -0.20, IDX_L_ANKLE_P: 0.06},
    )


def nudge_a_pose() -> np.ndarray:
    return _apply_joint_values(lift_pose(), {IDX_L_HIP_PITCH: -0.04})


def nudge_b_pose() -> np.ndarray:
    return _apply_joint_values(lift_pose(), {IDX_L_HIP_PITCH: -0.07})


def build_trajectory() -> list[tuple[Phase, np.ndarray, int, bool]]:
    return [
        (Phase.STAND, DEFAULT_POSE.copy(), STAND_STEPS, False),
        (Phase.UNLOAD, lateral_unload_pose(), UNLOAD_STEPS, True),
        (Phase.LIFT, lift_pose(), LIFT_STEPS, False),
        (Phase.HOLD, lift_pose(), HOLD_STEPS, False),
        (Phase.NUDGE_A, nudge_a_pose(), NUDGE_A_STEPS, False),
        (Phase.NUDGE_B, nudge_b_pose(), NUDGE_B_STEPS, False),
        (Phase.STOP, nudge_b_pose(), STOP_STEPS, False),
    ]


def _stance_ok(phase: Phase, r_drift_mm: float, tilt_rad: float) -> tuple[bool, str]:
    if r_drift_mm > MAX_R_DRIFT_MM:
        return False, f"R drift {r_drift_mm:.1f} mm > {MAX_R_DRIFT_MM:.0f} mm"
    if phase in (Phase.HOLD, Phase.LIFT) and tilt_rad > MAX_TILT_HOLD_RAD:
        return False, f"tilt {tilt_rad:.3f} rad > {MAX_TILT_HOLD_RAD:.2f} rad"
    if phase in (Phase.NUDGE_A, Phase.NUDGE_B, Phase.STOP) and tilt_rad > MAX_TILT_NUDGE_RAD:
        return False, f"tilt {tilt_rad:.3f} rad > {MAX_TILT_NUDGE_RAD:.2f} rad"
    return True, ""


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
    tilt = env._quat_tilt_rad()
    r_drift = float(np.linalg.norm(r_pos[:2] - stand_r_xy) * 1000.0)
    ok, note = _stance_ok(phase, r_drift, tilt)
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
        torso_tilt_rad=tilt,
        torso_angvel_rad_s=data.qvel[3:6].copy(),
        l_rel_r_forward_mm=_rel_forward_mm(l_pos[1], r_pos[1]),
        l_rel_r_lateral_mm=float((l_pos[0] - r_pos[0]) * 1000.0),
        l_world_forward_mm=float(-(l_pos[1] - stand_l_y) * 1000.0),
        l_clearance_mm=float((l_pos[2] - FLOOR_Z) * 1000.0),
        r_drift_mm=r_drift,
        l_hip_cmd=float(data.ctrl[IDX_L_HIP_PITCH]),
        l_hip_qpos=float(data.qpos[QPOS_L_HIP]),
        l_knee_cmd=float(data.ctrl[IDX_L_KNEE]),
        l_knee_qpos=float(data.qpos[QPOS_L_KNEE]),
        l_ankle_cmd=float(data.ctrl[IDX_L_ANKLE_P]),
        l_ankle_qpos=float(data.qpos[QPOS_L_ANKLE]),
        stance_ok=ok,
        limit_note=note,
    )


def _print_snapshot(snap: PhaseSnapshot) -> None:
    flag = "OK" if snap.stance_ok else f"BREACH: {snap.limit_note}"
    print(f"\n--- {snap.phase.value} [{flag}] ---")
    print(f"pelvis XYZ: {snap.pelvis_xyz}")
    print(f"pelvis pitch (rad): {snap.pelvis_pitch_rad:.4f}")
    print(f"L foot XYZ: {snap.l_foot_xyz}")
    print(f"R foot XYZ: {snap.r_foot_xyz}")
    print(f"L foot relative to pelvis: {snap.l_rel_pelvis}")
    print(f"L/R contact: {snap.l_contact} / {snap.r_contact}")
    print(f"L/R normal force: {snap.l_normal_force:.2f} / {snap.r_normal_force:.2f} N")
    print(f"R load fraction: {snap.r_load_fraction:.3f}")
    print(f"torso tilt: {snap.torso_tilt_rad:.4f} rad")
    print(f"torso angular velocity (rad/s): {snap.torso_angvel_rad_s}")
    print(f"L relative-to-R forward (mm): {snap.l_rel_r_forward_mm:.2f}")
    print(f"L relative-to-R lateral (mm): {snap.l_rel_r_lateral_mm:.2f}")
    print(f"L world forward (mm, secondary): {snap.l_world_forward_mm:.2f}")
    print(f"L clearance (mm): {snap.l_clearance_mm:.2f}")
    print(f"R drift from STAND (mm): {snap.r_drift_mm:.2f}")
    print(
        f"L joints cmd/qpos: hip {snap.l_hip_cmd:+.3f}/{snap.l_hip_qpos:+.3f}, "
        f"knee {snap.l_knee_cmd:+.3f}/{snap.l_knee_qpos:+.3f}, "
        f"ankle {snap.l_ankle_cmd:+.3f}/{snap.l_ankle_qpos:+.3f}"
    )


def _evaluate_success(metrics: StepMetrics) -> tuple[bool, list[str], list[str]]:
    ok: list[str] = []
    fail: list[str] = []

    if metrics.left_foot_left_ground:
        ok.append("L foot left the ground at least briefly")
    else:
        fail.append("L foot never lost contact")

    if metrics.peak_l_clearance_mm >= MIN_CLEARANCE_MM:
        ok.append(f"peak L clearance {metrics.peak_l_clearance_mm:.1f} mm")
    else:
        fail.append(f"insufficient L clearance ({metrics.peak_l_clearance_mm:.1f} mm)")

    if metrics.r_stance_locked_through_hold:
        ok.append(f"R drift through HOLD end {metrics.r_drift_at_hold_end_mm:.1f} mm")
    else:
        fail.append(
            f"R drift exceeded {MAX_R_DRIFT_MM:.0f} mm during HOLD "
            f"({metrics.r_drift_at_hold_end_mm:.1f} mm)"
        )

    if metrics.r_stance_locked_through_nudges:
        ok.append(f"R drift stayed within limit through nudges (max {metrics.r_max_drift_mm:.1f} mm)")
    else:
        fail.append(
            f"R drift exceeded {MAX_R_DRIFT_MM:.0f} mm during hip nudges "
            f"(max {metrics.r_max_drift_mm:.1f} mm at {metrics.first_r_drift_breach_phase})"
        )

    if metrics.tilt_at_hold_end_rad <= MAX_TILT_HOLD_RAD:
        ok.append(f"tilt at HOLD end {metrics.tilt_at_hold_end_rad:.3f} rad")
    else:
        fail.append(f"tilt at HOLD end {metrics.tilt_at_hold_end_rad:.3f} rad > {MAX_TILT_HOLD_RAD:.2f}")

    if metrics.rel_forward_gain_nudge_a_mm >= MIN_REL_FORWARD_GAIN_MM:
        ok.append(
            f"nudge A gained {metrics.rel_forward_gain_nudge_a_mm:.1f} mm relative to HOLD"
        )
    else:
        fail.append(
            f"nudge A did not gain {MIN_REL_FORWARD_GAIN_MM:.0f} mm relative forward "
            f"(delta {metrics.rel_forward_gain_nudge_a_mm:+.1f} mm)"
        )

    if metrics.rel_forward_gain_nudge_b_mm >= MIN_REL_FORWARD_GAIN_MM:
        ok.append(
            f"nudge B gained {metrics.rel_forward_gain_nudge_b_mm:.1f} mm relative to nudge A"
        )
    else:
        fail.append(
            f"nudge B did not gain {MIN_REL_FORWARD_GAIN_MM:.0f} mm relative forward "
            f"(delta {metrics.rel_forward_gain_nudge_b_mm:+.1f} mm)"
        )

    if metrics.min_l_rel_r_forward_mm < 0.0:
        fail.append(
            f"L moved backward relative to R at some point "
            f"(min {metrics.min_l_rel_r_forward_mm:.1f} mm)"
        )

    return len(fail) == 0, ok, fail


def run_stance_locked_swing(
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

    monitor_phases = {
        Phase.LIFT, Phase.HOLD, Phase.NUDGE_A, Phase.NUDGE_B, Phase.STOP,
    }

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
            rel_fwd = _rel_forward_mm(l_pos[1], r_pos[1])
            clearance = (l_pos[2] - FLOOR_Z) * 1000.0
            r_drift = float(np.linalg.norm(r_pos[:2] - stand_r_xy) * 1000.0)
            tilt = env._quat_tilt_rad()

            metrics.peak_l_clearance_mm = max(metrics.peak_l_clearance_mm, clearance)
            metrics.peak_l_rel_r_forward_mm = max(metrics.peak_l_rel_r_forward_mm, rel_fwd)
            metrics.min_l_rel_r_forward_mm = min(metrics.min_l_rel_r_forward_mm, rel_fwd)
            metrics.r_max_drift_mm = max(metrics.r_max_drift_mm, r_drift)

            if l_c == 0 and l_pos[2] > FOOT_CONTACT_Z:
                metrics.left_foot_left_ground = True

            if phase in monitor_phases:
                if r_drift > MAX_R_DRIFT_MM:
                    if metrics.first_r_drift_breach_mm is None:
                        metrics.first_r_drift_breach_mm = r_drift
                        metrics.first_r_drift_breach_phase = phase.value
                    if phase in (Phase.NUDGE_A, Phase.NUDGE_B, Phase.STOP):
                        metrics.r_stance_locked_through_nudges = False
                if phase in (Phase.LIFT, Phase.HOLD) and r_drift > MAX_R_DRIFT_MM:
                    metrics.r_stance_locked_through_hold = False

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

        snap = _snapshot(env, model, data, phase, stand_l_y, stand_r_xy)
        snapshots.append(snap)
        _print_snapshot(snap)

        if phase == Phase.LIFT:
            metrics.l_rel_r_after_lift_mm = snap.l_rel_r_forward_mm
        elif phase == Phase.HOLD:
            metrics.l_rel_r_after_hold_mm = snap.l_rel_r_forward_mm
            metrics.r_drift_at_hold_end_mm = snap.r_drift_mm
            metrics.tilt_at_hold_end_rad = snap.torso_tilt_rad
        elif phase == Phase.NUDGE_A:
            metrics.l_rel_r_after_nudge_a_mm = snap.l_rel_r_forward_mm
            metrics.rel_forward_gain_nudge_a_mm = (
                snap.l_rel_r_forward_mm - metrics.l_rel_r_after_hold_mm
            )
        elif phase == Phase.NUDGE_B:
            metrics.l_rel_r_after_nudge_b_mm = snap.l_rel_r_forward_mm
            metrics.rel_forward_gain_nudge_b_mm = (
                snap.l_rel_r_forward_mm - metrics.l_rel_r_after_nudge_a_mm
            )
            metrics.tilt_at_nudge_b_rad = snap.torso_tilt_rad

    final_l = _foot_pos(model, data, "L")
    final_r = _foot_pos(model, data, "R")
    metrics.final_rel_forward_mm = _rel_forward_mm(final_l[1], final_r[1])
    metrics.final_torso_tilt_rad = env._quat_tilt_rad()

    success, ok_reasons, fail_reasons = _evaluate_success(metrics)
    return RunResult(
        snapshots=snapshots,
        metrics=metrics,
        success=success,
        success_reasons=ok_reasons,
        failure_reasons=fail_reasons,
    )


def print_summary(result: RunResult) -> None:
    m = result.metrics
    print("\n" + "=" * 60)
    print("STANCE-LOCKED SWING TEST SUMMARY")
    print("=" * 60)
    print(f"LEFT_FOOT_LEFT_GROUND = {m.left_foot_left_ground}")
    print(f"PEAK_L_CLEARANCE_MM = {m.peak_l_clearance_mm:.1f}")
    print(f"L_REL_R_AFTER_LIFT_MM = {m.l_rel_r_after_lift_mm:.1f}")
    print(f"L_REL_R_AFTER_HOLD_MM = {m.l_rel_r_after_hold_mm:.1f}")
    print(f"L_REL_R_AFTER_NUDGE_A_MM = {m.l_rel_r_after_nudge_a_mm:.1f}")
    print(f"L_REL_R_AFTER_NUDGE_B_MM = {m.l_rel_r_after_nudge_b_mm:.1f}")
    print(f"GAIN_NUDGE_A_MM = {m.rel_forward_gain_nudge_a_mm:+.1f}")
    print(f"GAIN_NUDGE_B_MM = {m.rel_forward_gain_nudge_b_mm:+.1f}")
    print(f"MIN_L_REL_R_FORWARD_MM = {m.min_l_rel_r_forward_mm:.1f}")
    print(f"R_DRIFT_AT_HOLD_END_MM = {m.r_drift_at_hold_end_mm:.1f}")
    print(f"R_MAX_DRIFT_MM = {m.r_max_drift_mm:.1f}")
    print(f"FIRST_R_DRIFT_BREACH = {m.first_r_drift_breach_phase} ({m.first_r_drift_breach_mm} mm)")
    print(f"TILT_AT_HOLD_END_RAD = {m.tilt_at_hold_end_rad:.3f}")
    print(f"TILT_AT_NUDGE_B_RAD = {m.tilt_at_nudge_b_rad:.3f}")
    print(f"FINAL_REL_FORWARD_MM = {m.final_rel_forward_mm:.1f}")
    print(f"FINAL_TORSO_TILT = {m.final_torso_tilt_rad:.3f}")
    print()
    if result.success:
        print("VERDICT: SUCCESS - stance held while L moved forward relative to R.")
        for r in result.success_reasons:
            print(f"  + {r}")
    else:
        print("VERDICT: STANCE-LOCK FAILED (expected for current robot limits).")
        for r in result.failure_reasons:
            print(f"  - {r}")
        if result.success_reasons:
            print("\nPartial positives:")
            for r in result.success_reasons:
                print(f"  + {r}")
    print("\nPrimary metric: L_rel_R_forward = -(L_y - R_y)")
    print(f"R must stay within {MAX_R_DRIFT_MM:.0f} mm; tilt limits {MAX_TILT_HOLD_RAD:.2f}/{MAX_TILT_NUDGE_RAD:.2f} rad.")


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
    p = argparse.ArgumentParser(description="Stance-locked L swing mini-experiment.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Stance-locked swing test")
    print("Unload: lateral only | Lift: knee -0.20 | Nudges: hip -0.04 then -0.07")
    print("No R push-off, no touchdown phase, no stabilization.\n")

    if args.headless:
        result = run_stance_locked_swing(env, viewer=None, slow=False)
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
        result = run_stance_locked_swing(env, viewer=v, slow=args.slow)
    print_summary(result)


if __name__ == "__main__":
    main(sys.argv[1:])
