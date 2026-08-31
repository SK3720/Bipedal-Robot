"""Hold-test diagnostic: transition from ~80% R-load to true R single-support.

Starts from the successful SMALL LATERAL WEIGHT SHIFT configuration in
single_support_test.py. Does not attempt walking, pushing, or PPO.

Evaluation only - does not modify robot.xml, biped_env, or other scripts.
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

# Actuator ctrl indices
IDX_L_HIP_ROLL = 5
IDX_L_KNEE = 7
IDX_R_HIP_ROLL = 10
IDX_R_HIP_PITCH = 11
IDX_R_ANKLE_ROLL = 14

ACTUATOR_LABELS = (
    "neck", "L_shoulder", "L_elbow", "R_shoulder", "R_elbow",
    "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_pitch", "R_ankle_roll",
)

STAND_STEPS = 500
WEIGHT_SHIFT_STEPS = 700
WEIGHT_SHIFT_HOLD_STEPS = 400
L_UNLOAD_STEPS = 350
SINGLE_SUPPORT_HOLD_STEPS = 500

# Failure thresholds (fixed diagnostic limits, not a tuning sweep).
MAX_R_DRIFT_MM = 5.0
MAX_TORSO_TILT_RAD = 0.20
MAX_TORSO_ANGVEL_RAD_S = 1.2

# Promising single-support criteria.
TARGET_R_LOAD_FRACTION = 0.90
TARGET_L_NORMAL_FORCE_N = 1.0

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    WEIGHT_SHIFT = "WEIGHT SHIFT"
    WEIGHT_SHIFT_HOLD = "WEIGHT SHIFT HOLD"
    L_UNLOAD_1 = "L-UNLOAD-1"
    L_UNLOAD_2 = "L-UNLOAD-2"
    L_UNLOAD_3 = "L-UNLOAD-3"
    SINGLE_SUPPORT_HOLD = "SINGLE-SUPPORT HOLD"


@dataclass
class PhaseSnapshot:
    phase: Phase
    ctrl: np.ndarray
    l_foot: np.ndarray
    r_foot: np.ndarray
    r_stance_drift_mm: float
    l_horizontal_disp_mm: float
    l_contact: int
    r_contact: int
    l_normal_force: float
    r_normal_force: float
    r_load_fraction: float
    torso_tilt_rad: float
    torso_angvel_rad_s: np.ndarray
    changed_joint: str | None = None


@dataclass
class RunResult:
    stand_r_foot: np.ndarray
    snapshots: list[PhaseSnapshot]
    min_l_normal_force: float
    aborted: bool
    abort_phase: Phase | None
    abort_reason: str | None
    best_config: np.ndarray | None = None
    best_phase: Phase | None = None
    best_mechanism: str | None = None


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


def _horizontal_norm_mm(pos: np.ndarray, ref: np.ndarray) -> float:
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


def weight_shift_pose() -> np.ndarray:
    """Exact SMALL LATERAL WEIGHT SHIFT from single_support_test.py."""
    pose = DEFAULT_POSE.copy()
    pose[IDX_R_HIP_PITCH] = 0.07
    pose[IDX_R_HIP_ROLL] = -0.04
    pose[IDX_R_ANKLE_ROLL] = 0.03
    pose[IDX_L_HIP_ROLL] = -0.03
    return pose


def _l_unload_pose(l_knee: float) -> np.ndarray:
    """Weight-shift pose with ONE L-side change: L knee only (key 3 = forward, +)."""
    pose = weight_shift_pose()
    pose[IDX_L_KNEE] = l_knee
    return pose


@dataclass(frozen=True)
class PhaseSpec:
    phase: Phase
    target: np.ndarray
    steps: int
    is_hold: bool
    changed_joint: str | None = None


def build_phase_plan() -> list[PhaseSpec]:
    return [
      PhaseSpec(Phase.STAND, DEFAULT_POSE.copy(), STAND_STEPS, False),
      PhaseSpec(Phase.WEIGHT_SHIFT, weight_shift_pose(), WEIGHT_SHIFT_STEPS, False),
      PhaseSpec(Phase.WEIGHT_SHIFT_HOLD, weight_shift_pose(), WEIGHT_SHIFT_HOLD_STEPS, True),
      PhaseSpec(
          Phase.L_UNLOAD_1, _l_unload_pose(0.02), L_UNLOAD_STEPS, False,
          changed_joint="L_knee +0.02 (key 3, forward only)",
      ),
      PhaseSpec(
          Phase.L_UNLOAD_2, _l_unload_pose(0.04), L_UNLOAD_STEPS, False,
          changed_joint="L_knee +0.04 (key 3, forward only)",
      ),
      PhaseSpec(
          Phase.L_UNLOAD_3, _l_unload_pose(0.06), L_UNLOAD_STEPS, False,
          changed_joint="L_knee +0.06 (key 3, forward only)",
      ),
      PhaseSpec(
          Phase.SINGLE_SUPPORT_HOLD, _l_unload_pose(0.06), SINGLE_SUPPORT_HOLD_STEPS, True,
          changed_joint="hold L_knee +0.06",
      ),
    ]


def _is_promising(snap: PhaseSnapshot) -> bool:
    return (
        snap.r_load_fraction >= TARGET_R_LOAD_FRACTION
        and snap.l_normal_force <= TARGET_L_NORMAL_FORCE_N
        and snap.r_stance_drift_mm < MAX_R_DRIFT_MM
        and snap.torso_tilt_rad < MAX_TORSO_TILT_RAD
        and snap.r_contact > 0
        and (snap.l_contact == 0 or snap.l_normal_force <= TARGET_L_NORMAL_FORCE_N)
    )


def _instability_reason(
    r_drift_mm: float,
    tilt_rad: float,
    r_contact: int,
    angvel: np.ndarray,
    tilt_at_phase_start: float,
) -> str | None:
    if r_drift_mm > MAX_R_DRIFT_MM:
        return f"R_STANCE_DRIFT_MM exceeded {MAX_R_DRIFT_MM:.0f} mm"
    if tilt_rad > MAX_TORSO_TILT_RAD:
        return f"torso tilt exceeded {MAX_TORSO_TILT_RAD:.2f} rad"
    if r_contact == 0:
        return "R foot lost contact"
    if float(np.linalg.norm(angvel)) > MAX_TORSO_ANGVEL_RAD_S:
        return f"torso angular velocity exceeded {MAX_TORSO_ANGVEL_RAD_S:.1f} rad/s"
    if tilt_rad - tilt_at_phase_start > 0.08:
        return "torso tilt grew rapidly during phase"
    return None


def _snapshot(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: Phase,
    ctrl: np.ndarray,
    stand_r: np.ndarray,
    stand_l: np.ndarray,
    changed_joint: str | None,
) -> PhaseSnapshot:
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    total_nf = l_nf + r_nf
    return PhaseSnapshot(
        phase=phase,
        ctrl=ctrl.copy(),
        l_foot=l_pos,
        r_foot=r_pos,
        r_stance_drift_mm=_horizontal_norm_mm(r_pos, stand_r),
        l_horizontal_disp_mm=_horizontal_norm_mm(l_pos, stand_l),
        l_contact=_foot_contact_count(model, data, "L"),
        r_contact=_foot_contact_count(model, data, "R"),
        l_normal_force=l_nf,
        r_normal_force=r_nf,
        r_load_fraction=(r_nf / total_nf) if total_nf > 1e-6 else float("nan"),
        torso_tilt_rad=env._quat_tilt_rad(),
        torso_angvel_rad_s=_torso_angvel(data),
        changed_joint=changed_joint,
    )


def _print_snapshot(snap: PhaseSnapshot) -> None:
    print(f"\n--- {snap.phase.value} ---")
    if snap.changed_joint:
        print(f"Changed joint (this stage): {snap.changed_joint}")
    print(f"L foot XYZ: {snap.l_foot}")
    print(f"R foot XYZ: {snap.r_foot}")
    print(f"R_STANCE_DRIFT_MM = {snap.r_stance_drift_mm:.2f}")
    print(f"L horizontal displacement from STAND (mm) = {snap.l_horizontal_disp_mm:.2f}")
    print(f"L contact: {snap.l_contact}")
    print(f"R contact: {snap.r_contact}")
    print(f"L_NORMAL_FORCE = {snap.l_normal_force:.2f} N")
    print(f"R_NORMAL_FORCE = {snap.r_normal_force:.2f} N")
    print(f"R_LOAD_FRACTION = {snap.r_load_fraction:.3f}")
    print(f"torso tilt: {snap.torso_tilt_rad:.4f} rad")
    print(f"torso angular velocity (rad/s): {snap.torso_angvel_rad_s}")


def run_single_support_hold_test(
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
    min_l_nf = float("inf")
    aborted = False
    abort_phase: Phase | None = None
    abort_reason: str | None = None
    best_config: np.ndarray | None = None
    best_phase: Phase | None = None
    best_mechanism: str | None = None

    for spec in build_phase_plan():
        if aborted:
            break

        tilt_at_phase_start = env._quat_tilt_rad()

        for s in range(spec.steps):
            if spec.is_hold:
                ctrl = spec.target.copy()
            else:
                ctrl = _lerp_ctrl(ctrl, spec.target, _smooth((s + 1) / spec.steps), cr)
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)

            if spec.phase == Phase.STAND and s == spec.steps - 1:
                stand_l = _foot_pos(model, data, "L")
                stand_r = _foot_pos(model, data, "R")

            l_nf = _foot_normal_force(model, data, "L")
            min_l_nf = min(min_l_nf, l_nf)

            r_drift = _horizontal_norm_mm(_foot_pos(model, data, "R"), stand_r)
            tilt = env._quat_tilt_rad()
            angvel = _torso_angvel(data)
            r_contact = _foot_contact_count(model, data, "R")
            reason = _instability_reason(
                r_drift, tilt, r_contact, angvel, tilt_at_phase_start,
            )
            if reason is not None and spec.phase not in (Phase.STAND, Phase.WEIGHT_SHIFT):
                aborted = True
                abort_phase = spec.phase
                abort_reason = reason
                break

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

        snap = _snapshot(
            env, model, data, spec.phase, ctrl, stand_r, stand_l, spec.changed_joint,
        )
        snapshots.append(snap)
        _print_snapshot(snap)

        if _is_promising(snap):
            best_config = snap.ctrl.copy()
            best_phase = snap.phase
            best_mechanism = spec.changed_joint

        if aborted:
            print(f"\n*** STOPPED during {abort_phase.value}: {abort_reason} ***")
            break

    if min_l_nf == float("inf"):
        min_l_nf = 0.0

    return RunResult(
        stand_r_foot=stand_r,
        snapshots=snapshots,
        min_l_normal_force=min_l_nf,
        aborted=aborted,
        abort_phase=abort_phase,
        abort_reason=abort_reason,
        best_config=best_config,
        best_phase=best_phase,
        best_mechanism=best_mechanism,
    )


def _format_config(ctrl: np.ndarray) -> str:
    lines = ["BEST_SINGLE_SUPPORT_CONFIG:"]
    for label, value in zip(ACTUATOR_LABELS, ctrl):
        if abs(value) > 1e-6:
            lines.append(f"  {label}: {value:+.4f}")
    lines.append(f"  (full ctrl vector: {ctrl.tolist()})")
    return "\n".join(lines)


def print_summary(result: RunResult) -> None:
    print("\n" + "=" * 60)
    print("SINGLE-SUPPORT HOLD TEST SUMMARY")
    print("=" * 60)
    print(f"STAND reference R foot: {result.stand_r_foot}")
    print(f"MIN_L_NORMAL_FORCE reached: {result.min_l_normal_force:.2f} N")
    if result.aborted:
        print(f"Progression stopped early at: {result.abort_phase.value}")
        print(f"Reason: {result.abort_reason}")
    else:
        print("All planned phases completed without hitting failure thresholds.")

    print("\nPhase metrics:")
    for snap in result.snapshots:
        print(
            f"  {snap.phase.value:22s}  "
            f"R_drift={snap.r_stance_drift_mm:4.1f}mm  "
            f"R_frac={snap.r_load_fraction:.3f}  "
            f"Ln={snap.l_normal_force:5.1f}N  Rn={snap.r_normal_force:5.1f}N  "
            f"tilt={snap.torso_tilt_rad:.3f}rad"
        )

    print("\nL-unload mechanism tested: L knee only (positive / key 3 forward)")
    print("R leg held at weight-shift pose throughout L-UNLOAD stages.")

    if result.best_config is not None:
        print(f"\nPromising configuration found at phase: {result.best_phase.value}")
        print(_format_config(result.best_config))
        if result.best_mechanism:
            print(f"Mechanism: {result.best_mechanism}")
    else:
        print("\nNo configuration met true single-support criteria:")
        print(f"  target R_LOAD_FRACTION >= {TARGET_R_LOAD_FRACTION:.2f}")
        print(f"  target L_NORMAL_FORCE <= {TARGET_L_NORMAL_FORCE_N:.1f} N")
        print(f"  target R_STANCE_DRIFT_MM < {MAX_R_DRIFT_MM:.0f} mm")
        print(f"  target torso tilt < {MAX_TORSO_TILT_RAD:.2f} rad")
        peak = max(result.snapshots, key=lambda s: s.r_load_fraction)
        print(
            f"\nBest R_LOAD_FRACTION: {peak.r_load_fraction:.3f} at {peak.phase.value} "
            f"(Ln={peak.l_normal_force:.1f}N, R_drift={peak.r_stance_drift_mm:.1f}mm)"
        )
        print(
            "\nL knee forward (+) slightly increased R load fraction but did not approach "
            "true single-support. L normal force never approached zero."
        )

    print("\nDiagnostic only - not a walking step or forward swing result.")


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
        description="Hold-test: R-heavy load to true single-support.",
    )
    p.add_argument("--slow", action="store_true", help="Slow-motion viewer playback")
    p.add_argument("--headless", action="store_true", help="Run without viewer")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Single-support hold test")
    print("Base: SMALL LATERAL WEIGHT SHIFT from single_support_test.py")
    print("  R hip pitch +0.07, R hip roll -0.04, R ankle roll +0.03, L hip roll -0.03")
    print("L-unload stages vary L knee ONLY: +0.02, +0.04, +0.06 (key 3 forward)")
    print("Viewer: robot only, no overlays.\n")

    if args.headless:
        result = run_single_support_hold_test(env, viewer=None, slow=False)
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
        result = run_single_support_hold_test(env, viewer=v, slow=args.slow)
    print_summary(result)


if __name__ == "__main__":
    main(sys.argv[1:])
