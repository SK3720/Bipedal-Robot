"""Mechanism diagnostic: how to reach ~90%+ R-load from the lateral weight-shift base.

Tests candidate unload mechanisms ONE AT A TIME from the successful
SMALL LATERAL WEIGHT SHIFT configuration. No L knee, no forward swing, no PPO.

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

IDX_L_HIP_ROLL = 5
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
BASE_HOLD_STEPS = 400
MECH_RAMP_STEPS = 400
MECH_HOLD_STEPS = 300
RETURN_TO_BASE_STEPS = 300

MAX_R_DRIFT_MM = 5.0
MAX_TORSO_TILT_RAD = 0.20
MAX_TORSO_ANGVEL_RAD_S = 1.2

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
    BASE_HOLD = "BASE HOLD (reference)"
    MECH_TEST = "MECHANISM TEST"
    MECH_HOLD = "MECHANISM HOLD"
    RETURN_BASE = "RETURN TO BASE"
    COMBO_TEST = "COMBO TEST"
    COMBO_HOLD = "COMBO HOLD"


@dataclass(frozen=True)
class MechanismSpec:
    label: str
    family: str
    changes: dict[int, float]
    combo: bool = False


MECHANISM_TESTS: tuple[MechanismSpec, ...] = (
    MechanismSpec("R hip roll -0.05", "lateral R hip roll", {IDX_R_HIP_ROLL: -0.05}),
    MechanismSpec("R hip roll -0.06", "lateral R hip roll", {IDX_R_HIP_ROLL: -0.06}),
    MechanismSpec("R ankle roll +0.04", "R ankle roll", {IDX_R_ANKLE_ROLL: 0.04}),
    MechanismSpec("R ankle roll +0.05", "R ankle roll", {IDX_R_ANKLE_ROLL: 0.05}),
    MechanismSpec("L hip roll -0.04", "L hip roll unload", {IDX_L_HIP_ROLL: -0.04}),
    MechanismSpec("L hip roll -0.05", "L hip roll unload", {IDX_L_HIP_ROLL: -0.05}),
)


@dataclass
class PhaseSnapshot:
    phase: str
    mechanism: str | None
    ctrl: np.ndarray
    r_stance_drift_mm: float
    l_normal_force: float
    r_normal_force: float
    r_load_fraction: float
    l_contact: int
    r_contact: int
    torso_tilt_rad: float
    torso_angvel_rad_s: np.ndarray


@dataclass
class RunResult:
    stand_r_foot: np.ndarray
    snapshots: list[PhaseSnapshot]
    base_reference: PhaseSnapshot | None
    best_snapshot: PhaseSnapshot | None
    aborted: bool
    abort_phase: str | None
    abort_reason: str | None
    combo_snapshot: PhaseSnapshot | None


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


def base_weight_shift_pose() -> np.ndarray:
    """SMALL LATERAL WEIGHT SHIFT reference configuration."""
    pose = DEFAULT_POSE.copy()
    pose[IDX_R_HIP_PITCH] = 0.07
    pose[IDX_R_HIP_ROLL] = -0.04
    pose[IDX_R_ANKLE_ROLL] = 0.03
    pose[IDX_L_HIP_ROLL] = -0.03
    return pose


def mechanism_pose(changes: dict[int, float]) -> np.ndarray:
    pose = base_weight_shift_pose()
    for idx, value in changes.items():
        pose[idx] = value
    return pose


def _instability_reason(
    r_drift_mm: float,
    tilt_rad: float,
    r_contact: int,
    angvel: np.ndarray,
    tilt_at_start: float,
) -> str | None:
    if r_drift_mm > MAX_R_DRIFT_MM:
        return f"R stance drift exceeded {MAX_R_DRIFT_MM:.0f} mm"
    if tilt_rad > MAX_TORSO_TILT_RAD:
        return f"torso tilt exceeded {MAX_TORSO_TILT_RAD:.2f} rad"
    if r_contact == 0:
        return "R foot lost contact"
    if float(np.linalg.norm(angvel)) > MAX_TORSO_ANGVEL_RAD_S:
        return f"torso angular velocity exceeded {MAX_TORSO_ANGVEL_RAD_S:.1f} rad/s"
    if tilt_rad - tilt_at_start > 0.08:
        return "torso tilt grew rapidly"
    return None


def _is_success(snap: PhaseSnapshot) -> bool:
    return (
        snap.r_load_fraction >= TARGET_R_LOAD_FRACTION
        and snap.l_normal_force <= TARGET_L_NORMAL_FORCE_N
        and snap.r_stance_drift_mm < MAX_R_DRIFT_MM
        and snap.torso_tilt_rad < MAX_TORSO_TILT_RAD
        and snap.r_contact > 0
    )


def _take_snapshot(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: str,
    ctrl: np.ndarray,
    stand_r: np.ndarray,
    mechanism: str | None = None,
) -> PhaseSnapshot:
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    total = l_nf + r_nf
    return PhaseSnapshot(
        phase=phase,
        mechanism=mechanism,
        ctrl=ctrl.copy(),
        r_stance_drift_mm=_horizontal_drift_mm(_foot_pos(model, data, "R"), stand_r),
        l_normal_force=l_nf,
        r_normal_force=r_nf,
        r_load_fraction=(r_nf / total) if total > 1e-6 else float("nan"),
        l_contact=_foot_contact_count(model, data, "L"),
        r_contact=_foot_contact_count(model, data, "R"),
        torso_tilt_rad=env._quat_tilt_rad(),
        torso_angvel_rad_s=_torso_angvel(data),
    )


def _print_snapshot(snap: PhaseSnapshot) -> None:
    print(f"\n--- {snap.phase}" + (f" [{snap.mechanism}]" if snap.mechanism else "") + " ---")
    print(f"R_STANCE_DRIFT_MM = {snap.r_stance_drift_mm:.2f}")
    print(f"L_NORMAL_FORCE = {snap.l_normal_force:.2f} N")
    print(f"R_NORMAL_FORCE = {snap.r_normal_force:.2f} N")
    print(f"R_LOAD_FRACTION = {snap.r_load_fraction:.3f}")
    print(f"L contact: {snap.l_contact}")
    print(f"R contact: {snap.r_contact}")
    print(f"torso tilt: {snap.torso_tilt_rad:.4f} rad")
    print(f"torso angular velocity (rad/s): {snap.torso_angvel_rad_s}")


def _run_segment(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    cr: np.ndarray,
    stand_r: np.ndarray,
    *,
    is_hold: bool = False,
    check_stability: bool = False,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> tuple[np.ndarray, str | None]:
    tilt_start = env._quat_tilt_rad()
    for s in range(n_steps):
        if is_hold:
            ctrl = target.copy()
        else:
            ctrl = _lerp_ctrl(ctrl, target, _smooth((s + 1) / n_steps), cr)
        data.ctrl[:15] = ctrl
        mujoco.mj_step(model, data)

        if check_stability:
            snap = _take_snapshot(env, model, data, "", ctrl, stand_r)
            reason = _instability_reason(
                snap.r_stance_drift_mm,
                snap.torso_tilt_rad,
                snap.r_contact,
                snap.torso_angvel_rad_s,
                tilt_start,
            )
            if reason is not None:
                return ctrl, reason

        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)
    return ctrl, None


def _pick_combo(mechanism_holds: list[PhaseSnapshot]) -> dict[int, float] | None:
    """Combine only mechanisms that beat the base reference R load fraction."""
    if not mechanism_holds:
        return None
    base_frac = mechanism_holds[0].r_load_fraction
    winners = [s for s in mechanism_holds[1:] if s.r_load_fraction > base_frac + 0.005]
    if not winners:
        return None
    best = max(winners, key=lambda s: s.r_load_fraction)
    changes: dict[int, float] = {}
    base = base_weight_shift_pose()
    for idx, val in enumerate(best.ctrl):
        if abs(val - base[idx]) > 1e-6:
            changes[idx] = float(val)
    if len(changes) < 2:
        return None
    return changes


def run_mechanism_test(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> RunResult:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset_pose_only(model, data)

    ctrl = DEFAULT_POSE.copy()
    stand_r = _foot_pos(model, data, "R")
    snapshots: list[PhaseSnapshot] = []
    aborted = False
    abort_phase: str | None = None
    abort_reason: str | None = None
    base_reference: PhaseSnapshot | None = None
    best_snapshot: PhaseSnapshot | None = None
    combo_snapshot: PhaseSnapshot | None = None
    mechanism_holds: list[PhaseSnapshot] = []

    base_pose = base_weight_shift_pose()

    # Preamble: STAND -> WEIGHT SHIFT -> BASE HOLD
    ctrl, _ = _run_segment(
        env, model, data, ctrl, DEFAULT_POSE, STAND_STEPS, cr, stand_r,
        viewer=viewer, slow=slow,
    )
    if STAND_STEPS > 0:
        stand_r = _foot_pos(model, data, "R")

    snap = _take_snapshot(env, model, data, Phase.STAND.value, ctrl, stand_r)
    snapshots.append(snap)
    _print_snapshot(snap)

    ctrl, reason = _run_segment(
        env, model, data, ctrl, base_pose, WEIGHT_SHIFT_STEPS, cr, stand_r,
        viewer=viewer, slow=slow,
    )
    snap = _take_snapshot(env, model, data, Phase.WEIGHT_SHIFT.value, ctrl, stand_r)
    snapshots.append(snap)
    _print_snapshot(snap)

    ctrl, reason = _run_segment(
        env, model, data, ctrl, base_pose, BASE_HOLD_STEPS, cr, stand_r,
        is_hold=True, viewer=viewer, slow=slow,
    )
    snap = _take_snapshot(env, model, data, Phase.BASE_HOLD.value, ctrl, stand_r)
    snapshots.append(snap)
    _print_snapshot(snap)
    base_reference = snap
    best_snapshot = snap
    mechanism_holds.append(snap)

    # Individual mechanism tests: return to base between each test
    for mech in MECHANISM_TESTS:
        if aborted:
            break

        ctrl, _ = _run_segment(
            env, model, data, ctrl, base_pose, RETURN_TO_BASE_STEPS, cr, stand_r,
            viewer=viewer, slow=slow,
        )
        target = mechanism_pose(mech.changes)

        ctrl, reason = _run_segment(
            env, model, data, ctrl, target, MECH_RAMP_STEPS, cr, stand_r,
            check_stability=True, viewer=viewer, slow=slow,
        )
        if reason is not None:
            aborted = True
            abort_phase = f"{Phase.MECH_TEST.value} [{mech.label}]"
            abort_reason = reason
            break

        snap = _take_snapshot(
            env, model, data, Phase.MECH_TEST.value, ctrl, stand_r, mech.label,
        )
        snapshots.append(snap)
        _print_snapshot(snap)

        ctrl, reason = _run_segment(
            env, model, data, ctrl, target, MECH_HOLD_STEPS, cr, stand_r,
            is_hold=True, check_stability=True, viewer=viewer, slow=slow,
        )
        if reason is not None:
            aborted = True
            abort_phase = f"{Phase.MECH_HOLD.value} [{mech.label}]"
            abort_reason = reason
            break

        snap = _take_snapshot(
            env, model, data, Phase.MECH_HOLD.value, ctrl, stand_r, mech.label,
        )
        snapshots.append(snap)
        _print_snapshot(snap)
        mechanism_holds.append(snap)

        if best_snapshot is None or snap.r_load_fraction > best_snapshot.r_load_fraction:
            best_snapshot = snap

    # Combo test only if individual mechanisms beat baseline
    combo_changes = None if aborted else _pick_combo(mechanism_holds)
    if combo_changes and not aborted:
        ctrl, _ = _run_segment(
            env, model, data, ctrl, base_pose, RETURN_TO_BASE_STEPS, cr, stand_r,
            viewer=viewer, slow=slow,
        )
        combo_target = mechanism_pose(combo_changes)
        combo_label = ", ".join(
            f"{ACTUATOR_LABELS[i]}={v:+.3f}" for i, v in sorted(combo_changes.items())
        )

        ctrl, reason = _run_segment(
            env, model, data, ctrl, combo_target, MECH_RAMP_STEPS, cr, stand_r,
            check_stability=True, viewer=viewer, slow=slow,
        )
        if reason is None:
            snap = _take_snapshot(
                env, model, data, Phase.COMBO_TEST.value, ctrl, stand_r, combo_label,
            )
            snapshots.append(snap)
            _print_snapshot(snap)

            ctrl, reason = _run_segment(
                env, model, data, ctrl, combo_target, MECH_HOLD_STEPS, cr, stand_r,
                is_hold=True, check_stability=True, viewer=viewer, slow=slow,
            )
            if reason is None:
                snap = _take_snapshot(
                    env, model, data, Phase.COMBO_HOLD.value, ctrl, stand_r, combo_label,
                )
                snapshots.append(snap)
                _print_snapshot(snap)
                combo_snapshot = snap
                if snap.r_load_fraction > best_snapshot.r_load_fraction:
                    best_snapshot = snap
        if reason is not None:
            print(f"\n*** COMBO test stopped: {reason} ***")

    if aborted:
        print(f"\n*** STOPPED during {abort_phase}: {abort_reason} ***")

    return RunResult(
        stand_r_foot=stand_r,
        snapshots=snapshots,
        base_reference=base_reference,
        best_snapshot=best_snapshot,
        aborted=aborted,
        abort_phase=abort_phase,
        abort_reason=abort_reason,
        combo_snapshot=combo_snapshot,
    )


def _format_ctrl(ctrl: np.ndarray) -> str:
    parts = [f"  {label}: {val:+.4f}" for label, val in zip(ACTUATOR_LABELS, ctrl) if abs(val) > 1e-6]
    return "\n".join(parts) if parts else "  (all default)"


def print_summary(result: RunResult) -> None:
    print("\n" + "=" * 60)
    print("SINGLE-SUPPORT MECHANISM TEST SUMMARY")
    print("=" * 60)

    if result.base_reference is not None:
        b = result.base_reference
        print(
            f"BASE HOLD reference: R_frac={b.r_load_fraction:.3f}  "
            f"Ln={b.l_normal_force:.1f}N  Rn={b.r_normal_force:.1f}N  "
            f"drift={b.r_stance_drift_mm:.1f}mm"
        )

    print("\nMechanism hold results (vs base):")
    for snap in result.snapshots:
        if snap.phase != Phase.MECH_HOLD.value or snap.mechanism is None:
            continue
        base_frac = result.base_reference.r_load_fraction if result.base_reference else 0.0
        delta = snap.r_load_fraction - base_frac
        marker = " <-- best" if snap is result.best_snapshot else ""
        print(
            f"  {snap.mechanism:24s}  R_frac={snap.r_load_fraction:.3f} ({delta:+.3f})  "
            f"Ln={snap.l_normal_force:5.1f}N  drift={snap.r_stance_drift_mm:4.1f}mm  "
            f"tilt={snap.torso_tilt_rad:.3f}rad{marker}"
        )

    if result.combo_snapshot is not None:
        c = result.combo_snapshot
        print(
            f"\nCOMBO hold: R_frac={c.r_load_fraction:.3f}  Ln={c.l_normal_force:.1f}N  "
            f"drift={c.r_stance_drift_mm:.1f}mm"
        )
    elif result.base_reference is not None:
        print("\nNo combo test run (no individual mechanism beat baseline by >0.005).")

    print("\nSuccess criteria (target):")
    print(f"  R_LOAD_FRACTION >= {TARGET_R_LOAD_FRACTION:.2f}")
    print(f"  L_NORMAL_FORCE <= {TARGET_L_NORMAL_FORCE_N:.1f} N")
    print(f"  R_STANCE_DRIFT_MM < {MAX_R_DRIFT_MM:.0f} mm")
    print(f"  torso tilt < {MAX_TORSO_TILT_RAD:.2f} rad")

    if result.best_snapshot and _is_success(result.best_snapshot):
        print(f"\nCANDIDATE MECHANISM: {result.best_snapshot.mechanism}")
        print("Joint configuration:")
        print(_format_ctrl(result.best_snapshot.ctrl))
    elif result.best_snapshot:
        print(f"\nBest measured: {result.best_snapshot.mechanism or result.best_snapshot.phase}")
        print(f"  R_frac={result.best_snapshot.r_load_fraction:.3f}  "
              f"Ln={result.best_snapshot.l_normal_force:.1f}N  "
              f"drift={result.best_snapshot.r_stance_drift_mm:.1f}mm")
        print("True single-support (90%+ R load, Ln~0) was NOT achieved.")
    print("\nDiagnostic only - no foot lift, no forward swing.")


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
    p = argparse.ArgumentParser(description="Mechanism test for L-foot unload from R-heavy base.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Single-support mechanism test")
    print("Base: R pitch +0.07, R roll -0.04, R ankle roll +0.03, L roll -0.03")
    print("Tests one mechanism at a time. No L knee. Viewer: robot only.\n")

    if args.headless:
        result = run_mechanism_test(env, viewer=None, slow=False)
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
        result = run_mechanism_test(env, viewer=v, slow=args.slow)
    print_summary(result)


if __name__ == "__main__":
    main(sys.argv[1:])
