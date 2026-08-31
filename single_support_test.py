"""Single-support diagnostic — can R carry essentially all load with L unloaded?

Does not attempt walking, stepping, or pushing. Evaluation only.
Does not modify robot.xml, biped_env, or PPO artifacts.
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

STAND_STEPS = 500
UNLOAD_STEPS = 400
HOLD_STEPS = 300
SMALL_LATERAL_STEPS = 400
MODERATE_LATERAL_STEPS = 400
L_UNLOAD_ATTEMPT_STEPS = 350

# Abort progression if stance stability is lost (not a tuning sweep).
MAX_R_DRIFT_MM = 10.0
MAX_TORSO_TILT_RAD = 0.30

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    CURRENT_UNLOAD = "CURRENT UNLOAD"
    HOLD = "HOLD"
    SMALL_LATERAL = "SMALL LATERAL WEIGHT SHIFT"
    MODERATE_LATERAL = "MODERATE LATERAL WEIGHT SHIFT"
    L_FOOT_UNLOAD = "ATTEMPT L-FOOT UNLOAD"


@dataclass
class PhaseSnapshot:
    phase: Phase
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


@dataclass
class RunResult:
    stand_r_foot: np.ndarray
    snapshots: list[PhaseSnapshot]
    aborted: bool
    abort_phase: Phase | None
    abort_reason: str | None


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


def _unload_pose() -> np.ndarray:
    """STANCE LOCK configuration from prior experiment."""
    pose = DEFAULT_POSE.copy()
    pose[IDX_R_HIP_PITCH] = 0.07
    return pose


def _small_lateral_pose() -> np.ndarray:
    """Lean CoM toward R using lateral joints only (verified load-transfer signs)."""
    pose = _unload_pose()
    pose[IDX_R_HIP_ROLL] = -0.04
    pose[IDX_R_ANKLE_ROLL] = 0.03
    pose[IDX_L_HIP_ROLL] = -0.03
    return pose


def _moderate_lateral_pose() -> np.ndarray:
    pose = _unload_pose()
    pose[IDX_R_HIP_ROLL] = -0.06
    pose[IDX_R_ANKLE_ROLL] = 0.05
    pose[IDX_L_HIP_ROLL] = -0.05
    return pose


def _l_unload_attempt_pose() -> np.ndarray:
    """Very small L knee backward (interactive key 4) on top of moderate lateral shift."""
    pose = _moderate_lateral_pose()
    pose[IDX_L_KNEE] = -0.04
    return pose


def build_phase_plan() -> list[tuple[Phase, np.ndarray, int, bool]]:
    """(phase, target, steps, is_hold)."""
    return [
        (Phase.STAND, DEFAULT_POSE.copy(), STAND_STEPS, False),
        (Phase.CURRENT_UNLOAD, _unload_pose(), UNLOAD_STEPS, False),
        (Phase.HOLD, _unload_pose(), HOLD_STEPS, True),
        (Phase.SMALL_LATERAL, _small_lateral_pose(), SMALL_LATERAL_STEPS, False),
        (Phase.HOLD, _small_lateral_pose(), HOLD_STEPS, True),
        (Phase.MODERATE_LATERAL, _moderate_lateral_pose(), MODERATE_LATERAL_STEPS, False),
        (Phase.HOLD, _moderate_lateral_pose(), HOLD_STEPS, True),
        (Phase.L_FOOT_UNLOAD, _l_unload_attempt_pose(), L_UNLOAD_ATTEMPT_STEPS, False),
        (Phase.HOLD, _l_unload_attempt_pose(), HOLD_STEPS, True),
    ]


def _snapshot(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: Phase,
    stand_r: np.ndarray,
    stand_l: np.ndarray,
) -> PhaseSnapshot:
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    total_nf = l_nf + r_nf
    return PhaseSnapshot(
        phase=phase,
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
    )


def _print_snapshot(snap: PhaseSnapshot) -> None:
    print(f"\n--- {snap.phase.value} ---")
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


def _instability_reason(
    r_drift_mm: float,
    tilt_rad: float,
    tilt_start: float,
) -> str | None:
    if r_drift_mm > MAX_R_DRIFT_MM:
        return f"R_STANCE_DRIFT_MM exceeded {MAX_R_DRIFT_MM:.0f} mm"
    if tilt_rad > MAX_TORSO_TILT_RAD:
        return f"torso tilt exceeded {MAX_TORSO_TILT_RAD:.2f} rad"
    if tilt_rad - tilt_start > 0.12:
        return "torso tilt grew rapidly during phase"
    return None


def run_single_support_test(
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
    aborted = False
    abort_phase: Phase | None = None
    abort_reason: str | None = None

    for phase, target, n_steps, is_hold in build_phase_plan():
        if aborted:
            break

        tilt_at_phase_start = env._quat_tilt_rad()

        for s in range(n_steps):
            if is_hold:
                ctrl = target.copy()
            else:
                ctrl = _lerp_ctrl(ctrl, target, _smooth((s + 1) / n_steps), cr)
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)

            if phase == Phase.STAND and s == n_steps - 1:
                stand_l = _foot_pos(model, data, "L")
                stand_r = _foot_pos(model, data, "R")

            r_drift = _horizontal_norm_mm(_foot_pos(model, data, "R"), stand_r)
            tilt = env._quat_tilt_rad()
            reason = _instability_reason(r_drift, tilt, tilt_at_phase_start)
            if reason is not None and phase not in (Phase.STAND,):
                aborted = True
                abort_phase = phase
                abort_reason = reason
                break

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

        snap = _snapshot(env, model, data, phase, stand_r, stand_l)
        snapshots.append(snap)
        _print_snapshot(snap)

        if aborted:
            print(f"\n*** STOPPED during {abort_phase.value}: {abort_reason} ***")
            break

    return RunResult(
        stand_r_foot=stand_r,
        snapshots=snapshots,
        aborted=aborted,
        abort_phase=abort_phase,
        abort_reason=abort_reason,
    )


def print_summary(result: RunResult) -> None:
    print("\n" + "=" * 60)
    print("SINGLE-SUPPORT DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"STAND reference R foot: {result.stand_r_foot}")
    if result.aborted:
        print(f"Progression stopped early at: {result.abort_phase.value}")
        print(f"Reason: {result.abort_reason}")
    else:
        print("All planned phases completed without hitting stop thresholds.")

    print("\nKey phase metrics:")
    for snap in result.snapshots:
        print(
            f"  {snap.phase.value:32s}  "
            f"R_drift={snap.r_stance_drift_mm:5.1f}mm  "
            f"R_frac={snap.r_load_fraction:.3f}  "
            f"Ln={snap.l_normal_force:5.1f}N  Rn={snap.r_normal_force:5.1f}N  "
            f"tilt={snap.torso_tilt_rad:.3f}rad"
        )

    final = result.snapshots[-1]
    print("\nFinal assessment:")
    if final.r_load_fraction >= 0.70 and final.r_stance_drift_mm < MAX_R_DRIFT_MM:
        print(
            "R appears to carry most of the load with limited R-foot drift. "
            "Inspect viewer to confirm L is truly unloaded."
        )
    elif final.r_stance_drift_mm >= MAX_R_DRIFT_MM:
        print("R foot drifted too much - stable single-support not established.")
    else:
        print(
            "Load transfer incomplete: R_LOAD_FRACTION below ~0.70 and/or contacts "
            "do not indicate clean single-support."
        )
    print("\nThis is a diagnostic only — not a successful step or walking result.")


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
    p = argparse.ArgumentParser(description="Single-support diagnostic (R stance, L unload).")
    p.add_argument("--slow", action="store_true", help="Slow-motion viewer playback")
    p.add_argument("--headless", action="store_true", help="Run without viewer")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Single-support diagnostic - R stance / L unload")
    print("Lateral shift signs: R hip roll -, R ankle roll +, L hip roll -")
    print("L unload attempt: L knee -0.04 (key 4, backward)")
    print("Viewer: robot only, no overlays.\n")

    if args.headless:
        result = run_single_support_test(env, viewer=None, slow=False)
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
        result = run_single_support_test(env, viewer=v, slow=args.slow)
    print_summary(result)


if __name__ == "__main__":
    main(sys.argv[1:])
