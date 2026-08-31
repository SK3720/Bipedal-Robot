"""Dynamic single-step feasibility test for lateral_L (+Y) before RL stepping.

Tests whether faster swing timing can achieve true foot clearance where slow
quasi-static trajectories failed. Does not modify biped_env or any PPO artifacts.
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
    SETTLE_STEPS,
    STANDING_QUAT,
)

FLOOR_Z = 1.0
FOOT_CONTACT_Z = 1.042
MIN_CLEARANCE_M = 0.025
MIN_DISPLACEMENT_M = 0.05
MAX_UPRIGHT_TILT = 0.45
POST_STEP_HOLD_STEPS = 500

SWING_DURATIONS_MS = [50, 75, 100, 150, 200]
WEIGHT_SHIFT_MS = 400
UNLOAD_MS = 40
PLACE_MS = 100

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.015


class Phase(Enum):
    RESET_SETTLE = "RESET / SETTLE"
    WEIGHT_SHIFT = "WEIGHT SHIFT"
    FOOT_UNLOAD = "FOOT UNLOAD"
    SWING = "SWING"
    PLACEMENT = "PLACEMENT"
    STABILIZATION = "STABILIZATION"


@dataclass
class DynamicTrial:
    name: str
    swing_duration_ms: int
    unload_ms: int = UNLOAD_MS
    place_ms: int = PLACE_MS
    weight_shift_ms: int = WEIGHT_SHIFT_MS


@dataclass
class TrialMetrics:
    trial_name: str
    swing_duration_ms: int
    success: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    initial_swing_foot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    peak_displacement_m: float = 0.0
    peak_clearance_m: float = 0.0
    max_torso_tilt_rad: float = 0.0
    final_torso_tilt_rad: float = 0.0
    max_torso_angvel: float = 0.0
    started_in_contact: bool = False
    lost_contact: bool = False
    regained_contact: bool = False
    clearance_achieved: bool = False
    displacement_achieved: bool = False
    stance_maintained_during_swing: bool = True
    upright_after_hold: bool = False
    phase_summaries: list[dict] = field(default_factory=list)
    landing_step: int | None = None


def _smooth(alpha: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(alpha, 0.0, 1.0)))


def _ms_to_steps(ms: int) -> int:
    return max(1, int(ms))


def _foot_state(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[np.ndarray, int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
    pos = data.xpos[body_id].copy()
    contacts = 0
    for ci in range(data.ncon):
        contact = data.contact[ci]
        for geom_id in (contact.geom1, contact.geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if f"{side}_foot_collision" in name:
                contacts += 1
                break
    return pos, contacts


def _make_poses(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    cr = model.actuator_ctrlrange[:15]
    ws = DEFAULT_POSE.copy()
    ws[10] = -0.183

    unload = ws.copy()
    unload[7] = -0.18
    unload[6] = 0.05

    apex = ws.copy()
    apex[5] = cr[5, 0] * 0.95
    apex[6] = 0.45
    apex[7] = -0.55
    apex[8] = 0.40

    place = ws.copy()
    place[5] = cr[5, 0] * 0.90
    place[6] = 0.30
    place[7] = -0.22
    place[8] = 0.22

    return {"stand": DEFAULT_POSE.copy(), "ws": ws, "unload": unload, "apex": apex, "place": place}


def _reset_sim(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = DEFAULT_POSE
    data.qvel[:] = 0.0
    data.ctrl[:15] = DEFAULT_POSE
    for _ in range(SETTLE_STEPS):
        data.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_step(model, data)


def _run_trial(
    env: BipedalWalkEnv,
    trial: DynamicTrial,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    global_step: list[int] | None = None,
    slow_window: tuple[int, int] | None = None,
) -> TrialMetrics:
    model = env.model
    data = env.data
    poses = _make_poses(model)
    metrics = TrialMetrics(trial.name, trial.swing_duration_ms)

    _reset_sim(model, data)
    swing_init, init_c = _foot_state(model, data, "L")
    metrics.initial_swing_foot = swing_init
    metrics.started_in_contact = init_c > 0

    ctrl = DEFAULT_POSE.copy()
    step_counter = global_step if global_step is not None else [0]
    slow_lo, slow_hi = slow_window or (0, 10**9)

    segments: list[tuple[Phase, np.ndarray, int]] = [
        (Phase.WEIGHT_SHIFT, poses["ws"], _ms_to_steps(trial.weight_shift_ms)),
        (Phase.FOOT_UNLOAD, poses["unload"], _ms_to_steps(trial.unload_ms)),
        (Phase.SWING, poses["apex"], _ms_to_steps(trial.swing_duration_ms)),
        (Phase.PLACEMENT, poses["place"], _ms_to_steps(trial.place_ms)),
        (Phase.STABILIZATION, poses["place"], POST_STEP_HOLD_STEPS),
    ]

    landing_seen = False
    cr = model.actuator_ctrlrange[:15]

    for phase, target, n_steps in segments:
        for s in range(n_steps):
            alpha = _smooth((s + 1) / n_steps)
            ctrl = (1.0 - alpha) * ctrl + alpha * target
            ctrl = np.clip(ctrl, cr[:, 0], cr[:, 1])
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)

            swing_pos, swing_c = _foot_state(model, data, "L")
            _, stance_c = _foot_state(model, data, "R")
            tilt = env._quat_tilt_rad()
            angvel = float(np.linalg.norm(data.qvel[3:6]))
            clearance = swing_pos[2] - FLOOR_Z
            disp = swing_pos[1] - swing_init[1]

            metrics.max_torso_tilt_rad = max(metrics.max_torso_tilt_rad, tilt)
            metrics.max_torso_angvel = max(metrics.max_torso_angvel, angvel)
            metrics.peak_displacement_m = max(metrics.peak_displacement_m, disp)
            metrics.peak_clearance_m = max(metrics.peak_clearance_m, clearance)
            if swing_c == 0 and swing_pos[2] > FOOT_CONTACT_Z:
                metrics.lost_contact = True
            if clearance >= MIN_CLEARANCE_M:
                metrics.clearance_achieved = True
            if disp >= MIN_DISPLACEMENT_M:
                metrics.displacement_achieved = True
            if stance_c < 1 and phase == Phase.SWING:
                metrics.stance_maintained_during_swing = False

            if (
                not landing_seen
                and metrics.lost_contact
                and swing_c > 0
                and phase in (Phase.PLACEMENT, Phase.STABILIZATION)
            ):
                landing_seen = True
                metrics.regained_contact = True
                metrics.landing_step = step_counter[0]

            if viewer is not None and viewer.is_running():
                viewer.sync()
                if slow and slow_lo <= step_counter[0] <= slow_hi:
                    time.sleep(SLOW_SLEEP_S)
                else:
                    time.sleep(NORMAL_SLEEP_S)
            step_counter[0] += 1

    final_swing, final_c = _foot_state(model, data, "L")
    if metrics.lost_contact and final_c > 0:
        metrics.regained_contact = True
    metrics.final_torso_tilt_rad = env._quat_tilt_rad()
    metrics.upright_after_hold = metrics.final_torso_tilt_rad < MAX_UPRIGHT_TILT

    reasons: list[str] = []
    if not metrics.started_in_contact:
        reasons.append("swing foot not in contact at start")
    if not metrics.lost_contact:
        reasons.append("swing foot never lost contact")
    if not metrics.clearance_achieved:
        reasons.append(f"peak clearance {metrics.peak_clearance_m:.3f} m < {MIN_CLEARANCE_M:.3f} m")
    if not metrics.displacement_achieved:
        reasons.append(f"peak displacement {metrics.peak_displacement_m:.3f} m < {MIN_DISPLACEMENT_M:.3f} m")
    if not metrics.regained_contact:
        reasons.append("swing foot did not regain contact after swing")
    if not metrics.upright_after_hold:
        reasons.append(
            f"not upright after {POST_STEP_HOLD_STEPS} hold steps "
            f"(tilt {metrics.final_torso_tilt_rad:.3f} rad)"
        )
    if not metrics.stance_maintained_during_swing:
        reasons.append("stance foot lost contact during swing")

    metrics.success = not reasons
    metrics.failure_reasons = reasons
    return metrics


def _print_trial(metrics: TrialMetrics) -> None:
    print(f"\n--- {metrics.trial_name} ({metrics.swing_duration_ms} ms swing) ---")
    print(f"  started_in_contact: {metrics.started_in_contact}")
    print(f"  lost_contact:       {metrics.lost_contact}")
    print(f"  clearance_ok:       {metrics.clearance_achieved} (peak {metrics.peak_clearance_m:.3f} m)")
    print(f"  displacement_ok:    {metrics.displacement_achieved} (peak {metrics.peak_displacement_m:.3f} m)")
    print(f"  regained_contact:   {metrics.regained_contact}")
    print(f"  max_tilt:           {metrics.max_torso_tilt_rad:.3f} rad")
    print(f"  final_tilt:         {metrics.final_torso_tilt_rad:.3f} rad (after hold)")
    print(f"  max_angvel:         {metrics.max_torso_angvel:.3f} rad/s")
    print(f"  stance_ok_swing:    {metrics.stance_maintained_during_swing}")
    if metrics.landing_step is not None:
        print(f"  landing_step:       {metrics.landing_step}")
    print(f"  RESULT: {'SUCCESS' if metrics.success else 'FAILED'}")
    if metrics.failure_reasons:
        for r in metrics.failure_reasons:
            print(f"    - {r}")


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

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                    ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]

    user32 = ctypes.windll.user32
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 1), ctypes.byref(info))
    wa = info.rcWork
    x = wa.left + (wa.right - wa.left - VIEWER_WIDTH) // 2
    y = wa.top + (wa.bottom - wa.top - VIEWER_HEIGHT) // 2
    user32.SetWindowPos(hwnd, 0, x, y, VIEWER_WIDTH, VIEWER_HEIGHT, 0x0004)


def run_batch(headless: bool = True) -> list[TrialMetrics]:
    env = BipedalWalkEnv()
    trials = [DynamicTrial(f"lateral_L_{ms}ms", ms) for ms in SWING_DURATIONS_MS]
    results: list[TrialMetrics] = []
    print("=== Dynamic lateral_L (+Y) step feasibility ===")
    print(f"Swing durations tested: {SWING_DURATIONS_MS} ms")
    print(f"Strict: clearance >= {MIN_CLEARANCE_M} m, displacement >= {MIN_DISPLACEMENT_M} m, hold {POST_STEP_HOLD_STEPS} steps")

    for trial in trials:
        m = _run_trial(env, trial)
        results.append(m)
        if headless:
            _print_trial(m)

    successes = [m for m in results if m.success]
    print("\n=== Summary ===")
    print(f"Successful trials: {len(successes)} / {len(results)}")
    if successes:
        for m in successes:
            print(f"  {m.trial_name}: disp={m.peak_displacement_m:.3f} m, clear={m.peak_clearance_m:.3f} m")
    else:
        print("No trial met strict dynamic step criteria.")
        best_clear = max(results, key=lambda m: m.peak_clearance_m)
        best_disp = max(results, key=lambda m: m.peak_displacement_m)
        print(f"Best clearance: {best_clear.trial_name} -> {best_clear.peak_clearance_m:.3f} m (tilt {best_clear.max_torso_tilt_rad:.3f})")
        print(f"Best displacement: {best_disp.trial_name} -> {best_disp.peak_displacement_m:.3f} m (tilt {best_disp.max_torso_tilt_rad:.3f})")
        _print_limiting_mechanism(results)
    return results


def _print_limiting_mechanism(results: list[TrialMetrics]) -> None:
    print("\nLimiting mechanism analysis:")
    any_clear = any(m.clearance_achieved for m in results)
    any_disp = any(m.displacement_achieved for m in results)
    any_lost = any(m.lost_contact for m in results)
    any_regain = any(m.regained_contact for m in results)
    any_upright = any(m.upright_after_hold for m in results)

    if any_clear and not any_upright:
        print("  Foot can clear briefly, but torso cannot remain upright through landing/hold.")
    elif any_lost and any_disp and not any_regain:
        print("  Foot unloads and moves, but reliable re-contact fails.")
    elif not any_clear and any_disp:
        print("  Lateral displacement possible as shuffle only; clearance not achieved.")
    elif not any_lost:
        print("  Swing foot never unloads; actuation/timing insufficient to break contact.")
    else:
        print("  Combined failure: narrow support + weak torque limits prevent stable single-support phase.")


def run_viewer(swing_ms: int, slow: bool) -> TrialMetrics:
    env = BipedalWalkEnv()
    trial = DynamicTrial(f"lateral_L_{swing_ms}ms", swing_ms)
    slow_start = SETTLE_STEPS + _ms_to_steps(WEIGHT_SHIFT_MS) - 30
    slow_end = slow_start + _ms_to_steps(swing_ms + UNLOAD_MS + PLACE_MS) + 200

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, 0.0, 1.05]
        v.cam.distance = 1.25
        v.cam.azimuth = 115
        v.cam.elevation = -20
        _configure_viewer_window()
        metrics = _run_trial(
            env, trial, viewer=v, slow=slow,
            global_step=[0], slow_window=(slow_start, slow_end),
        )
    _print_trial(metrics)
    return metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic lateral step feasibility test")
    p.add_argument("--headless", action="store_true", help="Run all swing durations without viewer")
    p.add_argument("--swing-ms", type=int, default=100, help="Swing duration for viewer mode")
    p.add_argument("--slow", action="store_true", help="Slow-motion around swing phase")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless:
        run_batch(headless=True)
    else:
        run_viewer(args.swing_ms, slow=args.slow)


if __name__ == "__main__":
    main()
