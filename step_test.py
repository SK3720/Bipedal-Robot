"""Scripted single-step feasibility test for biped stepping recovery (Phase 2)."""

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
FOOT_CLEARANCE_Z = 1.045
MAX_UPRIGHT_TILT = 0.45
MIN_DISPLACEMENT_M = 0.05

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.015


class Phase(Enum):
    RESET_SETTLE = "RESET / SETTLE"
    WEIGHT_SHIFT = "WEIGHT SHIFT"
    SWING_LIFT = "SWING FOOT LIFT"
    SWING_MOVE = "SWING FOOT MOVE"
    FOOT_PLACE = "FOOT PLACEMENT"
    DOUBLE_SUPPORT = "DOUBLE-SUPPORT SETTLE"


@dataclass
class StepCase:
    name: str
    swing_leg: str
    axis: int
    direction_sign: float
    weight_shift: np.ndarray
    move_target: np.ndarray
    move_steps: int = 500
    weight_shift_steps: int = 800
    settle_steps: int = 300
    hold_steps: int = 0


@dataclass
class StepMetrics:
    case_name: str
    swing_leg: str
    success: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    initial_left_foot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    initial_right_foot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_foot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    final_swing_foot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    peak_displacement_m: float = 0.0
    final_displacement_m: float = 0.0
    min_foot_clearance_m: float = float("inf")
    max_torso_tilt_rad: float = 0.0
    final_torso_tilt_rad: float = 0.0
    max_torso_angvel: float = 0.0
    swing_lost_contact: bool = False
    swing_regained_contact: bool = False
    upright_at_end: bool = False
    stance_contact_maintained: bool = False
    phase_summaries: list[dict] = field(default_factory=list)
    trajectory_attempts: list[str] = field(default_factory=list)


def _smooth(alpha: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(alpha, 0.0, 1.0)))


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


def _make_cases(model: mujoco.MjModel) -> dict[str, StepCase]:
    cr = model.actuator_ctrlrange[:15]

    ws_l = DEFAULT_POSE.copy()
    ws_l[10] = -0.183

    move_l = ws_l.copy()
    move_l[6] = 0.20
    move_l[7] = -0.30
    move_l[8] = 0.20
    move_l[5] = cr[5, 0] * 0.9

    ws_r = DEFAULT_POSE.copy()
    ws_r[5] = cr[5, 1] * 0.20

    move_r = ws_r.copy()
    move_r[11] = 0.18
    move_r[12] = 0.32
    move_r[13] = 0.15
    move_r[10] = cr[10, 1] * 0.15

    move_sag = ws_l.copy()
    move_sag[6] = 0.20
    move_sag[7] = -0.10
    move_sag[8] = 0.25

    return {
        "lateral_L": StepCase(
            name="lateral_L",
            swing_leg="L",
            axis=1,
            direction_sign=1.0,
            weight_shift=ws_l,
            move_target=move_l,
        ),
        "lateral_R": StepCase(
            name="lateral_R",
            swing_leg="R",
            axis=1,
            direction_sign=1.0,
            weight_shift=ws_r,
            move_target=move_r,
        ),
        "sagittal_forward": StepCase(
            name="sagittal_forward",
            swing_leg="L",
            axis=0,
            direction_sign=1.0,
            weight_shift=ws_l,
            move_target=move_sag,
        ),
    }


def _foot_state(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[np.ndarray, int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
    pos = data.xpos[body_id].copy()
    contacts = 0
    for ci in range(data.ncon):
        contact = data.contact[ci]
        for geom_id in (contact.geom1, contact.geom2):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if f"{side}_foot_collision" in geom_name:
                contacts += 1
                break
    return pos, contacts


def _build_segments(case: StepCase) -> list[tuple[Phase, np.ndarray, int]]:
    ws = case.weight_shift
    move = case.move_target
    return [
        (Phase.RESET_SETTLE, DEFAULT_POSE.copy(), case.settle_steps),
        (Phase.WEIGHT_SHIFT, ws, case.weight_shift_steps),
        (Phase.SWING_MOVE, move, case.move_steps),
        (Phase.DOUBLE_SUPPORT, move, case.hold_steps),
    ]


def _build_segments_staged(case: StepCase) -> list[tuple[Phase, np.ndarray, int]]:
    ws = case.weight_shift
    move = case.move_target
    return [
        (Phase.RESET_SETTLE, DEFAULT_POSE.copy(), case.settle_steps),
        (Phase.WEIGHT_SHIFT, ws, case.weight_shift_steps),
        (Phase.SWING_LIFT, _lerp(ws, move, 0.20), case.move_steps // 4),
        (Phase.SWING_MOVE, _lerp(ws, move, 0.70), case.move_steps // 2),
        (Phase.FOOT_PLACE, move, case.move_steps // 4),
        (Phase.DOUBLE_SUPPORT, move, case.hold_steps),
    ]


def _variant_cases(base: StepCase) -> list[StepCase]:
    """Primary direct path plus one conservative fallback."""
    direct = StepCase(
        name=f"{base.name}_direct",
        swing_leg=base.swing_leg,
        axis=base.axis,
        direction_sign=base.direction_sign,
        weight_shift=base.weight_shift,
        move_target=base.move_target,
        move_steps=base.move_steps,
        weight_shift_steps=base.weight_shift_steps,
        settle_steps=base.settle_steps,
        hold_steps=0,
    )
    ws = base.weight_shift
    move = base.move_target
    leg_idxs = [5, 6, 7, 8] if base.swing_leg == "L" else [10, 11, 12, 13]
    conservative = move.copy()
    for idx in leg_idxs:
        conservative[idx] = ws[idx] + 0.80 * (move[idx] - ws[idx])
    fallback = StepCase(
        name=f"{base.name}_conservative",
        swing_leg=base.swing_leg,
        axis=base.axis,
        direction_sign=base.direction_sign,
        weight_shift=ws,
        move_target=conservative,
        move_steps=base.move_steps,
        weight_shift_steps=base.weight_shift_steps,
        settle_steps=base.settle_steps,
        hold_steps=0,
    )
    return [direct, fallback]


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


def _run_trajectory(
    env: BipedalWalkEnv,
    case: StepCase,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
) -> StepMetrics:
    model = env.model
    data = env.data
    cr = model.actuator_ctrlrange[:15]
    stance_leg = "R" if case.swing_leg == "L" else "L"

    _reset_sim(model, data)
    init_l, _ = _foot_state(model, data, "L")
    init_r, _ = _foot_state(model, data, "R")
    swing_init = init_l if case.swing_leg == "L" else init_r

    metrics = StepMetrics(
        case_name=case.name,
        swing_leg=case.swing_leg,
        initial_left_foot=init_l,
        initial_right_foot=init_r,
    )
    target_pos = swing_init.copy()
    target_pos[case.axis] += case.direction_sign * MIN_DISPLACEMENT_M
    metrics.target_foot = target_pos

    segments = _build_segments(case)
    ctrl = DEFAULT_POSE.copy()
    global_step = 0
    slow_start = case.settle_steps + case.weight_shift_steps - 40
    slow_end = sum(n for _, _, n in segments) - 60

    for phase, target, n_steps in segments:
        if n_steps <= 0:
            continue
        phase_tilts: list[float] = []
        phase_disps: list[float] = []
        phase_clearances: list[float] = []
        phase_stance_contact: list[int] = []

        for step in range(n_steps):
            alpha = _smooth((step + 1) / n_steps)
            ctrl = (1.0 - alpha) * ctrl + alpha * target
            ctrl = np.clip(ctrl, cr[:, 0], cr[:, 1])
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)

            swing_pos, swing_c = _foot_state(model, data, case.swing_leg)
            _, stance_c = _foot_state(model, data, stance_leg)
            tilt = env._quat_tilt_rad()
            angvel = float(np.linalg.norm(data.qvel[3:6]))
            clearance = swing_pos[2] - FLOOR_Z
            disp = (swing_pos[case.axis] - swing_init[case.axis]) * case.direction_sign

            metrics.max_torso_tilt_rad = max(metrics.max_torso_tilt_rad, tilt)
            metrics.max_torso_angvel = max(metrics.max_torso_angvel, angvel)
            metrics.peak_displacement_m = max(metrics.peak_displacement_m, disp)
            metrics.min_foot_clearance_m = min(metrics.min_foot_clearance_m, clearance)
            if swing_c == 0 and swing_pos[2] > FOOT_CLEARANCE_Z:
                metrics.swing_lost_contact = True

            phase_tilts.append(tilt)
            phase_disps.append(disp)
            phase_clearances.append(clearance)
            phase_stance_contact.append(stance_c)

            if viewer is not None and viewer.is_running():
                viewer.sync()
                if slow and slow_start <= global_step <= slow_end:
                    time.sleep(SLOW_SLEEP_S)
                else:
                    time.sleep(NORMAL_SLEEP_S)
            global_step += 1

        metrics.phase_summaries.append(
            {
                "phase": phase.value,
                "steps": n_steps,
                "max_tilt": float(max(phase_tilts)),
                "end_tilt": float(phase_tilts[-1]),
                "peak_disp_m": float(max(phase_disps)),
                "min_clearance_m": float(min(phase_clearances)),
                "min_stance_contacts": int(min(phase_stance_contact)),
            }
        )

    final_swing, final_swing_c = _foot_state(model, data, case.swing_leg)
    _, final_stance_c = _foot_state(model, data, stance_leg)
    metrics.final_swing_foot = final_swing
    metrics.final_displacement_m = (
        final_swing[case.axis] - swing_init[case.axis]
    ) * case.direction_sign
    metrics.final_torso_tilt_rad = env._quat_tilt_rad()
    metrics.swing_regained_contact = final_swing_c > 0
    metrics.upright_at_end = metrics.final_torso_tilt_rad < MAX_UPRIGHT_TILT
    metrics.stance_contact_maintained = final_stance_c > 0
    if metrics.min_foot_clearance_m == float("inf"):
        metrics.min_foot_clearance_m = 0.0

    reasons: list[str] = []
    if not metrics.swing_lost_contact:
        reasons.append("swing foot never cleared ground (>1.045 m)")
    if metrics.peak_displacement_m + 1e-4 < MIN_DISPLACEMENT_M:
        reasons.append(
            f"peak displacement {metrics.peak_displacement_m:.3f} m < {MIN_DISPLACEMENT_M:.3f} m"
        )
    if metrics.swing_lost_contact and not metrics.swing_regained_contact:
        reasons.append("swing foot lost contact and did not regain it")
    if not metrics.upright_at_end:
        reasons.append(
            f"final tilt {metrics.final_torso_tilt_rad:.3f} rad >= {MAX_UPRIGHT_TILT:.3f} rad"
        )
    if metrics.max_torso_tilt_rad >= 0.65:
        reasons.append(f"excessive peak tilt {metrics.max_torso_tilt_rad:.3f} rad")

    metrics.success = not reasons
    metrics.failure_reasons = reasons
    return metrics


def _find_mujoco_viewer_hwnd():
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    end_time = time.time() + VIEWER_POSITION_TIMEOUT_S
    while time.time() < end_time:
        matches: list[int] = []

        def enum_callback(hwnd, _lparam):
            if user32.IsWindowVisible(hwnd):
                n = user32.GetWindowTextLengthW(hwnd)
                if n > 0:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, buf, n + 1)
                    if buf.value.startswith(VIEWER_TITLE_PREFIX):
                        matches.append(hwnd)
            return True

        proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(enum_callback)
        user32.EnumWindows(proc, 0)
        if matches:
            return matches[0]
        time.sleep(0.05)
    return None


def _configure_viewer_window() -> None:
    if sys.platform != "win32":
        return
    hwnd = _find_mujoco_viewer_hwnd()
    if hwnd is None:
        return

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT), ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]

    user32 = ctypes.windll.user32
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 1), ctypes.byref(info))
    wa = info.rcWork
    x = wa.left + (wa.right - wa.left - VIEWER_WIDTH) // 2
    y = wa.top + (wa.bottom - wa.top - VIEWER_HEIGHT) // 2
    user32.SetWindowPos(hwnd, 0, x, y, VIEWER_WIDTH, VIEWER_HEIGHT, 0x0004)


def _print_metrics(metrics: StepMetrics) -> None:
    print(f"\n=== Step test: {metrics.case_name} ===")
    print(f"Swing leg: {metrics.swing_leg}")
    print(f"Initial L foot: {metrics.initial_left_foot}")
    print(f"Initial R foot: {metrics.initial_right_foot}")
    print(f"Target swing foot (approx): {metrics.target_foot}")
    print(f"Final swing foot:   {metrics.final_swing_foot}")
    print(f"Peak displacement:  {metrics.peak_displacement_m:.3f} m")
    print(f"Final displacement: {metrics.final_displacement_m:.3f} m")
    print(f"Min foot clearance: {metrics.min_foot_clearance_m:.3f} m")
    print(f"Max torso tilt:     {metrics.max_torso_tilt_rad:.3f} rad")
    print(f"Final torso tilt:   {metrics.final_torso_tilt_rad:.3f} rad")
    print(f"Max torso angvel:   {metrics.max_torso_angvel:.3f} rad/s")
    print(f"Swing lost contact: {metrics.swing_lost_contact}")
    print(f"Swing regained:     {metrics.swing_regained_contact}")
    print(f"Upright at end:     {metrics.upright_at_end}")

    print("\nPhase summary:")
    for row in metrics.phase_summaries:
        print(
            f"  {row['phase']:24s} | peak_disp {row['peak_disp_m']:.3f} m | "
            f"max_tilt {row['max_tilt']:.3f} | min_clear {row['min_clearance_m']:.3f} m | "
            f"stance_min_c {row['min_stance_contacts']}"
        )

    if metrics.trajectory_attempts:
        print("\nTrajectory attempts:")
        for name in metrics.trajectory_attempts:
            print(f"  - {name}")

    if metrics.success:
        print("\nRESULT: SUCCESS")
    else:
        print("\nRESULT: FAILED (strict step criteria)")
        for reason in metrics.failure_reasons:
            print(f"  - {reason}")

        shuffle_ok = (
            not metrics.swing_lost_contact
            and metrics.peak_displacement_m >= 0.04
            and metrics.upright_at_end
            and metrics.max_torso_tilt_rad < 0.65
        )
        if shuffle_ok:
            print(
                "\nPARTIAL: upright support-foot shuffle achieved "
                f"({metrics.peak_displacement_m:.3f} m displacement, both feet remained in contact). "
                "True stepping with foot clearance was not achieved."
            )
        else:
            print("\nPARTIAL: none - could not maintain upright shuffle either.")


def run_case(case_name: str, *, viewer: bool = True, slow: bool = False) -> StepMetrics:
    env = BipedalWalkEnv()
    cases = _make_cases(env.model)
    if case_name not in cases:
        raise ValueError(f"Unknown case {case_name!r}. Choose from: {list(cases)}")

    metrics: StepMetrics | None = None
    all_attempts: list[str] = []
    variants = _variant_cases(cases[case_name])
    best_metrics: StepMetrics | None = None
    best_score = -1.0

    def _score(metrics: StepMetrics) -> float:
        if not metrics.upright_at_end:
            return metrics.peak_displacement_m * 0.25
        return metrics.peak_displacement_m

    if not viewer:
        for variant in variants:
            metrics = _run_trajectory(env, variant, viewer=None, slow=False)
            all_attempts.append(variant.name)
            if _score(metrics) > best_score:
                best_metrics = metrics
                best_score = _score(metrics)
            if metrics.success:
                break
        metrics = best_metrics
        assert metrics is not None
        metrics.trajectory_attempts = all_attempts
        _print_metrics(metrics)
        return metrics

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, 0.0, 1.05]
        v.cam.distance = 1.25
        v.cam.azimuth = 115
        v.cam.elevation = -20
        _configure_viewer_window()
        for variant in variants:
            metrics = _run_trajectory(env, variant, viewer=v, slow=slow)
            all_attempts.append(variant.name)
            if _score(metrics) > best_score:
                best_metrics = metrics
                best_score = _score(metrics)
            if metrics.success:
                break
            if not v.is_running():
                break
        metrics = best_metrics
        assert metrics is not None
        metrics.trajectory_attempts = all_attempts
        _print_metrics(metrics)
        return metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scripted biped stepping feasibility test")
    parser.add_argument("--case", default="lateral_L", choices=["lateral_L", "lateral_R", "sagittal_forward"])
    parser.add_argument("--slow", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_case(args.case, viewer=not args.no_viewer, slow=args.slow)


if __name__ == "__main__":
    main()
