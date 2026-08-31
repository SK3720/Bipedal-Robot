"""Stance-width sweep for dynamic lateral_L stepping (evaluation only).

Tests whether widening the hip attachment spacing (40 / 80 / 120 mm foot
separation) enables strict dynamic stepping without modifying robot/robot.xml
or any PPO artifacts. Reuses trajectory logic from dynamic_step_test.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import mujoco
import mujoco.viewer

from biped_env import CHEST_Z_CONTACT, DEFAULT_POSE, SETTLE_STEPS
from dynamic_step_test import (
    DynamicTrial,
    TrialMetrics,
    WEIGHT_SHIFT_MS,
    UNLOAD_MS,
    SWING_DURATIONS_MS,
    _configure_viewer_window,
    _print_trial,
    _reset_sim,
    _run_trial,
)
from stance_width_utils import (
    STANCE_WIDTHS_MM,
    StanceStepEnv,
    load_model_with_stance_width,
    measure_foot_stance_width_m,
)


@dataclass
class WidthSummary:
    target_width_mm: int
    measured_width_mm: float
    n_success: int
    n_trials: int
    best_trial: str | None
    best_clearance_m: float
    best_displacement_m: float
    best_final_tilt_rad: float
    best_max_tilt_rad: float
    best_stance_ok: bool
    any_success: bool


def _make_env(target_width_m: float) -> StanceStepEnv:
    model = load_model_with_stance_width(target_width_m)
    data = mujoco.MjData(model)
    return StanceStepEnv(model=model, data=data)


def _measure_settled_width(env: StanceStepEnv) -> float:
    _reset_sim(env.model, env.data)
    return measure_foot_stance_width_m(env.model, env.data)


def _trial_name(width_mm: int, swing_ms: int) -> str:
    return f"w{width_mm}_lateral_L_{swing_ms}ms"


def run_batch(headless: bool = True) -> tuple[list[TrialMetrics], list[WidthSummary]]:
    all_results: list[TrialMetrics] = []
    summaries: list[WidthSummary] = []

    print("=== Stance-width dynamic lateral_L (+Y) step feasibility ===")
    print(f"Target stance widths: {list(STANCE_WIDTHS_MM)} mm")
    print(f"Swing durations: {SWING_DURATIONS_MS} ms")
    print("Strict criteria inherited from dynamic_step_test.py\n")

    for width_mm in STANCE_WIDTHS_MM:
        target_m = width_mm / 1000.0
        env = _make_env(target_m)
        measured_mm = _measure_settled_width(env) * 1000.0
        print(f"--- Stance width {width_mm} mm (measured {measured_mm:.1f} mm) ---")

        width_results: list[TrialMetrics] = []
        for swing_ms in SWING_DURATIONS_MS:
            trial = DynamicTrial(_trial_name(width_mm, swing_ms), swing_ms)
            metrics = _run_trial(env, trial)
            width_results.append(metrics)
            all_results.append(metrics)
            if headless:
                _print_trial(metrics)

        successes = [m for m in width_results if m.success]
        best = min(
            width_results,
            key=lambda m: (
                0 if m.success else 1,
                m.final_torso_tilt_rad,
                -m.peak_clearance_m,
            ),
        )
        summaries.append(
            WidthSummary(
                target_width_mm=width_mm,
                measured_width_mm=measured_mm,
                n_success=len(successes),
                n_trials=len(width_results),
                best_trial=best.trial_name,
                best_clearance_m=best.peak_clearance_m,
                best_displacement_m=best.peak_displacement_m,
                best_final_tilt_rad=best.final_torso_tilt_rad,
                best_max_tilt_rad=best.max_torso_tilt_rad,
                best_stance_ok=best.stance_maintained_during_swing,
                any_success=bool(successes),
            )
        )

    _print_comparison_table(summaries)
    _print_conclusion(summaries, all_results)
    return all_results, summaries


def _print_comparison_table(summaries: list[WidthSummary]) -> None:
    print("\n=== Comparison table (best trial per stance width) ===")
    header = (
        f"{'Width':>6} | {'Meas':>6} | {'OK':>3} | "
        f"{'Clear':>6} | {'Disp':>6} | {'MaxTilt':>7} | {'Final':>7} | "
        f"{'Stance':>6} | Best trial"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        ok = "YES" if s.any_success else "no"
        stance = "ok" if s.best_stance_ok else "lost"
        print(
            f"{s.target_width_mm:>5}mm | {s.measured_width_mm:>5.1f}mm | {ok:>3} | "
            f"{s.best_clearance_m*1000:>5.0f}mm | {s.best_displacement_m*1000:>5.0f}mm | "
            f"{s.best_max_tilt_rad:>6.2f} | {s.best_final_tilt_rad:>6.2f} | "
            f"{stance:>6} | {s.best_trial}"
        )


def _print_conclusion(summaries: list[WidthSummary], results: list[TrialMetrics]) -> None:
    total_success = sum(s.n_success for s in summaries)
    total_trials = sum(s.n_trials for s in summaries)
    print(f"\n=== Overall: {total_success} / {total_trials} strict successes ===")

    if any(s.any_success for s in summaries):
        winners = [s for s in summaries if s.any_success]
        print("Widening stance enabled at least one strict dynamic step.")
        for w in winners:
            print(f"  {w.target_width_mm} mm: {w.n_success}/{w.n_trials} successes")
        return

    print("No stance width produced a strict dynamic step.")
    baseline = next(s for s in summaries if s.target_width_mm == 40)
    widest = next(s for s in summaries if s.target_width_mm == 120)
    tilt_improved = widest.best_final_tilt_rad < baseline.best_final_tilt_rad
    stance_improved = widest.best_stance_ok and not baseline.best_stance_ok

    print("\nLimiting mechanism analysis:")
    if tilt_improved or stance_improved:
        print(
            "  Wider stance improves balance metrics but remains insufficient for strict success."
        )
    else:
        print(
            "  Wider stance did not materially improve single-support balance; "
            "geometry may not be the sole bottleneck."
        )

    any_regain = all(m.regained_contact for m in results)
    any_clear = all(m.clearance_achieved for m in results)
    if any_regain and any_clear:
        print(
            "  Kinematics (clearance, displacement, re-contact) succeed at all widths; "
            "torso collapse during/after landing is the consistent failure mode."
        )


def run_viewer(width_mm: int, swing_ms: int, slow: bool) -> TrialMetrics:
    env = _make_env(width_mm / 1000.0)
    measured_mm = _measure_settled_width(env) * 1000.0
    trial = DynamicTrial(_trial_name(width_mm, swing_ms), swing_ms)
    slow_start = SETTLE_STEPS + int(WEIGHT_SHIFT_MS) - 30
    slow_end = slow_start + int(swing_ms + UNLOAD_MS + 100) + 200

    print(f"Viewer: target {width_mm} mm, measured {measured_mm:.1f} mm, swing {swing_ms} ms")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 1.05]
        viewer.cam.distance = 1.25
        viewer.cam.azimuth = 115
        viewer.cam.elevation = -20
        _configure_viewer_window()
        metrics = _run_trial(
            env,
            trial,
            viewer=viewer,
            slow=slow,
            global_step=[0],
            slow_window=(slow_start, slow_end),
        )
    _print_trial(metrics)
    return metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stance-width sweep for dynamic lateral stepping")
    p.add_argument("--headless", action="store_true", help="Run all widths and swing durations")
    p.add_argument("--width-mm", type=int, default=80, choices=STANCE_WIDTHS_MM, help="Stance width for viewer")
    p.add_argument("--swing-ms", type=int, default=100, help="Swing duration for viewer mode")
    p.add_argument("--slow", action="store_true", help="Slow-motion around swing phase")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless:
        run_batch(headless=True)
    else:
        run_viewer(args.width_mm, args.swing_ms, slow=args.slow)


if __name__ == "__main__":
    main(sys.argv[1:])
