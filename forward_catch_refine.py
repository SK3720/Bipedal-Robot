"""Refine forward (+X) catch trajectory — weight-shift sweep on original robot geometry.

Baseline: L_fast @ 40 N from sagittal_step_test.py.
Evaluation only; does not modify robot.xml, biped_env, or PPO artifacts.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import product
from typing import Iterable

from biped_env import BipedalWalkEnv

from sagittal_step_test import (
    MAX_UPRIGHT_TILT,
    MIN_CLEARANCE_M,
    MIN_DISPLACEMENT_M,
    SagittalCase,
    TrajectoryParams,
    TrialMetrics,
    _configure_viewer_window,
    _print_trial,
    _run_trial,
)

import mujoco.viewer

PUSH_N = 40.0

# L_fast baseline (hip forward = negative ctrl per interactive keys 1/2)
BASELINE = TrajectoryParams(
    name="L_fast",
    swing_leg="L",
    hip_pitch=-0.22,
    knee=-0.35,
    ankle=0.30,
    ws_stance_roll=-0.183,
    swing_ms=100,
)

# INVALID — measured displacement along world +X (lateral), not anatomical forward.
# See forward_step_debug.py --audit for analysis.
BEST_FORWARD_CATCH = TrajectoryParams(
    name="best_forward_catch",
    swing_leg="L",
    hip_pitch=-0.20,
    knee=-0.35,
    ankle=0.25,
    ws_stance_roll=-0.14,
    swing_ms=120,
    stance_ankle_roll=0.10,
    stabilize_ms=300,
)

WS_SWEEP = [-0.08, -0.10, -0.12, -0.14, -0.16, -0.183]
STANCE_ANKLE_ROLL_SWEEP = [0.0, -0.05, 0.05, 0.10, -0.10]

VALIDATION_PUSH_NS = [35.0, 40.0, 45.0]


@dataclass(frozen=True)
class RefineCandidate:
    traj: TrajectoryParams
    label: str


def _case(traj: TrajectoryParams, push_n: float = PUSH_N) -> SagittalCase:
    return SagittalCase("forward", push_n, traj.swing_leg, traj, 1.0, 1.0)


def _rank_key(m: TrialMetrics) -> tuple:
    """Prefer strict success, then stance retained, then kinematics, then low tilt."""
    return (
        0 if m.success else 1,
        0 if m.stance_ok_swing else 1,
        0 if m.regained_contact else 1,
        0 if m.peak_clearance_m >= MIN_CLEARANCE_M else 1,
        0 if m.peak_displacement_m >= MIN_DISPLACEMENT_M else 1,
        m.final_tilt_rad,
        -m.peak_displacement_m,
    )


def _traj_label(traj: TrajectoryParams) -> str:
    return (
        f"ws{traj.ws_stance_roll:.3f}_ar{traj.stance_ankle_roll:+.2f}_"
        f"h{abs(traj.hip_pitch):.2f}_k{abs(traj.knee):.2f}_a{traj.ankle:.2f}_"
        f"t{traj.swing_ms}_stab{traj.stabilize_ms}"
    )


def _run_batch(env: BipedalWalkEnv, candidates: list[RefineCandidate], push_n: float) -> list[tuple[RefineCandidate, TrialMetrics]]:
    out: list[tuple[RefineCandidate, TrialMetrics]] = []
    for cand in candidates:
        case = _case(cand.traj, push_n)
        case.traj = TrajectoryParams(
            cand.traj.name,
            cand.traj.swing_leg,
            cand.traj.hip_pitch,
            cand.traj.knee,
            cand.traj.ankle,
            cand.traj.ws_stance_roll,
            cand.traj.swing_ms,
            cand.traj.stance_ankle_roll,
            cand.traj.stabilize_ms,
            cand.traj.place_hip_scale,
            cand.traj.place_knee_scale,
        )
        m = _run_trial(env, case)
        m.case_name = cand.label
        out.append((cand, m))
    return out


def _print_row(cand: RefineCandidate, m: TrialMetrics) -> None:
    ok = "PASS" if m.success else "fail"
    stance = "ok" if m.stance_ok_swing else "LOST"
    print(
        f"  [{ok}] {cand.label}: clear={m.peak_clearance_m*1000:.0f}mm "
        f"disp={m.peak_displacement_m*1000:.0f}mm stance={stance} "
        f"tilt_max={m.max_torso_tilt_rad:.3f} final={m.final_tilt_rad:.3f} "
        f"angvel={m.max_torso_angvel:.2f} | {', '.join(m.failure_reasons[:2])}"
    )


def phase1_weight_shift_sweep(env: BipedalWalkEnv) -> tuple[TrajectoryParams, list[tuple[RefineCandidate, TrialMetrics]]]:
    print("=== Phase 1: R hip-roll weight-shift sweep (L_fast swing fixed @ 40 N) ===")
    candidates: list[RefineCandidate] = []
    for ws, ar in product(WS_SWEEP, STANCE_ANKLE_ROLL_SWEEP):
        traj = TrajectoryParams(
            BASELINE.name,
            BASELINE.swing_leg,
            BASELINE.hip_pitch,
            BASELINE.knee,
            BASELINE.ankle,
            ws,
            BASELINE.swing_ms,
            ar,
            BASELINE.stabilize_ms,
        )
        label = f"ws_sweep_{_traj_label(traj)}"
        candidates.append(RefineCandidate(traj, label))

    results = _run_batch(env, candidates, PUSH_N)
    results.sort(key=lambda x: _rank_key(x[1]))
    for cand, m in results[:12]:
        _print_row(cand, m)
    best_traj = results[0][0].traj
    print(f"\nPhase 1 best: ws={best_traj.ws_stance_roll:.3f} ar={best_traj.stance_ankle_roll:+.2f} "
          f"success={results[0][1].success} stance_ok={results[0][1].stance_ok_swing}")
    return best_traj, results


def phase2_local_refine(env: BipedalWalkEnv, center: TrajectoryParams) -> list[tuple[RefineCandidate, TrialMetrics]]:
    print("\n=== Phase 2: local refine around best weight shift ===")
    swing_ms_vals = sorted({max(60, center.swing_ms - 20), center.swing_ms, center.swing_ms + 20})
    hip_vals = [center.hip_pitch + d for d in (-0.02, 0.0, 0.02)]
    knee_vals = [center.knee + d for d in (-0.03, 0.0, 0.03)]
    ankle_vals = [center.ankle + d for d in (-0.05, 0.0, 0.05)]
    stab_vals = [150, 200, 300]

    candidates: list[RefineCandidate] = []
    seen: set[str] = set()
    for sm, hip, knee, ankle, stab in product(swing_ms_vals, hip_vals, knee_vals, ankle_vals, stab_vals):
        traj = TrajectoryParams(
            "refined",
            center.swing_leg,
            hip,
            knee,
            ankle,
            center.ws_stance_roll,
            sm,
            center.stance_ankle_roll,
            stab,
        )
        label = f"local_{_traj_label(traj)}"
        if label in seen:
            continue
        seen.add(label)
        candidates.append(RefineCandidate(traj, label))

    print(f"  {len(candidates)} local candidates @ 40 N")
    results = _run_batch(env, candidates, PUSH_N)
    results.sort(key=lambda x: _rank_key(x[1]))
    successes = [r for r in results if r[1].success]
    print(f"  Strict successes: {len(successes)} / {len(results)}")
    for cand, m in results[:10]:
        _print_row(cand, m)
    return results


def validate_reliability(
    env: BipedalWalkEnv, traj: TrajectoryParams
) -> list[TrialMetrics]:
    print("\n=== Validation: push magnitude sweep ===")
    metrics: list[TrialMetrics] = []
    for push_n in VALIDATION_PUSH_NS:
        case = _case(traj, push_n)
        m = _run_trial(env, case)
        m.case_name = f"validate_{push_n:.0f}N_{_traj_label(traj)}"
        metrics.append(m)
        _print_row(RefineCandidate(traj, m.case_name), m)
    return metrics


def print_final_report(traj: TrajectoryParams, m: TrialMetrics, validation: list[TrialMetrics]) -> None:
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print("Best trajectory parameters:")
    print(f"  ws_stance_roll (R hip roll): {traj.ws_stance_roll:.4f}")
    print(f"  stance_ankle_roll (R ankle): {traj.stance_ankle_roll:+.3f}")
    print(f"  swing hip_pitch (L):         {traj.hip_pitch:.3f}  (negative = forward)")
    print(f"  swing knee (L):              {traj.knee:.3f}")
    print(f"  swing ankle_pitch (L):     {traj.ankle:.3f}")
    print(f"  swing_ms:                    {traj.swing_ms}")
    print(f"  stabilize_ms:                {traj.stabilize_ms}")
    print(f"  push_force_n:                {PUSH_N}")
    print()
    print(f"  Stance contact maintained:   {m.stance_ok_swing}")
    print(f"  Clearance:                   {m.peak_clearance_m*1000:.1f} mm")
    print(f"  Forward displacement:        {m.peak_displacement_m*1000:.1f} mm")
    print(f"  Re-contact:                  {m.regained_contact}")
    print(f"  Max torso tilt:              {m.max_torso_tilt_rad:.3f} rad")
    print(f"  Final torso tilt:            {m.final_tilt_rad:.3f} rad")
    print(f"  Max angular velocity:        {m.max_torso_angvel:.3f} rad/s")
    print(f"  Pre-swing max tilt:          {m.max_tilt_pre_swing:.3f} rad")
    print(f"  Strict success @ 40 N:       {m.success}")
    if validation:
        n_ok = sum(1 for v in validation if v.success)
        print(f"  Strict success validation:   {n_ok} / {len(validation)} pushes {VALIDATION_PUSH_NS}")
    if m.success and validation and all(v.success for v in validation):
        print("\nMechanism: reduced lateral weight-shift on stance hip roll keeps R foot")
        print("loaded while L hip pitch (forward) + knee flex achieve clearance/placement.")
        print("STOP — repeatable scripted forward catch demonstrated. Do not proceed to PPO yet.")
    elif m.success:
        print("\nPartial: passes at 40 N but not all validation pushes — tune further.")
    else:
        print("\nMechanism analysis:")
        if not m.stance_ok_swing:
            print("  Stance foot unloads during swing — weight shift still too aggressive or")
            print("  swing dynamics pull stance foot despite reduced R hip roll.")
        elif m.peak_displacement_m < MIN_DISPLACEMENT_M:
            print("  Stance retained but insufficient forward reach — hip/knee amplitude or timing.")
        elif m.final_tilt_rad >= MAX_UPRIGHT_TILT:
            print("  Step completes kinematically but landing/stabilization tilt too high.")
        else:
            print(f"  Remaining failures: {', '.join(m.failure_reasons)}")
        print("\nRecommended next experiment:")
        print("  - Split weight shift into slower lateral load + delayed swing")
        print("  - Or add stance ankle pitch (not roll) to press R toe into floor")
        print("  - Or reduce push to 30-35 N while refining trajectory")


def run_full_sweep() -> None:
    env = BipedalWalkEnv()
    print("Baseline L_fast @ 40 N:")
    base_m = _run_trial(env, _case(BASELINE))
    _print_trial(base_m)

    _, phase1 = phase1_weight_shift_sweep(env)
    best_ws_traj = phase1[0][0].traj

    phase2 = phase2_local_refine(env, best_ws_traj)
    best_cand, best_m = phase2[0]

    validation: list[TrialMetrics] = []
    if best_m.success:
        validation = validate_reliability(env, best_cand.traj)
    elif best_m.stance_ok_swing:
        validation = validate_reliability(env, best_cand.traj)

    print_final_report(best_cand.traj, best_m, validation)


def run_viewer(traj: TrajectoryParams | None, slow: bool) -> None:
    env = BipedalWalkEnv()
    t = traj or BASELINE
    case = _case(t)
    from sagittal_step_test import SETTLE_STEPS, STAND_MS, PUSH_MS, REACT_MS, WEIGHT_SHIFT_MS, UNLOAD_MS, PLACE_MS, _ms_to_steps

    slow_start = SETTLE_STEPS + _ms_to_steps(STAND_MS) - 20
    slow_end = slow_start + _ms_to_steps(
        PUSH_MS + REACT_MS + WEIGHT_SHIFT_MS + UNLOAD_MS + t.swing_ms + PLACE_MS + t.stabilize_ms + 400
    )
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.05, 0.0, 1.05]
        v.cam.distance = 1.35
        v.cam.azimuth = 140
        v.cam.elevation = -15
        _configure_viewer_window()
        m = _run_trial(env, case, viewer=v, slow=slow, global_step=[0], slow_window=(slow_start, slow_end))
    _print_trial(m)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward catch trajectory refinement sweep")
    p.add_argument("--headless", action="store_true", help="Run full two-phase sweep")
    p.add_argument("--slow", action="store_true", help="Viewer slow motion")
    p.add_argument("--best", action="store_true", help="Viewer: use refined best-forward-catch trajectory")
    p.add_argument("--ws", type=float, default=None, help="Override ws_stance_roll for viewer")
    p.add_argument("--ankle-roll", type=float, default=None, help="Stance ankle roll for viewer")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless:
        run_full_sweep()
    else:
        traj = BEST_FORWARD_CATCH if args.best else BASELINE
        if args.ws is not None or args.ankle_roll is not None:
            traj = TrajectoryParams(
                traj.name, traj.swing_leg, traj.hip_pitch, traj.knee,
                traj.ankle,
                args.ws if args.ws is not None else traj.ws_stance_roll,
                traj.swing_ms,
                args.ankle_roll if args.ankle_roll is not None else traj.stance_ankle_roll,
                traj.stabilize_ms,
            )
        run_viewer(traj, slow=args.slow)


if __name__ == "__main__":
    main(sys.argv[1:])
