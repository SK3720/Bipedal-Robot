"""Forward-push recovery — clean impulse from standing, correct signed metrics.

WHY THIS EXISTS (see recovery_metrics.py header + repo analysis):
  * The golden staged_forward_catch pipeline reaches its swing state via a
    RAPID LATERAL WEIGHT SHIFT. Verified with _arrest_probe2.py: at/after the
    "catch" the chest side-leans ~-60 deg and yaws ~-45 deg while chest_z barely
    changes — it topples/spins in 3D on one foot. That failure is dominated by
    the lateral CoM offset the shift creates, NOT by forward momentum, and the
    _quat_tilt_rad metric hides it (counts yaw as tilt, fires "fallen" at ~30 deg
    real lean).
  * A real forward push does none of that. Before building any recovery-step
    controller we need: (a) a real forward impulse, (b) signed metrics, and
    (c) the honest baseline — how big a forward push can the robot take with
    NO action, and with a simple in-place ankle/hip strategy, and where does it
    genuinely REQUIRE a step (capture point leaves the support polygon).

This script does not modify robot.xml, biped_env, or any golden experiment.
The golden swing primitive (left_leg_pose / knee-clearance + hip-sweep) is left
intact for reuse once the correct falling-forward entry state is characterised.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import mujoco
import mujoco.viewer
import numpy as np

from biped_env import (
    CHEST_Z_CONTACT,
    DEFAULT_POSE,
    PUSH_DURATION_STEPS,
    SETTLE_STEPS,
    STANDING_QUAT,
)
from recovery_metrics import (
    CHEST_BODY,
    NOMINAL_CHEST_Z,
    BalanceState,
    RecoveryVerdict,
    classify_run,
    sample_balance,
)

# ---- ctrl indices (from staged_forward_catch_test) ----
IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_R_HIP_PITCH = 11
IDX_R_KNEE = 12
IDX_R_ANKLE_P = 13

FORWARD_DIR_RAD = -np.pi / 2.0      # world -Y
WINDOW_STEPS = 6000                 # 6 s post-push
SAMPLE_EVERY = 10
PUSH_AT_STEP = 20                   # a few steps of clean sim, then the impulse
DEFAULT_SWEEP = (0.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0)

NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.02


# ---------------------------------------------------------------------------
# controllers: (model, data, k, BalanceState) -> ctrl[15]
# ---------------------------------------------------------------------------

def ctrl_zero(model, data, k, bs: BalanceState) -> np.ndarray:
    return DEFAULT_POSE.copy()


@dataclass
class AnkleHipGains:
    k_ankle_lean: float = 0.9      # rad ankle per rad lean
    k_ankle_rate: float = 0.12     # rad ankle per (rad/s) pitch rate
    k_hip_lean: float = 0.7
    k_hip_rate: float = 0.10
    ankle_clip: float = 0.6
    hip_clip: float = 0.6


def make_ctrl_ankle_hip(gains: AnkleHipGains = AnkleHipGains()) -> Callable:
    """In-place fixed-point strategy: plantarflex ankles + flex hips against a
    forward lean. Symmetric on both legs. No stepping, no weight shift."""
    def ctrl(model, data, k, bs: BalanceState) -> np.ndarray:
        lean = np.radians(bs.fwd_lean_deg)
        rate = bs.pitch_rate
        u_ankle = np.clip(
            gains.k_ankle_lean * lean + gains.k_ankle_rate * rate,
            -gains.ankle_clip, gains.ankle_clip,
        )
        u_hip = np.clip(
            gains.k_hip_lean * lean + gains.k_hip_rate * rate,
            -gains.hip_clip, gains.hip_clip,
        )
        c = DEFAULT_POSE.copy()
        # forward lean (+) -> push toes down (plantarflex) to move CoP forward,
        # and flex hips to throw the torso back. Sign convention verified by the
        # sweep (flip k_* if lean grows instead of shrinks).
        c[IDX_L_ANKLE_P] += -u_ankle
        c[IDX_R_ANKLE_P] += +u_ankle
        c[IDX_L_HIP_PITCH] += -u_hip
        c[IDX_R_HIP_PITCH] += +u_hip
        return c
    return ctrl


CONTROLLERS: dict[str, Callable] = {
    "zero": ctrl_zero,
    "ankle_hip": make_ctrl_ankle_hip(),
}


def _resolve_controller(name: str, model, data) -> Callable:
    """LQR needs the model to build its gain; others are static."""
    if name == "lqr":
        from standing_balance_lqr import get_controller
        return get_controller(model, data)
    return CONTROLLERS[name]


# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    controller: str
    push_n: float
    verdict: RecoveryVerdict
    peak_capture_past_support_mm: float
    peak_com_vfwd: float
    step_taken: bool               # did either foot leave the ground post-push?
    samples: list[tuple[int, BalanceState]] = field(default_factory=list)


def _reset_standing(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = DEFAULT_POSE
    data.qvel[:] = 0.0
    data.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(model, data)
    for _ in range(SETTLE_STEPS):
        data.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_step(model, data)


def run_push(model, data, controller: Callable, controller_name: str, push_n: float,
             direction_rad: float = FORWARD_DIR_RAD,
             viewer: mujoco.viewer.Handle | None = None, slow: bool = False) -> RunResult:
    _reset_standing(model, data)
    fxy = push_n * np.array([np.cos(direction_rad), np.sin(direction_rad)])

    samples: list[tuple[int, BalanceState]] = []
    peak_cap = -1e9
    peak_vfwd = -1e9
    step_taken = False
    both_down_at_push = True

    for k in range(WINDOW_STEPS):
        data.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT_STEP <= k < PUSH_AT_STEP + PUSH_DURATION_STEPS:
            data.xfrc_applied[CHEST_BODY, 0:2] = fxy

        bs = sample_balance(model, data)
        c = controller(model, data, k, bs)
        data.ctrl[:15] = np.clip(c, model.actuator_ctrlrange[:15, 0],
                                 model.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(model, data)

        if k == PUSH_AT_STEP:
            both_down_at_push = bs.l_contact and bs.r_contact
        if k > PUSH_AT_STEP + PUSH_DURATION_STEPS:
            peak_cap = max(peak_cap, bs.capture_fwd_rel_support_mm)
            peak_vfwd = max(peak_vfwd, bs.com_vfwd)
            if both_down_at_push and not (bs.l_contact and bs.r_contact):
                step_taken = True

        if k % SAMPLE_EVERY == 0:
            samples.append((k, bs))

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

    verdict = classify_run(samples)
    return RunResult(
        controller=controller_name, push_n=push_n, verdict=verdict,
        peak_capture_past_support_mm=peak_cap, peak_com_vfwd=peak_vfwd,
        step_taken=step_taken, samples=samples,
    )


def print_run(r: RunResult, timeline: bool = False) -> None:
    v = r.verdict
    print(f"\n[{r.controller}]  push {r.push_n:.0f} N forward")
    print(f"  peak fwd_lean {v.peak_fwd_lean_deg:6.1f}  peak |side_lean| {v.peak_side_lean_deg_abs:6.1f}  "
          f"peak up_tilt {v.peak_up_tilt_deg:6.1f}")
    print(f"  peak CoM v_fwd {r.peak_com_vfwd:+.3f} m/s   peak capture-past-support {r.peak_capture_past_support_mm:+.0f} mm")
    print(f"  end: up_tilt {v.end_up_tilt_deg:.1f}  fwd_lean {v.end_fwd_lean_deg:+.1f}  side_lean {v.end_side_lean_deg:+.1f}  "
          f"chestZ {v.end_chest_z:.3f}  feet L{int(v.end_l_contact)}/R{int(v.end_r_contact)}  CoM speed {v.end_com_speed:.3f}")
    print(f"  foot left ground post-push: {r.step_taken}")
    print(f"  => {v.label}")
    if timeline:
        print("    t(s)  up_tilt fwd_lean side_lean  yaw   chestZ  capt-supp  CoMvx  L/R")
        for step, s in r.samples:
            if step % 200 != 0:
                continue
            print(f"    {step/1000:4.2f}  {s.up_tilt_deg:6.1f} {s.fwd_lean_deg:+7.1f} {s.side_lean_deg:+8.1f} "
                  f"{s.yaw_deg:+6.1f} {s.chest_z:6.3f} {s.capture_fwd_rel_support_mm:+8.0f} "
                  f"{s.com_vfwd:+5.2f} {int(s.l_contact)}/{int(s.r_contact)}")


def print_sweep_table(results: list[RunResult]) -> None:
    print("\n" + "=" * 92)
    print("FORWARD-PUSH SWEEP  (clean impulse from standing, 6 s window, signed metrics)")
    print("=" * 92)
    by_ctrl: dict[str, list[RunResult]] = {}
    for r in results:
        by_ctrl.setdefault(r.controller, []).append(r)
    for name, rs in by_ctrl.items():
        print(f"\n--- controller: {name} ---")
        print(f"{'push N':>7} {'peak fwd':>9} {'peak side':>10} {'peak upT':>9} "
              f"{'capt>supp':>10} {'CoMvx':>7} {'stepped':>8}  outcome")
        for r in sorted(rs, key=lambda x: x.push_n):
            v = r.verdict
            print(f"{r.push_n:7.0f} {v.peak_fwd_lean_deg:9.1f} {v.peak_side_lean_deg_abs:10.1f} "
                  f"{v.peak_up_tilt_deg:9.1f} {r.peak_capture_past_support_mm:10.0f} "
                  f"{r.peak_com_vfwd:7.2f} {str(r.step_taken):>8}  {v.label}")

    print("\nReading this table:")
    print("  * 'capt>supp' > 0  => capture point left the front of the support polygon")
    print("    during the push: an in-place (ankle/hip) strategy is theoretically")
    print("    insufficient and a STEP is required to recover.")
    print("  * The largest push with a RECOVERED zero/ankle_hip outcome is the")
    print("    no-step ceiling. Above it is the regime the recovery step must own.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward-push recovery baseline sweep.")
    p.add_argument("--controllers", default="zero,ankle_hip",
                   help="comma list from: " + ",".join(CONTROLLERS))
    p.add_argument("--sweep", default=None,
                   help="comma list of push magnitudes (N); default is the built-in sweep")
    p.add_argument("--push", type=float, default=None,
                   help="single push magnitude (implies visual run unless --headless)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--timeline", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    model = mujoco.MjModel.from_xml_path("robot/robot.xml")
    data = mujoco.MjData(model)

    ctrl_names = [c.strip() for c in args.controllers.split(",") if c.strip()]
    known = list(CONTROLLERS) + ["lqr"]
    for cn in ctrl_names:
        if cn not in known:
            raise SystemExit(f"unknown controller {cn!r}; have {known}")

    if args.push is not None and not args.headless:
        cn = ctrl_names[0]
        ctrl_fn = _resolve_controller(cn, model, data)
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.lookat[:] = [0.0, -0.05, 1.05]
            v.cam.distance = 1.9
            v.cam.azimuth = 90
            v.cam.elevation = -10
            r = run_push(model, data, ctrl_fn, cn, args.push,
                         viewer=v, slow=args.slow)
            print_run(r, timeline=True)
            print("\nClose viewer to exit.")
            while v.is_running():
                v.sync()
                time.sleep(SLOW_SLEEP_S if args.slow else NORMAL_SLEEP_S)
        return

    sweep = DEFAULT_SWEEP
    if args.sweep:
        sweep = tuple(float(x) for x in args.sweep.split(","))
    if args.push is not None:
        sweep = (args.push,)

    results: list[RunResult] = []
    for cn in ctrl_names:
        ctrl_fn = _resolve_controller(cn, model, data)
        for pn in sweep:
            r = run_push(model, data, ctrl_fn, cn, pn)
            print_run(r, timeline=args.timeline)
            results.append(r)
    print_sweep_table(results)


if __name__ == "__main__":
    main(sys.argv[1:])
