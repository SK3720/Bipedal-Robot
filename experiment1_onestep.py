"""Experiment 1 - one dynamically feasible forward-push recovery step.

    push (~140 N) -> unload one foot -> swing forward -> plant ahead
    -> torso stays upright -> weight transfers -> stable double support -> LQR

Baseline / real foot geometry (robot/robot.xml).  Always steps at the test
magnitude (no trigger logic - Phase 3).  Isolates whether the STEP MANEUVER
itself can work with a feasible reference + TVLQR instead of fixed-point LQR
switching.

Stages (each prints an instrumented failure breakdown):
    seed    open-loop scripted unload/swing/plant tape (expected: fails)
    tvlqr   same tape, tracked with the iLQR backward-pass gains (no re-opt)
    ilqr    iLQR-optimised reference + TVLQR gains + terminal DS-LQR

Run:
    python experiment1_onestep.py --stage ilqr --push 140
    python experiment1_onestep.py --stage ilqr --push 140 --slow      # watch
    python experiment1_onestep.py --selftest                          # plumbing
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ilqr_recovery import RecoveryILQR, StepPlan, Weights, execute

CACHE = Path("_exp1_cache")


def build(push_n, swing, N, H):
    model = mujoco.MjModel.from_xml_path("robot/robot.xml")
    plan = StepPlan(swing=swing, push_n=push_n, N=N, H=H)
    prob = RecoveryILQR(model, plan, Weights(), verbose=True)
    return prob


def selftest(push_n, swing):
    prob = build(push_n, swing, N=40, H=10)
    U = prob.seed_controls()
    t0 = time.time()
    X = prob.rollout(U)
    print(f"rollout {len(X)} knots in {time.time()-t0:.2f}s   "
          f"x0 chestZ {prob.x0[2]:.3f}  end chestZ {X[-1][2]:.3f}")
    t0 = time.time()
    A, B = prob.sim.linearize_hold(X[0], U[0], prob.H)
    print(f"linearize_hold {A.shape} in {time.time()-t0:.2f}s   "
          f"|A| {np.linalg.norm(A):.1f}  |B| {np.linalg.norm(B):.3f}  "
          f"max|eig(A)| {np.max(np.abs(np.linalg.eigvals(A))):.3f}")
    t0 = time.time()
    r, Jx = prob._res_jac_x(5, X[5])
    print(f"res_jac_x r={r.shape} Jx={Jx.shape} in {time.time()-t0:.2f}s  |Jx| {np.linalg.norm(Jx):.2f}")
    J = prob.cost(X[:41], U[:40]) if len(X) > 40 else prob.cost(X, U)
    print(f"seed cost (40-knot) = {J:.2f}")
    print("plumbing OK")


def run(stage, push_n, swing, N, H, iters, slow, headless, reopt):
    key = CACHE / f"{stage}_{push_n:.0f}_{swing}_{N}_{H}.pkl"
    prob = build(push_n, swing, N, H)

    if stage == "seed":
        U = prob.seed_controls()
        X = prob.rollout(U)
        K = np.zeros((prob.N, prob.nu, prob.ndx))       # open loop
    elif stage == "tvlqr":
        U = prob.seed_controls()
        X = prob.rollout(U)
        AB = prob.linearize_all(X, U)
        bp = prob.backward(X, U, AB, 1e-3)
        K = bp[1] if bp is not None else np.zeros((prob.N, prob.nu, prob.ndx))
        prob.X_ref, prob.U_ref, prob.K_ref = X, U, K
    else:  # ilqr
        if key.exists() and not reopt:
            with open(key, "rb") as fh:
                X, U, K = pickle.load(fh)
            prob.X_ref, prob.U_ref, prob.K_ref = X, U, K
            print(f"[loaded cached optimum {key}]")
        else:
            X, U, K = prob.optimize(iters=iters)
            CACHE.mkdir(exist_ok=True)
            with open(key, "wb") as fh:
                pickle.dump((X, U, K), fh)
            print(f"[cached -> {key}]")

    if headless:
        execute(prob, X, U, K, viewer=None, verbose=True)
        return

    with mujoco.viewer.launch_passive(prob.sim.m, prob.sim.d) as vw:
        vw.cam.lookat[:] = [0.0, -0.15, 1.0]
        vw.cam.distance = 2.2
        vw.cam.azimuth = 90
        vw.cam.elevation = -8
        execute(prob, X, U, K, viewer=vw, slow=slow, verbose=True)
        print("\n  close viewer to exit")
        while vw.is_running():
            vw.sync()
            time.sleep(0.02 if slow else 0.002)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["seed", "tvlqr", "ilqr"], default="ilqr")
    p.add_argument("--push", type=float, default=140.0)
    p.add_argument("--swing", choices=["L", "R"], default="R")
    p.add_argument("--N", type=int, default=130)
    p.add_argument("--H", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--reopt", action="store_true", help="ignore cached optimum")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        selftest(a.push, a.swing)
        return
    run(a.stage, a.push, a.swing, a.N, a.H, a.iters, a.slow, a.headless, a.reopt)


if __name__ == "__main__":
    main(sys.argv[1:])
