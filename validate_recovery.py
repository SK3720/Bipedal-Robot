"""Robustness validation for the small-push recovery controller
(sagittal_recovery.py).  The nominal-sim milestone is done; before any
sim-to-real work we need to know whether the controller is genuinely robust or
overfit to one deterministic scenario.

Method
------
The physically meaningful disturbance is the post-push horizontal CoM speed, not
the push force in Newtons (force/mass, so a lighter robot gets a bigger kick from
the same force).  We calibrate, per model, the force that produces a target
post-push CoM speed, then sweep robustness axes one at a time and finally a
combined domain-randomised batch.

Axes: disturbance magnitude, disturbance direction (incl. BACKWARD), body mass,
floor friction, actuator torque limit, control latency, initial-pose jitter.

    python validate_recovery.py                # full matrix
    python validate_recovery.py --axis mass    # one axis
    python validate_recovery.py --random 60    # domain-randomised batch only
"""
from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from sagittal_recovery import Cfg, run
from recovery_metrics import CHEST_BODY, sample_balance
from biped_env import PUSH_DURATION_STEPS

PUSH_AT = 20
FWD = np.array([0.0, -1.0])          # world forward
BACK = np.array([0.0, 1.0])


# ---------------------------------------------------------------- model mutators
def mk_model_fn(mass=1.0, friction=1.0, actuator=1.0):
    def fn(m):
        if mass != 1.0:
            m.body_mass[:] *= mass
            m.body_inertia[:] *= mass
        if friction != 1.0:
            m.geom_friction[:, 0] *= friction        # tangential
        if actuator != 1.0:
            m.actuator_forcerange[:] *= actuator
            m.actuator_gainprm[:15, 0] *= 1.0        # keep kp; only clamp changes
    return fn


# --------------------------------------------------- calibrate force -> CoM speed
def _post_push_speed(push_n, push_dir, model_fn):
    """measure |horizontal CoM velocity| just after the push, no controller."""
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    if model_fn:
        model_fn(m)
    d = mujoco.MjData(m)
    from standing_balance_lqr import StandingLQR
    lqr = StandingLQR(m, d, verbose=False)
    d.qpos[:] = lqr.qpos0
    d.qvel[:] = lqr.qvel0
    mujoco.mj_forward(m, d)
    pd = np.asarray(push_dir, float)
    pd = pd / np.linalg.norm(pd)
    fxy = push_n * pd
    for k in range(PUSH_AT + PUSH_DURATION_STEPS + 3):
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT <= k < PUSH_AT + PUSH_DURATION_STEPS:
            d.xfrc_applied[CHEST_BODY, 0:2] = fxy
        d.ctrl[:15] = lqr.control(m, d)
        mujoco.mj_step(m, d)
    mujoco.mj_subtreeVel(m, d)
    return float(np.hypot(*d.subtree_linvel[CHEST_BODY][:2]))


def force_for_speed(v_target, push_dir=FWD, model_fn=None):
    """linear calibration: speed is ~linear in force for these small pushes."""
    lo = _post_push_speed(100.0, push_dir, model_fn)
    hi = _post_push_speed(160.0, push_dir, model_fn)
    slope = (hi - lo) / 60.0
    return float(np.clip(100.0 + (v_target - lo) / max(slope, 1e-6), 40, 400))


# ------------------------------------------------------------------- trial
def trial(v_target=0.24, push_dir=FWD, mass=1.0, friction=1.0, actuator=1.0,
          latency=0, jitter=0.0, seed=0, swing="R"):
    model_fn = mk_model_fn(mass, friction, actuator)
    pn = force_for_speed(v_target, push_dir, model_fn)
    r = run(pn, Cfg(swing=swing), verbose=False, push_dir=push_dir,
            model_fn=model_fn, ctrl_delay_ms=latency, init_jitter_deg=jitter,
            seed=seed)
    return r


def _rate(results):
    n = len(results)
    s = sum(1 for r in results if r["success"])
    # "recovered but no step needed" (LQR alone held it) counts as fine too
    held = sum(1 for r in results
               if not r["fell"] and not r["triggered"] and r["end_ds"]
               and r["end_up"] < 8)
    return s, held, n


def axis_sweep(name, cases):
    print(f"\n=== {name} ===")
    print(f"  {'case':<22} {'push_N':>7} {'v0':>5} {'step?':>6} {'peakUp':>7} "
          f"{'result':>9}")
    for label, kw in cases:
        r = trial(**kw)
        res = ("RECOVER" if r["success"] else
               ("held" if (not r["fell"] and not r["triggered"]) else
                ("FELL" if r["fell"] else "fail")))
        print(f"  {label:<22} {r['push_n']:7.0f} {kw.get('v_target',0.24):5.2f} "
              f"{str(r['n_planted']):>6} {r.get('peak_up',0):7.1f} {res:>9}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--axis", default="all")
    p.add_argument("--random", type=int, default=0)
    a = p.parse_args(argv)

    if a.random:
        rng = np.random.default_rng(1)
        res = []
        print(f"\n=== domain-randomised batch (n={a.random}) ===")
        for i in range(a.random):
            th = rng.uniform(-25, 25)                  # push dir: fwd +/- 25 deg
            pdir = np.array([np.sin(np.radians(th)), -np.cos(np.radians(th))])
            kw = dict(
                v_target=rng.uniform(0.17, 0.32),
                push_dir=pdir,
                mass=rng.uniform(0.88, 1.12),
                friction=rng.uniform(0.7, 1.3),
                actuator=rng.uniform(0.9, 1.1),
                latency=int(rng.integers(0, 7)),
                jitter=rng.uniform(0, 2.0),
                seed=i,
            )
            r = trial(**kw)
            res.append(r)
            if not r["success"] and not (not r["fell"] and not r["triggered"]):
                print(f"  #{i:2d} FAIL  v{kw['v_target']:.2f} dir{th:+.0f} "
                      f"m{kw['mass']:.2f} fr{kw['friction']:.2f} "
                      f"act{kw['actuator']:.2f} lat{kw['latency']} "
                      f"jit{kw['jitter']:.1f}  -> "
                      f"{'FELL' if r['fell'] else 'no-recover'}")
        s, held, n = _rate(res)
        print(f"\n  recovered via step: {s}/{n}   held w/o step: {held}/{n}   "
              f"total OK: {s + held}/{n}")
        return

    if a.axis in ("all", "magnitude"):
        axis_sweep("disturbance magnitude (forward)", [
            (f"v0={v:.2f}", dict(v_target=v)) for v in
            (0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36)])
    if a.axis in ("all", "direction"):
        axis_sweep("disturbance direction (v0=0.24)", [
            ("forward", dict(push_dir=FWD)),
            ("fwd +15deg", dict(push_dir=[np.sin(np.radians(15)), -np.cos(np.radians(15))])),
            ("fwd -15deg", dict(push_dir=[np.sin(np.radians(-15)), -np.cos(np.radians(-15))])),
            ("fwd +25deg", dict(push_dir=[np.sin(np.radians(25)), -np.cos(np.radians(25))])),
            ("BACKWARD", dict(push_dir=BACK)),
        ])
    if a.axis in ("all", "mass"):
        axis_sweep("body mass", [
            (f"mass x{s}", dict(mass=s)) for s in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)])
    if a.axis in ("all", "friction"):
        axis_sweep("floor friction", [
            (f"friction x{s}", dict(friction=s)) for s in (0.5, 0.7, 0.85, 1.0, 1.3, 1.6)])
    if a.axis in ("all", "actuator"):
        axis_sweep("actuator torque limit", [
            (f"actuator x{s}", dict(actuator=s)) for s in (0.8, 0.9, 0.95, 1.0, 1.1, 1.2)])
    if a.axis in ("all", "latency"):
        axis_sweep("control latency", [
            (f"latency {s} ms", dict(latency=s)) for s in (0, 1, 2, 3, 5, 8, 12)])
    if a.axis in ("all", "jitter"):
        axis_sweep("initial-pose jitter", [
            (f"jitter {s} deg", dict(jitter=s, seed=7)) for s in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)])


if __name__ == "__main__":
    main(sys.argv[1:])
