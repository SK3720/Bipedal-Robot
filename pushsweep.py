"""Push-magnitude sweep: how does the recovery scale with disturbance size?

For a grid of fixed straight-forward pushes, run the policy (or zero-action
reference) and report, per magnitude:
  * outcome (success / fell / timeout)
  * number of recovery steps taken
  * per-step: separation past the (previous) stance foot, forward foot travel,
    peak swing clearance, drag ms, and the CoM state at step onset
  * torso forward travel over the whole episode
  * peak forward CoM speed reached
  * end CoM speed, end capture-excess

    python pushsweep.py --policy runs/walk_w11 --lo 120 --hi 210 --step 10 --reps 4
    python pushsweep.py --lo 120 --hi 200 --step 10 --reps 3          # zero-action
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from biped_walk_env import BipedWalkEnv, _load_policy
from recovery_metrics import CHEST_BODY, sample_balance


class SweepEnv(BipedWalkEnv):
    def reset(self, **kw):
        self._log = dict(steps=[], y0=None, y_min=None, vfwd_peak=0.0)
        out = super().reset(**kw)
        self._log["y0"] = float(self.data.xpos[CHEST_BODY][1])
        self._log["y_min"] = self._log["y0"]
        self._seen = set()
        return out

    def step(self, a):
        out = super().step(a)
        d = self.data
        self._log["y_min"] = min(self._log["y_min"], float(d.xpos[CHEST_BODY][1]))
        bs = self._balance()
        self._log["vfwd_peak"] = max(self._log["vfwd_peak"], -float(bs.com_vel[1]))
        for k, td in self._td.items():
            if k not in self._seen:
                self._seen.add(k)
                dfwd, dlat, vf = self._capture()
                self._log["steps"].append(dict(
                    k=k, sep_mm=td["sep_mm"], fwd_mm=td["fwd_mm"],
                    peak_z_mm=td["peak_z_mm"], drag_ms=td["drag_ms"],
                    net_lat_mm=td["net_lat_mm"], planted=td["planted"]))
        return out


def run(env, pol, push, seed):
    opts = {"push_n": float(push), "push_dir_rad": -np.pi / 2}
    obs, _ = env.reset(seed=seed, options=opts)
    done = False
    info = {}
    while not done:
        if pol:
            import torch
            with torch.no_grad():
                a, _ = pol[0].predict(pol[1](obs), deterministic=True)
        else:
            a = np.zeros(10, np.float32)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    lg = env._log
    travel = (lg["y0"] - lg["y_min"]) * 1000.0
    return dict(
        push=push,
        outcome=("SUCC" if info.get("success") else
                 ("FELL" if info.get("fell") else
                  ("t/o" if info.get("timed_out") else info.get("done_reason", "?")))),
        flat=bool(info.get("flat_ok")),
        n=int(info.get("n_steps", 0)),
        steps=lg["steps"],
        travel_mm=travel,
        vfwd_peak=lg["vfwd_peak"],
        end_spd=float(info.get("end_spd", 9.9)),
        end_capt=float(info.get("end_capt_fwd_mm", 999)),
        peak_up=float(info.get("peak_up", 99)),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--lo", type=float, default=120)
    ap.add_argument("--hi", type=float, default=210)
    ap.add_argument("--step", type=float, default=10)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--band", default="126,144")
    ap.add_argument("--seed0", type=int, default=5000)
    a = ap.parse_args(argv)

    band = tuple(float(x) for x in a.band.split(","))
    pol = _load_policy(a.policy, band) if a.policy else None
    env = SweepEnv(push_band=band, max_steps=8)  # allow more steps so we can SEE the ceiling

    mags = np.arange(a.lo, a.hi + 0.1, a.step)
    tag = f"POLICY {a.policy}" if a.policy else "ZERO-ACTION reference"
    print(f"\n=== push sweep: {tag}   (MAX_STEPS raised to 8 for this probe) ===")
    print(f"{'push':>5} {'succ/reps':>10} {'steps(median)':>14} {'per-step sep mm':>32} "
          f"{'travel mm':>10} {'vpk m/s':>8} {'endSpd':>7}")
    for mag in mags:
        rs = [run(env, pol, mag, a.seed0 + int(mag) * 7 + i) for i in range(a.reps)]
        nsucc = sum(r["outcome"] == "SUCC" for r in rs)
        nfell = sum(r["outcome"] == "FELL" for r in rs)
        nsteps = [r["n"] for r in rs]
        seps = [[f"{s['sep_mm']:+.0f}" for s in r["steps"]] for r in rs]
        sep_str = " | ".join(",".join(s) for s in seps)[:32]
        trav = np.median([r["travel_mm"] for r in rs])
        vpk = np.median([r["vfwd_peak"] for r in rs])
        esp = np.median([r["end_spd"] for r in rs])
        flag = "  <-- FALLS" if nfell > a.reps // 2 else ""
        print(f"{mag:5.0f} {nsucc:>4}/{a.reps} f{nfell:<3} {str(nsteps):>14} {sep_str:>32} "
              f"{trav:10.0f} {vpk:8.2f} {esp:7.2f}{flag}")

    # detailed dump of a few magnitudes
    print("\n--- detail (1 rep each, selected magnitudes) ---")
    for mag in mags:
        if mag % 20 != 0 and mag != mags[-1]:
            continue
        r = run(env, pol, mag, a.seed0 + int(mag) * 7)
        print(f"\npush {mag:.0f} N  -> {r['outcome']}{' FLAT' if r['flat'] else ''}  "
              f"{r['n']} steps, travel {r['travel_mm']:.0f}mm, vpk {r['vfwd_peak']:.2f}, "
              f"endSpd {r['end_spd']:.2f}, endCapt {r['end_capt']:.0f}mm, peakUp {r['peak_up']:.0f}")
        for s in r["steps"]:
            print(f"    step {s['k']} ({'plant' if s['planted'] else 'NOplant'}): "
                  f"sep {s['sep_mm']:+.0f}mm  fwd {s['fwd_mm']:+.0f}mm  "
                  f"peakZ {s['peak_z_mm']:.0f}mm  drag {s['drag_ms']}ms  netLat {s['net_lat_mm']:+.0f}mm")
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
