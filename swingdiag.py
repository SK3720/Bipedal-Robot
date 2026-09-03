"""Swing-phase diagnostic: is the recovery a genuine step or a shuffle?

For every recovery step, sampled each physics step through swing+descend:
  * swing-foot height above its lift point  z(t)  -> peak clearance, time above 10 mm
  * swing-foot normal force  nf(t)          -> time genuinely unloaded (<2 N), does it reach ~0
  * swing-foot horizontal speed  |vxy|(t)   -> is it sliding while "swinging"
  * forward foot displacement                -> reach
  * touchdown time (nf first > 12 N)         -> when it plants
  * slip in the 30 ms before / after touchdown
  * step 1 vs step 2

    python swingdiag.py --policy runs/walk_w5 --n 20 --band 126,144
    python swingdiag.py --n 20 --band 126,144            # zero-action reference
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import mujoco

from biped_walk_env import BipedWalkEnv, LEG, _foot_xy_z, _load_policy
from recovery_metrics import sample_balance, _foot_normal_force, CHEST_BODY


class SD(BipedWalkEnv):
    def reset(self, **kw):
        self._steps = []
        self._cur = None
        return super().reset(**kw)

    def _begin_step(self):
        super()._begin_step()
        m, d = self.model, self.data
        sf0 = _foot_xy_z(m, d, self._swing).copy()
        bid = self._b_rf if self._swing == "R" else self._b_lf
        self._cur = dict(k=self._step_k, swing=self._swing, x0=float(sf0[0]),
                         y0=float(sf0[1]), z0=float(sf0[2]), bid=bid,
                         z=[], nf=[], vxy=[], phase=[])

    def step(self, a):
        out = super().step(a)
        c = self._cur
        if c is not None and self._phase in ("swing", "descend"):
            m, d = self.model, self.data
            sf = _foot_xy_z(m, d, c["swing"])
            c["z"].append((float(sf[2]) - c["z0"]) * 1000.0)
            c["nf"].append(_foot_normal_force(m, d, c["swing"]))
            c["vxy"].append(float(np.hypot(d.cvel[c["bid"]][3], d.cvel[c["bid"]][4])) * 1000.0)
            c["phase"].append(self._phase)
        if c is not None and c["k"] in self._td and not any(s["k"] == c["k"] for s in self._steps):
            m, d = self.model, self.data
            sf = _foot_xy_z(m, d, c["swing"])
            z = np.array(c["z"]); nf = np.array(c["nf"]); vxy = np.array(c["vxy"])
            n = len(z)
            # touchdown = first index nf>12 AFTER the foot has genuinely lifted
            lifted_by = next((i for i in range(n) if nf[i] < 2.0), n)
            td = next((i for i in range(lifted_by, n) if nf[i] > 12.0
                       and (i + 6 >= n or np.mean(nf[i:i + 6]) > 10.0)), n - 1)
            pre = slice(max(0, td - 30), td)
            post = slice(td, min(n, td + 30))
            c.update(
                dur_ms=n,
                peak_clear_mm=float(z.max()),
                clear_at_mid_mm=float(z[n // 2]) if n else 0.0,
                frac_above_10mm=float(np.mean(z > 10.0)),
                frac_above_20mm=float(np.mean(z > 20.0)),
                unloaded_ms=int(np.sum(nf < 2.0)),
                min_nf=float(nf.min()),
                frac_unloaded=float(np.mean(nf < 2.0)),
                # "shuffle" = low, loaded, sliding
                frac_dragging=float(np.mean((z < 8.0) & (nf > 4.0) & (vxy > 40.0))),
                td_ms=int(td),
                slip_pre_mm_s=float(np.median(vxy[pre])) if td > 0 else 0.0,
                slip_post_mm_s=float(np.median(vxy[post])) if td < n else 0.0,
                fwd_mm=float(-(sf[1] - c["y0"]) * 1000.0),
                net_lat_mm=float((sf[0] - c["x0"]) * 1000.0),
                zprofile=z,
            )
            self._steps.append(c)
            self._cur = None
        return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--band", default="126,144")
    ap.add_argument("--seed0", type=int, default=3000)
    a = ap.parse_args(argv)
    band = tuple(float(x) for x in a.band.split(","))
    pol = _load_policy(a.policy, band) if a.policy else None
    env = SD(push_band=band)

    rows = []
    succ = fell = flat = 0
    ncnt = []
    for i in range(a.n):
        obs, _ = env.reset(seed=a.seed0 + i)
        done = False
        info = {}
        while not done:
            if pol:
                import torch
                with torch.no_grad():
                    act, _ = pol[0].predict(pol[1](obs), deterministic=True)
            else:
                act = np.zeros(10, np.float32)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        succ += int(info.get("success", False))
        fell += int(info.get("fell", False))
        flat += int(info.get("flat_ok", False))
        ncnt.append(info.get("n_steps", 0))
        rows.extend(env._steps)

    tag = f"POLICY {a.policy}" if a.policy else "ZERO-ACTION"
    print(f"\n=== {tag}  band {band}  n={a.n} ===")
    print(f"outcomes: success {succ}/{a.n}  flat {flat}  fell {fell}   "
          f"step-count {np.bincount(ncnt, minlength=6)[1:].tolist()}")

    def st(name, vals, fmt="{:+7.1f}"):
        v = np.array(vals, float)
        if not len(v):
            return
        print(f"  {name:26s}  med " + fmt.format(np.median(v)) + "   p10 " + fmt.format(np.percentile(v, 10))
              + "   p90 " + fmt.format(np.percentile(v, 90)))

    for k in (1, 2, 3):
        sub = [r for r in rows if r["k"] == k]
        if not sub:
            continue
        print(f"\n-- STEP {k}  ({len(sub)} steps) --")
        st("swing duration ms", [r["dur_ms"] for r in sub])
        st("PEAK clearance mm", [r["peak_clear_mm"] for r in sub])
        st("clearance at mid-swing mm", [r["clear_at_mid_mm"] for r in sub])
        st("frac of swing z>10mm", [100 * r["frac_above_10mm"] for r in sub])
        st("frac of swing z>20mm", [100 * r["frac_above_20mm"] for r in sub])
        st("unloaded ms (nf<2N)", [r["unloaded_ms"] for r in sub])
        st("min swing-foot nf (N)", [r["min_nf"] for r in sub])
        st("frac swing DRAGGING", [100 * r["frac_dragging"] for r in sub])
        st("touchdown at ms", [r["td_ms"] for r in sub])
        st("forward displacement mm", [r["fwd_mm"] for r in sub])
        st("net lateral mm", [r["net_lat_mm"] for r in sub])
        st("slip 30ms PRE-td (mm/s)", [r["slip_pre_mm_s"] for r in sub])
        st("slip 30ms POST-td (mm/s)", [r["slip_post_mm_s"] for r in sub])
        # a compact z-profile (decimated) for the first example
        z = sub[0]["zprofile"]
        dec = z[:: max(1, len(z) // 20)]
        print("     z(t) mm: " + " ".join(f"{v:3.0f}" for v in dec))


if __name__ == "__main__":
    main(sys.argv[1:])
