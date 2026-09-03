"""Step-geometry diagnostic for biped_walk_env.

For each recovery step: swing-foot forward vs lateral excursion (lift -> plant),
PEAK lateral excursion mid-swing (the "wide swing"), clearance, duration, and the
CoM state at step onset (to judge whether the step was as small as necessary and
whether stepping continued after the robot was already caught).

    python stepdiag.py --policy runs/walk_w3 --n 30 --band 124,146
    python stepdiag.py --n 30 --band 124,146            # zero-action reference
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import mujoco

import biped_walk_env as W
from biped_walk_env import BipedWalkEnv, LEG, _foot_xy_z
from recovery_metrics import sample_balance, CHEST_BODY


class DiagEnv(BipedWalkEnv):
    def reset(self, **kw):
        self._diag = []
        self._cur = None
        return super().reset(**kw)

    def _begin_step(self):
        m, d = self.model, self.data
        mujoco.mj_subtreeVel(m, d)
        b = sample_balance(m, d)
        v = d.subtree_linvel[CHEST_BODY]
        stf = _foot_xy_z(m, d, self._stance)
        # capture-point fwd excess past the stance toe (m) at onset
        h = max(float(b.com[2]) - 1.0, 0.05)
        xi = -float(b.com[1]) + (-float(b.com_vel[1])) * np.sqrt(h / 9.81)
        excess = (xi - (-stf[1] + 0.044)) * 1000.0
        super()._begin_step()
        sf0 = _foot_xy_z(m, d, self._swing).copy()
        self._cur = dict(k=self._step_k, swing=self._swing,
                         v_fwd=-float(v[1]), v_lat=float(v[0]),
                         com_speed=float(b.com_speed_horiz),
                         capt_excess_mm=excess, lean=float(b.fwd_lean_deg),
                         x0=float(sf0[0]), y0=float(sf0[1]), z0=float(sf0[2]),
                         peak_lat=0.0, peak_clear=0.0, traj=[])

    def step(self, a):
        pre_phase = self._phase
        out = super().step(a)
        if self._cur is not None and self._phase in ("swing", "descend"):
            sf = _foot_xy_z(self.model, self.data, self._cur["swing"])
            self._cur["peak_lat"] = max(self._cur["peak_lat"],
                                        abs(float(sf[0]) - self._cur["x0"]))
            self._cur["peak_clear"] = max(self._cur["peak_clear"],
                                          float(sf[2]) - self._cur["z0"])
            self._cur["traj"].append((float(sf[0]) - self._cur["x0"],
                                      -(float(sf[1]) - self._cur["y0"])))
        # detect a plant just happened (step k now in self._td, not yet logged)
        if self._cur is not None and self._cur["k"] in self._td and \
                not any(dd["k"] == self._cur["k"] for dd in self._diag):
            sf = _foot_xy_z(self.model, self.data, self._cur["swing"])
            td = self._td[self._cur["k"]]
            self._cur.update(x1=float(sf[0]), y1=float(sf[1]),
                             fwd_mm=-(float(sf[1]) - self._cur["y0"]) * 1000.0,
                             net_lat_mm=(float(sf[0]) - self._cur["x0"]) * 1000.0,
                             peak_lat_mm=self._cur["peak_lat"] * 1000.0,
                             clear_mm=self._cur["peak_clear"] * 1000.0,
                             sep_mm=td["sep_mm"], planted=td["planted"])
            self._diag.append(self._cur)
            self._cur = None
        return out


def load_policy(run_dir, band):
    import json
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import os as _os
    hp = json.load(open(f"{run_dir}/hparams.json"))
    tmp = DummyVecEnv([lambda: DiagEnv(push_band=band)])
    _pf = f"{run_dir}/policy_best.pth" if _os.path.exists(f"{run_dir}/policy_best.pth") else f"{run_dir}/policy.pth"
    _vf = f"{run_dir}/vecnormalize_best.pkl" if _os.path.exists(f"{run_dir}/vecnormalize_best.pkl") else f"{run_dir}/vecnormalize.pkl"
    vn = VecNormalize.load(_vf, tmp)
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    mm = PPO("MlpPolicy", tmp, device="cpu",
             policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(_pf, map_location="cpu", weights_only=True))
    mm.policy.eval()
    return mm, lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--band", default="124,146")
    ap.add_argument("--seed0", type=int, default=4000)
    a = ap.parse_args(argv)
    band = tuple(float(x) for x in a.band.split(","))
    pol = load_policy(a.policy, band) if a.policy else None
    env = DiagEnv(push_band=band)

    rows = []
    outcomes = {"success": 0, "flat": 0, "fell": 0, "timeout": 0}
    nstep_hist = []
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
        outcomes["success"] += int(info.get("success", False))
        outcomes["flat"] += int(info.get("flat_ok", False))
        outcomes["fell"] += int(info.get("fell", False))
        outcomes["timeout"] += int(info.get("timed_out", False))
        nstep_hist.append(info.get("n_steps", 0))
        for dd in env._diag:
            dd["push_n"] = info.get("push_n", 0)
            dd["ep_success"] = info.get("success", False)
            rows.append(dd)

    tag = f"POLICY {a.policy}" if a.policy else "ZERO-ACTION"
    print(f"\n=== {tag}  band {band}  n={a.n} ===")
    print(f"outcomes: success {outcomes['success']}/{a.n}  flat {outcomes['flat']}  "
          f"fell {outcomes['fell']}  timeout {outcomes['timeout']}")
    print(f"step-count dist (1..5): {np.bincount(nstep_hist, minlength=6)[1:].tolist()}")

    def stats(name, vals):
        vals = np.array(vals, float)
        if len(vals) == 0:
            return
        print(f"  {name:22s} n={len(vals):3d}  med {np.median(vals):+7.1f}  "
              f"mean {vals.mean():+7.1f}  p10 {np.percentile(vals, 10):+7.1f}  p90 {np.percentile(vals, 90):+7.1f}")

    for k in (1, 2, 3, 4):
        sub = [d for d in rows if d["k"] == k]
        if not sub:
            continue
        print(f"\n-- STEP {k}  ({len(sub)} steps, {sum(d['swing']=='L' for d in sub)} L / "
              f"{sum(d['swing']=='R' for d in sub)} R) --")
        stats("fwd excursion mm", [d["fwd_mm"] for d in sub])
        stats("net lateral mm", [d["net_lat_mm"] for d in sub])
        stats("PEAK lateral mm", [d["peak_lat_mm"] for d in sub])
        stats("lat/fwd ratio %", [100 * abs(d["peak_lat_mm"]) / max(abs(d["fwd_mm"]), 1) for d in sub])
        stats("clearance mm", [d["clear_mm"] for d in sub])
        stats("sep past stance mm", [d["sep_mm"] for d in sub])
        stats("-- onset v_fwd mm/s", [d["v_fwd"] * 1000 for d in sub])
        stats("-- onset capt excess mm", [d["capt_excess_mm"] for d in sub])
        stats("-- onset com_speed mm/s", [d["com_speed"] * 1000 for d in sub])

    # gratuitous stepping: steps taken while already near-caught
    grat = [d for d in rows if d["k"] >= 2 and d["v_fwd"] < 0.10
            and d["com_speed"] < 0.15 and d["capt_excess_mm"] < 0]
    print(f"\ngratuitous steps (k>=2, onset v_fwd<0.10 & com_speed<0.15 & capt behind toe): "
          f"{len(grat)} / {len([d for d in rows if d['k']>=2])} k>=2 steps")


if __name__ == "__main__":
    main(sys.argv[1:])
