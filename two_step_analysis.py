"""Read-only analysis for the Push -> Step 1 -> Step 2 -> stable design.

Runs the CURRENT Stage-1 policy (runs/recovery_s1/policy.pth) and instruments
the state the robot actually reaches:
  * AT touchdown of step 1
  * through the transfer phase
  * ~600 ms into the settle (would-be step-2 window)

Measures: CoM pos/vel (fwd + lat), torso pitch/roll + rates, per-foot load,
swing/stance foot positions, forward capture point vs. each foot's toe, whole-body
angular momentum, and which foot is lighter (step-2 swing candidate).

Does NOT modify biped_recovery_env.py or touch the running training job.
    python two_step_analysis.py --n 50
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import mujoco

import biped_recovery_env as E
from biped_recovery_env import BipedRecoveryEnv, TRANSFER_MS
from recovery_metrics import CHEST_BODY, sample_balance, _foot_normal_force, _foot_xy_z
from wbtraj_opt import _limb_L_about_G, BODY

G = 9.81


def load_policy(run_dir):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    tmp = DummyVecEnv([lambda: BipedRecoveryEnv(push_band=(130., 140.))])
    model = PPO("MlpPolicy", tmp, device="cpu",
                policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.6))
    model.policy.load_state_dict(torch.load(f"{run_dir}/policy.pth", map_location="cpu",
                                            weights_only=True))
    model.policy.eval()
    vn = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", tmp)
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    return model, (lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32))


def _snap(m, d):
    bs = sample_balance(m, d)
    mujoco.mj_subtreeVel(m, d)
    comG = d.subtree_com[CHEST_BODY].copy()
    vG = d.subtree_linvel[CHEST_BODY].copy()
    Lwb = d.subtree_angmom[CHEST_BODY].copy()
    lf = _foot_xy_z(m, d, "L")
    rf = _foot_xy_z(m, d, "R")
    lnf = _foot_normal_force(m, d, "L")
    rnf = _foot_normal_force(m, d, "R")
    # forward capture point (world -Y is forward)
    h = max(float(bs.com[2]) - 1.0, 0.05)
    tc = np.sqrt(h / G)
    xi_fwd = -float(bs.com[1]) - float(bs.com_vel[1]) * tc
    xi_lat = float(bs.com[0]) + float(bs.com_vel[0]) * tc
    return dict(
        com_fwd=-float(bs.com[1]), com_lat=float(bs.com[0]),
        v_fwd=-float(bs.com_vel[1]), v_lat=float(bs.com_vel[0]),
        pitch=float(bs.fwd_lean_deg), roll=float(bs.side_lean_deg),
        pitch_rate=float(bs.pitch_rate),
        up_tilt=float(bs.up_tilt_deg),
        lnf=float(lnf), rnf=float(rnf),
        L_foot_fwd=-float(lf[1]), R_foot_fwd=-float(rf[1]),
        L_foot_lat=float(lf[0]), R_foot_lat=float(rf[0]),
        xi_fwd=xi_fwd, xi_lat=xi_lat,
        # xi ahead of each foot's toe (foot half-fwd ~0.044)
        xi_past_L_toe=(xi_fwd - (-float(lf[1]) + 0.044)) * 1000.0,
        xi_past_R_toe=(xi_fwd - (-float(rf[1]) + 0.044)) * 1000.0,
        Lwb_pitch=float(Lwb[0]),
        chest_z=float(bs.chest_z),
    )


class ProbeEnv(BipedRecoveryEnv):
    def _auto_finish(self):
        m, d = self.model, self.data
        self._probe = dict(td=_snap(m, d), transfer=[], settle=[])
        for i in range(TRANSFER_MS):
            u = self._scaffold_ctrl("transfer", self._sk, 0.0, np.zeros(3))
            d.ctrl[:15] = u
            mujoco.mj_step(m, d)
            self._sk += 1
            if i % 25 == 0:
                self._probe["transfer"].append((i, _snap(m, d)))
            if self._fallen():
                self._peak_up = max(self._peak_up, 55.0)
                return -20.0, self._info(fell=True)
        self._probe["transfer_end"] = _snap(m, d)
        self._phase = "settle"
        self._sk = 0
        for i in range(700):                       # probe longer than SETTLE_EVAL_MS
            u = self._scaffold_ctrl("settle", self._sk, 0.0, np.zeros(3))
            d.ctrl[:15] = u
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if i % 50 == 0:
                self._probe["settle"].append((i, _snap(m, d)))
            if self._fallen():
                return -20.0, self._info(fell=True)
        b = sample_balance(m, d)
        ds = b.l_contact and b.r_contact
        stable = (ds and b.up_tilt_deg < 8.0 and abs(b.side_lean_deg) < 8.0
                  and b.com_speed_horiz < 0.10 and b.chest_z > 1.26 - 0.10
                  and self._peak_up <= 35.0)
        success = stable and self._touchdown is not None and self._touchdown["planted"] \
            and self._touchdown["sep_mm"] >= 15.0
        info = self._info(fell=False)
        info["success"] = bool(success)
        info["end_sep_mm"] = float(self._touchdown["sep_mm"]) if self._touchdown else 0.0
        return (8.0 if success else 0.0), info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/recovery_s1")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--band", default="124,143")
    ap.add_argument("--spread", type=float, default=12.0)
    a = ap.parse_args(argv)
    band = tuple(float(x) for x in a.band.split(","))

    model, norm = load_policy(a.run)
    env = ProbeEnv(push_band=band, dir_spread_deg=a.spread)
    print(f"loaded {a.run}/policy.pth  band {band} +-{a.spread}deg  n={a.n}\n")

    succ_td, succ_tr, succ_st = [], [], []
    for i in range(a.n):
        o, _ = env.reset(seed=61000 + i)
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                act, _ = model.predict(norm(o), deterministic=True)
            o, r, term, trunc, info = env.step(act)
            done = term or trunc
        if not info.get("success"):
            continue
        P = env._probe
        succ_td.append(P["td"])
        succ_tr.append(P["transfer_end"])
        # settle snapshot nearest 300 ms (step-2 window)
        s300 = min(P["settle"], key=lambda kv: abs(kv[0] - 300))[1]
        succ_st.append(s300)

    def agg(rows, keys):
        return {k: (round(float(np.median([r[k] for r in rows])), 3),
                    round(float(np.percentile([r[k] for r in rows], 10)), 3),
                    round(float(np.percentile([r[k] for r in rows], 90)), 3)) for k in keys}

    KEYS = ["v_fwd", "v_lat", "com_fwd", "com_lat", "pitch", "roll", "pitch_rate",
            "up_tilt", "lnf", "rnf", "L_foot_fwd", "R_foot_fwd",
            "xi_past_L_toe", "xi_past_R_toe", "Lwb_pitch", "chest_z"]
    print(f"successful step-1 episodes analysed: {len(succ_td)}/{a.n}")
    for label, rows in (("AT TOUCHDOWN", succ_td),
                        (f"END of TRANSFER (+{TRANSFER_MS} ms)", succ_tr),
                        ("SETTLE +300 ms (step-2 window)", succ_st)):
        if not rows:
            continue
        A = agg(rows, KEYS)
        print(f"\n--- {label} ---  (median [p10, p90])")
        for k in KEYS:
            m_, lo_, hi_ = A[k]
            print(f"  {k:16s} {m_:+8.3f}   [{lo_:+.3f}, {hi_:+.3f}]")


if __name__ == "__main__":
    main(sys.argv[1:])
