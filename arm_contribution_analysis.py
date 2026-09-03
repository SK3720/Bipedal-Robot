"""Does the arm-waving in the learned recovery policy actually help?

The RL policy only controls the swing leg (R hip/knee/ankle).  The arms follow a
FIXED reference trajectory (the CMA-optimised 2x-hand solution's shoulder/elbow
knots) - a split-shoulder counter-rotation (L back, R forward).  That is the
"arm waving" seen in the eval videos; it is NOT policy-learned.

This runs the CURRENT trained policy (runs/recovery_s1/policy.pth) under 4 arm
conditions - full reference / arms held at sides / half amplitude / double
amplitude - on the same eval seeds, and measures recovery success + torso
excursion + the arm-vs-swing-leg angular momentum (about the global CoM) + arm
torque + whether the commanded arm trajectory is even tracked under +-2.3 N.m.

Does NOT modify biped_recovery_env.py or touch the running training job - it
mutates the module-level REF_KNOTS in *this* process only (training subprocs have
their own copy).

    python arm_contribution_analysis.py
    python arm_contribution_analysis.py --run runs/recovery_s1 --n 40
"""
from __future__ import annotations

import argparse
import copy
import sys

import numpy as np
import torch
import mujoco

import biped_recovery_env as E
from biped_recovery_env import BipedRecoveryEnv
from wbtraj_opt import _limb_L_about_G, BODY
from recovery_metrics import CHEST_BODY, sample_balance

ARM_CI = (1, 2, 3, 4)             # L_shoulder, L_elbow, R_shoulder, R_elbow
SW_HIP_BODY = BODY["r_hip"]       # swing = R
L_ARM_BODY, R_ARM_BODY = BODY["l_arm"], BODY["r_arm"]
_ORIG_ARM_KNOTS = {ci: E.REF_KNOTS[ci].copy() for ci in ARM_CI}


def set_arm_condition(cond):
    if cond == "full":
        for ci in ARM_CI:
            E.REF_KNOTS[ci] = _ORIG_ARM_KNOTS[ci].copy()
    elif cond == "sides":                 # arms commanded to DEFAULT_POSE (0) - no wave
        for ci in ARM_CI:
            E.REF_KNOTS[ci] = np.zeros(4)
    elif cond == "half":
        for ci in ARM_CI:
            E.REF_KNOTS[ci] = 0.5 * _ORIG_ARM_KNOTS[ci]
    elif cond == "double":
        for ci in ARM_CI:
            E.REF_KNOTS[ci] = 2.0 * _ORIG_ARM_KNOTS[ci]
    else:
        raise ValueError(cond)


class InstrumentedEnv(BipedRecoveryEnv):
    """samples angular momentum / arm tracking / torque every physics step of
    swing + descend, without changing any control behaviour."""

    def reset(self, *a, **k):
        self._I = dict(Lsw=[], Larm=[], oppose=0, n=0,
                       tau_arm=[], sh_cmd=[], sh_act=[])
        return super().reset(*a, **k)

    def _scaffold_ctrl(self, phase, sk, w, residual):
        u = super()._scaffold_ctrl(phase, sk, w, residual)
        if phase in ("swing", "descend"):
            m, d = self.model, self.data
            mujoco.mj_subtreeVel(m, d)
            comG = d.subtree_com[CHEST_BODY].copy()
            vG = d.subtree_linvel[CHEST_BODY].copy()
            Lsw = _limb_L_about_G(m, d, SW_HIP_BODY, comG, vG)[0]
            Larm = (_limb_L_about_G(m, d, L_ARM_BODY, comG, vG)
                    + _limb_L_about_G(m, d, R_ARM_BODY, comG, vG))[0]
            self._I["Lsw"].append(Lsw)
            self._I["Larm"].append(Larm)
            if Lsw * Larm < 0:
                self._I["oppose"] += 1
            self._I["n"] += 1
            self._I["tau_arm"].append(float(np.abs(d.actuator_force[[1, 3]]).max()))
            self._I["sh_cmd"].append((float(u[1]), float(u[3])))       # commanded L,R shoulder
            self._I["sh_act"].append((float(d.qpos[7 + 1]), float(d.qpos[7 + 3])))  # actual
        return u


def load_policy(run_dir):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    tmp = DummyVecEnv([lambda: BipedRecoveryEnv(push_band=(130., 140.))])
    model = PPO("MlpPolicy", tmp, device="cpu",
                policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.6))
    sd = torch.load(f"{run_dir}/policy.pth", map_location="cpu", weights_only=True)
    model.policy.load_state_dict(sd)
    model.policy.eval()
    vn = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", tmp)
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    return model, (lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32))


def run_condition(cond, model, norm, n, band, spread, seed0=42000):
    set_arm_condition(cond)
    env = InstrumentedEnv(push_band=band, dir_spread_deg=spread)
    rows = []
    for i in range(n):
        o, _ = env.reset(seed=seed0 + i)
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                a, _ = model.predict(norm(o), deterministic=True)
            o, r, term, trunc, info = env.step(a)
            done = term or trunc
        I = env._I
        Lsw = np.array(I["Lsw"]) if I["Lsw"] else np.array([0.0])
        Larm = np.array(I["Larm"]) if I["Larm"] else np.array([0.0])
        cmd = np.array(I["sh_cmd"]) if I["sh_cmd"] else np.zeros((1, 2))
        act = np.array(I["sh_act"]) if I["sh_act"] else np.zeros((1, 2))
        rows.append(dict(
            success=int(info.get("success", False)),
            sep=info.get("end_sep_mm", 0.0),
            peak_up=info.get("peak_up", 99.0),
            Lsw_peak=float(np.abs(Lsw).max()),
            Larm_peak=float(np.abs(Larm).max()),
            Larm_over_Lsw=float(np.abs(Larm).max() / max(np.abs(Lsw).max(), 1e-6)),
            oppose_frac=I["oppose"] / max(I["n"], 1),
            tau_arm_peak=float(max(I["tau_arm"])) if I["tau_arm"] else 0.0,
            sh_track_err=float(np.abs(cmd - act).mean()),
            sh_cmd_range=float(cmd.max() - cmd.min()),
            sh_act_range=float(act.max() - act.min()),
        ))
    set_arm_condition("full")
    return rows


def summarize(cond, rows):
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return dict(
        cond=cond, n=len(rows),
        success=round(float(a["success"].mean()), 3),
        med_sep=round(float(np.median(a["sep"])), 1),
        med_peak_up=round(float(np.median(a["peak_up"])), 1),
        Lsw_peak=round(float(np.median(a["Lsw_peak"])), 4),
        Larm_peak=round(float(np.median(a["Larm_peak"])), 4),
        Larm_over_Lsw=round(float(np.median(a["Larm_over_Lsw"])), 3),
        oppose_frac=round(float(np.median(a["oppose_frac"])), 2),
        tau_arm_peak=round(float(np.median(a["tau_arm_peak"])), 2),
        sh_track_err=round(float(np.median(a["sh_track_err"])), 3),
        sh_cmd_range=round(float(np.median(a["sh_cmd_range"])), 2),
        sh_act_range=round(float(np.median(a["sh_act_range"])), 2),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/recovery_s1")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--band", default="124,143")
    ap.add_argument("--spread", type=float, default=12.0)
    a = ap.parse_args(argv)
    band = tuple(float(x) for x in a.band.split(","))

    model, norm = load_policy(a.run)
    print(f"loaded {a.run}/policy.pth  |  eval band {band} +-{a.spread}deg  n={a.n}\n")

    outs = []
    for cond in ("full", "sides", "half", "double"):
        rows = run_condition(cond, model, norm, a.n, band, a.spread)
        s = summarize(cond, rows)
        outs.append(s)
        print(f"[{cond:6s}] success {s['success']:.2f}  medSep {s['med_sep']:5.1f}  "
              f"medPeakUp {s['med_peak_up']:4.1f}  |Larm| {s['Larm_peak']:.4f}  "
              f"|Lsw| {s['Lsw_peak']:.4f}  ratio {s['Larm_over_Lsw']:.3f}  "
              f"oppose {s['oppose_frac']:.2f}  tauArm {s['tau_arm_peak']:.2f}  "
              f"shoulderCmd~{s['sh_cmd_range']:.1f} act~{s['sh_act_range']:.1f} trackErr {s['sh_track_err']:.2f}")

    base = next(o for o in outs if o["cond"] == "full")
    sides = next(o for o in outs if o["cond"] == "sides")
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    d_succ = sides["success"] - base["success"]
    d_pitch = sides["med_peak_up"] - base["med_peak_up"]
    print(f"  full arm-wave -> arms-at-sides:  success {d_succ:+.2f}   "
          f"peak torso tilt {d_pitch:+.1f} deg")
    print(f"  arm angular momentum is {base['Larm_over_Lsw']*100:.0f}% of the swing "
          f"leg's; arms oppose the leg {base['oppose_frac']*100:.0f}% of swing steps")
    print(f"  shoulder: commanded ~{base['sh_cmd_range']:.1f} rad, actually moved "
          f"~{base['sh_act_range']:.1f} rad  (tracking error {base['sh_track_err']:.2f} rad)")
    if abs(d_succ) < 0.05 and abs(d_pitch) < 1.5:
        print("  => the arm motion is DECORATIVE - removing it changes nothing measurable.")
    elif d_succ < -0.05 or d_pitch > 1.5:
        print("  => the arm motion HELPS - removing it hurts recovery / raises torso excursion.")
    else:
        print("  => removing the arm motion IMPROVES things - the reference arm-wave is a mild net negative.")


if __name__ == "__main__":
    main(sys.argv[1:])
