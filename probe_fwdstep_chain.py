"""Mechanical probe (NO RL): why the recovery step does not chain.

Reuses biped_multistep_env's proven push -> trigger -> shift to reach swing onset,
then drives ONE forward-rolling step (capture-point-scaled hip flexion, monotonic
no-retract swing) and hands off to Stage-1's exact `_auto_finish` (transfer 150 +
settle 350).  Instruments the CoM velocity and torso pitch through it.

Finding (2026-09-02):
  * The forward step DOES arrest forward momentum: v_fwd ~ 0 at plant.
  * But it ends with the TORSO PITCHED FORWARD ~19 deg (135 N) .. ~30 deg (145 N)
    -- the whip-retract's retract is what rights the torso; a forward step lacks it.
  * From that pitched staggered / single-support state, the feet-together
    StandingLQR settle cannot right the torso (wrong linearisation + only
    +-2.3 N.m ankles) and instead converts the pitch into forward translation:
    v_fwd  +9 -> +185 -> +250 -> +391 mm/s, going BALLISTIC (feet leave the
    ground) by ~350 ms.
  * A partial retract (0 .. 0.35) barely changes the plant pitch (19.3 -> 18.0).

=> The scaffold (StandingLQR base + fixed swing ref + swing-residual policy)
   gives a robust ONE-step recovery and cannot be chained.  See
   memory/two-step-milestone.md.

    python probe_fwdstep_chain.py --pushes 132,138,145
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import mujoco

import biped_multistep_env as E
from biped_multistep_env import BipedMultiStepEnv, LEG_CTRL
from recovery_metrics import (
    CHEST_BODY, sample_balance, _foot_normal_force, _foot_xy_z,
)
from biped_recovery_env import TRANSFER_MS, SETTLE_EVAL_MS, _smooth
from step_primitive import FWD_HIP_SIGN as FHS, KNEE_FLEX_SIGN as KFS

G = 9.81
FS_SWING_MS, FS_DESCEND_MS = 150, 95
FS_HIP_BASE, FS_HIP_GAIN, FS_HIP_MAX = 0.18, 3.2, 0.62


def _fwd_targets(phase, sk, base, hf, hip_fwd, retract):
    h0, k0, a0 = base
    if phase == "swing":
        w = min(1.0, sk / FS_SWING_MS)
        s = _smooth(w)
        return (h0 + hf * hip_fwd * s,
                k0 + KFS * (0.05 + 0.35 * np.sin(np.pi * w)),
                a0 + 0.15 * np.sin(np.pi * w))
    s = _smooth(min(1.0, sk / FS_DESCEND_MS))
    return (h0 + hf * hip_fwd * (1.0 - retract * s),
            k0 + KFS * (0.12 + 0.22 * (1.0 - s)),
            a0 + 0.045 * (1.0 - s))


class Probe(BipedMultiStepEnv):
    def one_step_then_stage1_finish(self, seed, retract=0.0):
        self.reset(seed=seed)
        m, d = self.model, self.data
        sw, st, lat = "R", "L", 1.0
        c = LEG_CTRL[sw]
        base = self._ss[sw]

        b0 = sample_balance(m, d)
        h = max(float(b0.com[2]) - 1.0, 0.05)
        xi = -float(b0.com[1]) + (-float(b0.com_vel[1])) * np.sqrt(h / G)
        stf0 = _foot_xy_z(m, d, st)
        excess = xi - (-stf0[1] + 0.044)
        hip_fwd = float(np.clip(FS_HIP_BASE + FS_HIP_GAIN * max(0.0, excess), FS_HIP_BASE, FS_HIP_MAX))

        def snap(tag):
            mujoco.mj_subtreeVel(m, d)
            b = sample_balance(m, d)
            v = d.subtree_linvel[CHEST_BODY]
            print(f"    {tag:14s} up{b.up_tilt_deg:5.1f}  fwdLean{b.fwd_lean_deg:+5.1f}  "
                  f"vFwd{-v[1] * 1000:+7.1f}  spd{b.com_speed_horiz:.3f}  "
                  f"Lc{int(b.l_contact)} Rc{int(b.r_contact)}")

        phase, sk, lifted, streak = "swing", 0, False, 0
        while True:
            if phase == "swing" and sk >= FS_SWING_MS:
                phase, sk = "descend", 0
            u = self._scaffold_ctrl("swing", sk, min(1.0, sk / FS_SWING_MS), np.zeros(3))
            hh, kk, aa = _fwd_targets(phase, sk, base, FHS[sw], hip_fwd, retract)
            u[c["sw_hip"]], u[c["sw_knee"]], u[c["sw_ankle"]] = hh, kk, aa
            d.ctrl[:15] = np.clip(u, self.ctrl_low, self.ctrl_high)
            mujoco.mj_step(m, d)
            sk += 1
            b = sample_balance(m, d)
            nf = _foot_normal_force(m, d, sw)
            if phase == "swing" and nf < 3.0:
                lifted = True
            if phase == "descend":
                streak = streak + 1 if (lifted and b.r_contact and nf > 12.0) else 0
                if streak >= 10 or sk >= 260:
                    break
        plant_q = np.array([d.qpos[7 + c["sw_hip"]], d.qpos[7 + c["sw_knee"]], d.qpos[7 + c["sw_ankle"]]])
        snap("plant")

        # Stage-1's exact finish
        self._swing, self._stance, self._lat, self._step = sw, st, lat, 1
        self._plant_q = plant_q
        self._td = {1: {"planted": True, "sep_mm": 30.0}}
        self._phase, self._sk = "transfer", 0
        for _ in range(TRANSFER_MS):
            d.ctrl[:15] = self._scaffold_ctrl("transfer", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
        snap("transfer_end")
        self._phase, self._sk = "settle", 0
        for i in range(SETTLE_EVAL_MS):
            d.ctrl[:15] = self._scaffold_ctrl("settle", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            if i in (149, 349):
                snap(f"settle{i + 1}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pushes", default="132,138,145")
    ap.add_argument("--retract", type=float, default=0.0)
    a = ap.parse_args(argv)
    for pn in (float(x) for x in a.pushes.split(",")):
        print(f"  push {pn:.0f} N  (forward step, retract {a.retract}):")
        Probe(push_band=(pn, pn)).one_step_then_stage1_finish(1002, retract=a.retract)
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
