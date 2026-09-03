"""MECHANICAL PROBE (no RL): can a capture-point foot-placement stepper take
genuinely larger, state-dependent sequential recovery steps on this robot?

The walk-env base is a *whip-retract* step: fixed CMA knots, amplitude clipped to
1.15x, swing truncated at 150 ms, and the descend phase actively RETRACTS the
foot.  Result (see pushsweep.py): step-1 separation saturates at ~58 mm for every
push from 130 N to 210 N -- the robot physically cannot commit to a bigger step.

This probe replaces that base with a classic reactive stepper:
  * hold with StandingLQR until the sagittal capture point leaves the support
    polygon,
  * then swing the TRAILING foot forward to  x_place = capture_point + margin
    (step length grows with the capture-point excess -- bigger disturbance ->
    bigger step),
  * a real forward arc with a mid-swing clearance bump (step_primitive's
    _swing_leg_targets, NOT the whip),
  * plant, transfer, re-evaluate; keep stepping until caught; no fixed count.

Frontal-plane balance (ankle-roll CoP + lateral hip-roll) is reused verbatim from
the env so we isolate the sagittal stepping change.

    python probe_cp_stepper.py --lo 120 --hi 240 --step 10 --reps 3
    python probe_cp_stepper.py --push 180 --render probe180.mp4
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np
import mujoco

from biped_walk_env import BipedWalkEnv, LEG, _X_MID_M, HR_KX, HR_KV, HR_MAX
from recovery_metrics import (
    CHEST_BODY, FLOOR_Z, G, NOMINAL_CHEST_Z, sample_balance,
    _foot_normal_force, _foot_xy_z,
)
from step_primitive import _swing_leg_targets, StepCfg, _sole_pitch

AR = {"L": LEG["L"]["ar"], "R": LEG["R"]["ar"]}
FOOT_HALF = 0.044          # foot half-length (m) fwd/back of ankle


class CPStepper(BipedWalkEnv):
    # --- stepper config (the knobs this probe explores) ---
    TRIG_MARGIN_M = 0.000      # step once xi is this far past the front support edge
    TRIG_VFWD = 0.13           # ...or forward CoM speed exceeds this (whichever first)
    MIN_DWELL_MS = 25
    PLACE_BASE_M = 0.070       # nominal foot placement past the *current* stance foot
    PLACE_GAIN = 1.10         # + this * (capture excess) -> longer steps for bigger pushes
    PLACE_MARGIN_M = 0.045    # place the foot this far PAST the predicted capture point
    PLACE_MAX_M = 0.26
    SWING_MS = 150
    DESC_MS = 150
    XFER_MS = 80
    HOLD_CAUGHT_MS = 120
    CAUGHT_XI_M = 0.015
    CAUGHT_SPD = 0.16
    N_MAX = 8
    KNEE_RAD = 0.34           # mid-swing clearance bump

    def _hip_for_place(self, place_m):
        """Map a desired foot forward-placement (past stance) to peak hip flexion.
        Calibrated empirically (see sweep output): ~0.35 rad per 100 mm + offset."""
        return float(np.clip(0.28 + 3.4 * place_m, 0.30, 0.92))

    # ---- one scripted episode on the already-stood-and-pushed sim ----
    def rollout(self, push_n, seed, frames=None):
        m, d = self.model, self.data
        opts = {"push_n": float(push_n), "push_dir_rad": -np.pi / 2}
        self.reset(seed=seed, options=opts)

        phase, swing, stance, step_k, sk, dwell = "hold", "R", "L", 0, 0, 0
        caught_streak = 0
        an_prev = {"L": 0.0, "R": 0.0}
        y0_torso = float(d.xpos[CHEST_BODY][1])
        y_min = y0_torso
        vpk = 0.0
        steps = []
        cur = None
        hip_pk = 0.0
        result = "t/o"

        for it in range(1400):
            bs = sample_balance(m, d)
            mujoco.mj_subtreeVel(m, d)
            com = bs.com
            comv = bs.com_vel
            h = max(float(com[2]) - FLOOR_Z, 0.05)
            tc = np.sqrt(h / G)
            xi_fwd = -float(com[1]) + (-float(comv[1])) * tc      # capture pt, fwd coord (m)
            lf, rf = _foot_xy_z(m, d, "L"), _foot_xy_z(m, d, "R")
            # front support edge (toe of the most-forward *loaded* foot)
            loaded = []
            if bs.l_contact:
                loaded.append(-lf[1])
            if bs.r_contact:
                loaded.append(-rf[1])
            if not loaded:
                loaded = [-lf[1], -rf[1]]
            front_edge = max(loaded) + FOOT_HALF
            rear_edge = min(loaded) - FOOT_HALF
            xi_excess = xi_fwd - front_edge                       # >0 -> must step
            vfwd = -float(comv[1])
            vpk = max(vpk, vfwd)
            y_min = min(y_min, float(d.xpos[CHEST_BODY][1]))

            # ---------- control ----------
            u = np.array(self._stand.control(m, d), float)

            if phase in ("swing", "descend"):
                # roll-zeroed LQR base (matches env) for the torso, arc on the swing leg
                dq = np.zeros(m.nv)
                mujoco.mj_differentiatePos(m, dq, 1.0, self._stand.qpos0, d.qpos)
                dx = np.concatenate([dq, d.qvel - self._stand.qvel0])
                for ci in (AR["L"], AR["R"], LEG["L"]["hr"], LEG["R"]["hr"]):
                    dx[6 + ci] = 0.0
                    dx[m.nv + 6 + ci] = 0.0
                dx[0] = dx[m.nv + 0] = 0.0
                u = np.array(self._stand.ctrl0 - self._stand.K @ dx, float)

                g = LEG[swing]
                h0, k0, a0 = self._ss[swing]
                seg = "swing" if phase == "swing" else "descend"
                dur = self.SWING_MS if seg == "swing" else self.DESC_MS
                w = (min(1.0, sk / dur) if seg == "swing" else sk / dur)
                cfg = replace(StepCfg(swing=swing), swing_hip_rad=cur["hip_rad"],
                              swing_knee_rad=self.KNEE_RAD, knee_land_frac=0.10,
                              hip_retract_frac=0.10, sole_toe_up_mid=0.12)
                hip, knee, sole_tgt = _swing_leg_targets(cfg, seg, w, (h0, k0))
                u[g["hp"]] = hip
                u[g["kn"]] = knee
                u[g["ap"]] = a0 + sole_tgt * (1.0 if swing == "R" else -1.0)
                hip_pk = max(hip_pk, abs(hip - h0))

            # frontal-plane balance: reuse the env laws verbatim (both ankle rolls
            # carry the CoP bias, loaded hip-rolls carry the lateral strategy)
            swinging = phase in ("swing", "descend")
            from biped_walk_env import X_SS_MM, X_TR_MM
            x_ref = (X_SS_MM if swinging else X_TR_MM) / 1000.0
            lat = 1.0 if swing == "R" else -1.0
            arb = self._frontal_bias(bs, x_ref, lat)
            u[AR["L"]] = arb
            u[AR["R"]] = arb
            x = float(bs.com[0]); vx = float(bs.com_vel[0])
            if lat < 0:
                x = 2.0 * _X_MID_M - x; vx = -vx
            hr = float(np.clip(-(HR_KX * (x - _X_MID_M) + HR_KV * vx), -HR_MAX, HR_MAX))
            if lat < 0:
                hr = -hr
            for sd in ("L", "R"):
                if (sd == "L" and bs.l_contact) or (sd == "R" and bs.r_contact):
                    u[LEG[sd]["hr"]] += hr

            u[0:5] = 0.0
            u = np.clip(u, self.clow, self.chigh)
            d.ctrl[:15] = u
            mujoco.mj_step(m, d)
            self._vsync()
            sk += 1
            dwell += 1
            if frames is not None and it % 5 == 0:
                frames.append(self.render())

            # ---------- fall / caught ----------
            tilt = self._cheap_tilt()
            if tilt > 42.0 or d.qpos[2] < NOMINAL_CHEST_Z - 0.24:
                result = "FELL"
                break

            # ---------- transitions ----------
            if phase == "hold":
                inside = (rear_edge - self.CAUGHT_XI_M) < xi_fwd < (front_edge + self.CAUGHT_XI_M)
                caught = inside and bs.com_speed_horiz < self.CAUGHT_SPD
                caught_streak = caught_streak + 1 if caught else 0
                if (caught_streak >= self.HOLD_CAUGHT_MS and step_k >= 1
                        and bs.l_contact and bs.r_contact):
                    result = "SUCC"
                    break
                if ((xi_excess > self.TRIG_MARGIN_M or vfwd > self.TRIG_VFWD)
                        and step_k < self.N_MAX
                        and dwell >= self.MIN_DWELL_MS and not caught):
                    # begin a step with the TRAILING foot
                    swing = "L" if (-lf[1]) < (-rf[1]) else "R"
                    stance = "R" if swing == "L" else "L"
                    place = float(np.clip(
                        self.PLACE_BASE_M + self.PLACE_GAIN * max(0.0, xi_excess)
                        + self.PLACE_MARGIN_M, 0.04, self.PLACE_MAX_M))
                    cur = dict(k=step_k + 1, swing=swing, place_tgt_mm=place * 1000.0,
                               hip_rad=self._hip_for_place(place),
                               y0=float(_foot_xy_z(m, d, swing)[1]),
                               z0=float(_foot_xy_z(m, d, swing)[2]),
                               xi_excess_mm=xi_excess * 1000.0, vfwd=vfwd,
                               peak_z=0.0, drag=0)
                    step_k += 1
                    phase, sk, dwell = "swing", 0, 0
                    caught_streak = 0
            elif phase == "swing":
                sf = _foot_xy_z(m, d, swing)
                clr = float(sf[2]) - cur["z0"]
                cur["peak_z"] = max(cur["peak_z"], clr)
                swb = self._b_rf if swing == "R" else self._b_lf
                vxy = float(np.hypot(d.cvel[swb][3], d.cvel[swb][4]))
                if clr < 0.010 and _foot_normal_force(m, d, swing) > 3.0 and vxy > 0.03:
                    cur["drag"] += 1
                if sk >= self.SWING_MS:
                    phase, sk = "descend", 0
            elif phase == "descend":
                sf = _foot_xy_z(m, d, swing)
                cur["peak_z"] = max(cur["peak_z"], float(sf[2]) - cur["z0"])
                nf = _foot_normal_force(m, d, swing)
                swc = bs.r_contact if swing == "R" else bs.l_contact
                if (swc and nf > 12.0) or sk >= 260:
                    stf = _foot_xy_z(m, d, stance)
                    cur["sep_mm"] = float(-(sf[1] - stf[1]) * 1000.0)
                    cur["fwd_mm"] = float(-(sf[1] - cur["y0"]) * 1000.0)
                    cur["net_lat_mm"] = float((sf[0] - _foot_xy_z(m, d, swing)[0]) * 0 + 0.0)
                    cur["sole_pitch"] = float(np.degrees(_sole_pitch(m, d, swing)))
                    steps.append(cur)
                    phase, sk, dwell = "xfer", 0, 0
            elif phase == "xfer":
                if sk >= self.XFER_MS:
                    phase, sk, dwell = "hold", 0, 0

        bs = sample_balance(m, d)
        travel = (y0_torso - y_min) * 1000.0
        return dict(push=push_n, result=result, n=step_k, steps=steps,
                    travel_mm=travel, vpk=vpk, end_spd=float(bs.com_speed_horiz),
                    end_up=float(bs.up_tilt_deg), end_ds=bool(bs.l_contact and bs.r_contact))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=120)
    ap.add_argument("--hi", type=float, default=240)
    ap.add_argument("--step", type=float, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--push", type=float, default=None)
    ap.add_argument("--render", default=None)
    ap.add_argument("--seed0", type=int, default=6000)
    a = ap.parse_args(argv)

    env = CPStepper(push_band=(130.0, 140.0), max_steps=CPStepper.N_MAX)

    if a.push is not None:
        frames = [] if a.render else None
        r = env.rollout(a.push, a.seed0, frames=frames)
        print(f"\npush {a.push:.0f} N -> {r['result']}  {r['n']} steps  "
              f"travel {r['travel_mm']:.0f}mm  vpk {r['vpk']:.2f}  endSpd {r['end_spd']:.2f}  "
              f"endUp {r['end_up']:.0f}  ds {r['end_ds']}")
        for s in r["steps"]:
            print(f"  step {s['k']} {s['swing']}: place_tgt {s['place_tgt_mm']:.0f}mm  "
                  f"xi_excess {s['xi_excess_mm']:+.0f}mm  -> sep {s.get('sep_mm', -999):+.0f}mm  "
                  f"fwd {s.get('fwd_mm', -999):+.0f}mm  peakZ {s['peak_z']*1000:.0f}mm  "
                  f"drag {s['drag']}ms  sole {s.get('sole_pitch', 0):+.0f}deg")
        if a.render and frames:
            import imageio.v2 as imageio
            imageio.mimsave(a.render, frames, fps=40, macro_block_size=1)
            print(f"  wrote {a.render}")
        env.close()
        return

    mags = np.arange(a.lo, a.hi + 0.1, a.step)
    print(f"\n=== capture-point stepper probe (scripted, no RL) ===")
    print(f"{'push':>5} {'succ':>6} {'n steps':>16} {'step seps (mm)':>34} "
          f"{'travel':>7} {'vpk':>6} {'endSpd':>7}")
    for mag in mags:
        rs = [env.rollout(mag, a.seed0 + int(mag) * 5 + i) for i in range(a.reps)]
        ns = sum(r["result"] == "SUCC" for r in rs)
        nf = sum(r["result"] == "FELL" for r in rs)
        nsteps = [r["n"] for r in rs]
        seps = " | ".join(",".join(f"{s.get('sep_mm', 0):+.0f}" for s in r["steps"]) for r in rs)
        trav = np.median([r["travel_mm"] for r in rs])
        vpk = np.median([r["vpk"] for r in rs])
        esp = np.median([r["end_spd"] for r in rs])
        flag = "  <-- FALLS" if nf > a.reps // 2 else ""
        print(f"{mag:5.0f} {ns:>3}/{a.reps} f{nf:<2} {str(nsteps):>16} {seps[:34]:>34} "
              f"{trav:7.0f} {vpk:6.2f} {esp:7.2f}{flag}")
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
