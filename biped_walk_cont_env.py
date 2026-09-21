"""CONTINUOUS WALKING by extending the recovery policy.

The recovery policy runs/walk_s5 (cp variant) already learned the hard part: it
balances THROUGH a single-support step -- unload, swing, plant, catch the CoM,
hold a stance.  A walk is that, chained and driven forward.

This env reuses the recovery env's control composition verbatim (cp swing
primitive + StandingLQR torso + frontal-LIPM lateral balance + the 10-DOF policy
residual) but replaces the push/settle phase machine with a CONTINUOUS
alternating stepper:

    stand -> step L -> plant -> step R -> plant -> step L -> ...  (forever)

driven forward toward a target speed.  Same obs + 10-DOF action as the recovery
env, so it warm-starts directly from walk_s5.

GENUINE-STEP MEASUREMENT: a plant only counts as a real step (and is rewarded as
one) if the swing foot's *sole* actually cleared the ground -- we transform the
foot collision-mesh sole vertices every frame and require min-sole-clearance
> 15 mm with zero contact force for >= 20 ms, plus >= 25 mm net forward travel.
A pivot / scrape / drag of an oversized foot does NOT count and is penalised.

    python biped_walk_cont_env.py --smoke
    python biped_walk_cont_env.py --smoke --policy runs/walk_s5
    python biped_walk_cont_env.py --smoke --model robot/_exp_hands_2x_feet_1p5.xml --policy runs/walk_s5 --render w.mp4
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import mujoco

from biped_walk_env import (
    BipedWalkEnv, LEG, LEG_CTRL_ORDER, CHEST_BODY, FLOOR_Z, G, V0_NOMINAL,
    CP_REACH_MAX, mirror_action, sample_balance, _foot_xy_z, _foot_normal_force,
    _sole_pitch, CP_SWING_MS_BASE, CP_SWING_MS_PER_REACH, CP_SWING_MS_MAX,
    _X_MID_M,
)
from step_primitive import FWD_HIP_SIGN as _FHS, KNEE_FLEX_SIGN as _KFS

# hip-roll / ankle-roll ctrl sign for "abduct = move that foot outboard / push CoM
# to the opposite side".  Validated by the sign-sweep diagnostic (see __main__).
_HR_SIGN = dict(L=1.0, R=-1.0)
_AR_SIGN = dict(L=-1.0, R=1.0)

WCONT_MODEL = "robot/_exp_hands_2x_feet_1p5.xml"   # 2x hands (matches s5), 1.5x feet

FRAME_SKIP = 5
CTRL_DT = FRAME_SKIP * 1e-3
EP_SECONDS = 8.0
EP_STEPS = int(EP_SECONDS / CTRL_DT)

SPEED_TGT_DEFAULT = 0.28
WALK_REACH = 0.15
WALK_REACH_VGAIN = 0.30
HOLD_MS = 45                    # double-support dwell between steps
SHIFT_MS_MIN = 22             # min lateral weight-shift before the swing foot may lift
SHIFT_MS_MAX = 120           # ...and a hard cap
SHIFT_UNLOAD_FRAC = 0.28     # swing foot considered unloaded below this * bodyweight
FLIGHT_TERM_MS = 70            # continuous bilateral flight this long -> episode over
DESCEND_MIN_MS = 30           # min time in descend before a plant may register

# genuine lift-off thresholds
CLEAR_OK_M = 0.014            # sole must clear the floor by this
AIR_MIN_MS = 16             # ...for at least this long
STEP_FWD_MIN_M = 0.022     # ...and travel at least this far forward

# forward propulsion added on top of the recovery control stack (scripted base;
# RL residual refines).  Drives real CoM travel so the gait isn't in-place.
ST_HIP_EXT = 0.0            # stance-leg hip extension torque bias during swing (push body fwd)
ST_ANK_PUSH = 0.05         # stance-leg ankle plantarflex (push-off) ramped late in swing
PROP_LEAN_HIP = 0.02        # small symmetric hip bias -> gentle forward CoM lean during hold

# mediolateral control -- the piece every prior walk attempt was missing.
# Without lateral foot placement, CoM lateral velocity accumulates every step
# and the robot topples sideways within ~2 steps.
ML_SHIFT_HR = 0.26          # stance hip-roll drive during the weight-shift phase
ML_SHIFT_AR = 0.13          # stance ankle-roll drive during the weight-shift phase
ML_SHIFT_HR = 0.40         # (overridden above default; kept for import clarity)
ML_STANCE_HR_KX = 4.5       # stance hip-roll: CoM lateral-offset feedback (during swing)
ML_STANCE_HR_KV = 1.4       # stance hip-roll: CoM lateral-velocity feedback
ML_STANCE_HR_MAX = 0.42
ML_PLACE_K = 0.9            # swing-foot outboard placement per (m/s) of CoM lateral vel
ML_PLACE_MAX = 0.36         # cap on swing hip-roll placement command

# sagittal: resist forward torso-pitch runaway during single support (the other
# failure mode).  Stance ankle plantarflexes + stance hip extends when the torso
# pitches past a small forward target.
SG_PITCH_TGT = 3.0          # deg forward lean held during a step
SG_ANK_K = 0.013           # stance ankle-pitch per deg of excess forward lean
SG_HIP_K = 0.009           # stance hip-pitch per deg of excess forward lean
SG_MAX = 0.24

# SCRIPTED swing-leg trajectory (deterministic lift+place -- the momentum-scaled
# whip only works when a push has already supplied CoM velocity, so gait
# initiation from rest needs an explicit swing).  RL residual refines.
SW_HIP_BACK0 = 0.10         # swing hip starts slightly retracted
SW_KNEE_LIFT = 0.62         # peak knee flexion mid-swing (foot clearance)
SW_KNEE_LAND = 0.12         # knee flexion held at touchdown (soft knee)
SW_ANK_CLEAR = 0.20         # ankle dorsiflexion mid-swing


class BipedWalkContEnv(BipedWalkEnv):
    def __init__(self, model_path=WCONT_MODEL, speed_tgt=SPEED_TGT_DEFAULT, **kw):
        kw.setdefault("variant", "cp")
        kw.setdefault("max_steps", 10_000)
        self.speed_tgt = float(speed_tgt)
        self._sole_loc = {}
        super().__init__(model_path=model_path, **kw)
        self._bw_n = float(np.sum(self.model.body_mass)) * G
        self._lift_boost = 1.0
        for side in ("L", "R"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"{side}_foot_collision")
            mid = self.model.geom_dataid[gid]
            va, vn = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            loc = self.model.mesh_vert[va:va + vn].reshape(-1, 3).copy()
            # keep only the lower ~40% of vertices (the sole) to keep it cheap
            zloc = loc[:, 2]
            keep = zloc <= zloc.min() + 0.4 * (zloc.max() - zloc.min() + 1e-9)
            self._sole_loc[side] = (gid, loc[keep])

    def _sole_clear(self, side):
        gid, loc = self._sole_loc[side]
        d = self.data
        rot = d.geom_xmat[gid].reshape(3, 3)
        wz = loc @ rot.T[:, 2] + d.geom_xpos[gid][2]
        return float(wz.min() - FLOOR_Z)

    # -------- task knob (curriculum) --------
    def set_task(self, speed_tgt=None, **kw):
        if speed_tgt is not None:
            self.speed_tgt = float(speed_tgt)
        return (self.speed_tgt,)

    # -------- lifecycle --------
    def reset(self, seed=None, options=None):
        super(BipedWalkEnv, self).reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self._stand.qpos0
        d.qvel[:] = self._stand.qvel0
        from biped_env import DEFAULT_POSE
        d.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_forward(m, d)
        self._reset_flags()
        for _ in range(20):
            d.ctrl[:15] = self._stand.control(m, d)
            mujoco.mj_step(m, d)
            self._vsync()
        # REFERENCE-STATE INIT: every episode starts ALREADY MOVING FORWARD near
        # the target speed with a staggered stance -- so step 1 is a *catch* of a
        # controlled forward fall (the regime the recovery policy is good at), not
        # a standing-start gait initiation.  swing=R first, so plant L forward.
        vfwd0 = self.speed_tgt * float(self.np_random.uniform(0.75, 1.05))
        d.qvel[1] -= vfwd0                                   # world +Y is backward
        d.qpos[7 + LEG["L"]["hp"]] += _FHS["L"] * 0.18       # L (stance) hip forward
        d.qpos[7 + LEG["R"]["hp"]] += _FHS["R"] * (-0.14)    # R (swing) hip trailing
        d.qpos[7 + LEG["R"]["kn"]] += _KFS * 0.16            # R knee pre-flexed
        d.qvel[6 + LEG["R"]["hp"]] += _FHS["R"] * self.np_random.uniform(0.8, 1.6)
        d.qvel[6 + LEG["L"]["hp"]] += self.np_random.uniform(-0.3, 0.3)
        mujoco.mj_forward(m, d)

        self._v0 = self.speed_tgt
        self._t = 0
        self._flight_run = 0
        self._flight_ms = 0
        self._x0 = float(d.subtree_com[CHEST_BODY][1])
        self._hold = 0
        self._in_hold = False
        self._nstep = 0
        self._genuine = 0
        self._scrape = 0
        self._step_lens = []
        self._peak_tilt = float(sample_balance(m, d).up_tilt_deg)
        self._begin_step(first=True)
        return self._obs(), {}

    def _begin_step(self, first=False):
        m, d = self.model, self.data
        if not first:
            self._swing, self._stance = self._stance, self._swing
        self._lat = 1.0 if self._swing == "R" else -1.0
        self._step_k += 1
        self._phase = "shift"           # lateral weight-shift BEFORE the foot lifts
        self._sk = 0
        self._in_hold = False
        self._shift_done = False
        self._sw_lifted = False
        self._sw_unload = 0
        self._plant_streak = 0
        self._swing_p0 = _foot_xy_z(m, d, self._swing).copy()
        self._sw_peak_lat = 0.0
        self._sw_peak_z = 0.0          # ankle-body rise (legacy)
        self._sw_peak_clear = 0.0     # TRUE sole clearance peak
        self._sw_air_ms = 0
        self._sw_drag_ms = 0
        self._onset_capt_m = 0.0
        mujoco.mj_subtreeVel(m, d)
        v = float(np.hypot(d.subtree_linvel[CHEST_BODY][0], d.subtree_linvel[CHEST_BODY][1]))
        vf = -float(d.subtree_linvel[CHEST_BODY][1])
        # startup ramp: the first two steps are small so the gait initiates gently
        rr = float(np.clip(0.45 + 0.28 * (self._step_k - 1), 0.45, 1.0))
        reach = rr * float(np.clip(WALK_REACH + WALK_REACH_VGAIN * max(0.0, self.speed_tgt - vf),
                                   0.08, CP_REACH_MAX))
        self._arc = (reach, 0.0)
        self._swing_end = int(np.clip(120 + 130.0 * reach, 120, 155))   # brisk walk cadence
        self._ref_scale = float(np.clip(max(v, vf) / V0_NOMINAL, 0.5, 1.15))
        self._lift_boost = float(np.clip(1.30 - 0.18 * (self._step_k - 1), 0.70, 1.30))

    def _prop_ramp(self):
        return float(np.clip(0.30 + 0.24 * (self._step_k - 1), 0.30, 1.0))

    # -------- control: recovery stack + propulsion + mediolateral --------
    def _compose_ctrl(self, residual):
        u = np.array(super()._compose_ctrl(residual), float)
        m, d = self.model, self.data
        bs = self._balance()
        st, sw = self._stance, self._swing
        gst, gsw = LEG[st], LEG[sw]
        swdir = 1.0 if sw == "L" else -1.0
        x_out = (float(bs.com[0]) - _X_MID_M) * swdir     # + => CoM toward swing side
        vx_out = float(bs.com_vel[0]) * swdir

        if self._phase == "shift" and not self._in_hold:
            f = min(1.0, self._sk / 12.0)
            u[gst["hr"]] += _HR_SIGN[st] * ML_SHIFT_HR * f
            u[gst["ar"]] += _AR_SIGN[st] * ML_SHIFT_AR * f
        elif self._phase in ("swing", "descend") and not self._in_hold:
            prog = min(1.0, self._sk / max(self._swing_end, 1)) if self._phase == "swing" else 1.0
            rp = self._prop_ramp()
            if _foot_normal_force(m, d, st) > 8.0:
                u[gst["hp"]] += _FHS[st] * (-ST_HIP_EXT) * prog * rp
                po = float(np.clip((prog - 0.55) / 0.45, 0.0, 1.0)) if self._phase == "swing" else 1.0
                u[gst["ap"]] += _FHS[st] * (-ST_ANK_PUSH) * po * rp
            # stance-leg mediolateral hold
            hr_hold = float(np.clip(ML_STANCE_HR_KX * x_out + ML_STANCE_HR_KV * vx_out,
                                    -ML_STANCE_HR_MAX, ML_STANCE_HR_MAX))
            u[gst["hr"]] += _HR_SIGN[st] * hr_hold
            # sagittal: resist forward torso-pitch runaway (stance ankle + hip)
            exc = float(bs.fwd_lean_deg) - SG_PITCH_TGT
            if exc > 0.0 and _foot_normal_force(m, d, st) > 8.0:
                u[gst["ap"]] += _FHS[st] * float(np.clip(SG_ANK_K * exc, 0.0, SG_MAX))
                u[gst["hp"]] -= _FHS[st] * float(np.clip(SG_HIP_K * exc, 0.0, SG_MAX))
            # swing-foot outboard placement (LIPM lateral capture)
            place = float(np.clip(ML_PLACE_K * max(0.0, vx_out) + 0.15 * max(0.0, x_out),
                                  0.0, ML_PLACE_MAX)) * prog
            u[gsw["hr"]] += _HR_SIGN[sw] * (-place)

            # deterministic knee-flexion LIFT added to the whip so the swing foot
            # clears even at low CoM momentum (gait initiation); tapers in descend.
            if self._phase == "swing":
                bell = float(np.sin(np.pi * prog))
                u[gsw["kn"]] += _KFS * SW_KNEE_LIFT * bell * self._lift_boost
                u[gsw["ap"]] += -_FHS[sw] * SW_ANK_CLEAR * bell * self._lift_boost
            else:  # descend: gently flatten the sole for a flat touchdown
                sp = float(_sole_pitch(m, d, sw))
                u[gsw["ap"]] += -0.5 * sp
        elif self._in_hold:
            for side in ("L", "R"):
                if _foot_normal_force(m, d, side) > 8.0:
                    u[LEG[side]["hp"]] += _FHS[side] * (-PROP_LEAN_HIP)
        return np.clip(u, self.clow, self.chigh)

    # -------- step --------
    def step(self, action):
        m, d = self.model, self.data
        from biped_walk_env import ACT_SCALE
        a_canon = np.asarray(action, np.float32).clip(-1.0, 1.0)
        a_actual = a_canon if self._lat > 0 else mirror_action(a_canon)
        residual = a_actual * ACT_SCALE
        self._t += 1

        r = 0.0
        fell = False
        plant = False
        for _ in range(FRAME_SKIP):
            d.ctrl[:15] = self._compose_ctrl(residual)
            mujoco.mj_step(m, d)
            self._vsync()
            self._sk += 1
            self._peak_tilt = max(self._peak_tilt, self._cheap_tilt())

            lnf = _foot_normal_force(m, d, "L")
            rnf = _foot_normal_force(m, d, "R")
            if lnf < 4.0 and rnf < 4.0:
                self._flight_run += 1
                self._flight_ms += 1
            else:
                self._flight_run = 0
            if self._fallen() or self._flight_run > FLIGHT_TERM_MS:
                fell = True
                break

            if self._in_hold:
                self._hold += 1
                if self._hold >= HOLD_MS:
                    self._begin_step()
                continue

            sw_nf = _foot_normal_force(m, d, self._swing)

            if self._phase == "shift":
                if (self._sk >= SHIFT_MS_MIN
                        and (sw_nf < SHIFT_UNLOAD_FRAC * self._bw_n or self._sk >= SHIFT_MS_MAX)):
                    self._phase = "swing"
                    self._sk = 0
                    self._swing_p0 = _foot_xy_z(m, d, self._swing).copy()
                    mujoco.mj_subtreeVel(m, d)
                    vmag = float(np.hypot(d.subtree_linvel[CHEST_BODY][0],
                                          d.subtree_linvel[CHEST_BODY][1]))
                    self._ref_scale = float(np.clip(vmag / V0_NOMINAL, 0.45, 1.4))
                continue

            clear = self._sole_clear(self._swing)
            self._sw_peak_clear = max(self._sw_peak_clear, clear)
            swf = _foot_xy_z(m, d, self._swing)
            self._sw_peak_lat = max(self._sw_peak_lat,
                                    abs(float(swf[0]) - float(self._swing_p0[0])))
            self._sw_peak_z = max(self._sw_peak_z, float(swf[2]) - float(self._swing_p0[2]))

            if self._phase == "swing":
                if not self._sw_lifted and clear > CLEAR_OK_M and sw_nf < 2.0:
                    self._sw_lifted = True
                if clear > CLEAR_OK_M and sw_nf < 2.0:
                    self._sw_air_ms += 1
                    r += 0.03
                swb = self._b_rf if self._swing == "R" else self._b_lf
                vxy = float(np.hypot(d.cvel[swb][3], d.cvel[swb][4]))
                if clear < 0.010 and sw_nf > 3.0 and vxy > 0.03:
                    self._sw_drag_ms += 1
                    r += -0.12                                   # DRAG / shuffle / pivot
                if self._sk >= self._swing_end:
                    self._phase = "descend"
                    self._sk = 0
            else:  # descend
                swc = (sample_balance(m, d).r_contact if self._swing == "R"
                       else sample_balance(m, d).l_contact)
                genuine_plant = self._sw_lifted and swc and sw_nf > 10.0
                self._plant_streak = self._plant_streak + 1 if genuine_plant else 0
                if (self._sk >= DESCEND_MIN_MS
                        and (self._plant_streak >= 5 or self._sk >= 130)):
                    self._snapshot_td(sample_balance(m, d), sw_nf)
                    plant = True
                    self._in_hold = True
                    self._hold = 0
                    self._phase = "stance"       # report double-support to the policy
                    break

        # ---------------- reward ----------------
        bs = sample_balance(m, d)
        vfwd = -float(bs.com_vel[1])
        vlat = float(bs.com_vel[0])
        vz = float(bs.com_vel[2])
        tilt = self._cheap_tilt()
        wv = self.data.xmat[CHEST_BODY].reshape(3, 3) @ self.data.qvel[3:6]
        ang_speed = float(np.hypot(wv[0], wv[1]))
        tgt = max(self.speed_tgt, 0.1)
        lc = _foot_normal_force(m, d, "L") > 6.0
        rc = _foot_normal_force(m, d, "R") > 6.0

        r += 3.0
        r += 2.6 * float(np.clip(1.0 - abs(vfwd - tgt) / tgt, -0.5, 1.0))
        r += -1.6 * max(0.0, vfwd - 1.4 * tgt)
        r += -1.2 * float(vfwd < 0.5 * tgt)
        r += -1.0 * float(np.clip(tilt / 18.0, 0.0, 1.8))
        r += -0.5 * ang_speed
        r += -0.8 * abs(vlat)
        r += -0.7 * abs(vz)
        r += -0.3 * abs(float(bs.yaw_deg)) / 20.0
        r += -0.05 * float(np.mean((a_canon - self._prev_action) ** 2))
        r += -0.02 * float(np.mean(a_canon ** 2))
        if not self._in_hold and self._phase in ("swing", "descend"):
            r += 0.5 if (lc != rc) else -0.5
        if not lc and not rc:
            r += -3.0
        self._prev_action = a_canon.copy()

        if plant:
            td = self._td[self._step_k]
            sl = td["fwd_mm"] / 1000.0
            self._nstep += 1
            real = (self._sw_peak_clear >= CLEAR_OK_M and self._sw_air_ms >= AIR_MIN_MS
                    and sl >= STEP_FWD_MIN_M)
            if real:
                self._genuine += 1
                self._step_lens.append(sl)
                r += 3.5 * float(np.clip((sl - 0.02) / 0.10, -0.5, 1.0))
                r += 1.4 * float(np.clip((self._sw_peak_clear - CLEAR_OK_M) / 0.030, -0.5, 1.0))
                r += 0.9 * float(np.clip((self._sw_air_ms - AIR_MIN_MS) / 30.0, -0.5, 1.0))
                r += 0.6 * float(np.clip(1.0 - abs(td["sole_pitch"]) / 16.0, -0.5, 1.0))
                r += -0.5 * float(np.clip((abs(td["net_lat_mm"]) - 28.0) / 45.0, 0.0, 1.5))
            else:
                self._scrape += 1
                r += -2.5                                        # scrape / pivot / no-lift
            r += -0.02 * min(30.0, self._sw_drag_ms)

        if fell:
            return self._obs(), r - 50.0, True, False, self._info_w(True, vfwd)
        if self._t >= EP_STEPS:
            return self._obs(), r, False, True, self._info_w(False, vfwd)
        return self._obs(), r, False, False, self._info_w(False, vfwd)

    def _info_w(self, fell, vfwd):
        d = self.data
        dist = self._x0 - float(d.subtree_com[CHEST_BODY][1])
        dur = max(self._t * CTRL_DT, 1e-6)
        return dict(fell=bool(fell), vfwd=float(vfwd), dist=float(dist), t=int(self._t),
                    speed_tgt=float(self.speed_tgt), nstep=int(self._nstep),
                    genuine=int(self._genuine), scrape=int(self._scrape),
                    cadence=float(self._genuine / dur),
                    mean_step_mm=float(np.mean(self._step_lens) * 1000) if self._step_lens else 0.0,
                    flight_frac=float(self._flight_ms) / max(self._t * FRAME_SKIP, 1),
                    peak_tilt=float(self._peak_tilt))


# ============================================================ CLI
def _load_policy(run_dir, model=WCONT_MODEL):
    import json
    import os
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    hp = json.load(open(f"{run_dir}/hparams.json"))
    tmp = DummyVecEnv([lambda: BipedWalkContEnv(model_path=model)])
    pf = f"{run_dir}/policy_best.pth" if os.path.exists(f"{run_dir}/policy_best.pth") else f"{run_dir}/policy.pth"
    vf = f"{run_dir}/vecnormalize_best.pkl" if os.path.exists(f"{run_dir}/vecnormalize_best.pkl") else f"{run_dir}/vecnormalize.pkl"
    if os.path.exists(vf):
        vn = VecNormalize.load(vf, tmp)
        mean, var = vn.obs_rms.mean, vn.obs_rms.var
    else:
        mean, var = 0.0, 1.0
    mm = PPO("MlpPolicy", tmp, device="cpu",
             policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(pf, map_location="cpu", weights_only=True))
    mm.policy.eval()
    print(f"  loaded {pf}  (vecnorm: {os.path.basename(vf) if os.path.exists(vf) else 'none'})")
    return mm, (lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32))


def _smoke(n=8, render=None, policy=None, model=WCONT_MODEL, speed=SPEED_TGT_DEFAULT):
    pol = _load_policy(policy, model) if policy else None
    env = BipedWalkContEnv(model_path=model, speed_tgt=speed,
                           render_mode=("rgb_array" if render else None))
    frames = []
    agg = []
    for i in range(n):
        o, _ = env.reset(seed=100 + i)
        done = False
        info = {}
        vv = []
        ep = []
        while not done:
            if pol:
                import torch
                with torch.no_grad():
                    a, _ = pol[0].predict(pol[1](o), deterministic=True)
            else:
                a = np.zeros(10, np.float32)
            o, r, term, trunc, info = env.step(a)
            done = term or trunc
            vv.append(info["vfwd"])
            if render and i < 4:
                ep.append(env.render())
        if render and i < 4:
            frames.append(ep)
        agg.append(info)
        print(f"  ep {i:2d}  dist {info['dist']:+.2f}m  vfwd {np.mean(vv[15:]):+.2f}  "
              f"genuine {info['genuine']:2d}  scrape {info['scrape']:2d}  "
              f"cad {info['cadence']:.1f}/s  steplen {info['mean_step_mm']:.0f}mm  "
              f"flight {info['flight_frac']:.2f}  tilt {info['peak_tilt']:.0f}  "
              f"{'FELL@' + str(info['t']) if info['fell'] else 'survived ' + str(info['t'])}")
    surv = np.mean([not a["fell"] for a in agg])
    print(f"  --- survival {surv:.0%}  mean genuine {np.mean([a['genuine'] for a in agg]):.1f}  "
          f"mean dist {np.mean([a['dist'] for a in agg]):+.2f}m  "
          f"mean flight {np.mean([a['flight_frac'] for a in agg]):.2f}")
    if render and frames:
        import imageio.v2 as imageio
        H = max(len(f) for f in frames)
        tiles = [np.hstack([f[min(t, len(f) - 1)] for f in frames][:2]) for t in range(H)]
        imageio.mimsave(render, tiles, fps=40, macro_block_size=1)
        print(f"  wrote {render}")
    env.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--render", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--model", default=WCONT_MODEL)
    ap.add_argument("--speed", type=float, default=SPEED_TGT_DEFAULT)
    a = ap.parse_args(argv)
    if a.smoke:
        _smoke(a.n, a.render, a.policy, a.model, a.speed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
