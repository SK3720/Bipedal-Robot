"""Gym env for REACTIVE SEQUENTIAL STEP RECOVERY.

Push -> take a step to catch yourself -> if the capture point is still outside
the support polygon (momentum remains), take another step (opposite foot) ->
repeat until caught or MAX_STEPS.  The number of steps is state-dependent; a
scripted 2-plane capture-point trigger decides *whether* to step, the policy
decides *how* (its bounded swing-leg residual sets step length / shape).

This supersedes the fixed-2-step env.  It is Stage-1's architecture generalised:
Stage 1 already had a scripted capture-point trigger + a swing-residual policy;
here the trigger can fire again, alternating feet via the sagittal mirror, until
the capture point is inside the polygon.

Canonical frame: the policy always "thinks" swing = R.  Odd steps swing R
(identity); even steps swing L (obs sagittally reflected by `mirror_obs`, the
3-d residual reflected back by `mirror_action`).

Later steps enter with less momentum, so the whip-retract reference amplitude is
scaled by entry speed (`ref_scale`) -> steps naturally taper.  The policy
fine-tunes on top.

Reward (NOT softened): dense capture-point shaping + per-step touchdown quality
+ a per-step cost (be reactive, don't burn steps) + a terminal bonus that
requires the capture point genuinely INSIDE the polygon, low CoM speed, upright,
double support.  A robot still drifting at the end is a FAILURE.

Model: robot/_exp_hands_2x.xml.  robot/robot.xml is NOT touched.

    python biped_multistep_env.py --smoke
    python biped_multistep_env.py --smoke --policy runs/recovery_s1
    python biped_multistep_env.py --mirror-test
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import mujoco

from recovery_metrics import (
    FLOOR_Z, NOMINAL_CHEST_Z, sample_balance, _foot_normal_force, _foot_xy_z,
)
import biped_recovery_env as S1
from biped_recovery_env import (
    BipedRecoveryEnv, DEFAULT_MODEL, G,
    AR_L, AR_R, HIPR_L, HIPR_R, REF_KNOTS, REF_SRC,
    COP_PER_RAD, AR_BIAS_MAX, COP_LO_MM, COP_HI_MM, X_MID_MM, X_SS_MM, X_TRANSFER_MM,
    SWING_MS, DESCEND_MS, SHIFT_MS, IMP_AMP, IMP_RAMP_MS, IMP_HOLD_MS,
    TRANSFER_MS, SETTLE_BIAS_FLOOR,
    PLANT_NF, PLANT_HOLD, STEP_MAX_MS, HIPROLL_CAP, STANCE_CAP,
    FRAME_SKIP, ACT_SCALE, BODY_WEIGHT_N, _knot_val, _smooth,
    _FOOT_HALF_LAT, _FOOT_HALF_FWD,
)
from step_primitive import _sole_pitch

# ---------------------------------------------------------------- mirror maps
_MIRROR_PERM = np.array([0, 3, 4, 1, 2, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9])
_MIRROR_SIGN = np.array([-1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], float)
LEG_CTRL = {
    "R": dict(sw_hip=11, sw_knee=12, sw_ankle=13, st_hip=6,  st_knee=7,  st_ankle=8,  hipr_sw=10),
    "L": dict(sw_hip=6,  sw_knee=7,  sw_ankle=8,  st_hip=11, st_knee=12, st_ankle=13, hipr_sw=5),
}
_X_MID_M = X_MID_MM / 1000.0


def _mirror_ref(ref_knots, ref_src):
    kn = {int(_MIRROR_PERM[ci]): _MIRROR_SIGN[ci] * v.copy() for ci, v in ref_knots.items()}
    sr = {int(_MIRROR_PERM[ci]): s for ci, s in ref_src.items()}
    return kn, sr


REF_KNOTS_L, REF_SRC_L = _mirror_ref(REF_KNOTS, REF_SRC)

_OBS_NEG = (0, 3, 6, 7, 8, 12, 14, 22, 26, 27)
_OBS_SWAP = ((18, 19), (20, 21))
_OBS_JPOS = slice(28, 43)
_OBS_JVEL = slice(43, 58)
_OBS_DIM = 66


def mirror_obs(o):
    o = np.asarray(o, np.float32).copy()
    for i in _OBS_NEG:
        o[i] = -o[i]
    for a, b in _OBS_SWAP:
        o[a], o[b] = o[b], o[a]
    o[_OBS_JPOS] = (_MIRROR_SIGN * o[_OBS_JPOS][_MIRROR_PERM]).astype(np.float32)
    o[_OBS_JVEL] = (_MIRROR_SIGN * o[_OBS_JVEL][_MIRROR_PERM]).astype(np.float32)
    return o


def mirror_action(a):
    return np.asarray(a, np.float32) * np.array([-1.0, -1.0, -1.0], np.float32)


def _sole_roll(model, data, side):
    R = data.xmat[model.body(f"{side}_foot").id].reshape(3, 3)
    return float(np.arcsin(np.clip(R[0, 0], -1.0, 1.0)))


# ---------------------------------------------------------------- tuning
MAX_STEPS = 4
BETWEEN_TRANSFER_MS = 90      # short transfer after a plant
BETWEEN_SETTLE_MS = 120       # micro-settle before the step-again decision
SHIFT_K_MS = 95              # weight-shift onto the new stance before step k>=2
SHIFT_K_MIN_MS = 55
SWING_K_START_MS = 92
FINAL_SETTLE_MS = 420        # after the last step, before judging

# step-again trigger (evaluated after the micro-settle): still translating, or the
# capture point still past the front support edge.  d_fwd collapses after the
# momentum-killing whip so v/speed are the primary signal.
TRIG_CAPT_FWD_MM = 15.0
TRIG_VFWD = 0.14
TRIG_SPEED = 0.24
# "caught" (success): capture point inside the polygon, essentially at rest, upright
CAUGHT_CAPT_MM = 10.0
CAUGHT_SPEED = 0.13
V0_NOMINAL = 0.24            # typical latched post-push CoM speed -> ref_scale = 1
STEP_COST = 0.6             # reward penalty per step taken (be reactive)

# ---- forward-rolling step primitive (for chaining; whip-retract is arrest-only) ----
# monotonic hip flexion forward, knee sin-bump for clearance, NO end retract, ankle
# toe-up mid then flat.  The trailing ankle pushes off during the (short) roll.
FS_HIP_FWD = 0.52          # peak forward hip flexion (rad, sign per side)
FS_KNEE_PK = 0.42          # mid-swing knee flex bump
FS_KNEE_LAND = 0.12        # knee flex held at touchdown
FS_ANK_TOEUP = 0.16        # ankle toe-up mid-swing (clearance) -> 0 at TD
FS_SWING_MS = 150
FS_DESCEND_MS = 95
ROLL_MS = 55              # trailing-ankle push-off + weight-commit before the swing
ROLL_PUSH = 0.30

PUSH_BAND_MS = (128.0, 140.0)


class BipedMultiStepEnv(BipedRecoveryEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, model_path=DEFAULT_MODEL, seed=None,
                 push_band=PUSH_BAND_MS, dir_spread_deg=0.0, render_mode=None,
                 max_steps=MAX_STEPS):
        self._max_steps = int(max_steps)
        super().__init__(model_path=model_path, seed=None, push_band=push_band,
                         dir_spread_deg=dir_spread_deg, render_mode=render_mode)
        self._ss = {
            "R": (self._ss_hip, self._ss_knee, self._ss_ankle),
            "L": (_MIRROR_SIGN[11] * self._ss_hip,
                  _MIRROR_SIGN[12] * self._ss_knee,
                  _MIRROR_SIGN[13] * self._ss_ankle),
        }
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._stand.qpos0
        self.data.qvel[:] = self._stand.qvel0
        mujoco.mj_forward(self.model, self.data)
        self._reset_ms_flags()
        if seed is not None:
            self.reset(seed=seed)

    # -------------------------------------------------- lifecycle
    def _reset_ms_flags(self):
        self._swing, self._stance, self._lat = "R", "L", 1.0
        self._step_k = 1
        self._td = {}
        self._td_paid = set()
        self._rhold = None
        self._ref_scale = 1.0
        self._pending_fell = False
        self._steps_info = []          # per-step (push, sep, capt_after) for diagnostics

    def _init_episode_state(self):
        self._reset_ms_flags()
        super()._init_episode_state()

    def reset(self, seed=None, options=None):
        self._reset_ms_flags()
        return super().reset(seed=seed, options=options)

    def _init_after_shift(self):
        super()._init_after_shift()
        self._reset_ms_flags()

    # -------------------------------------------------- per-side helpers
    def _ref_val(self, ci):
        c = LEG_CTRL[self._swing]
        ss = getattr(self, "_ss", None) or {"R": (self._ss_hip, self._ss_knee, self._ss_ankle)}
        h, k, a = ss[self._swing]
        if ci == c["sw_hip"]:
            return h
        if ci == c["sw_knee"]:
            return k
        if ci == c["sw_ankle"]:
            return a
        return 0.0

    def _frontal_bias(self, bs, x_ref_m, lat=1.0):
        x = float(bs.com[0])
        v = float(bs.com_vel[0])
        if lat < 0:
            x = 2.0 * _X_MID_M - x
            v = -v
        cop = x_ref_m - self.kx * (x - x_ref_m) - self.kv * v
        cop = float(np.clip(cop, COP_LO_MM / 1000.0, COP_HI_MM / 1000.0))
        b = float(np.clip(-(cop - x) / COP_PER_RAD, -AR_BIAS_MAX, AR_BIAS_MAX))
        return -b if lat < 0 else b

    # -------------------------------------------------- capture point
    def _capture_fwd_lat(self):
        """(d_fwd, d_lat) of the 2-plane capture point OUTSIDE the current support
        polygon, in mm (>0 => must catch).  Also returns signed fwd excess and v_fwd."""
        bs = self._balance()
        h = max(float(bs.com[2]) - FLOOR_Z, 0.05)
        tc = np.sqrt(h / G)
        xi_fwd = -float(bs.com[1]) + (-float(bs.com_vel[1])) * tc
        xi_lat = float(bs.com[0]) + float(bs.com_vel[0]) * tc
        stf = _foot_xy_z(self.model, self.data, self._stance)
        swf = _foot_xy_z(self.model, self.data, self._swing)
        feet = [stf, swf] if self._td.get(self._step_k) is not None else [stf]
        fwd_lo = min(-f[1] for f in feet) - _FOOT_HALF_FWD
        fwd_hi = max(-f[1] for f in feet) + _FOOT_HALF_FWD
        lat_lo = min(f[0] for f in feet) - _FOOT_HALF_LAT
        lat_hi = max(f[0] for f in feet) + _FOOT_HALF_LAT
        d_fwd = max(0.0, xi_fwd - fwd_hi, fwd_lo - xi_fwd) * 1000.0
        d_lat = max(0.0, xi_lat - lat_hi, lat_lo - xi_lat) * 1000.0
        fwd_excess = (xi_fwd - fwd_hi) * 1000.0        # signed: >0 => past front edge
        return d_fwd, d_lat, fwd_excess, -float(bs.com_vel[1])

    def _capture_2plane(self):
        d_fwd, d_lat, _, _ = self._capture_fwd_lat()
        return d_fwd, d_lat

    # -------------------------------------------------- scaffold control
    def _scaffold_ctrl(self, phase, sk, w, residual):
        m, d = self.model, self.data
        bs = self._balance()
        s = self._swing
        c = LEG_CTRL[s]
        lat = self._lat
        ref_knots = REF_KNOTS if s == "R" else REF_KNOTS_L
        ref_src = REF_SRC if s == "R" else REF_SRC_L

        if phase == "shift":
            fr = _smooth(min(1.0, sk / SHIFT_MS))
            x_ref = (X_MID_MM + fr * (X_SS_MM - X_MID_MM)) / 1000.0
        elif phase == "shiftk":
            fr = _smooth(min(1.0, sk / SHIFT_K_MS))
            x_ref = (X_MID_MM + fr * (X_SS_MM - X_MID_MM)) / 1000.0
        elif phase in ("swing", "descend", "transfer"):
            x_ref = X_SS_MM / 1000.0
        else:  # settle
            x_ref = X_TRANSFER_MM / 1000.0

        # ---- settle: FULL StandingLQR + decaying frontal bias (Stage-1 verbatim) ----
        if phase == "settle":
            u = np.array(self._stand.control(m, d), float)
            for ci in (c["sw_hip"], c["sw_knee"], c["sw_ankle"],
                       c["st_hip"], c["st_knee"], c["st_ankle"]):
                u[ci] = float(np.clip(u[ci], self._stand.ctrl0[ci] - 0.7,
                                      self._stand.ctrl0[ci] + 0.7))
            decay = SETTLE_BIAS_FLOOR + (1.0 - SETTLE_BIAS_FLOOR) * max(0.0, 1.0 - sk / 500.0)
            bias = self._frontal_bias(bs, x_ref, lat) * decay
            u[AR_L] = bias
            u[AR_R] = bias
            return np.clip(u, self.ctrl_low, self.ctrl_high)

        dq = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, self._stand.qpos0, d.qpos)
        dx = np.concatenate([dq, d.qvel - self._stand.qvel0])
        for ci in (AR_L, AR_R, HIPR_L, HIPR_R):
            dx[6 + ci] = 0.0
            dx[m.nv + 6 + ci] = 0.0
        dx[0] = dx[m.nv + 0] = 0.0
        u = np.array(self._stand.ctrl0 - self._stand.K @ dx, float)

        unload_sign = -1.0 if s == "R" else 1.0
        imp_amp = IMP_AMP if phase == "shift" else 0.11
        if phase in ("shift", "shiftk") and sk < IMP_RAMP_MS + IMP_HOLD_MS:
            ramp = sk / IMP_RAMP_MS if sk < IMP_RAMP_MS else 1.0
            bias = unload_sign * imp_amp * ramp
        else:
            bias = self._frontal_bias(bs, x_ref, lat)

        if phase in ("swing", "descend"):
            rs = self._ref_scale
            for ci, kn in ref_knots.items():
                # taper only the swing-leg knots by entry momentum; arms/stance as-is
                kk = kn * rs if ci in (c["sw_hip"], c["sw_knee"], c["sw_ankle"]) else kn
                dval = _knot_val(kk, w)
                if ref_src[ci] == "lqr":
                    u[ci] = u[ci] + dval
                else:
                    u[ci] = self._ref_val(ci) + dval
        elif phase == "transfer":
            b = _smooth(min(1.0, sk / TRANSFER_MS))
            u[c["sw_hip"]] = self._plant_q[0]
            u[c["sw_knee"]] = self._plant_q[1] + S1.KNEE_FLEX_SIGN * 0.05 * b
            u[c["sw_ankle"]] = self._plant_q[2] + S1.FWD_HIP_SIGN * 0.05 * b

        if phase in ("swing", "descend", "transfer"):
            u[c["sw_hip"]] += residual[0]
            u[c["sw_knee"]] += residual[1]
            u[c["sw_ankle"]] += residual[2]

        u[AR_L] = bias
        u[AR_R] = bias

        # keep the previously-planted leg from being unwound by the feet-together LQR
        if self._step_k >= 2 and self._rhold is not None:
            rh = self._rhold
            if phase == "shiftk":
                for k, ci in enumerate((c["st_hip"], c["st_knee"], c["st_ankle"])):
                    u[ci] = float(np.clip(u[ci], rh[k] - 0.13, rh[k] + 0.13))
            elif phase in ("swing", "descend", "transfer"):
                for k, ci in enumerate((c["st_hip"], c["st_knee"])):
                    u[ci] = float(np.clip(u[ci], rh[k] - 0.30, rh[k] + 0.30))

        if phase in ("swing", "descend", "transfer"):
            u[c["hipr_sw"]] = float(np.clip(u[c["hipr_sw"]], -HIPROLL_CAP, HIPROLL_CAP))
            for ci in (c["st_hip"], c["st_knee"], c["st_ankle"]):
                if ci not in ref_knots:
                    u[ci] = float(np.clip(u[ci], -STANCE_CAP, STANCE_CAP))

        return np.clip(u, self.ctrl_low, self.ctrl_high)

    # -------------------------------------------------- obs
    def _obs(self):
        m, d = self.model, self.data
        up, fwd = self._chest_axes()
        bs = self._balance()
        lf = _foot_xy_z(m, d, "L")
        rf = _foot_xy_z(m, d, "R")
        stf = lf if self._stance == "L" else rf
        swf = rf if self._swing == "R" else lf
        swb = self._b_rfoot if self._swing == "R" else self._b_lfoot
        swvel = d.cvel[swb][3:6].copy()
        lnf = _foot_normal_force(m, d, "L")
        rnf = _foot_normal_force(m, d, "R")
        d_fwd, d_lat = self._capture_2plane()
        h_err = float(d.qpos[2]) - NOMINAL_CHEST_Z
        phase_oh = np.array([self._phase == "swing", self._phase == "descend",
                             self._phase in ("transfer", "settle", "shiftk")], np.float32)
        dur = SWING_MS if self._phase == "swing" else DESCEND_MS
        obs = np.concatenate([
            up, fwd[:2],
            d.qvel[3:6], d.qvel[0:3],
            [h_err],
            [float(bs.com[0] - stf[0]), float(-bs.com[1] - (-stf[1]))],
            [float(bs.com_vel[0]), float(-bs.com_vel[1])],
            [d_fwd / 100.0, d_lat / 100.0],
            [float(bs.l_contact), float(bs.r_contact),
             lnf / BODY_WEIGHT_N, rnf / BODY_WEIGHT_N],
            [swf[0] - stf[0], -(swf[1] - stf[1]), swf[2] - stf[2]],
            swvel,
            d.qpos[7:22], d.qvel[6:21],
            phase_oh,
            [min(1.0, self._sk / max(dur, 1))],
            self._prev_action,
            [self._v0],
        ]).astype(np.float32)
        return mirror_obs(obs) if self._lat < 0 else obs

    # -------------------------------------------------- step
    def step(self, action):
        m, d = self.model, self.data
        action = np.asarray(action, np.float32).clip(-1.0, 1.0)
        residual = (action * ACT_SCALE) if self._lat > 0 else mirror_action(action * ACT_SCALE)
        self._ep_step += 1

        r_shape = 0.0
        fell = False
        plant_now = False
        for _ in range(FRAME_SKIP):
            if self._phase == "swing":
                self._w = min(1.0, self._sk / SWING_MS)
            elif self._phase == "descend":
                self._w = 1.0 + self._sk / DESCEND_MS
            d.ctrl[:15] = self._scaffold_ctrl(self._phase, self._sk, self._w, residual)
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                fell = True
                break
            bs = self._balance()
            sw_nf = _foot_normal_force(m, d, self._swing)
            d_fwd, d_lat = self._capture_2plane()
            r_shape += -0.004 * min(120.0, d_fwd) / 120.0 - 0.006 * min(120.0, d_lat) / 120.0

            if self._phase == "swing":
                if not self._foot_lifted and sw_nf < 3.0:
                    self._foot_lifted = True
                if sw_nf < 3.0:
                    self._unloaded_streak += 1
                if self._sk >= SWING_MS:
                    self._phase = "descend"
                    self._sk = 0
            elif self._phase == "descend":
                sw_contact = bs.r_contact if self._swing == "R" else bs.l_contact
                genuine = (self._foot_lifted and sw_contact and sw_nf > PLANT_NF)
                self._plant_streak = self._plant_streak + 1 if genuine else 0
                if self._plant_streak >= PLANT_HOLD or self._sk >= STEP_MAX_MS:
                    self._snapshot_touchdown(bs, sw_nf)
                    plant_now = True
                    break

        self._prev_action = action.copy()

        if fell:
            return self._obs(), r_shape - 20.0, True, False, self._info(fell=True)

        if plant_now:
            td_r = self._touchdown_reward(self._step_k)
            again = self._between_steps()          # None -> another step queued; else -> terminal (r, info)
            if again is None:
                return self._obs(), r_shape + td_r - STEP_COST + 0.3, False, False, self._info(fell=False)
            term_r, info = again
            return self._obs(), r_shape + td_r - STEP_COST + term_r, True, False, info

        if self._ep_step > 900:
            return self._obs(), r_shape, False, True, self._info(fell=False)
        return self._obs(), r_shape + 0.3, False, False, self._info(fell=False)

    # -------------------------------------------------- between steps: step again?
    def _between_steps(self):
        """Just planted step k.  Short transfer + micro-settle, then decide: is the
        capture point caught (=> final settle + terminate) or still out and
        k < max (=> flip feet, shift, hand back to the policy for step k+1)?"""
        m, d = self.model, self.data
        self._phase = "transfer"
        self._sk = 0
        for _ in range(BETWEEN_TRANSFER_MS):
            d.ctrl[:15] = self._scaffold_ctrl("transfer", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                return self._terminal(fell=True)
        self._phase = "settle"
        self._sk = 0
        for _ in range(BETWEEN_SETTLE_MS):
            d.ctrl[:15] = self._scaffold_ctrl("settle", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                return self._terminal(fell=True)

        d_fwd, d_lat, fwd_excess, v_fwd = self._capture_fwd_lat()
        b = sample_balance(m, d)
        self._steps_info.append(dict(k=self._step_k, sep=self._td[self._step_k]["sep_mm"],
                                     capt_fwd=d_fwd, v_fwd=v_fwd, spd=b.com_speed_horiz))
        need_step = (d_fwd > TRIG_CAPT_FWD_MM or v_fwd > TRIG_VFWD
                     or b.com_speed_horiz > TRIG_SPEED)
        # the just-planted foot bears weight next; the trailing foot (current
        # stance) is about to swing, so it may already be light -- don't require it.
        planted_contact = b.r_contact if self._swing == "R" else b.l_contact
        can_step = (self._step_k < self._max_steps and planted_contact
                    and b.up_tilt_deg < 14.0 and abs(b.side_lean_deg) < 12.0
                    and b.com_speed_horiz < 0.75)
        if need_step and can_step:
            self._begin_next_step()
            if self._pending_fell:
                return self._terminal(fell=True)
            return None
        return self._terminal(fell=False)

    def _begin_next_step(self):
        m, d = self.model, self.data
        new_swing = self._stance          # opposite foot
        new_stance = self._swing
        cS = LEG_CTRL[new_swing]
        # hold the leg that just planted (now the stance)
        cH = LEG_CTRL[new_stance]
        self._rhold = np.array([d.qpos[7 + cH["sw_hip"]],
                                d.qpos[7 + cH["sw_knee"]],
                                d.qpos[7 + cH["sw_ankle"]]])
        self._swing, self._stance = new_swing, new_stance
        self._lat = 1.0 if new_swing == "R" else -1.0
        self._step_k += 1
        self._sk = 0
        self._w = 0.0
        self._foot_lifted = False
        self._plant_streak = 0
        self._unloaded_streak = 0
        self._prev_action = np.zeros(3, np.float32)
        self._swing_p0 = _foot_xy_z(m, d, new_swing).copy()
        self._stance_p0 = _foot_xy_z(m, d, new_stance).copy()
        self._plant_q = np.array([self._ref_val(cS["sw_hip"]),
                                  self._ref_val(cS["sw_knee"]),
                                  self._ref_val(cS["sw_ankle"])])
        # entry-momentum taper for the whip reference
        mujoco.mj_subtreeVel(m, d)
        from recovery_metrics import CHEST_BODY
        v_entry = float(np.hypot(d.subtree_linvel[CHEST_BODY][0], d.subtree_linvel[CHEST_BODY][1]))
        self._ref_scale = float(np.clip(v_entry / V0_NOMINAL, 0.45, 1.0))
        # weight-shift onto the new stance foot
        self._phase = "shiftk"
        for _ in range(SHIFT_K_MS + 45):
            d.ctrl[:15] = self._scaffold_ctrl("shiftk", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                self._pending_fell = True
                return
            sw_nf = _foot_normal_force(m, d, new_swing)
            b = self._balance()
            if (self._sk >= SHIFT_K_MIN_MS
                    and (sw_nf < 8.0 or self._sk >= SWING_K_START_MS)
                    and (abs(b.pitch_rate) < 0.6 or self._sk >= SWING_K_START_MS)):
                break
        self._phase = "swing"
        self._sk = 0
        self._w = 0.0

    # -------------------------------------------------- touchdown
    def _snapshot_touchdown(self, bs, sw_nf):
        sw, st = self._swing, self._stance
        sf = _foot_xy_z(self.model, self.data, sw)
        stf = _foot_xy_z(self.model, self.data, st)
        c = LEG_CTRL[sw]
        self._plant_q = np.array([self.data.qpos[7 + c["sw_hip"]],
                                  self.data.qpos[7 + c["sw_knee"]],
                                  self.data.qpos[7 + c["sw_ankle"]]])
        swb = self._b_rfoot if sw == "R" else self._b_lfoot
        self._td[self._step_k] = dict(
            planted=self._plant_streak >= PLANT_HOLD,
            sep_mm=float(-(sf[1] - stf[1]) * 1000.0),
            sole_pitch=float(np.degrees(_sole_pitch(self.model, self.data, sw))),
            sole_roll=float(np.degrees(_sole_roll(self.model, self.data, sw))),
            nf=float(sw_nf),
            vz=float(self.data.cvel[swb][5]),
            unloaded_ok=self._unloaded_streak >= 15,
        )
        self._touchdown = self._td[self._step_k]

    def _touchdown_reward(self, k):
        if k in self._td_paid or k not in self._td:
            return 0.0
        self._td_paid.add(k)
        td = self._td[k]
        r = 1.0 * float(np.clip(1.0 - abs(td["sole_pitch"]) / 12.0, 0.0, 1.0))
        r += 0.5 * float(np.clip(1.0 - abs(td["sole_roll"]) / 12.0, 0.0, 1.0))
        r += 1.5 * float(np.clip(td["nf"] / 20.0, 0.0, 1.0))
        r += -0.6 * float(np.clip(abs(td["vz"]) / 0.35, 0.0, 1.0))
        r += 0.5 if td["unloaded_ok"] else -0.5
        return r

    # -------------------------------------------------- terminal
    def _terminal(self, fell):
        m, d = self.model, self.data
        if fell:
            self._peak_up = max(self._peak_up, 55.0)
            return -20.0, self._info(fell=True)
        # final settle
        self._phase = "settle"
        self._sk = 0
        spd_hist = []
        for _ in range(FINAL_SETTLE_MS):
            d.ctrl[:15] = self._scaffold_ctrl("settle", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            b = self._balance()
            spd_hist.append(b.com_speed_horiz)
            if self._fallen():
                return -20.0, self._info(fell=True)

        b = sample_balance(m, d)
        ds = b.l_contact and b.r_contact
        d_fwd, d_lat, fwd_excess, v_fwd = self._capture_fwd_lat()
        lf = _foot_xy_z(m, d, "L")
        rf = _foot_xy_z(m, d, "R")
        sp_l = abs(np.degrees(_sole_pitch(m, d, "L")))
        sp_r = abs(np.degrees(_sole_pitch(m, d, "R")))
        sr_l = abs(np.degrees(_sole_roll(m, d, "L")))
        sr_r = abs(np.degrees(_sole_roll(m, d, "R")))
        flat = 0.25 * (np.clip(1 - sp_l / 8, 0, 1) + np.clip(1 - sp_r / 8, 0, 1)
                       + np.clip(1 - sr_l / 8, 0, 1) + np.clip(1 - sr_r / 8, 0, 1))
        spd_late = float(np.mean(spd_hist[-20:]))       # last ~20 ms, instantaneous-ish
        all_planted = all(self._td[j]["planted"] for j in self._td)

        # --- success: the capture point is genuinely CAUGHT and the robot is
        #     upright + at rest + double support.  Still drifting => NOT success. ---
        caught = (d_fwd < CAUGHT_CAPT_MM and spd_late < CAUGHT_SPEED)
        upright = (b.up_tilt_deg < 12.0 and abs(b.side_lean_deg) < 10.0
                   and b.chest_z > NOMINAL_CHEST_Z - 0.12 and self._peak_up <= 32.0)
        success = bool(caught and upright and ds and all_planted)
        flat_ok = success and max(sp_l, sp_r) < 15.0 and max(sr_l, sr_r) < 15.0

        r = 0.0
        r += 12.0 if success else 0.0
        r += 3.0 if flat_ok else 0.0
        r += 2.0 * float(np.clip(1.0 - b.up_tilt_deg / 12.0, 0.0, 1.0))
        r += 1.5 * float(np.clip(1.0 - abs(b.side_lean_deg) / 12.0, 0.0, 1.0))
        r += 2.5 * float(np.clip(1.0 - spd_late / 0.30, 0.0, 1.0))        # come to rest
        r += 2.5 * float(np.clip(1.0 - d_fwd / 60.0, 0.0, 1.0))          # capture caught
        r += 1.5 if ds else -1.0
        r += 1.5 * float(flat)

        info = self._info(fell=False)
        info.update(success=success, flat_ok=bool(flat_ok), n_steps=int(self._step_k),
                    end_capt_fwd_mm=float(d_fwd), end_vfwd=float(v_fwd),
                    end_spd=float(spd_late), end_up=float(b.up_tilt_deg),
                    end_side=float(b.side_lean_deg), sole_pitch_max=float(max(sp_l, sp_r)),
                    sole_roll_max=float(max(sr_l, sr_r)), all_planted=bool(all_planted))
        return r, info

    def _info(self, fell):
        info = super()._info(fell)
        info["n_steps"] = int(getattr(self, "_step_k", 1))
        info.setdefault("success", False)
        info.setdefault("flat_ok", False)
        info.setdefault("end_capt_fwd_mm", 999.0)
        info.setdefault("end_vfwd", 9.9)
        info.setdefault("end_spd", 9.9)
        info.setdefault("sole_pitch_max", 99.0)
        info.setdefault("sole_roll_max", 99.0)
        info.setdefault("all_planted", False)
        return info

    # after _begin_next_step, an in-shift fall is flagged; convert on next step()
    def _fallen(self):
        if getattr(self, "_pending_fell", False):
            return True
        return super()._fallen()


# ============================================================ smoke / mirror test
def _load_policy(run_dir, band):
    import json
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    hp = json.load(open(f"{run_dir}/hparams.json"))
    tmp = DummyVecEnv([lambda: BipedMultiStepEnv(push_band=band)])
    vn = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", tmp)
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    mm = PPO("MlpPolicy", tmp, device="cpu",
             policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(f"{run_dir}/policy.pth", map_location="cpu", weights_only=True))
    mm.policy.eval()
    return mm, lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)


def _smoke(n=24, render_path=None, band=PUSH_BAND_MS, model=DEFAULT_MODEL, policy=None):
    pol = _load_policy(policy, band) if policy else None
    env = BipedMultiStepEnv(model_path=model, push_band=band,
                            render_mode=("rgb_array" if render_path else None))
    frames = []
    succ = flat = fell = 0
    nsteps = []
    by_push = {}
    for i in range(n):
        obs, _ = env.reset(seed=1000 + i)
        done = False
        ep_f = []
        info = {}
        while not done:
            if pol:
                import torch
                with torch.no_grad():
                    a, _ = pol[0].predict(pol[1](obs), deterministic=True)
            else:
                a = np.zeros(3, np.float32)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            if render_path and i < 6:
                ep_f.append(env.render())
        succ += int(info.get("success", False))
        flat += int(info.get("flat_ok", False))
        fell += int(info.get("fell", False))
        nsteps.append(info.get("n_steps", 0))
        pk = int(round(info.get("push_n", 0) / 3) * 3)
        by_push.setdefault(pk, []).append((info.get("n_steps", 0), int(info.get("success", False))))
        if render_path and i < 6:
            frames.append(ep_f)
        si = env._steps_info
        chain = " ".join(f"k{s['k']}:sep{s['sep']:.0f},capt{s['capt_fwd']:.0f},v{s['v_fwd']:.2f}" for s in si)
        print(f"  ep {i:2d} push {info.get('push_n', 0):6.1f}  steps {info.get('n_steps', 0)}  "
              f"{'SUCC' if info.get('success') else ('fell' if info.get('fell') else 'fail')}"
              f"{' FLAT' if info.get('flat_ok') else ''}  "
              f"endCapt {info.get('end_capt_fwd_mm', 0):5.1f}  endVf {info.get('end_vfwd', 0):+.2f}  "
              f"endSpd {info.get('end_spd', 0):.2f}  peakUp {info.get('peak_up', 0):4.1f}")
        if si:
            print(f"        chain: {chain}")
    tag = f"POLICY {policy}" if policy else "ZERO-ACTION"
    print(f"\n  {tag}  multi-step  ({n} eps, band {band}, max {env._max_steps}):")
    print(f"    SUCCESS {succ}/{n} = {succ/n:.2f}   flat_ok {flat}/{n}   fell {fell}/{n}")
    print(f"    step-count dist: {np.bincount(nsteps, minlength=env._max_steps+1)[1:].tolist()}  (1..{env._max_steps})")
    for pk in sorted(by_push):
        rows = by_push[pk]
        ms = np.mean([r[0] for r in rows])
        sr = np.mean([r[1] for r in rows])
        print(f"      push ~{pk}N: n={len(rows)}  mean steps {ms:.1f}  success {sr:.2f}")
    if render_path and frames:
        try:
            import imageio.v2 as imageio
            H = max(len(f) for f in frames)
            tiles = []
            for t in range(H):
                row = [f[min(t, len(f) - 1)] for f in frames]
                grid = (np.vstack([np.hstack(row[:3]), np.hstack(row[3:6])])
                        if len(row) >= 6 else np.hstack(row))
                tiles.append(grid)
            imageio.mimsave(render_path, tiles, fps=40)
            print(f"    wrote {render_path}")
        except Exception as e:
            print(f"    render skipped: {e}")
    env.close()


def _mirror_test():
    rng = np.random.default_rng(0)
    bad = sum(np.abs(mirror_obs(mirror_obs(o)) - o).max() > 1e-5
              for o in [rng.normal(size=_OBS_DIM).astype(np.float32) for _ in range(2000)])
    print(f"  mirror_obs involution: {'OK' if bad == 0 else f'{bad} FAIL'}")
    assert list(_MIRROR_PERM[_MIRROR_PERM]) == list(range(15))
    print("  joint PERM involution: OK")
    for tag, ang in (("R", -np.pi / 2 - 0.15), ("L", -np.pi / 2 + 0.15)):
        e = BipedMultiStepEnv(push_band=(134.0, 134.0))
        o, _ = e.reset(options={"push_n": 134.0, "push_dir_rad": ang})
        done = False
        info = {}
        while not done:
            o, r, t1, t2, info = e.step(np.zeros(3, np.float32))
            done = t1 or t2
        print(f"  push-{tag}: steps {info.get('n_steps')}  succ {info.get('success')}  "
              f"endCapt {info.get('end_capt_fwd_mm', 0):.0f}  peakUp {info.get('peak_up', 0):.1f}")
        e.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mirror-test", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--render", default=None)
    ap.add_argument("--band", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args(argv)
    band = tuple(float(x) for x in a.band.split(",")) if a.band else PUSH_BAND_MS
    if a.mirror_test:
        _mirror_test()
    if a.smoke:
        _smoke(a.n, a.render, band, a.model, a.policy)
    if not (a.smoke or a.mirror_test):
        ap.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
