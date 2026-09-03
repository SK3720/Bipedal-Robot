"""Gym env for Push -> Step 1 (R) -> Step 2 (L) -> stable  (2-step milestone).

Clean rewrite (v2).  Extends the Stage-1 recovery env; the policy is reused in a
CANONICAL "swing = R" frame for BOTH steps:
  * step 1  : identical to Stage 1 (swing R, identity mirror).
  * bridge  : transfer + full-LQR settle onto the R foot, then FLIP to a mirrored
              canonical-L step + a short mirrored weight-shift onto R.
  * step 2  : swing L, but the policy still "thinks" swing R — `mirror_obs()`
              sagittally reflects the 66-d obs, `mirror_action()` reflects the
              3-d residual back onto the real L swing leg.
  * finish  : transfer + settle, verbatim from Stage 1 (its settle is proven).

Success is reframed for what this short-footed robot can actually do mid-gait:
TWO CONTROLLED ALTERNATING FORWARD STEPS that stay upright and don't run away —
not a dead stop (that needs a step 3 or a dedicated halting controller;
`clean_stop` tracks it when it happens).

Model: robot/_exp_hands_2x.xml.  robot/robot.xml is NOT touched.

    python biped_2step_env.py --smoke
    python biped_2step_env.py --mirror-test
    python biped_2step_env.py --smoke --render out.mp4
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import mujoco
from gymnasium import spaces

from recovery_metrics import (
    FLOOR_Z, NOMINAL_CHEST_Z, sample_balance, _foot_normal_force, _foot_xy_z,
)
import biped_recovery_env as S1
from biped_recovery_env import (
    BipedRecoveryEnv, DEFAULT_MODEL, G,
    AR_L, AR_R, HIPR_L, HIPR_R, REF_KNOTS, REF_SRC,
    COP_PER_RAD, AR_BIAS_MAX, COP_LO_MM, COP_HI_MM, X_MID_MM, X_SS_MM, X_TRANSFER_MM,
    SWING_MS, DESCEND_MS, SHIFT_MS, IMP_AMP, IMP_RAMP_MS, IMP_HOLD_MS,
    TRANSFER_MS, SETTLE_EVAL_MS, SETTLE_BIAS_FLOOR,
    PLANT_NF, PLANT_HOLD, STEP_MAX_MS, HIPROLL_CAP, STANCE_CAP,
    FRAME_SKIP, ACT_SCALE, BODY_WEIGHT_N, _knot_val, _smooth,
    _FOOT_HALF_LAT, _FOOT_HALF_FWD,
)
from step_primitive import _sole_pitch

# ---------------------------------------------------------------- mirror maps
# ctrl index i  <->  qpos[7+i], qvel[6+i].  Sagittal reflection about the robot
# centreline.  PERM is an involution; SIGN from the joint axes.
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
    """Medial-lateral sole tilt (rad), 0 == flat.  Foot body local x = sole normal."""
    R = data.xmat[model.body(f"{side}_foot").id].reshape(3, 3)
    return float(np.arcsin(np.clip(R[0, 0], -1.0, 1.0)))


# ---------------------------------------------------------------- 2-step tuning
BRIDGE_SETTLE_MS = 190     # full-LQR settle after step 1 before the flip
SHIFT2_MS = 95
SHIFT2_MIN_MS = 58
SWING2_START_MS = 92
SETTLE2_MS = 300
STAGGER_OK_MM = -18.0      # step 2 must bring L at least ~level with R
W_FWD_PROGRESS = 1.5
PUSH_BAND_2STEP = (132.0, 138.0)


class Biped2StepEnv(BipedRecoveryEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, model_path=DEFAULT_MODEL, seed=None,
                 push_band=PUSH_BAND_2STEP, dir_spread_deg=0.0, render_mode=None):
        super().__init__(model_path=model_path, seed=None, push_band=push_band,
                         dir_spread_deg=dir_spread_deg, render_mode=render_mode)
        # L swing reference = sagittal mirror of the R one (SS-LQR(swing="L")
        # converges to a saturated pose; only the pose is used downstream anyway).
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
        self._reset_2step_flags()
        if seed is not None:
            self.reset(seed=seed)

    # -------------------------------------------------- lifecycle
    def _reset_2step_flags(self):
        self._swing, self._stance, self._lat, self._step = "R", "L", 1.0, 1
        self._td = {}
        self._td_paid = set()
        self._ds_hold = 0
        self._step1_only = False
        self._rhold = self._lhold = None

    def _init_episode_state(self):
        self._reset_2step_flags()
        super()._init_episode_state()

    def reset(self, seed=None, options=None):
        self._reset_2step_flags()
        return super().reset(seed=seed, options=options)

    def _init_after_shift(self):
        super()._init_after_shift()
        self._reset_2step_flags()

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
        elif phase == "shift2":
            fr = _smooth(min(1.0, sk / SHIFT2_MS))
            x_ref = (X_MID_MM + fr * (X_SS_MM - X_MID_MM)) / 1000.0
        elif phase in ("swing", "descend", "transfer"):
            x_ref = X_SS_MM / 1000.0
        else:  # settle
            x_ref = X_TRANSFER_MM / 1000.0

        # ---- settle: FULL StandingLQR + decaying frontal bias  (Stage-1 verbatim) ----
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

        # ---- roll-zeroed LQR base ----
        dq = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, self._stand.qpos0, d.qpos)
        dx = np.concatenate([dq, d.qvel - self._stand.qvel0])
        for ci in (AR_L, AR_R, HIPR_L, HIPR_R):
            dx[6 + ci] = 0.0
            dx[m.nv + 6 + ci] = 0.0
        dx[0] = dx[m.nv + 0] = 0.0
        u = np.array(self._stand.ctrl0 - self._stand.K @ dx, float)

        # ---- lateral bias ----
        unload_sign = -1.0 if s == "R" else 1.0
        imp_amp = IMP_AMP if phase == "shift" else 0.11
        if phase in ("shift", "shift2") and sk < IMP_RAMP_MS + IMP_HOLD_MS:
            ramp = sk / IMP_RAMP_MS if sk < IMP_RAMP_MS else 1.0
            bias = unload_sign * imp_amp * ramp
        else:
            bias = self._frontal_bias(bs, x_ref, lat)

        # ---- swing-leg reference + residual ----
        if phase in ("swing", "descend"):
            for ci, kn in ref_knots.items():
                dval = _knot_val(kn, w)
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

        # ---- step 2: keep the already-planted R leg from being unwound by the
        #      feet-together LQR.  Tight during the brief shift2, loose (hip+knee
        #      only, R ankle free for balance) during the L step. ----
        if self._step == 2:
            rh, lh = self._rhold, self._lhold
            if phase == "shift2":
                if rh is not None:
                    for k, ci in enumerate((c["st_hip"], c["st_knee"], c["st_ankle"])):
                        u[ci] = float(np.clip(u[ci], rh[k] - 0.13, rh[k] + 0.13))
                if lh is not None:
                    for k, ci in enumerate((c["sw_hip"], c["sw_knee"], c["sw_ankle"])):
                        u[ci] = float(np.clip(u[ci], lh[k] - 0.13, lh[k] + 0.13))
            elif phase in ("swing", "descend", "transfer") and rh is not None:
                for k, ci in enumerate((c["st_hip"], c["st_knee"])):
                    u[ci] = float(np.clip(u[ci], rh[k] - 0.30, rh[k] + 0.30))

        if phase in ("swing", "descend", "transfer"):
            u[c["hipr_sw"]] = float(np.clip(u[c["hipr_sw"]], -HIPROLL_CAP, HIPROLL_CAP))
            for ci in (c["st_hip"], c["st_knee"], c["st_ankle"]):
                if ci not in ref_knots:
                    u[ci] = float(np.clip(u[ci], -STANCE_CAP, STANCE_CAP))

        return np.clip(u, self.ctrl_low, self.ctrl_high)

    # -------------------------------------------------- capture point (role-aware)
    def _capture_2plane(self):
        bs = self._balance()
        h = max(float(bs.com[2]) - FLOOR_Z, 0.05)
        tc = np.sqrt(h / G)
        xi_fwd = -float(bs.com[1]) + (-float(bs.com_vel[1])) * tc
        xi_lat = float(bs.com[0]) + float(bs.com_vel[0]) * tc
        stf = _foot_xy_z(self.model, self.data, self._stance)
        swf = _foot_xy_z(self.model, self.data, self._swing)
        feet = [stf, swf] if self._td.get(self._step) is not None else [stf]
        fwd_lo = min(-f[1] for f in feet) - _FOOT_HALF_FWD
        fwd_hi = max(-f[1] for f in feet) + _FOOT_HALF_FWD
        lat_lo = min(f[0] for f in feet) - _FOOT_HALF_LAT
        lat_hi = max(f[0] for f in feet) + _FOOT_HALF_LAT
        d_fwd = max(0.0, xi_fwd - fwd_hi, fwd_lo - xi_fwd) * 1000.0
        d_lat = max(0.0, xi_lat - lat_hi, lat_lo - xi_lat) * 1000.0
        return d_fwd, d_lat

    # -------------------------------------------------- obs (role-aware + mirror)
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
                             self._phase in ("transfer", "settle", "shift2")], np.float32)
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

        if plant_now and self._step == 1:
            td_r = self._touchdown_reward(1)
            broke = self._bridge_to_step2()
            if broke is not None:
                return self._obs(), r_shape + td_r + broke, True, False, self._info(fell=self._step1_only is False)
            return self._obs(), r_shape + td_r + 0.3, False, False, self._info(fell=False)

        if plant_now and self._step == 2:
            td_r = self._touchdown_reward(2)
            term_r, info = self._auto_finish()
            return self._obs(), r_shape + td_r + term_r, True, False, info

        if self._ep_step > 600:
            return self._obs(), r_shape, False, True, self._info(fell=False)
        return self._obs(), r_shape + 0.3, False, False, self._info(fell=False)

    # -------------------------------------------------- bridge
    def _bridge_to_step2(self):
        m, d = self.model, self.data
        self._phase = "transfer"
        self._sk = 0
        for _ in range(TRANSFER_MS):
            d.ctrl[:15] = self._scaffold_ctrl("transfer", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                return -20.0
        self._phase = "settle"
        self._sk = 0
        for _ in range(BRIDGE_SETTLE_MS):
            d.ctrl[:15] = self._scaffold_ctrl("settle", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                return -20.0
        b = sample_balance(m, d)
        if not (b.r_contact and b.up_tilt_deg < 9.0 and abs(b.side_lean_deg) < 9.0
                and b.com_speed_horiz < 0.16):
            self._step1_only = True
            return -3.0
        # FLIP
        self._swing, self._stance, self._lat, self._step = "L", "R", -1.0, 2
        self._sk = 0
        self._w = 0.0
        self._foot_lifted = False
        self._plant_streak = 0
        self._unloaded_streak = 0
        self._prev_action = np.zeros(3, np.float32)
        self._swing_p0 = _foot_xy_z(m, d, "L").copy()
        self._stance_p0 = _foot_xy_z(m, d, "R").copy()
        self._rhold = np.array([d.qpos[7 + 11], d.qpos[7 + 12], d.qpos[7 + 13]])
        self._lhold = np.array([d.qpos[7 + 6], d.qpos[7 + 7], d.qpos[7 + 8]])
        cL = LEG_CTRL["L"]
        self._plant_q = np.array([self._ref_val(cL["sw_hip"]),
                                  self._ref_val(cL["sw_knee"]),
                                  self._ref_val(cL["sw_ankle"])])
        self._phase = "shift2"
        for _ in range(SHIFT2_MS + 45):
            d.ctrl[:15] = self._scaffold_ctrl("shift2", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                return -20.0
            sw_nf = _foot_normal_force(m, d, "L")
            b = self._balance()
            if (self._sk >= SHIFT2_MIN_MS
                    and (sw_nf < 8.0 or self._sk >= SWING2_START_MS)
                    and (abs(b.pitch_rate) < 0.6 or self._sk >= SWING2_START_MS)):
                break
        self._phase = "swing"
        self._sk = 0
        self._w = 0.0
        return None

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
        self._td[self._step] = dict(
            planted=self._plant_streak >= PLANT_HOLD,
            sep_mm=float(-(sf[1] - stf[1]) * 1000.0),
            sole_pitch=float(np.degrees(_sole_pitch(self.model, self.data, sw))),
            sole_roll=float(np.degrees(_sole_roll(self.model, self.data, sw))),
            nf=float(sw_nf),
            vz=float(self.data.cvel[swb][5]),
            unloaded_ok=self._unloaded_streak >= 15,
        )
        self._touchdown = self._td[self._step]

    def _touchdown_reward(self, step):
        if step in self._td_paid or step not in self._td:
            return 0.0
        self._td_paid.add(step)
        td = self._td[step]
        r = 1.0 * float(np.clip(1.0 - abs(td["sole_pitch"]) / 12.0, 0.0, 1.0))
        r += 0.5 * float(np.clip(1.0 - abs(td["sole_roll"]) / 12.0, 0.0, 1.0))
        r += 1.5 * float(np.clip(td["nf"] / 20.0, 0.0, 1.0))
        r += -0.6 * float(np.clip(abs(td["vz"]) / 0.35, 0.0, 1.0))
        r += 0.5 if td["unloaded_ok"] else -0.5
        if step == 2:
            r += 2.4 * float(np.clip((td["sep_mm"] - 15.0) / 55.0, 0.0, 1.0))
        else:
            r += 1.2 * float(np.clip((td["sep_mm"] - 20.0) / 40.0, 0.0, 1.0))
        return r

    # -------------------------------------------------- finish
    def _auto_finish(self):
        m, d = self.model, self.data
        self._phase = "transfer"
        self._sk = 0
        for _ in range(TRANSFER_MS):
            d.ctrl[:15] = self._scaffold_ctrl("transfer", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                self._peak_up = max(self._peak_up, 55.0)
                return -20.0, self._info(fell=True)
        self._phase = "settle"
        self._sk = 0
        spd_hist = []
        for _ in range(SETTLE2_MS):
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
        lf = _foot_xy_z(m, d, "L")
        rf = _foot_xy_z(m, d, "R")
        stagger_mm = float((-lf[1]) - (-rf[1]))
        sp_l = abs(np.degrees(_sole_pitch(m, d, "L")))
        sp_r = abs(np.degrees(_sole_pitch(m, d, "R")))
        sr_l = abs(np.degrees(_sole_roll(m, d, "L")))
        sr_r = abs(np.degrees(_sole_roll(m, d, "R")))
        flat = 0.25 * (np.clip(1 - sp_l / 8, 0, 1) + np.clip(1 - sp_r / 8, 0, 1)
                       + np.clip(1 - sr_l / 8, 0, 1) + np.clip(1 - sr_r / 8, 0, 1))
        com_fwd = -float(b.com[1])
        front = max(-lf[1], -rf[1]) + _FOOT_HALF_FWD
        back = min(-lf[1], -rf[1]) - _FOOT_HALF_FWD
        span = max(front - back, 1e-3)
        com_support = float(np.clip(0.5 + min(front - com_fwd, com_fwd - back) / span, 0.0, 1.0))
        n3 = max(1, len(spd_hist) // 3)
        spd_early = float(np.mean(spd_hist[:n3]))
        spd_late = float(np.mean(spd_hist[-n3:]))
        decel = spd_late <= spd_early + 0.03

        planted1 = self._td.get(1, {}).get("planted", False)
        planted2 = self._td.get(2, {}).get("planted", False)
        sep2 = self._td.get(2, {}).get("sep_mm", 0.0)

        upright = (b.up_tilt_deg < 16.0 and abs(b.side_lean_deg) < 13.0
                   and b.chest_z > NOMINAL_CHEST_Z - 0.13 and self._peak_up <= 30.0)
        bounded = (spd_late < 0.55 and (decel or spd_late < 0.30))
        success = (planted1 and planted2 and upright and bounded
                   and stagger_mm >= STAGGER_OK_MM and ds)
        clean_stop = (success and spd_late < 0.16 and b.up_tilt_deg < 9.0
                      and max(sp_l, sp_r) < 16.0 and max(sr_l, sr_r) < 16.0)

        r = 0.0
        r += 10.0 if success else 0.0
        r += 4.0 if clean_stop else 0.0
        r += 2.0 * float(np.clip(1.0 - b.up_tilt_deg / 12.0, 0.0, 1.0))
        r += 1.5 * float(np.clip(1.0 - abs(b.side_lean_deg) / 12.0, 0.0, 1.0))
        r += 2.0 * float(np.clip(1.0 - spd_late / 0.35, 0.0, 1.0))
        r += 1.5 if ds else -1.0
        r += 1.5 * float(flat)
        r += 1.5 * com_support
        r += 1.2 * float(np.clip((stagger_mm + 25.0) / 60.0, 0.0, 1.0))
        if success:
            r += W_FWD_PROGRESS * float(np.clip((stagger_mm + 15.0) / 55.0, 0.0, 1.0))

        info = self._info(fell=False)
        info.update(success=bool(success), clean_stop=bool(clean_stop),
                    end_sep_mm=float(sep2), end_stagger_mm=float(stagger_mm),
                    end_up=float(b.up_tilt_deg), end_side=float(b.side_lean_deg),
                    end_spd=float(spd_late), sole_pitch_max=float(max(sp_l, sp_r)),
                    sole_roll_max=float(max(sr_l, sr_r)), com_support=float(com_support),
                    planted1=bool(planted1), planted2=bool(planted2),
                    sep1_mm=float(self._td.get(1, {}).get("sep_mm", 0.0)))
        return r, info

    def _info(self, fell):
        info = super()._info(fell)
        info["step_reached"] = int(self._step)
        info["step1_only"] = bool(getattr(self, "_step1_only", False))
        info.setdefault("end_stagger_mm", 0.0)
        info.setdefault("sole_pitch_max", 99.0)
        info.setdefault("sole_roll_max", 99.0)
        info.setdefault("com_support", 0.0)
        info.setdefault("clean_stop", False)
        info.setdefault("end_spd", 9.9)
        info["sep1_mm"] = float(self._td.get(1, {}).get("sep_mm", 0.0)) if self._td else 0.0
        info["planted1"] = bool(self._td.get(1, {}).get("planted", False)) if self._td else False
        info["planted2"] = bool(self._td.get(2, {}).get("planted", False)) if self._td else False
        return info


# ============================================================ smoke / mirror test
def _smoke(n=30, render_path=None, band=PUSH_BAND_2STEP, model=DEFAULT_MODEL, policy=None):
    from_policy = None
    if policy:
        import torch
        import json
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        hp = json.load(open(f"{policy}/hparams.json"))
        tmp = DummyVecEnv([lambda: Biped2StepEnv(push_band=band)])
        vn = VecNormalize.load(f"{policy}/vecnormalize.pkl", tmp)
        mean, var = vn.obs_rms.mean, vn.obs_rms.var
        mm = PPO("MlpPolicy", tmp, device="cpu",
                 policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
        mm.policy.load_state_dict(torch.load(f"{policy}/policy.pth", map_location="cpu", weights_only=True))
        mm.policy.eval()
        from_policy = (mm, lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32))

    env = Biped2StepEnv(model_path=model, push_band=band,
                        render_mode=("rgb_array" if render_path else None))
    frames, succ, clean, s1o, fell, p1, p2 = [], 0, 0, 0, 0, 0, 0
    seps2, stags, ups, spds = [], [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=1000 + i)
        done = False
        ep_f = []
        info = {}
        while not done:
            if from_policy:
                import torch
                with torch.no_grad():
                    a, _ = from_policy[0].predict(from_policy[1](obs), deterministic=True)
            else:
                a = np.zeros(3, np.float32)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            if render_path and i < 6:
                ep_f.append(env.render())
        succ += int(info.get("success", False))
        clean += int(info.get("clean_stop", False))
        s1o += int(info.get("step1_only", False))
        fell += int(info.get("fell", False))
        p1 += int(info.get("planted1", False))
        p2 += int(info.get("planted2", False))
        seps2.append(info.get("end_sep_mm", 0.0))
        stags.append(info.get("end_stagger_mm", 0.0))
        ups.append(info.get("peak_up", 99.0))
        spds.append(info.get("end_spd", 9.9))
        if render_path and i < 6:
            frames.append(ep_f)
        print(f"  ep {i:2d}  push {info.get('push_n', 0):6.1f}  "
              f"{'SUCC' if info.get('success') else ('S1  ' if info.get('step1_only') else 'fail')}"
              f"{' CLEAN' if info.get('clean_stop') else '     '}  "
              f"sep1 {info.get('sep1_mm', 0):5.1f}  sep2 {info.get('end_sep_mm', 0):+6.1f}  "
              f"stag {info.get('end_stagger_mm', 0):+6.1f}  peakUp {info.get('peak_up', 0):4.1f}  "
              f"endSpd {info.get('end_spd', 0):.2f}")
    tag = f"POLICY {policy}" if policy else "ZERO-ACTION"
    print(f"\n  {tag} 2-step over {n} eps (band {band}):")
    print(f"    plant1 {p1}/{n}  plant2 {p2}/{n}  step1_only {s1o}  fell {fell}")
    print(f"    SUCCESS {succ}/{n} = {succ/n:.2f}   clean_stop {clean}/{n}")
    print(f"    median sep2 {np.median(seps2):+.1f}  stagger {np.median(stags):+.1f}  "
          f"peakUp {np.median(ups):.1f}  endSpd {np.median(spds):.2f}")
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
    ok = all(np.abs(mirror_obs(mirror_obs(rng.normal(size=_OBS_DIM).astype(np.float32)))
                    - o).max() < 1e-5 for o in [rng.normal(size=_OBS_DIM).astype(np.float32)])
    bad = 0
    for _ in range(2000):
        o = rng.normal(size=_OBS_DIM).astype(np.float32)
        if np.abs(mirror_obs(mirror_obs(o)) - o).max() > 1e-5:
            bad += 1
    print(f"  mirror_obs involution: {'OK' if bad == 0 else f'{bad} FAIL'}")
    assert list(_MIRROR_PERM[_MIRROR_PERM]) == list(range(15))
    print("  joint PERM involution: OK")
    # physics: mirrored-direction pushes, zero action, compare
    for tag, ang in (("R", -np.pi / 2 - 0.16), ("L", -np.pi / 2 + 0.16)):
        e = Biped2StepEnv(push_band=(134.0, 134.0))
        o, _ = e.reset(options={"push_n": 133.0, "push_dir_rad": ang})
        done = False
        info = {}
        while not done:
            o, r, t1, t2, info = e.step(np.zeros(3, np.float32))
            done = t1 or t2
        print(f"  push-{tag}: step->{info.get('step_reached')} sep1 {info.get('sep1_mm', 0):.0f} "
              f"sep2 {info.get('end_sep_mm', 0):+.0f} peakUp {info.get('peak_up', 0):.1f} "
              f"succ {info.get('success')}")
        e.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mirror-test", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--render", default=None)
    ap.add_argument("--band", default=None)
    ap.add_argument("--policy", default=None, help="run dir to load policy from")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args(argv)
    band = tuple(float(x) for x in a.band.split(",")) if a.band else PUSH_BAND_2STEP
    if a.mirror_test:
        _mirror_test()
    if a.smoke:
        _smoke(a.n, a.render, band, a.model, a.policy)
    if not (a.smoke or a.mirror_test):
        ap.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
