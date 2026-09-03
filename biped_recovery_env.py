"""Gym env for RL push-recovery stepping — v1.

Design (from the assessment; not the old recovery_env.py):
  * The lateral plane is handled by the FRONTAL LIPM REGULATOR (the missing
    scaffold in the 6 failed runs).  The sagittal/posture base is StandingLQR
    with roll+lateral columns zeroed.  The recovery is triggered and sequenced
    by the fixed phase machine from wbtraj_opt (stand -> shift -> swing ->
    descend -> [plant] -> transfer -> settle).
  * reset() fast-forwards stand -> push -> trigger -> LIPM weight-shift under the
    scaffold and hands control to the policy AT SWING ONSET.
  * The POLICY output is a bounded RESIDUAL on the swing-leg (R) joint targets
    (hip/knee/ankle), added to the whip-and-retract REFERENCE trajectory (the
    CMA-optimised 2x-hand solution that already recovers ~69%).  Zero action ==
    that fixed trajectory.  The policy acts during SWING + DESCEND only; on plant
    the env auto-runs transfer + settle and ends the episode.
  * Reward centres on the 2-plane CAPTURE POINT vs. the support polygon, plus
    touchdown quality and a terminal stable-stance bonus.
  * Model: robot/_exp_hands_2x.xml (experimental 2x-hand morphology).
    robot/robot.xml is NOT touched.

    python biped_recovery_env.py --smoke        # zero-action baseline check
    python biped_recovery_env.py --smoke --render out.mp4
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from recovery_metrics import (
    CHEST_BODY, FLOOR_Z, NOMINAL_CHEST_Z, sample_balance,
    _foot_normal_force, _foot_xy_z,
)
from standing_balance_lqr import StandingLQR
from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS, STANDING_QUAT

G = 9.81
DEFAULT_MODEL = "robot/_exp_hands_2x.xml"

# ---- ctrl indices (swing = R) ----
AR_L, AR_R = 9, 14
HIPR_L, HIPR_R = 5, 10
SW_HIP, SW_KNEE, SW_ANKLE = 11, 12, 13
ST_HIP, ST_KNEE, ST_ANKLE = 6, 7, 8
AR_UNLOAD = -1.0            # both-ankle-roll sign that unloads the R foot
KNEE_FLEX_SIGN = -1.0
FWD_HIP_SIGN = 1.0          # R hip flexes + to swing the foot forward

# ---- scaffold constants (frontal LIPM + phase machine; from wbtraj_opt TrajCfg
#      defaults + the CMA-optimised 2x solution) ----
FRONT_POLE_RE, FRONT_POLE_IM = -5.7, 2.3
COP_PER_RAD, AR_BIAS_MAX = 0.62, 0.16
COP_LO_MM, COP_HI_MM, X_MID_MM = 5.0, 96.0, 35.0
X_SS_MM, X_TRANSFER_MM = 73.95, 20.0
SWING_MS, DESCEND_MS = 186, 56
SHIFT_MS, SHIFT_MIN_MS, SWING_START_MS = 135, 122, 110
IMP_AMP, IMP_RAMP_MS, IMP_HOLD_MS = 0.30, 8, 40
TRANSFER_MS, SETTLE_EVAL_MS = 150, 350
SETTLE_BIAS_FLOOR = 0.15      # frontal LIPM bias not fully released in settle
# The recovery is judged at ~SETTLE_EVAL_MS after touchdown: foot planted, both
# feet loaded, torso upright, low CoM speed, held.  "Hold standing forever" is
# the separate marginal-StandingLQR problem and is NOT what this task trains.
PLANT_NF, PLANT_HOLD, STEP_MAX_MS = 12.0, 10, 520
HIPROLL_CAP, STANCE_CAP = 0.06, 0.40
TRIG_CAPT_MM, TRIG_VFWD, TRIG_HOLD, TRIG_DEADLINE_MS = 6.0, 0.05, 12, 1200
PUSH_AT = 20

# ---- reference swing-leg + arm + stance trajectory: CMA-optimised 2x-hand
#      solution (results/wb_followup.json '2.0x'), knots at w=[.35,.70,1.0,1.4] ----
REF_KNOTS = {
    SW_KNEE:  np.array([-0.927,  0.745,  0.607, -0.825]),
    SW_HIP:   np.array([-1.094,  0.509,  0.409, -0.207]),
    SW_ANKLE: np.array([ 0.440, -0.394,  0.382, -0.680]),
    1:        np.array([-0.348, -1.308, -0.975, -1.713]),   # L shoulder
    3:        np.array([ 1.696,  0.944,  1.584,  1.753]),   # R shoulder
    2:        np.array([ 1.596, -0.207, -0.198, -1.240]),   # L elbow
    4:        np.array([ 1.160,  0.991,  1.281, -1.415]),   # R elbow
    ST_HIP:   np.array([-0.109,  0.410,  0.295,  0.048]),   # L hip  (add to LQR)
    ST_ANKLE: np.array([ 0.378,  0.357,  0.367,  0.322]),   # L ankle (add to LQR)
}
REF_SRC = {SW_KNEE: "ss", SW_HIP: "ss", SW_ANKLE: "ss",
           1: "def", 3: "def", 2: "def", 4: "def",
           ST_HIP: "lqr", ST_ANKLE: "lqr"}
_WK = np.array([0.0, 0.35, 0.70, 1.00, 1.40])

# ---- RL ----
FRAME_SKIP = 5                       # -> 200 Hz control
ACT_SCALE = np.array([0.35, 0.35, 0.20])   # rad residual: hip, knee, ankle
PUSH_BAND_STAGE1 = (128.0, 136.0)
BODY_WEIGHT_N = 2.97 * G             # 2x-hand total mass ~2.97 kg

FALL_UPTILT_DEG = 45.0
FALL_CHEST_DROP = 0.22


def _knot_val(kn4, w):
    full = np.concatenate([[0.0], kn4])
    if w <= 0.0:
        return 0.0
    if w >= _WK[-1]:
        return float(full[-1])
    j = int(np.searchsorted(_WK, w)) - 1
    j = max(0, min(j, 3))
    t = (w - _WK[j]) / (_WK[j + 1] - _WK[j])
    return float(full[j] * (1.0 - t) + full[j + 1] * t)


def _frontal_gains(omega2, p_re, p_im):
    mag2 = p_re * p_re + p_im * p_im
    kv = (2.0 * p_re) / omega2
    kx = -(mag2 / omega2) - 1.0
    return kx, kv


def _smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * t))


# rough foot half-extents (m) for the support-polygon reward (66 mm wide, 88 mm long)
_FOOT_HALF_LAT = 0.033
_FOOT_HALF_FWD = 0.044


class BipedRecoveryEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, model_path: str = DEFAULT_MODEL, seed: int | None = None,
                 push_band=PUSH_BAND_STAGE1, dir_spread_deg: float = 0.0,
                 render_mode: str | None = None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._stand = StandingLQR(self.model, self.data, verbose=False)
        self._ss_hip = self._ss_knee = self._ss_ankle = None
        self._build_ss_reference()

        omega2 = G / (NOMINAL_CHEST_Z - FLOOR_Z)
        self.omega = float(np.sqrt(omega2))
        self.kx, self.kv = _frontal_gains(omega2, FRONT_POLE_RE, FRONT_POLE_IM)

        self.push_band = tuple(push_band)
        self.dir_spread = np.radians(dir_spread_deg)
        self.render_mode = render_mode
        self._renderer = None

        self.ctrl_low = self.model.actuator_ctrlrange[:15, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:15, 1].copy()
        self._up_local = np.array([0.0, 1.0, 0.0])
        self._fwd_local = np.array([0.0, 0.0, -1.0])
        self._b_lfoot = self.model.body("L_foot").id
        self._b_rfoot = self.model.body("R_foot").id

        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self._prev_action = np.zeros(3, np.float32)
        # build one obs to size the space
        self._init_episode_state()
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            (self._obs().shape[0],), np.float32)
        if seed is not None:
            self.reset(seed=seed)

    # -------------------------------------------------- ss reference pose
    def _build_ss_reference(self):
        """swing-leg reference joint values at the leaned single-support state,
        via the same quasi-static ankle-roll ramp _SingleSupportLQR uses."""
        from push_step_recovery_test import (
            StepConfig, _LQRAbout, _SingleSupportLQR,
        )
        ab = _LQRAbout(self.model, self.data, DEFAULT_POSE.copy(), tag="ab", verbose=False)
        ss = _SingleSupportLQR(self.model, self.data, ab,
                               StepConfig(swing="R", ankle_roll_amp_rad=0.12,
                                          ss_reach_steps=340), verbose=False)
        self._ss_hip = float(ss.qpos0[7 + SW_HIP])
        self._ss_knee = float(ss.qpos0[7 + SW_KNEE])
        self._ss_ankle = float(ss.qpos0[7 + SW_ANKLE])

    # -------------------------------------------------- cached balance / fall
    def _balance(self):
        key = id(self.data), self.data.time
        if getattr(self, "_bs_key", None) != key:
            self._bs = sample_balance(self.model, self.data)
            self._bs_key = key
        return self._bs

    def _cheap_tilt(self):
        R = self.data.xmat[CHEST_BODY].reshape(3, 3)
        up_z = float((R @ self._up_local)[2])
        return np.degrees(np.arccos(min(1.0, max(-1.0, up_z))))

    def _fallen(self):
        return (self._cheap_tilt() > FALL_UPTILT_DEG
                or self.data.qpos[2] < NOMINAL_CHEST_Z - FALL_CHEST_DROP)

    def _ref_val(self, ci):
        if ci == SW_HIP:
            return self._ss_hip
        if ci == SW_KNEE:
            return self._ss_knee
        if ci == SW_ANKLE:
            return self._ss_ankle
        return 0.0

    # -------------------------------------------------- scaffold control
    def _frontal_bias(self, bs, x_ref_m):
        x = float(bs.com[0])
        v = float(bs.com_vel[0])
        cop = x_ref_m - self.kx * (x - x_ref_m) - self.kv * v
        lo, hi = COP_LO_MM / 1000.0, COP_HI_MM / 1000.0
        cop = float(np.clip(cop, lo, hi))
        return float(np.clip(-(cop - x) / COP_PER_RAD, -AR_BIAS_MAX, AR_BIAS_MAX))

    def _scaffold_ctrl(self, phase, sk, w, residual):
        m, d = self.model, self.data
        bs = self._balance()

        if phase == "shift":
            fr = _smooth(min(1.0, sk / SHIFT_MS))
            x_ref = (X_MID_MM + fr * (X_SS_MM - X_MID_MM)) / 1000.0
        elif phase in ("swing", "descend", "transfer"):
            x_ref = X_SS_MM / 1000.0
        else:  # settle
            x_ref = X_TRANSFER_MM / 1000.0

        if phase == "settle":
            # settle uses the FULL StandingLQR (roll + lateral feedback active) -
            # matches wbtraj_opt.run_traj; the roll-zeroed base below would leave
            # the settle with no roll authority.
            u = np.array(self._stand.control(m, d), float)
            for ci in (SW_HIP, SW_KNEE, SW_ANKLE, ST_HIP, ST_KNEE, ST_ANKLE):
                u[ci] = float(np.clip(u[ci], self._stand.ctrl0[ci] - 0.7,
                                      self._stand.ctrl0[ci] + 0.7))
            decay = SETTLE_BIAS_FLOOR + (1.0 - SETTLE_BIAS_FLOOR) * max(0.0, 1.0 - sk / 500.0)
            bias = self._frontal_bias(bs, x_ref) * decay
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

        if phase == "shift" and sk < IMP_RAMP_MS + IMP_HOLD_MS:
            bias = AR_UNLOAD * IMP_AMP * (sk / IMP_RAMP_MS if sk < IMP_RAMP_MS else 1.0)
        else:
            bias = self._frontal_bias(bs, x_ref)

        if phase in ("swing", "descend"):
            for ci, kn in REF_KNOTS.items():
                dval = _knot_val(kn, w)
                if REF_SRC[ci] == "lqr":
                    u[ci] = u[ci] + dval
                else:
                    u[ci] = self._ref_val(ci) + dval
        elif phase == "transfer":
            b = _smooth(min(1.0, sk / TRANSFER_MS))
            u[SW_HIP] = self._plant_q[0]
            u[SW_KNEE] = self._plant_q[1] + KNEE_FLEX_SIGN * 0.05 * b
            u[SW_ANKLE] = self._plant_q[2] + FWD_HIP_SIGN * 0.05 * b

        if phase in ("swing", "descend", "transfer"):
            u[SW_HIP] += residual[0]
            u[SW_KNEE] += residual[1]
            u[SW_ANKLE] += residual[2]

        u[AR_L] = bias
        u[AR_R] = bias

        if phase in ("swing", "descend", "transfer"):
            u[HIPR_R] = float(np.clip(u[HIPR_R], -HIPROLL_CAP, HIPROLL_CAP))
            for ci in (ST_HIP, ST_KNEE, ST_ANKLE):
                if ci not in REF_KNOTS:
                    u[ci] = float(np.clip(u[ci], -STANCE_CAP, STANCE_CAP))

        return np.clip(u, self.ctrl_low, self.ctrl_high)

    # -------------------------------------------------- episode state
    def _init_episode_state(self):
        self.data.qpos[:] = self._stand.qpos0
        self.data.qvel[:] = self._stand.qvel0
        mujoco.mj_forward(self.model, self.data)
        self._phase = "swing"
        self._sk = 0
        self._w = 0.0
        self._plant_q = np.array([self._ref_val(SW_HIP), self._ref_val(SW_KNEE),
                                  self._ref_val(SW_ANKLE)])
        self._foot_lifted = False
        self._plant_streak = 0
        self._unloaded_streak = 0
        self._v0 = 0.0
        self._swing_p0 = _foot_xy_z(self.model, self.data, "R").copy()
        self._stance_p0 = _foot_xy_z(self.model, self.data, "L").copy()
        self._peak_up = 0.0
        self._touchdown = None
        self._td_bonus_paid = False
        self._prev_action = np.zeros(3, np.float32)
        self._ep_step = 0

    # -------------------------------------------------- obs
    def _chest_axes(self):
        R = self.data.xmat[CHEST_BODY].reshape(3, 3)
        return R @ self._up_local, R @ self._fwd_local

    def _capture_2plane(self):
        """(fwd, lat) capture point relative to the current support polygon edge,
        in mm (>0 => outside => must-catch).  polygon = stance foot pre-td,
        both feet post-td."""
        bs = self._balance()
        h = max(float(bs.com[2]) - FLOOR_Z, 0.05)
        tc = np.sqrt(h / G)
        xi_fwd = -float(bs.com[1]) + (-float(bs.com_vel[1])) * tc
        xi_lat = float(bs.com[0]) + float(bs.com_vel[0]) * tc
        lf = _foot_xy_z(self.model, self.data, "L")
        rf = _foot_xy_z(self.model, self.data, "R")
        post_td = self._touchdown is not None
        feet = [lf, rf] if post_td else [lf]
        fwd_lo = min(-f[1] for f in feet) - _FOOT_HALF_FWD
        fwd_hi = max(-f[1] for f in feet) + _FOOT_HALF_FWD
        lat_lo = min(f[0] for f in feet) - _FOOT_HALF_LAT
        lat_hi = max(f[0] for f in feet) + _FOOT_HALF_LAT
        d_fwd = max(0.0, xi_fwd - fwd_hi, fwd_lo - xi_fwd) * 1000.0
        d_lat = max(0.0, xi_lat - lat_hi, lat_lo - xi_lat) * 1000.0
        return d_fwd, d_lat

    def _obs(self):
        m, d = self.model, self.data
        up, fwd = self._chest_axes()
        bs = self._balance()
        lf = _foot_xy_z(m, d, "L")
        rf = _foot_xy_z(m, d, "R")
        rvel = d.cvel[self._b_rfoot][3:6].copy()
        lnf = _foot_normal_force(m, d, "L")
        rnf = _foot_normal_force(m, d, "R")
        d_fwd, d_lat = self._capture_2plane()
        h_err = float(d.qpos[2]) - NOMINAL_CHEST_Z
        phase_oh = np.array([self._phase == "swing", self._phase == "descend",
                             self._phase in ("transfer", "settle")], np.float32)
        dur = SWING_MS if self._phase == "swing" else DESCEND_MS
        obs = np.concatenate([
            up, fwd[:2],                                        # 5
            d.qvel[3:6], d.qvel[0:3],                           # 6
            [h_err],                                            # 1
            [float(bs.com[0] - lf[0]), float(-bs.com[1] - (-lf[1]))],   # 2 com rel stance
            [float(bs.com_vel[0]), float(-bs.com_vel[1])],      # 2
            [d_fwd / 100.0, d_lat / 100.0],                     # 2 capture pt
            [float(bs.l_contact), float(bs.r_contact),
             lnf / BODY_WEIGHT_N, rnf / BODY_WEIGHT_N],         # 4
            [rf[0] - lf[0], -(rf[1] - lf[1]), rf[2] - lf[2]],   # 3 swing foot rel stance
            rvel,                                               # 3
            d.qpos[7:22], d.qvel[6:21],                         # 30
            phase_oh,                                           # 3
            [min(1.0, self._sk / max(dur, 1))],                 # 1
            self._prev_action,                                  # 3
            [self._v0],                                         # 1
        ]).astype(np.float32)
        return obs

    # -------------------------------------------------- curriculum
    def set_task(self, push_band=None, dir_spread_deg=None):
        if push_band is not None:
            self.push_band = tuple(push_band)
        if dir_spread_deg is not None:
            self.dir_spread = np.radians(dir_spread_deg)
        return (self.push_band, float(np.degrees(self.dir_spread)))

    # -------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        for attempt in range(5):
            ok = self._run_to_swing(options)
            if ok:
                break
        else:
            raise RuntimeError("reset: robot fell before swing onset 5x in a row")
        self._init_after_shift()
        return self._obs(), {"push_n": self._push_n}

    def _run_to_swing(self, options):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self._stand.qpos0
        d.qvel[:] = self._stand.qvel0
        d.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_forward(m, d)
        for _ in range(12):
            d.ctrl[:15] = self._stand.control(m, d)
            mujoco.mj_step(m, d)

        if "push_n" in options:
            self._push_n = float(options["push_n"])
            ang = float(options.get("push_dir_rad", -np.pi / 2))
        else:
            self._push_n = float(self.np_random.uniform(*self.push_band))
            ang = -np.pi / 2 + float(self.np_random.uniform(-self.dir_spread, self.dir_spread))
        fxy = self._push_n * np.array([np.cos(ang), np.sin(ang)])
        push_at = PUSH_AT + int(self.np_random.integers(0, 40))

        phase = "stand"
        sk = 0
        trig_streak = 0
        k = 0
        while k < push_at + PUSH_DURATION_STEPS + TRIG_DEADLINE_MS + 400:
            d.xfrc_applied[CHEST_BODY, :] = 0.0
            if push_at <= k < push_at + PUSH_DURATION_STEPS:
                d.xfrc_applied[CHEST_BODY, 0:2] = fxy
            post = k > push_at + PUSH_DURATION_STEPS
            if phase == "stand":
                u = self._stand.control(m, d)
            else:  # shift
                u = self._scaffold_ctrl("shift", sk, 0.0, np.zeros(3))
            d.ctrl[:15] = u
            mujoco.mj_step(m, d)
            k += 1
            if phase != "stand":
                sk += 1
            if self._fallen():
                return False
            bs = self._balance()
            if phase == "stand":
                if post and bs.capture_fwd_rel_support_mm > TRIG_CAPT_MM and bs.com_vfwd > TRIG_VFWD:
                    trig_streak += 1
                else:
                    trig_streak = 0
                if trig_streak >= TRIG_HOLD:
                    phase = "shift"
                    sk = 0
                    mujoco.mj_subtreeVel(m, d)
                    self._v0 = float(np.hypot(d.subtree_linvel[CHEST_BODY][0],
                                              d.subtree_linvel[CHEST_BODY][1]))
                elif post and k > push_at + PUSH_DURATION_STEPS + TRIG_DEADLINE_MS:
                    return False        # LQR held it -> no step needed; skip episode
            elif phase == "shift":
                sw_nf = _foot_normal_force(m, d, "R")
                if (sk >= SHIFT_MIN_MS and (sw_nf < 6.0 or sk >= SWING_START_MS)
                        and (abs(bs.pitch_rate) < 0.5 or sk >= SWING_START_MS)):
                    return True
        return False

    def _init_after_shift(self):
        self._phase = "swing"
        self._sk = 0
        self._w = 0.0
        self._foot_lifted = False
        self._plant_streak = 0
        self._unloaded_streak = 0
        self._peak_up = float(self._balance().up_tilt_deg)
        self._touchdown = None
        self._td_bonus_paid = False
        self._prev_action = np.zeros(3, np.float32)
        self._ep_step = 0
        self._swing_p0 = _foot_xy_z(self.model, self.data, "R").copy()
        self._stance_p0 = _foot_xy_z(self.model, self.data, "L").copy()
        self._plant_q = np.array([self._ref_val(SW_HIP), self._ref_val(SW_KNEE),
                                  self._ref_val(SW_ANKLE)])

    def step(self, action):
        m, d = self.model, self.data
        action = np.asarray(action, np.float32).clip(-1.0, 1.0)
        residual = action * ACT_SCALE
        self._ep_step += 1

        r_shape = 0.0
        fell = False
        entered_transfer = False
        for _ in range(FRAME_SKIP):
            if self._phase == "swing":
                self._w = min(1.0, self._sk / SWING_MS)
            elif self._phase == "descend":
                self._w = 1.0 + self._sk / DESCEND_MS
            u = self._scaffold_ctrl(self._phase, self._sk, self._w, residual)
            d.ctrl[:15] = u
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                fell = True
                break
            bs = self._balance()
            sw_nf = _foot_normal_force(m, d, "R")

            # capture-point shaping (dense, small)
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
                genuine = (self._foot_lifted and bs.r_contact and sw_nf > PLANT_NF)
                self._plant_streak = self._plant_streak + 1 if genuine else 0
                if self._plant_streak >= PLANT_HOLD or self._sk >= STEP_MAX_MS:
                    self._snapshot_touchdown(bs, sw_nf)
                    self._phase = "transfer"
                    self._sk = 0
                    entered_transfer = True
                    break

        self._prev_action = action.copy()

        if fell:
            obs = self._obs()
            return obs, r_shape - 20.0, True, False, self._info(fell=True)

        if entered_transfer:
            # auto-run transfer + settle, then terminate with the terminal reward
            term_r, info = self._auto_finish()
            obs = self._obs()
            return obs, r_shape + self._touchdown_reward() + term_r, True, False, info

        # safety truncation (shouldn't happen: swing+descend is bounded)
        if self._ep_step > 400:
            return self._obs(), r_shape, False, True, self._info(fell=False)

        return self._obs(), r_shape + 0.3, False, False, self._info(fell=False)

    def _snapshot_touchdown(self, bs, sw_nf):
        sf = _foot_xy_z(self.model, self.data, "R")
        stf = _foot_xy_z(self.model, self.data, "L")
        self._plant_q = np.array([self.data.qpos[7 + SW_HIP],
                                  self.data.qpos[7 + SW_KNEE],
                                  self.data.qpos[7 + SW_ANKLE]])
        from step_primitive import _sole_pitch
        self._touchdown = dict(
            planted=self._plant_streak >= PLANT_HOLD,
            sep_mm=float(-(sf[1] - stf[1]) * 1000.0),
            lat_mm=float((sf[0] - self._swing_p0[0]) * 1000.0),
            sole_deg=float(np.degrees(_sole_pitch(self.model, self.data, "R"))),
            nf=float(sw_nf),
            side=float(bs.side_lean_deg),
            vz=float(self.data.cvel[self._b_rfoot][5]),
            unloaded_ok=self._unloaded_streak >= 15,
        )

    def _touchdown_reward(self):
        if self._td_bonus_paid or self._touchdown is None:
            return 0.0
        self._td_bonus_paid = True
        td = self._touchdown
        r = 0.0
        r += 2.0 * float(np.clip(1.0 - abs(td["sole_deg"]) / 15.0, 0.0, 1.0))
        r += 1.5 * float(np.clip(td["nf"] / 20.0, 0.0, 1.0))
        r += -0.6 * float(np.clip(abs(td["vz"]) / 0.35, 0.0, 1.0))
        r += 0.5 if td["unloaded_ok"] else -0.5
        r += 1.2 * float(np.clip((td["sep_mm"] - 20.0) / 40.0, 0.0, 1.0))
        return r

    def _auto_finish(self):
        m, d = self.model, self.data
        # transfer
        for _ in range(TRANSFER_MS):
            u = self._scaffold_ctrl("transfer", self._sk, 0.0, np.zeros(3))
            d.ctrl[:15] = u
            mujoco.mj_step(m, d)
            self._sk += 1
            if self._fallen():
                self._peak_up = max(self._peak_up, 55.0)
                return -20.0, self._info(fell=True)
        # settle
        self._phase = "settle"
        self._sk = 0
        for _ in range(SETTLE_EVAL_MS):
            u = self._scaffold_ctrl("settle", self._sk, 0.0, np.zeros(3))
            d.ctrl[:15] = u
            mujoco.mj_step(m, d)
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                return -20.0, self._info(fell=True)
        b = sample_balance(m, d)
        ds = b.l_contact and b.r_contact
        stable = (ds and b.up_tilt_deg < 8.0 and abs(b.side_lean_deg) < 8.0
                  and b.com_speed_horiz < 0.10 and b.chest_z > NOMINAL_CHEST_Z - 0.10
                  and self._peak_up <= 35.0)
        success = stable and self._touchdown is not None and self._touchdown["planted"] \
            and self._touchdown["sep_mm"] >= 15.0
        r = 0.0
        r += 8.0 if success else 0.0
        r += 2.0 * float(np.clip(1.0 - b.up_tilt_deg / 10.0, 0.0, 1.0))
        r += 1.5 * float(np.clip(1.0 - abs(b.side_lean_deg) / 10.0, 0.0, 1.0))
        r += 1.0 * float(np.clip(1.0 - b.com_speed_horiz / 0.2, 0.0, 1.0))
        r += 1.0 if ds else -1.0
        info = self._info(fell=False)
        info["success"] = bool(success)
        info["end_sep_mm"] = float(self._touchdown["sep_mm"]) if self._touchdown else 0.0
        info["end_up"] = float(b.up_tilt_deg)
        info["end_side"] = float(b.side_lean_deg)
        return r, info

    def _info(self, fell):
        info = dict(push_n=self._push_n, phase=self._phase,
                    peak_up=float(self._peak_up), v0=float(self._v0),
                    fell=bool(fell), success=False,
                    end_sep_mm=(float(self._touchdown["sep_mm"])
                                if self._touchdown else 0.0))
        return info

    # -------------------------------------------------- render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 240, 320)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ============================================================ smoke test
def _smoke(n=40, render_path=None, band=PUSH_BAND_STAGE1, model=DEFAULT_MODEL):
    env = BipedRecoveryEnv(model_path=model, push_band=band,
                           render_mode=("rgb_array" if render_path else None))
    frames = []
    succ = 0
    seps = []
    peak_ups = []
    n_ep = 0
    for i in range(n):
        obs, _ = env.reset(seed=1000 + i)
        done = False
        ep_frames = []
        while not done:
            obs, r, term, trunc, info = env.step(np.zeros(3, np.float32))
            done = term or trunc
            if render_path and i < 6:
                ep_frames.append(env.render())
        n_ep += 1
        if info.get("success"):
            succ += 1
        seps.append(info.get("end_sep_mm", 0.0))
        peak_ups.append(info.get("peak_up", 99.0))
        if render_path and i < 6:
            frames.append(ep_frames)
        print(f"  ep {i:2d}  push {info['push_n']:6.1f}  v0 {info['v0']:.3f}  "
              f"{'SUCCESS' if info.get('success') else 'fail   '}  "
              f"sep {info.get('end_sep_mm', 0):5.1f}  peakUp {info.get('peak_up', 0):5.1f}  "
              f"endUp {info.get('end_up', -1):5.1f}")
    print(f"\n  ZERO-ACTION baseline over {n_ep} eps (band {band}, model {model}):")
    print(f"    success {succ}/{n_ep} = {succ / n_ep:.2f}")
    print(f"    median sep {np.median(seps):.1f} mm   median peak up-tilt {np.median(peak_ups):.1f} deg")
    if render_path and frames:
        try:
            import imageio.v2 as imageio
            H = max(len(f) for f in frames)
            tiles = []
            for t in range(H):
                row = []
                for f in frames:
                    row.append(f[min(t, len(f) - 1)])
                grid = np.vstack([np.hstack(row[:3]), np.hstack(row[3:6])]) \
                    if len(row) >= 6 else np.hstack(row)
                tiles.append(grid)
            imageio.mimsave(render_path, tiles, fps=40)
            print(f"    wrote {render_path}")
        except Exception as e:
            print(f"    render skipped: {e}")
    env.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--render", default=None)
    ap.add_argument("--band", default=None, help="lo,hi")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args(argv)
    band = tuple(float(x) for x in a.band.split(",")) if a.band else PUSH_BAND_STAGE1
    if a.smoke:
        _smoke(a.n, a.render, band, a.model)


if __name__ == "__main__":
    main(sys.argv[1:])
