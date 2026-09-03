"""Full-authority RL env for REACTIVE SEQUENTIAL PUSH RECOVERY (route to walking).

The scaffold architecture (StandingLQR base + fixed swing ref + 3-joint residual)
gave a robust ONE-step recovery but cannot chain: the feet-together LQR is only
valid near standing, and every step passes through pitched / staggered /
single-support states it destabilises (see memory/two-step-milestone.md).

Here the POLICY has full authority over all 10 leg joints (position residual on a
minimal phase-indexed reference).  There is NO base stabiliser during the policy
phase -- the policy learns torso righting, lateral balance, momentum modulation
and the step transitions itself.  StandingLQR is used ONLY to hold the robot
standing and absorb the push impulse before hand-off.

  * ACTION (10): bounded position residual on the L/R hip-roll, hip-pitch, knee,
    ankle-pitch, ankle-roll targets, in a CANONICAL "swing = R" frame (obs
    sagittally mirrored, action mirrored back).  Zero action == the reference.
  * REFERENCE (minimal): stance leg held near a leaned neutral; swing leg a
    forward arc scaled by the sagittal capture-point excess; both ankle-rolls
    carry the frontal-LIPM CoP command (a feedback law, not a scripted motion).
    Arms + neck are servo'd to neutral (ctrl 0) -- passive 2x-hand inertia only,
    NO scripted upper-body.
  * A scripted 2-plane capture-point trigger decides WHEN to start each step and
    WHICH foot (alternating); the policy does everything else.  Steps continue
    until the capture point is caught or MAX_STEPS.
  * REWARD (not softened): alive + upright + drive the capture point into support
    + penalise excess CoM speed + effort/smoothness + a decaying reference-track
    term + per-step cost + touchdown quality; terminal +big for a genuine caught
    & stable state, -20 for a fall.

Model: robot/_exp_hands_2x.xml.  robot/robot.xml is NOT touched.

    python biped_walk_env.py --smoke
    python biped_walk_env.py --smoke --render walk_zero.mp4
    python biped_walk_env.py --mirror-test
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
import gymnasium as gym
from gymnasium import spaces

from recovery_metrics import (
    CHEST_BODY, FLOOR_Z, NOMINAL_CHEST_Z, sample_balance,
    _foot_normal_force, _foot_xy_z,
)
from standing_balance_lqr import StandingLQR
from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS
from step_primitive import _sole_pitch, FWD_HIP_SIGN as _FHS, KNEE_FLEX_SIGN as _KFS
from biped_recovery_env import REF_KNOTS as _RK, _knot_val as _kv

# whip-retract swing knots (CMA-optimised recovery step) in ctrl-index terms:
#  11 R hip-pitch, 12 R knee, 13 R ankle-pitch  (applied as _ss + knot)
#   6 L hip-pitch,  8 L ankle-pitch             (stance assist, applied as _ss + knot)
_SW_KN = {"hp": _RK[11], "kn": _RK[12], "ap": _RK[13]}          # for the R (canonical) swing
_ST_KN = {"hp": _RK[6], "ap": _RK[8]}                           # for the L (canonical) stance
_RK_SWING_MS, _RK_DESC_MS = 186, 56
_SWING_END_MS = 150      # swing->descend earlier than the whip's full arc (truncated whip)

G = 9.81
DEFAULT_MODEL = "robot/_exp_hands_2x.xml"

# ---- leg ctrl indices ----
# 0 neck | 1..4 arms | 5..9 L(hipRoll,hipPitch,knee,anklePitch,ankleRoll) | 10..14 R
LEG = {
    "L": dict(hr=5, hp=6, kn=7, ap=8, ar=9),
    "R": dict(hr=10, hp=11, kn=12, ap=13, ar=14),
}
LEG_CTRL_ORDER = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]     # action maps to these (canonical)

# sagittal mirror over the 15 ctrl indices (from joint axes)
_MIRROR_PERM = np.array([0, 3, 4, 1, 2, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9])
_MIRROR_SIGN = np.array([-1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], float)
# 10-vector (leg-joint) mirror: index j of the action <-> _LEGMIR_PERM[j], sign _LEGMIR_SIGN[j]
_LEGMIR_PERM = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])   # L<->R block swap
_LEGMIR_SIGN = _MIRROR_SIGN[LEG_CTRL_ORDER]               # [-1,-1,-1,-1,-1, -1,-1,-1,-1,-1]

# ---- frontal LIPM (lateral CoP feedback, kept as a reference bias) ----
FRONT_POLE_RE, FRONT_POLE_IM = -5.7, 2.3
COP_PER_RAD, AR_BIAS_MAX = 0.62, 0.16
COP_LO_MM, COP_HI_MM, X_MID_MM, X_SS_MM, X_TR_MM = 5.0, 96.0, 35.0, 73.95, 22.0
_X_MID_M = X_MID_MM / 1000.0

# ---- reference swing arc ----
FS_SWING_MS, FS_DESC_MS = 150, 95
FS_HIP_BASE, FS_HIP_GAIN, FS_HIP_MAX = 0.34, 3.0, 0.68
FS_KNEE_PK, FS_KNEE_LAND, FS_ANK_TOEUP = 0.48, 0.10, 0.16
FS_RETRACT = 0.12

# ---- lateral hip-roll strategy (damps step-to-step lateral velocity build-up) ----
HR_KX, HR_KV, HR_MAX = 2.2, 0.45, 0.32

# ---- pre-swing frontal weight-shift (restores the Stage-1 shift) ----
SHIFT_MS, SHIFT_RAMP_MS, SHIFT_UNLOAD = 75, 45, 0.20

# ---- brief swing-foot unload before each step (foot must genuinely leave the
#      ground before the whip drives it forward, otherwise it drags/shuffles) ----
UNLOAD_MS, UNLOAD_RAMP_MS, UNLOAD_BIAS = 45, 30, 0.17

# ---- action / episode ----
FRAME_SKIP = 5
ACT_SCALE = np.array([0.15, 0.30, 0.30, 0.20, 0.15,          # L hr,hp,kn,ap,ar
                      0.15, 0.30, 0.30, 0.20, 0.15])         # R
BODY_WEIGHT_N = 2.97 * G
FALL_UPTILT_DEG = 42.0
FALL_CHEST_DROP = 0.24
MAX_STEPS = 4  # was 5
EP_TIMEOUT_CTRL = 800
HANDOFF_MS = 25                    # StandingLQR ms after the push ends, before hand-off
STEP_COST = 1.0                    # per recovery step -- be reactive, don't over-step

# ---- trigger / caught ----
#  step 1: the robot is falling from the push -> step promptly (any disturbance).
#  step k>=2: ONLY a genuine FORWARD problem warrants another step -- the capture
#  point past the front support edge, or real forward CoM velocity.  Lateral
#  drift and a static forward lean are handled IN-step / in the settle, NOT by
#  taking another step (a forward step doesn't fix lateral drift -- it turns into
#  the wide sideways swing we're trying to eliminate).
TRIG_CAPT_FWD_MM = 14.0
TRIG_VFWD = 0.16
TRIG_SPEED = 0.24
STANCE_MIN_MS = 40
CAUGHT_CAPT_MM = 10.0
CAUGHT_SPEED = 0.13
CAUGHT_HOLD_MS = 60
SETTLE_MS = 300
V0_NOMINAL = 0.24

# ---- stepping-mechanics reward (compact FORWARD steps, human-stumble-like) ----
W_FWD_PLACE = 2.4        # reward the swing foot planted forward of the stance foot
W_LAT_SWING = 3.0        # penalise peak/net mid-swing lateral foot excursion
W_FOOT_SLIP = 0.04       # penalise planted-foot sliding (per physics step)
LAT_SWING_OK_MM = 24.0   # lateral foot excursion up to here is free
# genuine (not shuffled, not high-lift) swing-foot clearance
W_CLEARANCE = 1.4        # reward peak clearance being IN the window
W_AIR = 1.4              # reward genuine airborne time (foot truly off the ground)
CLEAR_LO_MM = 20.0       # below this -> a shuffle
CLEAR_HI_MM = 60.0       # above this -> an exaggerated leg lift

PUSH_BAND = (126.0, 138.0)

# ==== "cp" variant: capture-point forward-arc swing (commit to larger stumbles) ====
# The default "whip" swing is amplitude-capped (_ref_scale <= 1.15), truncated at
# 150 ms and RETRACTS in descend -> step-1 separation saturates at ~58 mm for every
# push from 130 N to 210 N (pushsweep.py).  The "cp" variant replaces the sagittal
# swing reference with step_primitive's forward arc, sized to the capture-point
# excess at step onset, with an adaptive duration and a gentle (non-retracting)
# descend.  Frontal-plane balance, torso LQR base and the policy residual are
# unchanged.
# Design intent: at LOW push the "cp" base must behave ~exactly like the whip so
# the warm-started policy stays ~100%; the "commit bigger" behaviour is DRAWN OUT
# by the reward + curriculum, only ENABLED (not forced) by the reach term, which
# stays ~0 until the capture-point excess / forward speed is genuinely large.
CP_REACH_PER_M = 2.6         # extra forward hip flexion (rad) per m of capture excess
CP_REACH_VGAIN = 0.55        # extra forward hip flexion (rad) per m/s of excess vfwd
CP_REACH_V0 = 0.20           # ...excess measured above this vfwd (below -> no reach)
CP_REACH_MAX = 0.24          # cap on the added reach -- small steps; the 2x foot
                             # polygon does the arresting, not a big lunge
CP_SWING_MS_BASE = 150
CP_SWING_MS_PER_REACH = 150.0  # + this * reach(rad)  -> longer swing for a longer step
CP_SWING_MS_MAX = 230
CP_PLACE_MARGIN_M = 0.030    # target the foot this far PAST the onset capture point
CP_XSS_VFWD_LO, CP_XSS_VFWD_HI = 0.28, 0.55   # ease frontal shift X_SS -> X_TR only once truly committed
CP_TRIG_CAPT_FWD_MM = 12.0   # k>=2 step trigger: capture excess past support
CP_CAUGHT_CAPT_MM = 18.0
CP_CAUGHT_SPEED = 0.16
CP_W_COMMIT = 0.018          # reward forward CoM progress while the capture pt is uncaught
CP_W_PLACE_CAPT = 1.8        # reward the foot planted near (capture pt + margin)
CP_SWING_LEAN = 0.12         # rad of forward pitch the base LQR tolerates in a big committed
                             # swing -- so it does not fight the CoM back to bolt-upright and
                             # topple.  No measurable effect off-policy; co-trained by RL.
# flat-foot touchdown: in descend, servo the CALF (shin->ankle) toward vertical
# with the knee, and trim the sole flat with the (weak) ankle -- so the foot
# lands flat by leg posture, not by the ankle scrambling after contact.
CP_KNEE_SHANK_K = 1.10       # knee gain on shank-fwd-lean error (rad/rad)
CP_KNEE_DIR = -1.0           # knee-flex direction that pulls the ankle back under the knee
CP_ANK_SOLE_K = 0.45         # ankle gain on sole-pitch error (rad/rad)
CP_FLAT_BLEND_MS = 45        # blend the flat-foot shaping in over the first ms of descend
# return to a NEUTRAL stance after the recovery: in settle, close up a staggered
# stance (rear foot steps up to the front foot) and settle the legs back to the
# standing pose.
CP_SETTLE_MS = 460           # longer settle so the robot has time to recentre
CP_RECENTER_DEADBAND = 0.030  # fore-aft foot stagger below this is "close enough"
CP_RECENTER_GAIN = 0.55      # rear-foot forward drive during recentre
CP_W_NEUTRAL = 0.05          # settle reward: legs near standing pose + feet level + nominal width
CP_SETTLE_BRAKE = 0.9        # settle: ankle/hip gain braking residual forward CoM velocity


def _frontal_gains(omega2, p_re, p_im):
    mag2 = p_re * p_re + p_im * p_im
    return -(mag2 / omega2) - 1.0, (2.0 * p_re) / omega2      # kx, kv


def _smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * t))


def _sole_roll(model, data, side):
    R = data.xmat[model.body(f"{side}_foot").id].reshape(3, 3)
    return float(np.arcsin(np.clip(R[0, 0], -1.0, 1.0)))


def _shank_fwd_lean(model, data, side):
    """Fore-aft lean of the shank (shin->ankle segment) from vertical, radians.
    + = shank leaning forward (ankle ahead of knee).  0 => calf points straight
    down, which lands the foot flat regardless of the hip angle."""
    p_shin = data.xpos[model.body(f"{side}_shin").id]
    p_ank = data.xpos[model.body(f"{side}_ankle").id]
    v = p_ank - p_shin
    return float(np.arctan2(-(v[1]), -(v[2]) + 1e-9))


def mirror_obs(o, jp, jv):
    """Sagittal reflection of the observation.  jp,jv are the slices of the 10
    leg-joint pos / vel entries; the rest are handled by fixed index lists."""
    o = np.asarray(o, np.float32).copy()
    for i in _OBS_NEG:
        o[i] = -o[i]
    for a, b in _OBS_SWAP:
        o[a], o[b] = o[b], o[a]
    o[jp] = (_LEGMIR_SIGN * o[jp][_LEGMIR_PERM]).astype(np.float32)
    o[jv] = (_LEGMIR_SIGN * o[jv][_LEGMIR_PERM]).astype(np.float32)
    return o


def mirror_action(a):
    return (_LEGMIR_SIGN * np.asarray(a, np.float32)[_LEGMIR_PERM]).astype(np.float32)


# obs layout (filled after first build)
_OBS_NEG = ()
_OBS_SWAP = ()
_OBS_JP = slice(0, 0)
_OBS_JV = slice(0, 0)


class BipedWalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, model_path=DEFAULT_MODEL, seed=None, push_band=PUSH_BAND,
                 dir_spread_deg=0.0, render_mode=None, max_steps=MAX_STEPS,
                 variant="whip"):
        super().__init__()
        self._variant = variant                 # "whip" (default, unchanged) | "cp"
        if not os.path.isabs(model_path):
            model_path = os.path.join(_HERE, model_path)   # robust under SubprocVecEnv
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._stand = StandingLQR(self.model, self.data, verbose=False)
        self._max_steps = int(max_steps)
        self.push_band = tuple(push_band)
        self.dir_spread = np.radians(dir_spread_deg)
        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None
        self._view_slow = 1.0          # >1 = slow motion in the interactive viewer
        self._ref_track_w = 1.0                 # decayed by the trainer via set_ref_track

        omega2 = G / (NOMINAL_CHEST_Z - FLOOR_Z)
        self.omega = float(np.sqrt(omega2))
        self.kx, self.kv = _frontal_gains(omega2, FRONT_POLE_RE, FRONT_POLE_IM)

        self.clow = self.model.actuator_ctrlrange[:15, 0].copy()
        self.chigh = self.model.actuator_ctrlrange[:15, 1].copy()
        self._up_local = np.array([0.0, 1.0, 0.0])
        self._fwd_local = np.array([0.0, 0.0, -1.0])
        self._b_lf = self.model.body("L_foot").id
        self._b_rf = self.model.body("R_foot").id
        self._foot_ext = self._measure_foot_extents()   # sole edges rel. foot body origin
        # nominal standing pose (for "return to neutral" shaping)
        self._leg_qadr = [7 + i for i in LEG_CTRL_ORDER]
        self._nominal_legpos = self._stand.qpos0[self._leg_qadr].copy()
        _d0 = mujoco.MjData(self.model)
        _d0.qpos[:] = self._stand.qpos0
        mujoco.mj_forward(self.model, _d0)
        self._nominal_foot_dx = float(_d0.xpos[self._b_lf][0] - _d0.xpos[self._b_rf][0])

        # leaned neutral leg pose (hip_pitch, knee, ankle_pitch) — small, symmetric.
        self._ss = {"R": np.array([0.126, -0.058, 0.015]),
                    "L": np.array([-0.126, 0.058, -0.015])}

        self.action_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        self._prev_action = np.zeros(10, np.float32)
        self._reset_flags()
        self.data.qpos[:] = self._stand.qpos0
        self.data.qvel[:] = self._stand.qvel0
        mujoco.mj_forward(self.model, self.data)
        o0 = self._build_obs()
        self._install_obs_layout()
        self.observation_space = spaces.Box(-np.inf, np.inf, (o0.shape[0],), np.float32)
        if seed is not None:
            self.reset(seed=seed)

    # -------------------------------------------------- obs layout
    def _install_obs_layout(self):
        global _OBS_NEG, _OBS_SWAP, _OBS_JP, _OBS_JV
        L = self._obs_index
        _OBS_NEG = tuple(L["neg"])
        _OBS_SWAP = tuple(L["swap"])
        _OBS_JP = slice(L["jp0"], L["jp0"] + 10)
        _OBS_JV = slice(L["jv0"], L["jv0"] + 10)

    # -------------------------------------------------- lifecycle
    def _reset_flags(self):
        self._swing, self._stance, self._lat = "R", "L", 1.0
        self._step_k = 0
        self._phase = "stance"
        self._sk = 0
        self._ep_ctrl = 0
        self._td = {}
        self._peak_up = 0.0
        self._v0 = 0.0
        self._caught_streak = 0
        self._stance_dwell = 0
        self._prev_action = np.zeros(10, np.float32)
        self._ref_scale = 1.0
        self._done_reason = ""
        self._swing_p0 = np.zeros(3)
        self._settle_ms = 0
        self._settle_spd = []
        self._sw_lifted = False
        self._sw_unload = 0
        self._plant_streak = 0
        self._sw_peak_lat = 0.0
        self._sw_peak_z = 0.0
        self._sw_air_ms = 0
        self._sw_drag_ms = 0
        self._swing_end = _SWING_END_MS      # per-step (adaptive in the "cp" variant)
        self._onset_capt_m = 0.0             # capture-point excess at step onset (m)
        self._arc = None                    # (hip_rad, knee_rad) for the "cp" swing arc
        self._com_fwd_prev = None
        self._spd_prev = None

    def set_task(self, push_band=None, dir_spread_deg=None):
        if push_band is not None:
            self.push_band = tuple(push_band)
        if dir_spread_deg is not None:
            self.dir_spread = np.radians(dir_spread_deg)
        return self.push_band, float(np.degrees(self.dir_spread))

    def set_ref_track(self, w):
        self._ref_track_w = float(w)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        for _ in range(6):
            if self._stand_and_push(options):
                break
        else:
            raise RuntimeError("reset: fell during stand+push 6x")
        return self._obs(), {"push_n": self._push_n}

    def _stand_and_push(self, options):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self._stand.qpos0
        d.qvel[:] = self._stand.qvel0
        d.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_forward(m, d)
        self._reset_flags()
        for _ in range(15):
            d.ctrl[:15] = self._stand.control(m, d)
            mujoco.mj_step(m, d)
            self._vsync()

        if "push_n" in options:
            self._push_n = float(options["push_n"])
            ang = float(options.get("push_dir_rad", -np.pi / 2))
        else:
            self._push_n = float(self.np_random.uniform(*self.push_band))
            ang = -np.pi / 2 + float(self.np_random.uniform(-self.dir_spread, self.dir_spread))
        fxy = self._push_n * np.array([np.cos(ang), np.sin(ang)])
        push_at = 15 + int(self.np_random.integers(0, 25))
        total = push_at + PUSH_DURATION_STEPS + HANDOFF_MS
        for k in range(total):
            d.xfrc_applied[CHEST_BODY, :] = 0.0
            if push_at <= k < push_at + PUSH_DURATION_STEPS:
                d.xfrc_applied[CHEST_BODY, 0:2] = fxy
            d.ctrl[:15] = self._stand.control(m, d)
            mujoco.mj_step(m, d)
            self._vsync()
            if self._fallen():
                return False
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        mujoco.mj_subtreeVel(m, d)
        self._v0 = float(np.hypot(d.subtree_linvel[CHEST_BODY][0], d.subtree_linvel[CHEST_BODY][1]))
        self._phase = "stance"
        self._sk = 0
        self._stance_dwell = 0
        self._peak_up = float(sample_balance(m, d).up_tilt_deg)
        return True

    # -------------------------------------------------- fall / balance
    def _cheap_tilt(self):
        R = self.data.xmat[CHEST_BODY].reshape(3, 3)
        return float(np.degrees(np.arccos(min(1.0, max(-1.0, (R @ self._up_local)[2])))))

    def _fallen(self):
        return (self._cheap_tilt() > FALL_UPTILT_DEG
                or self.data.qpos[2] < NOMINAL_CHEST_Z - FALL_CHEST_DROP)

    def _balance(self):
        key = self.data.time
        if getattr(self, "_bs_key", None) != key:
            self._bs = sample_balance(self.model, self.data)
            self._bs_key = key
        return self._bs

    def _foot_slip_penalty(self):
        """Penalise a foot sliding horizontally while it bears load -- keeps the
        planted foot planted (the late steps were dragging / pivoting)."""
        d = self.data
        pen = 0.0
        for side, bid in (("L", self._b_lf), ("R", self._b_rf)):
            if _foot_normal_force(self.model, d, side) > 20.0:
                vxy = float(np.hypot(d.cvel[bid][3], d.cvel[bid][4]))
                pen += -W_FOOT_SLIP * min(1.0, vxy / 0.15)
        return pen

    # -------------------------------------------------- frontal LIPM bias (reference)
    def _frontal_bias(self, bs, x_ref_m, lat):
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
    def _measure_foot_extents(self):
        """Sole footprint edges relative to each foot body origin (world axes, m),
        measured at the settled standing pose.  Lets the support-polygon / capture
        maths track any foot size (baseline, +30% 'duck feet', ...)."""
        m = mujoco.MjData(self.model)
        m.qpos[:] = self._stand.qpos0
        m.qvel[:] = self._stand.qvel0
        mujoco.mj_forward(self.model, m)
        ext = {}
        for side, bid in (("L", self._b_lf), ("R", self._b_rf)):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"{side}_foot_collision")
            mid = self.model.geom_dataid[gid]
            va, vn = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            loc = self.model.mesh_vert[va:va + vn].reshape(-1, 3)
            rot = m.geom_xmat[gid].reshape(3, 3)
            w = (rot @ loc.T).T + m.geom_xpos[gid]
            sole = w[w[:, 2] <= w[:, 2].min() + 0.003]
            b = m.xpos[bid]
            ext[side] = dict(
                fwd=float(-(sole[:, 1].min() - b[1])),   # sole extends this far fwd (-Y) of body
                back=float(sole[:, 1].max() - b[1]),
                xlo=float(sole[:, 0].min() - b[0]),
                xhi=float(sole[:, 0].max() - b[0]),
            )
        return ext

    def _capture(self):
        bs = self._balance()
        h = max(float(bs.com[2]) - FLOOR_Z, 0.05)
        tc = np.sqrt(h / G)
        xi_fwd = -float(bs.com[1]) + (-float(bs.com_vel[1])) * tc
        xi_lat = float(bs.com[0]) + float(bs.com_vel[0]) * tc
        lf, rf = _foot_xy_z(self.model, self.data, "L"), _foot_xy_z(self.model, self.data, "R")
        both = (self._phase in ("stance", "settle")) or self._td.get(self._step_k) is not None
        sides = ["L", "R"] if both else [self._stance]
        fmap = {"L": lf, "R": rf}
        e = self._foot_ext
        fwd_hi = max(-fmap[s][1] + e[s]["fwd"] for s in sides)
        fwd_lo = min(-fmap[s][1] - e[s]["back"] for s in sides)
        lat_hi = max(fmap[s][0] + e[s]["xhi"] for s in sides)
        lat_lo = min(fmap[s][0] + e[s]["xlo"] for s in sides)
        d_fwd = max(0.0, xi_fwd - fwd_hi, fwd_lo - xi_fwd) * 1000.0
        d_lat = max(0.0, xi_lat - lat_hi, lat_lo - xi_lat) * 1000.0
        return d_fwd, d_lat, -float(bs.com_vel[1])

    # -------------------------------------------------- control composition
    #  base = StandingLQR (roll-zeroed during a swing, full otherwise) -- provides
    #  whole-body torso stabilisation.  On top: the whip-retract swing knots
    #  (momentum-scaled, mirrored), the frontal-LIPM CoP command on both ankle
    #  rolls, and the FULL 10-joint policy residual (authority to correct the LQR
    #  wherever its feet-together linearisation is wrong -- staggered / single
    #  support).  Arms + neck are forced to neutral (ctrl 0).
    def _compose_ctrl(self, residual):
        m, d = self.model, self.data
        bs = self._balance()
        s = self._swing
        swinging = self._phase in ("swing", "descend")

        if self._phase in ("stance", "settle"):
            u = np.array(self._stand.control(m, d), float)
            if self._phase == "settle":
                # a whip step leaves the trailing foot light/lifted; nudge it back
                # to the ground for a genuine double-support finish (feedback law)
                for side in ("L", "R"):
                    if _foot_normal_force(m, d, side) < 6.0 and self._settle_ms > 8:
                        g = LEG[side]
                        u[g["hp"]] += _FHS[side] * 0.12       # flex hip -> foot forward+down
                        u[g["kn"]] += _KFS * (-0.10)          # slight knee bend
                        u[g["ap"]] += _FHS[side] * 0.06
                if self._variant == "cp" and self._settle_ms > 15:
                    # RECENTRE: ease a staggered stance together by relaxing BOTH
                    # legs toward the standing pose (a soft pull, no active
                    # stepping -- an aggressive rear-foot drive lurches the CoM).
                    relax = 0.22 * min(1.0, self._settle_ms / 160.0)
                    for j, ci in enumerate(LEG_CTRL_ORDER):
                        u[ci] += relax * (self._nominal_legpos[j] - float(d.qpos[self._leg_qadr[j]]))
                    # gentle forward-velocity brake, with a deadband so it does not
                    # fight an already-settled robot.
                    vf = -float(bs.com_vel[1])
                    if abs(vf) > 0.06:
                        brake = float(np.clip(vf, -0.30, 0.30)) * CP_SETTLE_BRAKE * 0.6
                        for side in ("L", "R"):
                            if _foot_normal_force(m, d, side) > 15.0:
                                u[LEG[side]["ap"]] += _FHS[side] * brake
                                u[LEG[side]["hp"]] += _FHS[side] * 0.5 * brake
        else:  # unload / swing / descend: roll-zeroed LQR base
            dq = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, self._stand.qpos0, d.qpos)
            dx = np.concatenate([dq, d.qvel - self._stand.qvel0])
            for ci in (LEG["L"]["ar"], LEG["R"]["ar"], LEG["L"]["hr"], LEG["R"]["hr"]):
                dx[6 + ci] = 0.0
                dx[m.nv + 6 + ci] = 0.0
            dx[0] = dx[m.nv + 0] = 0.0
            if self._variant == "cp" and self._arc:
                # let the torso LEAN into a committed stumble instead of the base
                # LQR fighting the forward pitch back to bolt-upright (which whips
                # the CoM back and topples).  Bias the pitch target forward,
                # proportional to the step's reach.
                lean = CP_SWING_LEAN * float(np.clip(self._arc[0] / CP_REACH_MAX, 0.0, 1.0))
                dx[3] -= lean
                dx[m.nv + 3] -= 0.0
            u = np.array(self._stand.ctrl0 - self._stand.K @ dx, float)

        # whip-retract swing knots (momentum-scaled, mirrored) on the swing leg;
        # CMA stance-assist knots on the stance leg
        if swinging:
            msign = 1.0 if s == "R" else -1.0
            mst = 1.0 if self._stance == "R" else -1.0
            gs, gst = LEG[s], LEG[self._stance]
            h0, k0, a0 = self._ss[s]
            if self._variant == "cp":
                # WHIP backbone (validated joint trajectory -> warm-startable) but:
                #  - amplitude uncapped (rs up to ~1.7 for a big committed step),
                #  - progress fraction stretched over the adaptive swing duration,
                #  - descend does NOT reverse hard (w clamped) -- the whip's w>1.15
                #    knots retract the foot; that is what pulls the trailing foot
                #    back in,
                #  - plus a capture-scaled forward hip/knee REACH ramped over swing
                #    so the step SIZE tracks the disturbance.
                rs = self._ref_scale
                reach = self._arc[0] if self._arc else 0.0
                # descend retract clamp: full whip retract (w->1.4) when reach~0 so
                # small pushes behave exactly like the whip; no retract (w->1.12)
                # for a big committed step so the foot stays out front.
                w_cap = 1.40 - 0.28 * float(np.clip(reach / CP_REACH_MAX, 0.0, 1.0))
                if self._phase == "swing":
                    w = min(1.0, self._sk / max(self._swing_end, 1)) * (_SWING_END_MS / _RK_SWING_MS)
                else:
                    w = min(1.0 + self._sk / _RK_DESC_MS, w_cap)
                ramp = min(1.0, self._sk / max(self._swing_end, 1)) if self._phase == "swing" else 1.0
                u[gs["hp"]] = h0 + msign * (rs * _kv(_SW_KN["hp"], w) + reach * ramp)
                u[gs["kn"]] = k0 + msign * (rs * _kv(_SW_KN["kn"], w)
                                            - 0.35 * reach * ramp * np.sin(np.pi * ramp))
                u[gs["ap"]] = a0 + msign * rs * _kv(_SW_KN["ap"], w)
                u[gst["hp"]] += mst * _kv(_ST_KN["hp"], min(w, 1.4))
                u[gst["ap"]] += mst * _kv(_ST_KN["ap"], min(w, 1.4))
                # flat-foot touchdown: pre-orient the CALF vertical (knee) and the
                # sole flat (ankle) over the last part of swing + all of descend,
                # so the foot is already flat when it contacts (descend is only
                # ~20 ms -- too late to fix orientation after that).
                if self._phase == "descend":
                    bd = 1.0
                else:  # swing: ramp in over the last 40% of the swing
                    bd = float(np.clip((ramp - 0.6) / 0.4, 0.0, 1.0))
                if bd > 0.0:
                    sfl = _shank_fwd_lean(m, d, s)
                    spitch = _sole_pitch(m, d, s)
                    u[gs["kn"]] += CP_KNEE_DIR * CP_KNEE_SHANK_K * bd * sfl
                    u[gs["ap"]] += -CP_ANK_SOLE_K * bd * spitch
            else:
                w = (min(1.0, self._sk / _RK_SWING_MS) if self._phase == "swing"
                     else 1.0 + self._sk / _RK_DESC_MS)
                rs = self._ref_scale
                rs_kn = rs
                u[gs["hp"]] = h0 + msign * rs * _kv(_SW_KN["hp"], w)
                u[gs["kn"]] = k0 + msign * rs_kn * _kv(_SW_KN["kn"], w)
                u[gs["ap"]] = a0 + msign * rs * _kv(_SW_KN["ap"], w)
                u[gst["hp"]] += mst * _kv(_ST_KN["hp"], w)
                u[gst["ap"]] += mst * _kv(_ST_KN["ap"], w)

        # frontal-LIPM CoP command on both ankle rolls.  In the "cp" variant, when
        # the robot is already committed forward, ease the big lateral weight-shift
        # (X_SS) back toward X_TR so step 1 stops bleeding forward momentum sideways.
        if self._variant == "cp" and swinging:
            vf = -float(bs.com_vel[1])
            bl = float(np.clip((vf - CP_XSS_VFWD_LO) / (CP_XSS_VFWD_HI - CP_XSS_VFWD_LO),
                               0.0, 1.0))
            x_ref = ((1.0 - bl) * X_SS_MM + bl * X_TR_MM) / 1000.0
        else:
            x_ref = (X_SS_MM if swinging else X_TR_MM) / 1000.0
        arb = self._frontal_bias(bs, x_ref, self._lat)
        u[LEG["L"]["ar"]] = arb
        u[LEG["R"]["ar"]] = arb
        # lateral hip-roll strategy on the loaded leg(s): drive the CoM back to
        # the centreline (more authority than ankle roll; damps step-to-step
        # lateral velocity accumulation).  Feedback law, not choreography.
        x = float(bs.com[0])
        vx = float(bs.com_vel[0])
        if self._lat < 0:
            x = 2.0 * _X_MID_M - x
            vx = -vx
        hr_cmd = float(np.clip(-(HR_KX * (x - _X_MID_M) + HR_KV * vx), -HR_MAX, HR_MAX))
        if self._lat < 0:
            hr_cmd = -hr_cmd
        for side in ("L", "R"):
            if (side == "L" and bs.l_contact) or (side == "R" and bs.r_contact):
                u[LEG[side]["hr"]] += hr_cmd

        # full 10-joint policy residual (already in the actual frame, scaled)
        u[LEG_CTRL_ORDER] = u[LEG_CTRL_ORDER] + residual

        u[0:5] = 0.0                                    # neck + arms neutral
        return np.clip(u, self.clow, self.chigh)

    # -------------------------------------------------- obs
    def _chest_axes(self):
        R = self.data.xmat[CHEST_BODY].reshape(3, 3)
        return R @ self._up_local, R @ self._fwd_local

    def _build_obs(self):
        m, d = self.model, self.data
        up, fwd = self._chest_axes()
        bs = self._balance()
        s, st = self._swing, self._stance
        lf, rf = _foot_xy_z(m, d, "L"), _foot_xy_z(m, d, "R")
        stf = lf if st == "L" else rf
        swf = rf if s == "R" else lf
        lnf, rnf = _foot_normal_force(m, d, "L"), _foot_normal_force(m, d, "R")
        d_fwd, d_lat, v_fwd = self._capture()
        h_err = float(d.qpos[2]) - NOMINAL_CHEST_Z
        arb = self._frontal_bias(bs, (X_SS_MM if s == "R" else X_TR_MM) / 1000.0, self._lat)
        legpos = d.qpos[[7 + i for i in LEG_CTRL_ORDER]].copy()
        legvel = d.qvel[[6 + i for i in LEG_CTRL_ORDER]].copy()
        phase_oh = np.array([self._phase == "stance", self._phase == "shift",
                             self._phase == "swing", self._phase == "descend",
                             self._phase == "settle"], np.float32)
        parts = []
        idx = {}

        def add(name, arr):
            arr = np.atleast_1d(np.asarray(arr, np.float32))
            idx[name] = (sum(len(p) for p in parts), len(arr))
            parts.append(arr)

        add("up", up)                                   # 3  (neg 0)
        add("fwd", fwd[:2])                              # 2  (neg 0)
        add("wvel", d.qvel[3:6])                         # 3  (neg 1,2)
        add("lvel", d.qvel[0:3])                         # 3  (neg 0)
        add("h_err", [h_err])                            # 1
        add("com_rel_st", [bs.com[0] - stf[0], -bs.com[1] - (-stf[1])])   # 2 (neg 0)
        add("com_vel", [bs.com_vel[0], -bs.com_vel[1]])  # 2  (neg 0)
        add("capt", [d_fwd / 100.0, d_lat / 100.0, v_fwd])   # 3 (neg none; d_lat symmetric)
        add("contacts", [float(bs.l_contact), float(bs.r_contact),
                         lnf / BODY_WEIGHT_N, rnf / BODY_WEIGHT_N])       # 4 (swap L/R)
        add("swf_rel_st", [swf[0] - stf[0], -(swf[1] - stf[1]), swf[2] - stf[2]])  # 3 (neg 0)
        add("arb_hint", [arb])                           # 1  (neg 0)
        jp0 = sum(len(p) for p in parts)
        add("legpos", legpos)                            # 10 leg-joint pos
        jv0 = sum(len(p) for p in parts)
        add("legvel", legvel)                            # 10 leg-joint vel
        add("phase", phase_oh)                           # 4
        add("gaitfrac", [min(1.0, self._sk / max(self._swing_end, 1)),
                         self._step_k / max(self._max_steps, 1)])       # 2
        add("prev_a", self._prev_action)                 # 10 (mirror-perm/sign)
        add("v0", [self._v0])                            # 1

        obs = np.concatenate(parts).astype(np.float32)
        # index lists for the mirror
        neg = []
        for nm, off in (("up", 0), ("fwd", 0), ("lvel", 0), ("com_rel_st", 0),
                        ("com_vel", 0), ("swf_rel_st", 0), ("arb_hint", 0)):
            neg.append(idx[nm][0] + off)
        neg += [idx["wvel"][0] + 1, idx["wvel"][0] + 2]
        swap = [(idx["contacts"][0], idx["contacts"][0] + 1),
                (idx["contacts"][0] + 2, idx["contacts"][0] + 3)]
        # prev_action mirrors like an action: handle via a dedicated remap
        self._obs_index = dict(neg=neg, swap=swap, jp0=jp0, jv0=jv0,
                               preva0=idx["prev_a"][0])
        return obs

    def _obs(self):
        o = self._build_obs()
        if self._lat < 0:
            o = mirror_obs(o, _OBS_JP, _OBS_JV)
            p0 = self._obs_index["preva0"]
            o[p0:p0 + 10] = mirror_action(o[p0:p0 + 10])
        return o

    # -------------------------------------------------- step
    def step(self, action):
        m, d = self.model, self.data
        a_canon = np.asarray(action, np.float32).clip(-1.0, 1.0)
        a_actual = a_canon if self._lat > 0 else mirror_action(a_canon)
        residual = a_actual * ACT_SCALE
        self._ep_ctrl += 1

        r = 0.0
        fell = plant = settled = False
        for _ in range(FRAME_SKIP):
            d.ctrl[:15] = self._compose_ctrl(residual)
            mujoco.mj_step(m, d)
            self._vsync()
            self._sk += 1
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                fell = True
                break
            bs = self._balance()
            d_fwd, d_lat, v_fwd = self._capture()

            # dense shaping per physics step
            r += 0.02                                              # alive
            r += -0.012 * min(1.0, self._cheap_tilt() / 25.0)
            r += -0.008 * min(1.0, d_fwd / 100.0) - 0.012 * min(1.0, d_lat / 100.0)
            if self._variant == "cp":
                # only penalise forward speed the robot does NOT need -- i.e. once
                # the capture point is (nearly) back inside support.  While it is
                # still well outside, carrying momentum into the step is CORRECT,
                # and a small reward for forward CoM progress keeps the policy
                # committing instead of aborting the step.
                near = float(np.clip(1.0 - d_fwd / 45.0, 0.0, 1.0))
                r += -0.015 * near * max(0.0, bs.com_speed_horiz - 0.18)
                cf = bs.com_fwd
                if self._com_fwd_prev is not None and d_fwd > 12.0 and self._phase != "settle":
                    r += CP_W_COMMIT * float(np.clip((cf - self._com_fwd_prev) / 0.004,
                                                     -1.0, 1.0))
                self._com_fwd_prev = cf
                # gentle deceleration reward once the capture point is nearly
                # caught -- a continuous gradient toward coming to REST.
                spd = bs.com_speed_horiz
                if self._spd_prev is not None and d_fwd < 20.0 and self._phase != "settle":
                    r += 0.25 * float(np.clip((self._spd_prev - spd) / 0.006, -1.0, 1.0))
                self._spd_prev = spd
            else:
                r += -0.015 * max(0.0, bs.com_speed_horiz - 0.18)
            r += -0.008 * float(np.mean(a_canon ** 2))             # stay near the reference
            r += self._foot_slip_penalty()

            # phase machine
            if self._phase == "stance":
                self._stance_dwell += 1
                lean = float(bs.fwd_lean_deg)
                cp = self._variant == "cp"
                if self._step_k == 0:
                    # first step: the robot is falling from the push -> step promptly
                    need = (lean > 4.0 or v_fwd > 0.09 or bs.com_speed_horiz > 0.12)
                    min_dwell = 8
                elif cp:
                    # k>=1: another step only while a real forward problem remains
                    # -- residual forward speed / capture out / persistent lean.
                    # Step COUNT tracks the push (big -> several small fwd steps).
                    need = (d_fwd > 9.0 or v_fwd > 0.12 or bs.com_speed_horiz > 0.15
                            or (lean > 7.0 and v_fwd > 0.05))
                    min_dwell = STANCE_MIN_MS
                else:
                    need = (lean > 7.0 or d_fwd > TRIG_CAPT_FWD_MM or v_fwd > TRIG_VFWD
                            or bs.com_speed_horiz > TRIG_SPEED)
                    min_dwell = STANCE_MIN_MS
                if cp:
                    caught = (d_fwd < 15.0 and v_fwd < 0.12 and bs.com_speed_horiz < 0.15
                              and abs(lean) < 8.0)
                    hold_needed = 30
                elif self._step_k >= 2:
                    caught = (d_fwd < 20.0 and v_fwd < 0.16 and bs.com_speed_horiz < 0.20
                              and abs(lean) < 11.0)
                    hold_needed = 25
                else:
                    caught = (d_fwd < CAUGHT_CAPT_MM and bs.com_speed_horiz < CAUGHT_SPEED
                              and v_fwd < CAUGHT_SPEED and abs(lean) < 7.0)
                    hold_needed = CAUGHT_HOLD_MS
                self._caught_streak = self._caught_streak + 1 if caught else 0
                # cp: settle allowed from step 1 (the 2x-foot polygon can catch a
                # small push in one step); the trigger adds steps 2..N when a
                # forward problem remains.  Force-settle stops any coasting.
                settle_min_k = 1
                cp_force_settle = (cp and self._step_k >= settle_min_k
                                   and self._stance_dwell > 200
                                   and bs.up_tilt_deg < 16.0 and bs.l_contact and bs.r_contact
                                   and not need)
                if ((self._caught_streak >= hold_needed or cp_force_settle)
                        and self._step_k >= settle_min_k
                        and bs.l_contact and bs.r_contact):
                    self._phase = "settle"
                    self._sk = 0
                    self._settle_ms = 0
                    self._settle_spd = []
                elif (need and self._step_k < self._max_steps
                      and self._stance_dwell >= min_dwell
                      and not (caught and self._step_k >= settle_min_k)
                      and bs.up_tilt_deg < 24.0):
                    self._begin_step()
            elif self._phase in ("swing", "descend"):
                sw_nf = _foot_normal_force(m, d, self._swing)
                swf = _foot_xy_z(m, d, self._swing)
                swb = self._b_rf if self._swing == "R" else self._b_lf
                clr = float(swf[2]) - float(self._swing_p0[2])
                self._sw_peak_lat = max(self._sw_peak_lat,
                                        abs(float(swf[0]) - float(self._swing_p0[0])))
                self._sw_peak_z = max(self._sw_peak_z, clr)
                if self._phase == "swing":
                    if not self._sw_lifted and sw_nf < 3.0:
                        self._sw_lifted = True
                    if sw_nf < 3.0:
                        self._sw_unload += 1
                    # GENUINE swing = foot off the ground AND meaningfully clear
                    vxy = float(np.hypot(d.cvel[swb][3], d.cvel[swb][4]))
                    airborne = (sw_nf < 2.0 and clr > 0.012)
                    self._sw_air_ms += int(airborne)
                    if airborne:
                        r += 0.03
                    # SHUFFLE = low + loaded + sliding forward -> penalise directly
                    if clr < 0.010 and sw_nf > 3.0 and vxy > 0.03:
                        self._sw_drag_ms += 1
                        r += -0.10
                    if self._sk >= self._swing_end:
                        self._phase = "descend"
                        self._sk = 0
                else:  # descend
                    swc = bs.r_contact if self._swing == "R" else bs.l_contact
                    genuine = self._sw_lifted and swc and sw_nf > 12.0
                    self._plant_streak = self._plant_streak + 1 if genuine else 0
                    if self._plant_streak >= 10 or self._sk >= 260:
                        self._snapshot_td(bs, sw_nf)
                        plant = True
                        break
            elif self._phase == "settle":
                self._settle_ms += 1
                r += 0.05 * float(np.clip(1.0 - bs.com_speed_horiz / 0.20, 0.0, 1.0))
                r += -0.06 * max(0.0, bs.com_speed_horiz - 0.14)
                self._settle_spd.append(bs.com_speed_horiz)
                settle_len = CP_SETTLE_MS if self._variant == "cp" else SETTLE_MS
                if self._variant == "cp":
                    # return to NEUTRAL stance: feet level (no fore-aft stagger),
                    # nominal width, legs near the standing pose.
                    lf, rf = _foot_xy_z(m, d, "L"), _foot_xy_z(m, d, "R")
                    fa_sep = abs(float(lf[1] - rf[1]))
                    wid_err = abs(float(lf[0] - rf[0]) - self._nominal_foot_dx)
                    legdev = float(np.mean(np.abs(
                        d.qpos[self._leg_qadr] - self._nominal_legpos)))
                    r += CP_W_NEUTRAL * 0.02 * (
                        float(np.clip(1.0 - fa_sep / 0.05, -1.0, 1.0))
                        + float(np.clip(1.0 - wid_err / 0.04, -1.0, 1.0))
                        + float(np.clip(1.0 - legdev / 0.25, -1.0, 1.0)))
                if self._settle_ms >= settle_len:
                    settled = True
                    break

        self._prev_action = a_canon.copy()

        if fell:
            return self._obs(), r - 40.0, True, False, self._info(fell=True)

        if plant:
            r += self._td_reward(self._step_k)
            #  per-step cost.  Flat for the first 4 steps so the count is free to
            #  track the push size; rising past the 4th to discourage a long shuffle.
            if self._variant == "cp":
                r -= STEP_COST * (1.0 + 0.5 * max(0, self._step_k - 4))
            else:
                r -= STEP_COST
            self._phase = "stance"
            self._sk = 0
            self._stance_dwell = 0
            self._caught_streak = 0
            return self._obs(), r + 0.3, False, False, self._info(fell=False)

        if settled:
            self._done_reason = "settled"
            tr, info = self._terminal_eval(timed_out=False, settle_spd=self._settle_spd)
            return self._obs(), r + tr, True, False, info

        if self._ep_ctrl > EP_TIMEOUT_CTRL:
            self._done_reason = "timeout"
            tr, info = self._terminal_eval(timed_out=True)
            return self._obs(), r + tr, True, False, info
        return self._obs(), r + 0.3, False, False, self._info(fell=False)

    def _begin_step(self):
        m, d = self.model, self.data
        if self._step_k >= 1:                       # step 1 keeps swing=R; alternate after
            self._swing, self._stance = self._stance, self._swing
        self._lat = 1.0 if self._swing == "R" else -1.0
        self._step_k += 1
        self._phase = "swing"
        self._sk = 0
        self._sw_lifted = False
        self._sw_unload = 0
        self._plant_streak = 0
        self._swing_p0 = _foot_xy_z(m, d, self._swing).copy()
        self._sw_peak_lat = 0.0
        self._sw_peak_z = 0.0
        self._sw_air_ms = 0
        self._sw_drag_ms = 0
        # whip amplitude scaled by the CoM speed at onset (known-good: step 1 at
        # ~full scale catches the push; later steps enter slower -> smaller).
        mujoco.mj_subtreeVel(m, d)
        v = float(np.hypot(d.subtree_linvel[CHEST_BODY][0], d.subtree_linvel[CHEST_BODY][1]))
        if self._variant == "cp":
            d_fwd_mm, _, v_fwd = self._capture()
            d_fwd_m = d_fwd_mm / 1000.0
            self._onset_capt_m = d_fwd_m
            # extra forward reach sized to the capture-point excess + the
            # developing forward velocity (step 1 often shows d_fwd~0 while
            # already toppling forward).
            reach = float(np.clip(
                CP_REACH_PER_M * d_fwd_m + CP_REACH_VGAIN * max(0.0, v_fwd - CP_REACH_V0),
                0.0, CP_REACH_MAX))
            self._arc = (reach, 0.0)
            self._swing_end = int(np.clip(
                CP_SWING_MS_BASE + CP_SWING_MS_PER_REACH * reach,
                CP_SWING_MS_BASE, CP_SWING_MS_MAX))
            self._ref_scale = float(np.clip(v / V0_NOMINAL, 0.40, 1.7))
        else:
            self._ref_scale = float(np.clip(v / V0_NOMINAL, 0.40, 1.15))
            self._swing_end = _SWING_END_MS

    def _snapshot_td(self, bs, sw_nf):
        s, st = self._swing, self._stance
        sf = _foot_xy_z(self.model, self.data, s)
        stf = _foot_xy_z(self.model, self.data, st)
        swb = self._b_rf if s == "R" else self._b_lf
        self._td[self._step_k] = dict(
            planted=self._plant_streak >= 10,
            sep_mm=float(-(sf[1] - stf[1]) * 1000.0),               # swing fwd of stance
            fwd_mm=float(-(sf[1] - self._swing_p0[1]) * 1000.0),    # swing-foot fwd travel
            net_lat_mm=float((sf[0] - self._swing_p0[0]) * 1000.0),
            peak_lat_mm=float(self._sw_peak_lat * 1000.0),
            peak_z_mm=float(self._sw_peak_z * 1000.0),
            air_ms=int(self._sw_air_ms),
            drag_ms=int(self._sw_drag_ms),
            sole_pitch=float(np.degrees(_sole_pitch(self.model, self.data, s))),
            sole_roll=float(np.degrees(_sole_roll(self.model, self.data, s))),
            nf=float(sw_nf), vz=float(self.data.cvel[swb][5]),
            unloaded_ok=self._sw_unload >= 15,
            onset_capt_mm=float(self._onset_capt_m * 1000.0),
            capt_at_td_mm=float(self._capture()[0]),
        )

    def _td_reward(self, k):
        td = self._td[k]
        r = 1.0 * float(np.clip(1.0 - abs(td["sole_pitch"]) / 12.0, 0.0, 1.0))
        r += 0.5 * float(np.clip(1.0 - abs(td["sole_roll"]) / 12.0, 0.0, 1.0))
        r += 1.2 * float(np.clip(td["nf"] / 20.0, 0.0, 1.0))
        r += -0.5 * float(np.clip(abs(td["vz"]) / 0.35, 0.0, 1.0))
        r += 0.5 if td["unloaded_ok"] else -0.5
        # --- stepping mechanics: forward, compact, human-stumble-like ---
        #  reward the foot planted AHEAD of the stance foot; saturates ~50 mm.
        r += W_FWD_PLACE * float(np.clip((td["sep_mm"] + 8.0) / 50.0, -1.2, 1.0))
        if self._variant == "cp":
            # commit: place the foot near where the capture point was at onset
            # (+ a margin) -- this is what makes the step SIZE track the push size.
            tgt = td["onset_capt_mm"] + CP_PLACE_MARGIN_M * 1000.0 + 40.0
            r += CP_W_PLACE_CAPT * float(np.clip(
                1.0 - abs(td["sep_mm"] - tgt) / 60.0, -1.0, 1.0))
            # ...and reward the step actually reducing the capture-point excess.
            r += 1.2 * float(np.clip(
                (td["onset_capt_mm"] - td["capt_at_td_mm"]) / 40.0, -1.0, 1.0))
        #  wide sideways swing -- peak & net lateral foot excursion
        r += -W_LAT_SWING * float(np.clip(
            (abs(td["peak_lat_mm"]) - LAT_SWING_OK_MM) / 40.0, 0.0, 1.6))
        r += -0.6 * W_LAT_SWING * float(np.clip(
            (abs(td["net_lat_mm"]) - LAT_SWING_OK_MM) / 40.0, 0.0, 1.3))
        #  --- GENUINE step: peak foot clearance in a WINDOW (not maximised) ---
        #  below CLEAR_LO -> shuffle (steep penalty); in [LO,HI] -> full reward;
        #  above HI -> a high leg lift (mild penalty).
        pz = td["peak_z_mm"]
        if pz < CLEAR_LO_MM:
            r += -W_CLEARANCE * (1.0 + (CLEAR_LO_MM - pz) / CLEAR_LO_MM)      # up to -2*W
        elif pz <= CLEAR_HI_MM:
            r += W_CLEARANCE
        else:
            r += W_CLEARANCE * max(-0.6, 1.0 - (pz - CLEAR_HI_MM) / 35.0)     # mild over-lift
        #  genuine airborne time (foot truly off the ground)
        r += W_AIR * float(np.clip((td["air_ms"] - 12) / 22.0, -0.8, 1.0))
        #  the SHUFFLE: foot dragging (low + loaded + sliding) -- penalise per-step total
        r += -0.14 * min(30.0, td["drag_ms"])
        return r

    # -------------------------------------------------- settle + terminal
    def _terminal_eval(self, timed_out, settle_spd=None):
        m, d = self.model, self.data
        b = sample_balance(m, d)
        ds = b.l_contact and b.r_contact
        d_fwd, d_lat, v_fwd = self._capture()
        sp = max(abs(np.degrees(_sole_pitch(m, d, "L"))), abs(np.degrees(_sole_pitch(m, d, "R"))))
        sr = max(abs(np.degrees(_sole_roll(m, d, "L"))), abs(np.degrees(_sole_roll(m, d, "R"))))
        spd_end = float(np.mean(settle_spd[-40:])) if settle_spd else b.com_speed_horiz
        all_pl = all(self._td[j]["planted"] for j in self._td) if self._td else False

        # SUCCESS: the robot comes to rest, upright, without falling, having taken
        # the steps.  Ending balanced-and-still on ONE foot is a legitimate push
        # recovery (common for real recovery); a flat double-support finish is the
        # higher-quality outcome tracked separately as `flat_ok`.
        caught = (d_fwd < CAUGHT_CAPT_MM and spd_end < 0.14)
        upright = (b.up_tilt_deg < 11.0 and abs(b.side_lean_deg) < 10.0
                   and b.chest_z > NOMINAL_CHEST_Z - 0.11 and self._peak_up <= 30.0)
        success = bool(caught and upright and all_pl and not timed_out and self._step_k >= 1)
        flat_ok = bool(success and ds and sp < 14.0 and sr < 14.0)

        # returned to a NEUTRAL stance: feet level + nominal width + legs near the
        # standing pose (the highest-quality finish -- "back to standing").
        lf, rf = _foot_xy_z(m, d, "L"), _foot_xy_z(m, d, "R")
        fa_sep_mm = abs(float(lf[1] - rf[1])) * 1000.0
        wid_err_mm = abs(abs(float(lf[0] - rf[0])) - abs(self._nominal_foot_dx)) * 1000.0
        legdev = float(np.mean(np.abs(d.qpos[self._leg_qadr] - self._nominal_legpos)))
        neutral_ok = bool(flat_ok and fa_sep_mm < 45.0 and wid_err_mm < 40.0 and legdev < 0.22)

        r = 0.0
        r += 26.0 if success else 0.0
        r += 9.0 if flat_ok else 0.0
        r += (6.0 if neutral_ok else 0.0) if self._variant == "cp" else 0.0
        if self._variant == "cp":       # graded pull toward the neutral stance
            r += 3.0 * float(np.clip(1.0 - fa_sep_mm / 90.0, 0.0, 1.0))
            r += 2.0 * float(np.clip(1.0 - legdev / 0.35, 0.0, 1.0))
        r += -10.0 if timed_out else 0.0
        r += 2.0 * float(np.clip(1.0 - b.up_tilt_deg / 12.0, 0.0, 1.0))
        r += 1.5 * float(np.clip(1.0 - abs(b.side_lean_deg) / 12.0, 0.0, 1.0))
        r += 4.0 * float(np.clip(1.0 - spd_end / 0.24, 0.0, 1.0))
        r += -10.0 * float(np.clip((spd_end - 0.16) / 0.20, 0.0, 1.0))   # not at rest
        r += 2.5 * float(np.clip(1.0 - d_fwd / 60.0, 0.0, 1.0))
        r += 1.5 if ds else -1.0
        info = self._info(fell=False)
        info.update(success=success, flat_ok=flat_ok, neutral_ok=neutral_ok,
                    n_steps=int(self._step_k),
                    end_capt_fwd_mm=float(d_fwd), end_spd=float(spd_end),
                    end_up=float(b.up_tilt_deg), end_side=float(b.side_lean_deg),
                    end_fa_sep_mm=float(fa_sep_mm), end_legdev=float(legdev),
                    sole_pitch_max=float(sp), sole_roll_max=float(sr),
                    all_planted=bool(all_pl), timed_out=bool(timed_out))
        return r, info

    def _info(self, fell):
        return dict(push_n=float(self._push_n), phase=self._phase, n_steps=int(self._step_k),
                    peak_up=float(self._peak_up), v0=float(self._v0), fell=bool(fell),
                    success=False, flat_ok=False, neutral_ok=False, end_capt_fwd_mm=999.0,
                    end_spd=9.9, end_up=99.0, end_side=99.0, end_fa_sep_mm=999.0,
                    end_legdev=9.9, sole_pitch_max=99.0, sole_roll_max=99.0,
                    all_planted=False, timed_out=False, done_reason=self._done_reason)

    # -------------------------------------------------- render
    def _vsync(self):
        """Interactive-viewer sync (render_mode='human').  Real-time-ish pacing."""
        if self.render_mode != "human":
            return
        if getattr(self, "_viewer", None) is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.azimuth = 90
            self._viewer.cam.elevation = -8
            self._viewer.cam.distance = 2.0
            self._t_last = time.perf_counter()
        if not self._viewer.is_running():
            raise KeyboardInterrupt("viewer closed")
        self._viewer.cam.lookat[:] = [0.0, float(self.data.xpos[CHEST_BODY][1]), 0.95]
        self._viewer.sync()
        dt = self.model.opt.timestep * self._view_slow
        now = time.perf_counter()
        sleep = dt - (now - self._t_last)
        if sleep > 0:
            time.sleep(sleep)
        self._t_last = time.perf_counter()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 320, 300)
            self._cam = mujoco.MjvCamera()
            self._cam.azimuth = 110        # look along +X -> robot's forward (-Y) goes left->right
            self._cam.elevation = -28
            self._cam.distance = 1.3
        # track the torso so multi-step forward travel stays in frame
        p = self.data.xpos[CHEST_BODY]
        self._cam.lookat[:] = [0.0, float(p[1]), 0.95]
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None


# ============================================================ smoke / mirror
def _load_policy(run_dir, band):
    import json
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    hp = json.load(open(f"{run_dir}/hparams.json"))
    tmp = DummyVecEnv([lambda: BipedWalkEnv(push_band=band)])
    # prefer the best-by-eval checkpoint if present
    pol_f = f"{run_dir}/policy_best.pth" if os.path.exists(f"{run_dir}/policy_best.pth") else f"{run_dir}/policy.pth"
    vn_f = f"{run_dir}/vecnormalize_best.pkl" if os.path.exists(f"{run_dir}/vecnormalize_best.pkl") else f"{run_dir}/vecnormalize.pkl"
    vn = VecNormalize.load(vn_f, tmp)
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    mm = PPO("MlpPolicy", tmp, device="cpu",
             policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(pol_f, map_location="cpu", weights_only=True))
    print(f"  loaded {pol_f}")
    mm.policy.eval()
    return mm, lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)


def _smoke(n=20, render_path=None, band=PUSH_BAND, model=DEFAULT_MODEL, policy=None):
    pol = _load_policy(policy, band) if policy else None
    env = BipedWalkEnv(model_path=model, push_band=band,
                       render_mode=("rgb_array" if render_path else None))
    frames = []
    succ = fell = flat = 0
    nsteps, ups, spds = [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=1000 + i)
        done = False
        info = {}
        ep = []
        while not done:
            if pol:
                import torch
                with torch.no_grad():
                    a, _ = pol[0].predict(pol[1](obs), deterministic=True)
            else:
                a = np.zeros(10, np.float32)
            obs, rr, term, trunc, info = env.step(a)
            done = term or trunc
            if render_path and i < 6:
                ep.append(env.render())
        succ += int(info.get("success", False))
        fell += int(info.get("fell", False))
        flat += int(info.get("flat_ok", False))
        nsteps.append(info.get("n_steps", 0))
        ups.append(info.get("peak_up", 99))
        spds.append(info.get("end_spd", 9.9))
        if render_path and i < 6:
            frames.append(ep)
        print(f"  ep {i:2d} push {info.get('push_n', 0):6.1f}  steps {info.get('n_steps', 0)}  "
              f"{'SUCC' if info.get('success') else ('fell' if info.get('fell') else info.get('done_reason', '?'))}"
              f"{' FLAT' if info.get('flat_ok') else ''}  peakUp {info.get('peak_up', 0):4.1f}  "
              f"endCapt {info.get('end_capt_fwd_mm', 0):5.1f}  endSpd {info.get('end_spd', 0):.2f}")
    tag = f"POLICY {policy}" if policy else "ZERO-ACTION (reference gait)"
    print(f"\n  {tag}  ({n} eps, band {band}):")
    print(f"    SUCCESS {succ}/{n}   fell {fell}/{n}   flat {flat}/{n}")
    print(f"    step-count: {np.bincount(nsteps, minlength=MAX_STEPS + 1).tolist()}")
    print(f"    median peakUp {np.median(ups):.1f}   median endSpd {np.median(spds):.2f}")
    if render_path and frames:
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
    env.close()


def _mirror_test():
    env = BipedWalkEnv(push_band=(133.0, 133.0))
    rng = np.random.default_rng(0)
    # obs involution on random obs-shaped vectors
    dim = env.observation_space.shape[0]
    p0 = env._obs_index["preva0"]

    def full_mirror(o):
        o = mirror_obs(o, _OBS_JP, _OBS_JV)
        o[p0:p0 + 10] = mirror_action(o[p0:p0 + 10])
        return o
    bad = sum(np.abs(full_mirror(full_mirror(o)) - o).max() > 1e-5
              for o in [rng.normal(size=dim).astype(np.float32) for _ in range(1000)])
    print(f"  obs mirror involution: {'OK' if bad == 0 else f'{bad} FAIL'}")
    assert np.array_equal(_LEGMIR_PERM[_LEGMIR_PERM], np.arange(10))
    print("  leg-joint PERM involution: OK")
    for tag, ang in (("R", -np.pi / 2 - 0.16), ("L", -np.pi / 2 + 0.16)):
        e = BipedWalkEnv(push_band=(133.0, 133.0))
        o, _ = e.reset(options={"push_n": 133.0, "push_dir_rad": ang})
        done = False
        info = {}
        while not done:
            o, r, t1, t2, info = e.step(np.zeros(10, np.float32))
            done = t1 or t2
        print(f"  push-{tag}: steps {info['n_steps']}  {info['done_reason'] or ('fell' if info['fell'] else '')}  "
              f"peakUp {info['peak_up']:.1f}  endSpd {info.get('end_spd', 0):.2f}")
        e.close()
    env.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mirror-test", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--render", default=None)
    ap.add_argument("--band", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args(argv)
    band = tuple(float(x) for x in a.band.split(",")) if a.band else PUSH_BAND
    if a.mirror_test:
        _mirror_test()
    if a.smoke:
        _smoke(a.n, a.render, band, a.model, a.policy)
    if not (a.smoke or a.mirror_test):
        ap.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
