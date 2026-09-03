"""Reactive recovery via a RAPID ALTERNATING SMALL-STEP GAIT.

Rationale (measured -- see the round-2 investigation):
  * one big committed step cannot be stabilised (torso pitches over) and the
    +-2.3 N.m knee cannot hold the shank vertical for a big hip flexion;
  * a KNEE-driven unload of the swing foot works (the ankle-roll unload did not);
  * a small hip flexion (<~0.25 rad) DOES let the knee keep the calf vertical, so
    the foot lands flat by leg posture -- no ankle scrambling;
  * scripted, a fast (~90 ms) sequence of such small flat steps chains 3-5 steps
    and travels 100-240 mm before the torso topples -- the topple is because the
    StandingLQR base fights the forward lean.

So: the recovery is push -> a fast run of small, flat, alternating forward steps,
each removing a little energy, until the capture point is back inside support ->
then close the feet and settle to a NEUTRAL stance.  Step COUNT (not step size)
tracks the disturbance.  The torso base is given a forward-lean reference that
GOES WITH the travel instead of fighting it.

  * ACTION (10): bounded position residual on L/R hip-roll,hip-pitch,knee,
    ankle-pitch,ankle-roll, in a canonical "swing = R" frame (obs+action
    mirrored per step).  Zero action == the reference gait.
  * MODEL: robot/_exp_hands_2x_bigfeet.xml (+30% "duck feet"; robot.xml untouched).

    python biped_gait_env.py --smoke                 # zero-action reference
    python biped_gait_env.py --smoke --render g.mp4
    python biped_gait_env.py --scripted --lo 120 --hi 190
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

_HERE = os.path.dirname(os.path.abspath(__file__))

from recovery_metrics import (
    CHEST_BODY, FLOOR_Z, G, NOMINAL_CHEST_Z, sample_balance,
    _foot_normal_force, _foot_xy_z,
)
from standing_balance_lqr import StandingLQR
from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS
from step_primitive import _sole_pitch
from biped_walk_env import (
    LEG, LEG_CTRL_ORDER, _shank_fwd_lean, _sole_roll,
    mirror_obs, mirror_action, _LEGMIR_PERM, _LEGMIR_SIGN,
    _frontal_gains, FRONT_POLE_RE, FRONT_POLE_IM,
    COP_PER_RAD, AR_BIAS_MAX, COP_LO_MM, COP_HI_MM, X_MID_MM,
    _X_MID_M, HR_KX, HR_KV, HR_MAX,
)

DEFAULT_MODEL = "robot/_exp_hands_2x_bigfeet.xml"

# ---- gait primitive ----
UNLOAD_MS = 20            # swing knee flexes hard -> foot lifts
SWING_MS = 60            # hip drives forward to hip_amp, calf held vertical
REACH_MS = 40            # after SWING: extend the knee, drive the foot DOWN to plant
STEP_MAX_MS = 130        # hard cap on one step (fast cadence -- topple has less time)
PLANT_NF = 12.0
PLANT_HOLD = 4
HIP_AMP0 = 0.22          # nominal forward hip flexion per step (rad)
HIP_AMP_VGAIN = 0.45     # + this * max(0, vfwd - 0.15)   (a touch bigger when fast)
HIP_AMP_MAX = 0.34
KNEE_UNLOAD = 0.85       # peak swing-knee flex during unload
KNEE_BASE_FLEX = 0.14    # baseline swing-knee flex through the step
KNEE_SHANK_KP = 1.35     # knee gain holding the calf vertical
KNEE_REACH_EXT = 0.9     # knee extension drive in the reach phase (find the ground)
ANK_SOLE_KP = 1.0        # ankle gain keeping the sole flat
TOEOFF = 0.10            # trailing-leg plantarflex push at the end of its stance
STANCE_RIGHT_KP = 1.1    # stance HIP torque righting the torso pitch in single support

# ---- torso lean reference: RIDE a forward lean during the gait (like a run),
#      right it during settle -- do NOT fight it back to vertical mid-recovery ----
TORSO_LEAN_GAIN = 1.1    # rad of fwd pitch target per m/s of forward CoM speed
TORSO_LEAN_MAX = 0.32

# ---- frontal LIPM (lateral CoP), reused ----
FOOT_HALF_LAT = 0.033

# ---- episode ----
FRAME_SKIP = 5
ACT_SCALE = np.array([0.14, 0.26, 0.26, 0.18, 0.14, 0.14, 0.26, 0.26, 0.18, 0.14])
BODY_WEIGHT_N = 2.97 * G
FALL_TILT_DEG = 45.0
FALL_CHEST_DROP = 0.26
MAX_STEPS = 10
EP_TIMEOUT_CTRL = 900
HANDOFF_MS = 20
SETTLE_MS = 420
STEP_COST = 0.5

CAUGHT_SPD = 0.14
CAUGHT_CAPT_MM = 20.0
CAUGHT_LEAN = 7.0
CAUGHT_HOLD = 40
GAIT_ENTER_SPD = 0.11

PUSH_BAND = (122.0, 140.0)
V0_NOMINAL = 0.24

_LQR_ZERO_CI = (LEG["L"]["ar"], LEG["R"]["ar"], LEG["L"]["hr"], LEG["R"]["hr"])

# leaned-neutral leg pose (hip_pitch, knee, ankle_pitch)
_SS = {"R": np.array([0.10, -0.05, 0.012]), "L": np.array([-0.10, 0.05, -0.012])}
_KNEE_SGN = {"R": -1.0, "L": 1.0}     # sign that FLEXES the knee
_ANK_SGN = {"R": 1.0, "L": -1.0}      # sign for toe-up on the ankle-pitch
_HIP_FWD_SGN = {"R": 1.0, "L": -1.0}  # sign that FLEXES the hip forward


def _obs_layout_probe(env):
    o = env._build_obs()
    return o.shape[0]


class BipedGaitEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, model_path=DEFAULT_MODEL, seed=None, push_band=PUSH_BAND,
                 dir_spread_deg=0.0, render_mode=None, max_steps=MAX_STEPS):
        super().__init__()
        if not os.path.isabs(model_path):
            model_path = os.path.join(_HERE, model_path)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._stand = StandingLQR(self.model, self.data, verbose=False)
        self._max_steps = int(max_steps)
        self.push_band = tuple(push_band)
        self.dir_spread = np.radians(dir_spread_deg)
        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None
        self._view_slow = 1.0

        omega2 = G / (NOMINAL_CHEST_Z - FLOOR_Z)
        self.omega = float(np.sqrt(omega2))
        self.kx, self.kv = _frontal_gains(omega2, FRONT_POLE_RE, FRONT_POLE_IM)

        self.clow = self.model.actuator_ctrlrange[:15, 0].copy()
        self.chigh = self.model.actuator_ctrlrange[:15, 1].copy()
        self._up_local = np.array([0.0, 1.0, 0.0])
        self._fwd_local = np.array([0.0, 0.0, -1.0])
        self._b_lf = self.model.body("L_foot").id
        self._b_rf = self.model.body("R_foot").id

        self._leg_qadr = [7 + i for i in LEG_CTRL_ORDER]
        self._nominal_legpos = self._stand.qpos0[self._leg_qadr].copy()
        self._foot_ext = self._measure_foot_extents()
        d0 = mujoco.MjData(self.model)
        d0.qpos[:] = self._stand.qpos0
        mujoco.mj_forward(self.model, d0)
        self._nominal_foot_dx = abs(float(d0.xpos[self._b_lf][0] - d0.xpos[self._b_rf][0]))

        self.action_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        self._reset_flags()
        self.data.qpos[:] = self._stand.qpos0
        self.data.qvel[:] = self._stand.qvel0
        mujoco.mj_forward(self.model, self.data)
        n = _obs_layout_probe(self)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)
        if seed is not None:
            self.reset(seed=seed)

    # ---------------------------------------------------- foot geometry
    def _measure_foot_extents(self):
        m = mujoco.MjData(self.model)
        m.qpos[:] = self._stand.qpos0
        mujoco.mj_forward(self.model, m)
        ext = {}
        for side, bid in (("L", self._b_lf), ("R", self._b_rf)):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
            mid = self.model.geom_dataid[gid]
            va, vn = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            loc = self.model.mesh_vert[va:va + vn].reshape(-1, 3)
            rot = m.geom_xmat[gid].reshape(3, 3)
            w = (rot @ loc.T).T + m.geom_xpos[gid]
            sole = w[w[:, 2] <= w[:, 2].min() + 0.003]
            b = m.xpos[bid]
            ext[side] = dict(fwd=float(-(sole[:, 1].min() - b[1])),
                             back=float(sole[:, 1].max() - b[1]),
                             xlo=float(sole[:, 0].min() - b[0]),
                             xhi=float(sole[:, 0].max() - b[0]))
        return ext

    # ---------------------------------------------------- lifecycle
    def _reset_flags(self):
        self._swing, self._stance, self._lat = "R", "L", 1.0
        self._phase = "gait"
        self._step_k = 0
        self._sk = 0
        self._ep_ctrl = 0
        self._peak_tilt = 0.0
        self._v0 = 0.0
        self._caught_streak = 0
        self._plant_streak = 0
        self._sw_lifted = False
        self._prev_action = np.zeros(10, np.float32)
        self._hip_amp = HIP_AMP0
        self._swing_p0 = np.zeros(3)
        self._settle_ms = 0
        self._settle_spd = []
        self._done_reason = ""
        self._td = {}
        self._com_fwd_prev = None
        self._an_prev = {"L": 0.0, "R": 0.0}

    def set_task(self, push_band=None, dir_spread_deg=None):
        if push_band is not None:
            self.push_band = tuple(push_band)
        if dir_spread_deg is not None:
            self.dir_spread = np.radians(dir_spread_deg)
        return self.push_band, float(np.degrees(self.dir_spread))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        for _ in range(6):
            if self._stand_and_push(options):
                break
        else:
            raise RuntimeError("reset: fell during stand+push")
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
        push_at = 15 + int(self.np_random.integers(0, 22))
        for k in range(push_at + PUSH_DURATION_STEPS + HANDOFF_MS):
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
        bs = sample_balance(m, d)
        self._peak_tilt = float(bs.up_tilt_deg)
        # if the push barely moved us, no step is needed -> go straight to settle
        if bs.com_speed_horiz < GAIT_ENTER_SPD and abs(bs.fwd_lean_deg) < 3.0:
            self._phase = "settle"
        else:
            self._phase = "gait"
            self._begin_step(first=True)
        return True

    # ---------------------------------------------------- fall / capture
    def _tilt(self):
        R = self.data.xmat[CHEST_BODY].reshape(3, 3)
        return float(np.degrees(np.arccos(min(1.0, max(-1.0, (R @ self._up_local)[2])))))

    def _fallen(self):
        return (self._tilt() > FALL_TILT_DEG
                or self.data.qpos[2] < NOMINAL_CHEST_Z - FALL_CHEST_DROP)

    def _balance(self):
        key = self.data.time
        if getattr(self, "_bs_key", None) != key:
            self._bs = sample_balance(self.model, self.data)
            self._bs_key = key
        return self._bs

    def _capture(self):
        bs = self._balance()
        h = max(float(bs.com[2]) - FLOOR_Z, 0.05)
        tc = np.sqrt(h / G)
        xi_fwd = -float(bs.com[1]) + (-float(bs.com_vel[1])) * tc
        xi_lat = float(bs.com[0]) + float(bs.com_vel[0]) * tc
        lf, rf = _foot_xy_z(self.model, self.data, "L"), _foot_xy_z(self.model, self.data, "R")
        e = self._foot_ext
        fmap = {"L": lf, "R": rf}
        both = self._phase == "settle" or (self._sk > SWING_MS + 10)
        sides = ["L", "R"] if both else [self._stance]
        fwd_hi = max(-fmap[s][1] + e[s]["fwd"] for s in sides)
        fwd_lo = min(-fmap[s][1] - e[s]["back"] for s in sides)
        lat_hi = max(fmap[s][0] + e[s]["xhi"] for s in sides)
        lat_lo = min(fmap[s][0] + e[s]["xlo"] for s in sides)
        d_fwd = max(0.0, xi_fwd - fwd_hi, fwd_lo - xi_fwd) * 1000.0
        d_lat = max(0.0, xi_lat - lat_hi, lat_lo - xi_lat) * 1000.0
        return d_fwd, d_lat, -float(bs.com_vel[1])

    # ---------------------------------------------------- frontal bias (reused)
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

    # ---------------------------------------------------- control
    def _begin_step(self, first=False):
        m, d = self.model, self.data
        if not first:
            self._swing, self._stance = self._stance, self._swing
        self._lat = 1.0 if self._swing == "R" else -1.0
        self._step_k += 1
        self._sk = 0
        self._sw_lifted = False
        self._plant_streak = 0
        self._swing_p0 = _foot_xy_z(m, d, self._swing).copy()
        mujoco.mj_subtreeVel(m, d)
        vf = -float(d.subtree_linvel[CHEST_BODY][1])
        self._hip_amp = float(np.clip(HIP_AMP0 + HIP_AMP_VGAIN * max(0.0, vf - 0.15),
                                      HIP_AMP0, HIP_AMP_MAX))

    def _compose_ctrl(self, residual):
        m, d = self.model, self.data
        bs = self._balance()
        s, st = self._swing, self._stance

        # ---- torso base: LQR, but with a forward-lean reference during the gait
        dq = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, self._stand.qpos0, d.qpos)
        dx = np.concatenate([dq, d.qvel - self._stand.qvel0])
        for ci in _LQR_ZERO_CI:
            dx[6 + ci] = 0.0
            dx[m.nv + 6 + ci] = 0.0
        dx[0] = dx[m.nv + 0] = 0.0
        if self._phase == "gait":
            vf = -float(bs.com_vel[1])
            lean = float(np.clip(TORSO_LEAN_GAIN * vf, 0.0, TORSO_LEAN_MAX))
            dx[3] -= lean                       # pitch target leaned forward
        u = np.array(self._stand.ctrl0 - self._stand.K @ dx, float)

        if self._phase == "gait":
            gs, gst = LEG[s], LEG[st]
            h0, k0, a0 = _SS[s]
            hs0, ks0, as0 = _SS[st]
            ksg = _KNEE_SGN[s]
            asg = _ANK_SGN[s]
            sk = self._sk

            # swing-leg reference
            if sk < UNLOAD_MS:
                w = sk / UNLOAD_MS
                hip = h0 + self._lat * 0.06 * w
                knee = k0 + ksg * (KNEE_BASE_FLEX + (KNEE_UNLOAD - KNEE_BASE_FLEX) * np.sin(0.5 * np.pi * w))
                sole_t = 0.10 * w
            elif sk < UNLOAD_MS + SWING_MS:
                w = (sk - UNLOAD_MS) / SWING_MS
                hip = h0 + self._lat * self._hip_amp * (0.4 + 0.6 * w)
                # knee holds the CALF vertical (proven OK for small hip amp)
                sfl = _shank_fwd_lean(m, d, s)
                knee = k0 + ksg * (KNEE_BASE_FLEX + KNEE_SHANK_KP * max(0.0, sfl))
                sole_t = 0.10 * (1.0 - w)
            else:
                # reach for the ground: hold hip forward, EXTEND the knee to drive
                # the foot down and plant, flat sole
                w = float(np.clip((sk - (UNLOAD_MS + SWING_MS)) / REACH_MS, 0.0, 1.0))
                hip = h0 + self._lat * self._hip_amp
                sfl = _shank_fwd_lean(m, d, s)
                knee = k0 + ksg * max(0.02, KNEE_BASE_FLEX + KNEE_SHANK_KP * max(0.0, sfl)
                                      - KNEE_REACH_EXT * w)
                sole_t = 0.0
            # flatten the sole with the ankle (feedback)
            spitch = _sole_pitch(m, d, s)
            ank = a0 + asg * sole_t - ANK_SOLE_KP * spitch
            u[gs["hp"]] = hip
            u[gs["kn"]] = knee
            u[gs["ap"]] = ank

            # stance leg: support + toe-off push near hand-off + HIP torque righting
            # the torso pitch (the ankle alone can't hold it in single support)
            near_end = float(np.clip((sk - (UNLOAD_MS + SWING_MS - 20)) / 40.0, 0.0, 1.0))
            u[gst["ap"]] += _ANK_SGN[st] * (-TOEOFF) * near_end   # plantarflex = push down/back
            fwd_lean_rad = np.radians(bs.fwd_lean_deg)
            lean_ref = float(np.clip(TORSO_LEAN_GAIN * max(0.0, -float(bs.com_vel[1])),
                                     0.0, TORSO_LEAN_MAX))
            # hip EXTENSION rights a forward lean; FWD_HIP_SIGN gives the flexion
            # direction, so extension is the negative of it.
            u[gst["hp"]] += -_HIP_FWD_SGN[st] * STANCE_RIGHT_KP * (fwd_lean_rad - lean_ref)

            # frontal-plane balance (both ankle rolls + loaded hip rolls)
            x_ref = X_MID_MM / 1000.0
            arb = self._frontal_bias(bs, x_ref, self._lat)
            u[LEG["L"]["ar"]] = arb
            u[LEG["R"]["ar"]] = arb
            x = float(bs.com[0]); vx = float(bs.com_vel[0])
            if self._lat < 0:
                x = 2.0 * _X_MID_M - x; vx = -vx
            hr = float(np.clip(-(HR_KX * (x - _X_MID_M) + HR_KV * vx), -HR_MAX, HR_MAX))
            if self._lat < 0:
                hr = -hr
            for sd in ("L", "R"):
                if (sd == "L" and bs.l_contact) or (sd == "R" and bs.r_contact):
                    u[LEG[sd]["hr"]] += hr

        elif self._phase == "settle":
            u = np.array(self._stand.control(m, d), float)
            if self._settle_ms > 8:
                lf, rf = _foot_xy_z(m, d, "L"), _foot_xy_z(m, d, "R")
                fa = float(lf[1] - rf[1])
                if abs(fa) > 0.030:
                    rear = "L" if fa > 0 else "R"
                    g = LEG[rear]
                    drive = float(np.clip((abs(fa) - 0.030) / 0.06, 0.0, 1.0)) * 0.5 \
                        * min(1.0, self._settle_ms / 120.0)
                    u[g["hp"]] += (-1.0 if rear == "L" else 1.0) * drive
                    u[g["kn"]] += _KNEE_SGN[rear] * 0.5 * drive
                relax = 0.30 * min(1.0, self._settle_ms / 200.0)
                for j, ci in enumerate(LEG_CTRL_ORDER):
                    u[ci] += relax * (self._nominal_legpos[j] - float(d.qpos[self._leg_qadr[j]]))
                vf = -float(bs.com_vel[1])
                brake = float(np.clip(vf, -0.35, 0.35)) * 0.9
                for side in ("L", "R"):
                    if _foot_normal_force(m, d, side) > 15.0:
                        g = LEG[side]
                        u[g["ap"]] += _ANK_SGN[side] * brake
                        u[g["hp"]] += (-1.0 if side == "L" else 1.0) * 0.5 * brake

        u[LEG_CTRL_ORDER] = u[LEG_CTRL_ORDER] + residual
        u[0:5] = 0.0
        return np.clip(u, self.clow, self.chigh)

    # ---------------------------------------------------- obs
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
        legpos = d.qpos[self._leg_qadr].copy()
        legvel = d.qvel[[6 + i for i in LEG_CTRL_ORDER]].copy()
        step_frac = min(1.0, self._sk / (UNLOAD_MS + SWING_MS + REACH_MS))
        phase_oh = np.array([self._phase == "gait", self._phase == "settle"], np.float32)

        parts, idx = [], {}

        def add(name, arr):
            arr = np.atleast_1d(np.asarray(arr, np.float32))
            idx[name] = (sum(len(p) for p in parts), len(arr))
            parts.append(arr)

        add("up", up)                                    # 3  neg 0
        add("fwd", fwd[:2])                              # 2  neg 0
        add("wvel", d.qvel[3:6])                         # 3  neg 1,2
        add("lvel", d.qvel[0:3])                         # 3  neg 0
        add("h_err", [float(d.qpos[2]) - NOMINAL_CHEST_Z])   # 1
        add("com_rel_st", [bs.com[0] - stf[0], -bs.com[1] - (-stf[1])])   # 2 neg 0
        add("com_vel", [bs.com_vel[0], -bs.com_vel[1]])  # 2 neg 0
        add("capt", [d_fwd / 100.0, d_lat / 100.0, v_fwd])   # 3
        add("contacts", [float(bs.l_contact), float(bs.r_contact),
                         lnf / BODY_WEIGHT_N, rnf / BODY_WEIGHT_N])       # 4 swap
        add("swf_rel_st", [swf[0] - stf[0], -(swf[1] - stf[1]), swf[2] - stf[2]])  # 3 neg 0
        jp0 = sum(len(p) for p in parts)
        add("legpos", legpos)                            # 10
        jv0 = sum(len(p) for p in parts)
        add("legvel", legvel)                            # 10
        add("phase", phase_oh)                           # 2
        add("gaitfrac", [step_frac, self._step_k / max(self._max_steps, 1)])   # 2
        add("prev_a", self._prev_action)                 # 10
        add("v0", [self._v0])                            # 1

        obs = np.concatenate(parts).astype(np.float32)
        neg = []
        for nm in ("up", "fwd", "lvel", "com_rel_st", "com_vel", "swf_rel_st"):
            neg.append(idx[nm][0])
        neg += [idx["wvel"][0] + 1, idx["wvel"][0] + 2]
        swap = [(idx["contacts"][0], idx["contacts"][0] + 1),
                (idx["contacts"][0] + 2, idx["contacts"][0] + 3)]
        self._obs_index = dict(neg=neg, swap=swap, jp0=jp0, jv0=jv0, preva0=idx["prev_a"][0])
        return obs

    def _obs(self):
        o = self._build_obs()
        global _OBS_NEG, _OBS_SWAP, _OBS_JP, _OBS_JV
        L = self._obs_index
        _OBS_NEG = tuple(L["neg"]); _OBS_SWAP = tuple(L["swap"])
        _OBS_JP = slice(L["jp0"], L["jp0"] + 10); _OBS_JV = slice(L["jv0"], L["jv0"] + 10)
        if self._lat < 0:
            o = _gait_mirror_obs(o, L)
        return o

    # ---------------------------------------------------- step
    def step(self, action):
        m, d = self.model, self.data
        a_canon = np.asarray(action, np.float32).clip(-1.0, 1.0)
        a_actual = a_canon if self._lat > 0 else mirror_action(a_canon)
        residual = a_actual * ACT_SCALE
        self._ep_ctrl += 1

        r = 0.0
        fell = settled = False
        for _ in range(FRAME_SKIP):
            d.ctrl[:15] = self._compose_ctrl(residual)
            mujoco.mj_step(m, d)
            self._vsync()
            self._sk += 1
            self._peak_tilt = max(self._peak_tilt, self._tilt())
            if self._fallen():
                fell = True
                break
            bs = self._balance()
            d_fwd, d_lat, v_fwd = self._capture()

            # dense shaping
            r += 0.02                                             # alive
            r += -0.010 * min(1.0, self._tilt() / 22.0)
            r += -0.014 * min(1.0, d_lat / 90.0)                  # lateral capture excess is bad
            r += -0.010 * min(1.0, abs(bs.side_lean_deg) / 14.0)
            r += -0.006 * float(np.mean(a_canon ** 2))
            if self._phase == "gait":
                # forward progress toward catching the capture point is GOOD:
                cf = bs.com_fwd
                if self._com_fwd_prev is not None and d_fwd > 8.0:
                    r += 0.9 * float(np.clip((cf - self._com_fwd_prev) / 0.004, -1.0, 1.0))
                self._com_fwd_prev = cf
                r += -0.010 * min(1.0, d_fwd / 120.0)             # but still want it caught
                r += self._foot_slip_pen()
            elif self._phase == "settle":
                self._settle_ms += 1
                self._settle_spd.append(bs.com_speed_horiz)
                r += 0.05 * float(np.clip(1.0 - bs.com_speed_horiz / 0.20, 0.0, 1.0))
                r += -0.06 * max(0.0, bs.com_speed_horiz - 0.14)
                lf, rf = _foot_xy_z(m, d, "L"), _foot_xy_z(m, d, "R")
                r += 0.02 * float(np.clip(1.0 - abs(lf[1] - rf[1]) / 0.05, -1.0, 1.0))
                if self._settle_ms >= SETTLE_MS:
                    settled = True
                    break
                continue

            # gait phase machine: detect plant, alternate, or become caught
            if self._phase == "gait":
                sw_nf = _foot_normal_force(m, d, self._swing)
                if not self._sw_lifted and sw_nf < 3.0:
                    self._sw_lifted = True
                swc = bs.r_contact if self._swing == "R" else bs.l_contact
                planted = self._sw_lifted and swc and sw_nf > PLANT_NF
                self._plant_streak = self._plant_streak + 1 if planted else 0
                lean = float(bs.fwd_lean_deg)
                caught = (bs.com_speed_horiz < CAUGHT_SPD and d_fwd < CAUGHT_CAPT_MM
                          and abs(lean) < CAUGHT_LEAN and bs.l_contact and bs.r_contact)
                self._caught_streak = self._caught_streak + 1 if caught else 0
                step_done = self._plant_streak >= PLANT_HOLD or self._sk >= STEP_MAX_MS
                if step_done:
                    self._snapshot_td()
                    r += self._step_reward()
                    r -= STEP_COST
                    if (self._caught_streak >= CAUGHT_HOLD or self._step_k >= self._max_steps) \
                            and bs.l_contact and bs.r_contact:
                        self._phase = "settle"
                        self._sk = 0
                        self._settle_ms = 0
                        self._settle_spd = []
                        self._com_fwd_prev = None
                    else:
                        self._begin_step()

        self._prev_action = a_canon.copy()

        if fell:
            return self._obs(), r - 45.0, True, False, self._info(fell=True)
        if settled:
            self._done_reason = "settled"
            tr, info = self._terminal(timed_out=False)
            return self._obs(), r + tr, True, False, info
        if self._ep_ctrl > EP_TIMEOUT_CTRL:
            self._done_reason = "timeout"
            tr, info = self._terminal(timed_out=True)
            return self._obs(), r + tr, True, False, info
        return self._obs(), r + 0.2, False, False, self._info(fell=False)

    def _foot_slip_pen(self):
        d = self.data
        pen = 0.0
        for side, bid in (("L", self._b_lf), ("R", self._b_rf)):
            if _foot_normal_force(self.model, d, side) > 20.0:
                vxy = float(np.hypot(d.cvel[bid][3], d.cvel[bid][4]))
                pen += -0.03 * min(1.0, vxy / 0.15)
        return pen

    def _snapshot_td(self):
        m, d = self.model, self.data
        s, st = self._swing, self._stance
        sf = _foot_xy_z(m, d, s)
        stf = _foot_xy_z(m, d, st)
        self._td[self._step_k] = dict(
            sep_mm=float(-(sf[1] - stf[1]) * 1000.0),
            fwd_mm=float(-(sf[1] - self._swing_p0[1]) * 1000.0),
            lat_mm=float((sf[0] - self._swing_p0[0]) * 1000.0),
            sole_pitch=float(np.degrees(_sole_pitch(m, d, s))),
            sole_roll=float(np.degrees(_sole_roll(m, d, s))),
            nf=float(_foot_normal_force(m, d, s)),
            planted=self._plant_streak >= PLANT_HOLD,
        )

    def _step_reward(self):
        td = self._td[self._step_k]
        r = 0.0
        r += 0.6 * float(np.clip(1.0 - abs(td["sole_pitch"]) / 12.0, -0.5, 1.0))   # FLAT foot
        r += 0.3 * float(np.clip(1.0 - abs(td["sole_roll"]) / 12.0, -0.5, 1.0))
        r += 0.5 * float(np.clip(td["nf"] / 20.0, 0.0, 1.0))
        r += 0.5 * float(np.clip((td["fwd_mm"] - 10.0) / 60.0, -0.5, 1.0))         # a real forward step
        r += -0.5 * float(np.clip((abs(td["lat_mm"]) - 22.0) / 40.0, 0.0, 1.5))    # not sideways
        r += 0.3 if td["planted"] else -0.3
        return r

    def _terminal(self, timed_out):
        m, d = self.model, self.data
        b = sample_balance(m, d)
        ds = b.l_contact and b.r_contact
        d_fwd, d_lat, v_fwd = self._capture()
        sp = max(abs(np.degrees(_sole_pitch(m, d, "L"))), abs(np.degrees(_sole_pitch(m, d, "R"))))
        sr = max(abs(np.degrees(_sole_roll(m, d, "L"))), abs(np.degrees(_sole_roll(m, d, "R"))))
        spd_end = float(np.mean(self._settle_spd[-40:])) if self._settle_spd else b.com_speed_horiz
        all_pl = all(self._td[j]["planted"] for j in self._td) if self._td else True
        lf, rf = _foot_xy_z(m, d, "L"), _foot_xy_z(m, d, "R")
        fa_sep_mm = abs(float(lf[1] - rf[1])) * 1000.0
        wid_err_mm = abs(abs(float(lf[0] - rf[0])) - self._nominal_foot_dx) * 1000.0
        legdev = float(np.mean(np.abs(d.qpos[self._leg_qadr] - self._nominal_legpos)))

        caught = d_fwd < CAUGHT_CAPT_MM and spd_end < 0.14
        upright = (b.up_tilt_deg < 11.0 and abs(b.side_lean_deg) < 10.0
                   and b.chest_z > NOMINAL_CHEST_Z - 0.11 and self._peak_tilt <= 34.0)
        success = bool(caught and upright and all_pl and not timed_out)
        flat_ok = bool(success and ds and sp < 14.0 and sr < 14.0)
        neutral_ok = bool(flat_ok and fa_sep_mm < 45.0 and wid_err_mm < 42.0 and legdev < 0.22)

        r = 0.0
        r += 26.0 if success else 0.0
        r += 8.0 if flat_ok else 0.0
        r += 6.0 if neutral_ok else 0.0
        r += -10.0 if timed_out else 0.0
        r += 3.0 * float(np.clip(1.0 - fa_sep_mm / 90.0, 0.0, 1.0))
        r += 2.0 * float(np.clip(1.0 - legdev / 0.35, 0.0, 1.0))
        r += 2.0 * float(np.clip(1.0 - b.up_tilt_deg / 12.0, 0.0, 1.0))
        r += 1.5 * float(np.clip(1.0 - abs(b.side_lean_deg) / 12.0, 0.0, 1.0))
        r += 4.0 * float(np.clip(1.0 - spd_end / 0.24, 0.0, 1.0))
        r += -10.0 * float(np.clip((spd_end - 0.16) / 0.20, 0.0, 1.0))
        r += 1.5 if ds else -1.0
        info = self._info(fell=False)
        info.update(success=success, flat_ok=flat_ok, neutral_ok=neutral_ok,
                    n_steps=int(self._step_k), end_spd=float(spd_end),
                    end_capt_fwd_mm=float(d_fwd), end_fa_sep_mm=float(fa_sep_mm),
                    end_legdev=float(legdev), sole_pitch_max=float(sp),
                    all_planted=bool(all_pl), timed_out=bool(timed_out))
        return r, info

    def _info(self, fell):
        return dict(push_n=float(self._push_n), phase=self._phase, n_steps=int(self._step_k),
                    peak_up=float(self._peak_tilt), v0=float(self._v0), fell=bool(fell),
                    success=False, flat_ok=False, neutral_ok=False, end_spd=9.9,
                    end_capt_fwd_mm=999.0, end_fa_sep_mm=999.0, end_legdev=9.9,
                    sole_pitch_max=99.0, all_planted=False, timed_out=False,
                    done_reason=self._done_reason)

    # ---------------------------------------------------- render
    def _vsync(self):
        if self.render_mode != "human":
            return
        import time
        if getattr(self, "_viewer", None) is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.azimuth = 90
            self._viewer.cam.elevation = -8
            self._viewer.cam.distance = 2.2
            self._t_last = time.perf_counter()
        if not self._viewer.is_running():
            raise KeyboardInterrupt("viewer closed")
        self._viewer.cam.lookat[:] = [0.0, float(self.data.xpos[CHEST_BODY][1]), 0.95]
        self._viewer.sync()
        dt = self.model.opt.timestep * self._view_slow
        now = time.perf_counter()
        if dt - (now - self._t_last) > 0:
            time.sleep(dt - (now - self._t_last))
        self._t_last = time.perf_counter()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 320, 300)
            self._cam = mujoco.MjvCamera()
            self._cam.azimuth = 110
            self._cam.elevation = -22
            self._cam.distance = 1.5
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


# obs-mirror indices (filled per build)
_OBS_NEG = ()
_OBS_SWAP = ()
_OBS_JP = slice(0, 0)
_OBS_JV = slice(0, 0)


def _gait_mirror_obs(o, L):
    o = np.asarray(o, np.float32).copy()
    for i in L["neg"]:
        o[i] = -o[i]
    for a, b in L["swap"]:
        o[a], o[b] = o[b], o[a]
    jp = slice(L["jp0"], L["jp0"] + 10)
    jv = slice(L["jv0"], L["jv0"] + 10)
    o[jp] = (_LEGMIR_SIGN * o[jp][_LEGMIR_PERM]).astype(np.float32)
    o[jv] = (_LEGMIR_SIGN * o[jv][_LEGMIR_PERM]).astype(np.float32)
    p0 = L["preva0"]
    o[p0:p0 + 10] = mirror_action(o[p0:p0 + 10])
    return o


# ============================================================ CLI
def _load_policy(run_dir, band, model=DEFAULT_MODEL):
    import json
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    hp = json.load(open(f"{run_dir}/hparams.json"))
    tmp = DummyVecEnv([lambda: BipedGaitEnv(model_path=model, push_band=band)])
    pf = f"{run_dir}/policy_best.pth" if os.path.exists(f"{run_dir}/policy_best.pth") else f"{run_dir}/policy.pth"
    vf = f"{run_dir}/vecnormalize_best.pkl" if os.path.exists(f"{run_dir}/vecnormalize_best.pkl") else f"{run_dir}/vecnormalize.pkl"
    vn = VecNormalize.load(vf, tmp)
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    mm = PPO("MlpPolicy", tmp, device="cpu",
             policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(pf, map_location="cpu", weights_only=True))
    mm.policy.eval()
    print(f"  loaded {pf}")
    return mm, lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)


def _run(env, pol, n, band, seed0=1000, render=None):
    frames = []
    succ = fell = flat = neut = 0
    nsteps, spds, travels = [], [], []
    import torch
    for i in range(n):
        o, _ = env.reset(seed=seed0 + i)
        y0 = float(env.data.xpos[CHEST_BODY][1])
        ymin = y0
        done = False
        info = {}
        ep = []
        while not done:
            if pol:
                with torch.no_grad():
                    a, _ = pol[0].predict(pol[1](o), deterministic=True)
            else:
                a = np.zeros(10, np.float32)
            o, rr, term, trunc, info = env.step(a)
            done = term or trunc
            ymin = min(ymin, float(env.data.xpos[CHEST_BODY][1]))
            if render and i < 6:
                ep.append(env.render())
        succ += int(info.get("success", False))
        fell += int(info.get("fell", False))
        flat += int(info.get("flat_ok", False))
        neut += int(info.get("neutral_ok", False))
        nsteps.append(info.get("n_steps", 0))
        spds.append(info.get("end_spd", 9.9))
        travels.append((y0 - ymin) * 1000.0)
        if render and i < 6:
            frames.append(ep)
        print(f"  ep {i:2d} push {info.get('push_n', 0):6.1f}  {info.get('n_steps', 0)} steps  "
              f"{'SUCC' if info.get('success') else ('FELL' if info.get('fell') else info.get('done_reason', '?'))}"
              f"{' FLAT' if info.get('flat_ok') else ''}{' NEUTRAL' if info.get('neutral_ok') else ''}  "
              f"travel {travels[-1]:.0f}mm  endSpd {info.get('end_spd', 0):.2f}  peakTilt {info.get('peak_up', 0):.0f}")
    print(f"\n  {'POLICY' if pol else 'ZERO-ACTION'}  ({n} eps, band {band}):")
    print(f"    SUCCESS {succ}/{n}  fell {fell}/{n}  flat {flat}/{n}  neutral {neut}/{n}")
    print(f"    steps {np.bincount(nsteps, minlength=6).tolist()}  median travel {np.median(travels):.0f}mm  "
          f"median endSpd {np.median(spds):.2f}")
    if render and frames:
        import imageio.v2 as imageio
        H = max(len(f) for f in frames)
        tiles = []
        for t in range(H):
            row = [f[min(t, len(f) - 1)] for f in frames]
            grid = (np.vstack([np.hstack(row[:3]), np.hstack(row[3:6])]) if len(row) >= 6
                    else np.hstack(row))
            tiles.append(grid)
        imageio.mimsave(render, tiles, fps=40, macro_block_size=1)
        print(f"    wrote {render}")


def _scripted_sweep(lo, hi, step, reps, model):
    print(f"\n=== scripted gait primitive sweep ({model}) ===")
    print(f"{'push':>5} {'succ':>7} {'steps':>18} {'travel':>8} {'endSpd':>7} {'soleP':>7}")
    for mag in np.arange(lo, hi + 0.1, step):
        env = BipedGaitEnv(model_path=model, push_band=(mag, mag), max_steps=MAX_STEPS)
        S = F = 0
        ns, tr, es, sps = [], [], [], []
        for i in range(reps):
            o, _ = env.reset(seed=500 + int(mag) * 3 + i,
                             options={"push_n": float(mag), "push_dir_rad": -np.pi / 2})
            y0 = float(env.data.xpos[CHEST_BODY][1]); ymin = y0
            done = False; info = {}
            while not done:
                o, rr, t1, t2, info = env.step(np.zeros(10, np.float32))
                done = t1 or t2
                ymin = min(ymin, float(env.data.xpos[CHEST_BODY][1]))
            S += int(info.get("success", False)); F += int(info.get("fell", False))
            ns.append(info.get("n_steps", 0)); tr.append((y0 - ymin) * 1000.0)
            es.append(info.get("end_spd", 9.9)); sps.append(info.get("sole_pitch_max", 99))
        env.close()
        flag = "  <-- FALLS" if F > reps // 2 else ""
        print(f"{mag:5.0f} {S:>3}/{reps} f{F:<2} {str(ns):>18} {np.median(tr):8.0f} "
              f"{np.median(es):7.2f} {np.median(sps):7.0f}{flag}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--render", default=None)
    ap.add_argument("--band", default="122,140")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lo", type=float, default=120)
    ap.add_argument("--hi", type=float, default=190)
    ap.add_argument("--step", type=float, default=10)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    band = tuple(float(x) for x in a.band.split(","))
    if a.scripted:
        _scripted_sweep(a.lo, a.hi, a.step, a.reps, a.model)
    if a.smoke:
        pol = _load_policy(a.policy, band, a.model) if a.policy else None
        env = BipedGaitEnv(model_path=a.model, push_band=band,
                           render_mode=("rgb_array" if a.render else None))
        _run(env, pol, a.n, band, render=a.render)
        env.close()
    if not (a.smoke or a.scripted):
        ap.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
