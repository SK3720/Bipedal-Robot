"""Learn to WALK.

The push-recovery milestone is done (runs/walk_s5).  This env is for the actual
goal: a forward walking gait.

  * MODEL: robot/_exp_hands_5x_feet_1p5.xml -- foot mesh x1.5 (clears the ground,
    unlike the 2x "duck feet"), hand mass x5 with the ARMS UNLOCKED so the policy
    can use them as counterweights / for anti-phase swing.  robot.xml and the
    frozen runs/recovery_s1 checkpoint are untouched.
  * ACTION (14): bounded position residual on a CPG gait reference --
    L/R hip-roll,hip-pitch,knee,ankle-pitch,ankle-roll  +  L/R shoulder,elbow.
    Neck fixed.  Zero action == the open-loop reference gait.
  * REFERENCE: a phase-clock walking trajectory (hip/knee/ankle per leg,
    contralateral arm swing) at a commanded frequency.  A light torso-attitude PD
    holds pitch/roll near a slightly-forward target -- NOT the feet-together LQR,
    which fights walking.
  * REWARD (standard locomotion recipe): track a forward-speed target + alive +
    upright + feet-air-time (real alternating steps, no shuffle / no standing
    still) + foot clearance - foot slip - lateral/yaw drift - action rate -
    torque; big negative on a fall.
  * Episode: start standing, walk; ends on a fall or after ~6 s.

    python biped_locomotion_env.py --smoke                 # zero-action reference
    python biped_locomotion_env.py --smoke --render w.mp4
    python biped_locomotion_env.py --smoke --policy runs/loco_x1
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

from recovery_metrics import CHEST_BODY, FLOOR_Z, G, NOMINAL_CHEST_Z, sample_balance, \
    _foot_normal_force, _foot_xy_z
from standing_balance_lqr import StandingLQR
from biped_env import DEFAULT_POSE
from step_primitive import _sole_pitch

DEFAULT_MODEL = "robot/_exp_hands_3x.xml"   # 3x hands (unlocked), NORMAL (canonical)
# foot geometry (sole ~0.088 m fore-aft x 0.066 m).  Harder: the small sagittal
# support polygon is the pitch/torque wall the whole project has fought; the
# long-foot run (loco_w2) beat it with a 0.21 m foot.  Trying normal feet with a
# heavier forward-progress reward + more steps.

# ---- ctrl indices: 0 neck | 1..4 arms | 5..9 L leg | 10..14 R leg ----
NECK = 0
ARM = dict(L=dict(sh=1, el=2), R=dict(sh=3, el=4))
LEG = dict(L=dict(hr=5, hp=6, kn=7, ap=8, ar=9),
           R=dict(hr=10, hp=11, kn=12, ap=13, ar=14))
ACT_CTRL = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4]     # 14 actuated (order of the action)

# joint-axis signs (measured): + hip-pitch value flexes the hip forward for R,
# backward for L; knee flex is negative for R / positive for L; etc.
HP_FWD = dict(L=-1.0, R=1.0)     # sign that flexes the hip forward
KN_FLEX = dict(L=1.0, R=-1.0)    # sign that flexes the knee
AP_DORSI = dict(L=-1.0, R=1.0)   # sign that dorsiflexes the ankle (toe up)
SH_FWD = dict(L=-1.0, R=1.0)     # sign that swings the shoulder forward

# ---- gait clock ----
GAIT_HZ = 0.80           # steps per second per leg (full cycle = 2 steps)
DUTY = 0.64             # per-foot stance fraction; 2*0.64 -> ~28% double support, 0 flight
HIP_AMP = 0.24         # peak fore-aft hip excursion (rad)
HIP_BIAS = 0.0
KNEE_SWING = 0.66     # peak knee flexion mid-swing
KNEE_STANCE = 0.12    # knee flex held through stance (soft knee)
ANK_CLEAR = 0.18     # ankle dorsiflexion for toe clearance mid-swing
ANK_PUSH = 0.22     # ankle plantarflexion push at toe-off
ARM_AMP = 0.5      # shoulder fore-aft swing (contralateral to the leg)
ELB_FLEX = 0.30   # static elbow flex
LAT_ROLL = 0.20   # ankle/hip-roll lean toward the stance leg (weight shift so the
                  # swing leg can unload) -- synced to the gait phase
LAT_HR = 0.12

# ---- torso PD (attitude only) ----
WALK_LEAN = 0.04             # forward-pitch bias in the attitude target while walking
                             # (small on normal feet -- little margin for pitch)
STEP_FWD = 0.12             # extra forward reach of the swing foot -> net step length

# ---- BIGGER-STEPS mode (stride_mode=True) ----
# measured w3 (normal-feet walk) stride ~0.16 m/step, cadence ~2.7 steps/s.
STRIDE_REF = 0.16           # per-foot step length the walking policy already does
STRIDE_TGT_DEFAULT = 0.17   # curriculum raises this toward STRIDE_MAX
STRIDE_MAX = 0.25          # small feet + torque limit -> a real ceiling on stride
BIGSTEP_GAIT_HZ = 0.56      # slower reference cadence -> more time per (longer) step
BIGSTEP_SPEED_TGT = 0.28    # fixed while stride grows -> cadence must DROP, not steps shrink

# ---- frontal-plane feedback stabiliser (the piece that makes single support
#      survivable -- drives the CoP toward the loaded foot, ankle-roll + hip-roll)
FR_KX, FR_KV = 3.2, 0.9     # CoP feedback on lateral CoM position / velocity
FR_AR_MAX = 0.28           # ankle-roll authority (rad)
FR_HR_KX, FR_HR_KV, FR_HR_MAX = 2.6, 0.5, 0.34   # lateral hip-roll strategy
# ---- sagittal ankle CoP feedback (helps fore-aft single-support balance) ----
SG_KV, SG_AP_MAX = 0.6, 0.20

# ---- episode / reward ----
FRAME_SKIP = 5                                        # 1 kHz sim -> 200 Hz control
EP_SECONDS = 7.0
EP_STEPS = int(EP_SECONDS * 1000 / FRAME_SKIP)        # control steps per episode
# full-authority leg residual (no CPG rhythm imposed -- the policy IS the gait);
# generous so it can actually step.  arms wide open for counterweight use.
ACT_SCALE = np.array([0.20, 0.52, 0.55, 0.34, 0.20,    # L leg (hr,hp,kn,ap,ar)
                      0.20, 0.52, 0.55, 0.34, 0.20,    # R leg
                      0.55, 0.42, 0.55, 0.42])         # arms
                      # enough authority to add the forward drive + catch pitch
CPG_GAIN = float(os.environ.get("LOCO_CPG_GAIN", "0.75"))
                         # 0 -> pure full-authority; >0 -> blend in the CPG rhythm.
                         # 0.75: rhythm + step shape are a prior the policy can
                         # still meaningfully reshape (it needs to add the forward
                         # drive the open-loop reference lacks).  Override per-run
                         # with the LOCO_CPG_GAIN env var.

FALL_TILT_DEG = 48.0
FALL_CHEST_DROP = 0.30
AIR_TIME_TGT = 0.18          # s -- reward each foot for ~this much swing air-time
SPEED_TGT_DEFAULT = 0.25     # m/s forward (curriculum raises this)

BODY_WEIGHT_N = 3.075 * G


def _smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * t))


class BipedLocomotionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, model_path=DEFAULT_MODEL, seed=None, render_mode=None,
                 speed_tgt=SPEED_TGT_DEFAULT, gait_hz=GAIT_HZ, stride_mode=False,
                 robust=0.0):
        super().__init__()
        if not os.path.isabs(model_path):
            model_path = os.path.join(_HERE, model_path)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._stand = StandingLQR(self.model, self.data, verbose=False)
        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None
        self._view_slow = 1.0
        self.speed_tgt = float(speed_tgt)
        # BIGGER-STEPS mode: a slower reference cadence + a stride-length target
        # the reward peaks on, + a cadence cap.  Off by default -> env is byte-
        # identical to the walking (w3) setup.
        self.stride_mode = bool(stride_mode)
        self.stride_tgt = STRIDE_TGT_DEFAULT
        self.gait_hz = float(BIGSTEP_GAIT_HZ if self.stride_mode else gait_hz)
        self.assist = 0.0        # 1 -> full training-wheel torso help, 0 -> none
        # SIM-TO-REAL robustness: 0 -> nominal (w3 setup unchanged); >0 scales the
        # domain randomisation + disturbance + smoothness pressure (curriculum 0->1).
        self.robust = float(robust)
        self._nom_friction = self.model.geom_friction.copy()
        self._nom_damping = self.model.dof_damping.copy()
        self._nom_gain = self.model.actuator_gainprm.copy()
        self._nom_bias = self.model.actuator_biasprm.copy()
        self._nom_bmass = self.model.body_mass.copy()

        self.clow = self.model.actuator_ctrlrange[:15, 0].copy()
        self.chigh = self.model.actuator_ctrlrange[:15, 1].copy()
        self._up_local = np.array([0.0, 1.0, 0.0])
        self._fwd_local = np.array([0.0, 0.0, -1.0])
        self._b_lf = self.model.body("L_foot").id
        self._b_rf = self.model.body("R_foot").id
        self._q0 = self._stand.qpos0.copy()
        # sole collision-mesh vertices (local) -> TRUE sole-to-floor clearance, so
        # a scrape / pivot of an enlarged foot is never scored as a step.
        self._sole_loc = {}
        for side in ("L", "R"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
            mid = self.model.geom_dataid[gid]
            va, vn2 = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            loc = self.model.mesh_vert[va:va + vn2].reshape(-1, 3).copy()
            zl = loc[:, 2]
            keep = zl <= zl.min() + 0.4 * (zl.max() - zl.min() + 1e-9)
            self._sole_loc[side] = (gid, loc[keep])
        self._leg_qadr = [7 + i for i in ACT_CTRL]
        self._leg_vadr = [6 + i for i in ACT_CTRL]
        # attitude-only mask on the LQR state error: keep base height + base
        # orientation (+ their rates), drop everything else so the torso
        # stabiliser does NOT fight the gait's leg motion or forward travel.
        nv = self.model.nv
        self._att_mask = np.zeros(2 * nv)
        for i in (2, 3, 4, 5):
            self._att_mask[i] = 1.0
            self._att_mask[nv + i] = 1.0

        self.action_space = spaces.Box(-1.0, 1.0, (14,), np.float32)
        self._reset_state()
        self.data.qpos[:] = self._q0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        n = self._build_obs().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)
        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------ lifecycle
    def _reset_state(self):
        self._phase = 0.0
        self._t = 0
        self._prev_action = np.zeros(14, np.float32)
        self._air = dict(L=0.0, R=0.0)          # running swing air-time (s)
        self._air_true = dict(L=0.0, R=0.0)     # air-time with the WHOLE sole clear
        self._peak_clr = dict(L=0.0, R=0.0)     # peak true sole clearance this swing
        self._air_cred = dict(L=0.0, R=0.0)     # air-time already rewarded this swing
        self._was_stance = dict(L=True, R=True)
        self._lift_fwd = dict(L=0.0, R=0.0)
        # chatter-robust per-foot swing state machine (for genuine step / stride)
        self._foot_state = dict(L="stance", R="stance")
        self._contact_run = dict(L=99, R=99)     # consecutive solid-contact substeps
        self._swing_run = dict(L=0, R=0)         # consecutive genuinely-airborne substeps
        self._last_land_fwd = dict(L=None, R=None)
        self._swing_air_ms = dict(L=0.0, R=0.0)
        self._stride_ema = STRIDE_REF            # smoothed recent step length
        self._step_ct = 0
        self._act_delay = 0                      # control-step action latency (robust)
        self._act_buf = []
        self._obs_noise = 0.0
        self._push = np.zeros(3)                 # active torso disturbance force (N)
        self._push_until = -1
        self._next_push = 10 ** 9
        self._lc_prev = True
        self._rc_prev = True
        self._chatter = 0
        self._x0 = 0.0
        self._flight_ms = 0
        self._flight_run = 0
        self._dived = False
        self._last_step_t = dict(L=0.0, R=0.0)   # sim-time of the last genuine landing
        self._strides = []                       # landed step lengths (m)

    def set_task(self, speed_tgt=None, gait_hz=None, assist=None, stride_tgt=None,
                 robust=None):
        if stride_tgt is not None:
            self.stride_tgt = float(np.clip(stride_tgt, STRIDE_REF, STRIDE_MAX))
        if speed_tgt is not None:
            self.speed_tgt = float(speed_tgt)
        if gait_hz is not None:
            self.gait_hz = float(gait_hz)
        if assist is not None:
            self.assist = float(np.clip(assist, 0.0, 1.0))
        if robust is not None:
            self.robust = float(np.clip(robust, 0.0, 1.0))
        return self.speed_tgt, self.gait_hz, self.assist, self.robust

    def _pitch_roll_signed(self):
        """(forward-pitch, side-roll) of the chest in rad, small-angle signed."""
        R = self.data.xmat[CHEST_BODY].reshape(3, 3)
        up = R @ self._up_local
        return float(up[1]), float(-up[0])   # +pitch => leaning forward (-Y up-tilt)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self._q0
        d.qvel[:] = 0.0
        self._reset_state()

        # ---- SIM-TO-REAL domain randomisation (scaled by self.robust) ----
        rb = self.robust
        m.geom_friction[:] = self._nom_friction
        m.dof_damping[:] = self._nom_damping
        m.actuator_gainprm[:] = self._nom_gain
        m.actuator_biasprm[:] = self._nom_bias
        m.body_mass[:] = self._nom_bmass
        if rb > 0.0:
            u = self.np_random.uniform
            m.geom_friction[:, 0] = self._nom_friction[:, 0] * (1.0 + rb * u(-0.35, 0.35))
            m.dof_damping[6:] = self._nom_damping[6:] * (1.0 + rb * u(-0.4, 0.6, size=m.nv - 6))
            # position-servo gain spread (kp lives in gainprm[0] / biasprm[1])
            gsc = 1.0 + rb * u(-0.18, 0.18, size=self._nom_gain.shape[0])
            m.actuator_gainprm[:, 0] = self._nom_gain[:, 0] * gsc
            m.actuator_biasprm[:, 1] = self._nom_bias[:, 1] * gsc
            # link masses +-, plus a small random torso payload
            m.body_mass[:] = self._nom_bmass * (1.0 + rb * u(-0.12, 0.12, size=m.nbody))
            m.body_mass[CHEST_BODY] += rb * u(0.0, 0.25) * float(self._nom_bmass[CHEST_BODY])
            # (no mj_setConst -- it corrupts the servo/spring reference; mj_forward
            #  in the warmup below propagates the mass change through qM fine)
            # control channel: action latency 0-2 steps, obs gaussian noise
            self._act_delay = int(self.np_random.integers(0, 1 + round(2 * rb) + 1))
            self._act_buf = [np.zeros(14, np.float32)] * self._act_delay
            self._obs_noise = rb * 0.02
            self._next_push = int(self.np_random.integers(60, 180))
        rsi = self.np_random.random() < 0.5
        # non-RSI: start at a hand-off phase so the CPG doesn't lurch a standing
        # robot straight into a swing.
        self._phase = 2.0 * np.pi * float(self.np_random.choice([0.0, 0.5]))
        d.qpos[7:22] += self.np_random.uniform(-0.02, 0.02, 15)
        d.qpos[2] += self.np_random.uniform(-0.005, 0.005)

        # REFERENCE-STATE INITIALISATION: half the episodes start MID-STRIDE
        # (legs offset per the gait phase, CoM already moving forward, one foot
        # unloaded).  This breaks the "just stand still" local optimum -- the
        # policy has to learn to CONTINUE a walk it's dropped into.
        if rsi:
            # start MID-STRIDE but only near a DOUBLE-SUPPORT hand-off (both feet
            # ~down): phase in the last part of one leg's stance / start of the
            # other's -> avoids dropping the robot into a deep single-support pose
            # it can't catch.  CoM already travelling forward near target speed.
            self._phase = 2.0 * np.pi * float(self.np_random.choice([0.0, 0.5])
                                              + self.np_random.uniform(-0.04, 0.04))
            for side in ("L", "R"):
                ph = (self._phase / (2.0 * np.pi) + (0.0 if side == "L" else 0.5)) % 1.0
                hp, kn, ap = self._leg_ref(side, ph)
                d.qpos[7 + LEG[side]["hp"]] = hp
                d.qpos[7 + LEG[side]["kn"]] = kn
                d.qpos[7 + LEG[side]["ap"]] = ap
            d.qvel[1] = -float(self.np_random.uniform(0.9, 1.15)) * self.speed_tgt
            d.qvel[6 + LEG["L"]["hp"]] += self.np_random.uniform(-0.4, 0.4)
            d.qvel[6 + LEG["R"]["hp"]] += self.np_random.uniform(-0.4, 0.4)

        d.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_forward(m, d)
        for _ in range(4):
            d.ctrl[:15] = self._compose_ctrl(np.zeros(14, np.float32))
            mujoco.mj_step(m, d)
            self._vsync()
        self._x0 = float(d.subtree_com[CHEST_BODY][1])   # world Y; forward = -Y
        return self._obs(), {}

    # ------------------------------------------------ gait reference
    def _leg_ref(self, side, ph):
        """(hip, knee, ankle) joint targets for `side` at cycle phase ph in [0,1),
        ph=0 == heel-strike.  Reference is open-loop; the policy corrects it."""
        hp0 = self._q0[7 + LEG[side]["hp"]]
        kn0 = self._q0[7 + LEG[side]["kn"]]
        ap0 = self._q0[7 + LEG[side]["ap"]]
        if ph < DUTY:                      # STANCE: hip rolls gently front -> back
            s = ph / DUTY
            hip = HP_FWD[side] * HIP_AMP * (0.5 * np.cos(np.pi * s))       # +A/2 -> -A/2
            knee = KN_FLEX[side] * (KNEE_STANCE + 0.08 * np.sin(np.pi * s))
            push = _smooth((s - 0.70) / 0.30) if s > 0.70 else 0.0
            ank = AP_DORSI[side] * (0.03 - (ANK_PUSH + 0.03) * push)
        else:                             # SWING: knee flexes for clearance, hip swings fwd
            s = (ph - DUTY) / (1.0 - DUTY)
            # in bigger-steps mode, scale the swing reach / lift with the stride
            # target so the open-loop reference already takes longer strides.
            sc = float(np.clip(self.stride_tgt / STRIDE_REF, 1.0, 2.2)) if self.stride_mode else 1.0
            hip = HP_FWD[side] * (HIP_AMP * (-0.5 + _smooth(s)) * (0.6 + 0.4 * sc)
                                  + STEP_FWD * sc * _smooth(s))
            knee = KN_FLEX[side] * (KNEE_STANCE + KNEE_SWING * (0.75 + 0.25 * sc) * np.sin(np.pi * s))
            ank = AP_DORSI[side] * (ANK_CLEAR * (0.8 + 0.2 * sc) * np.sin(np.pi * s))
        return hp0 + hip, kn0 + knee, ap0 + ank

    def _compose_ctrl(self, residual, settle=False):
        m, d = self.model, self.data
        nv = m.nv

        # ---- nominal: standing ctrl0 (+ optional CPG rhythm blended in) ----
        u = np.array(self._stand.ctrl0, float)
        if not settle and CPG_GAIN > 0.0:
            g = CPG_GAIN
            lat = np.sin(self._phase)
            for side in ("L", "R"):
                ph = (self._phase / (2.0 * np.pi) + (0.0 if side == "L" else 0.5)) % 1.0
                hp, kn, ap = self._leg_ref(side, ph)
                sgn = 1.0 if side == "L" else -1.0
                u[LEG[side]["hp"]] += g * (hp - u[LEG[side]["hp"]])
                u[LEG[side]["kn"]] += g * (kn - u[LEG[side]["kn"]])
                u[LEG[side]["ap"]] += g * (ap - u[LEG[side]["ap"]])
                u[LEG[side]["ar"]] += g * sgn * LAT_ROLL * lat
                u[LEG[side]["hr"]] += g * sgn * LAT_HR * lat
                oph = (self._phase / (2.0 * np.pi) + (0.5 if side == "L" else 0.0)) % 1.0
                u[ARM[side]["sh"]] += g * SH_FWD[side] * ARM_AMP * np.sin(2.0 * np.pi * oph)
                u[ARM[side]["el"]] += g * (1.0 if side == "L" else -1.0) * ELB_FLEX

        # ---- attitude-only LQR correction (keeps the torso upright without
        #      fighting the gait or forward travel).  A small forward-pitch bias
        #      in the target makes the CoM fall forward, which the alternating
        #      steps catch -> net forward walking. ----
        dq = np.zeros(nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, self._stand.qpos0, d.qpos)
        if not settle:
            dq[3] -= WALK_LEAN
        dx = np.concatenate([dq, d.qvel - self._stand.qvel0]) * self._att_mask
        u = u + (-(self._stand.K @ dx))

        # ---- frontal-plane feedback stabiliser: drive the CoP toward the LOADED
        #      foot (ankle-roll) + a lateral hip-roll strategy.  This is what
        #      makes the single-support phase of a step survivable. ----
        if not settle:
            bs = sample_balance(m, d)
            lnf = _foot_normal_force(m, d, "L")
            rnf = _foot_normal_force(m, d, "R")
            lf, rf = _foot_xy_z(m, d, "L")[0], _foot_xy_z(m, d, "R")[0]
            tot = lnf + rnf + 1e-6
            x_ref = (lnf * lf + rnf * rf) / tot        # lateral CoP target = load-weighted foot x
            cx = float(bs.com[0]); cvx = float(bs.com_vel[0])
            ar = float(np.clip(-(FR_KX * (cx - x_ref) + FR_KV * cvx) / 6.0, -FR_AR_MAX, FR_AR_MAX))
            u[LEG["L"]["ar"]] += ar
            u[LEG["R"]["ar"]] += ar
            hr = float(np.clip(-(FR_HR_KX * (cx - x_ref) + FR_HR_KV * cvx), -FR_HR_MAX, FR_HR_MAX))
            if lnf > 12.0:
                u[LEG["L"]["hr"]] += hr
            if rnf > 12.0:
                u[LEG["R"]["hr"]] += hr

        # ---- policy residual ----
        u[ACT_CTRL] = u[ACT_CTRL] + residual
        u[NECK] = 0.0
        return np.clip(u, self.clow, self.chigh)

    # ------------------------------------------------ obs
    def _chest_axes(self):
        R = self.data.xmat[CHEST_BODY].reshape(3, 3)
        return R @ self._up_local, R @ self._fwd_local

    def _build_obs(self):
        m, d = self.model, self.data
        up, fwd = self._chest_axes()
        bs = sample_balance(m, d)
        lf, rf = _foot_xy_z(m, d, "L"), _foot_xy_z(m, d, "R")
        lnf, rnf = _foot_normal_force(m, d, "L"), _foot_normal_force(m, d, "R")
        com_v = bs.com_vel
        legpos = d.qpos[self._leg_qadr]
        legvel = d.qvel[self._leg_vadr]
        clk = np.array([np.sin(self._phase), np.cos(self._phase)], np.float32)
        parts = [
            up,                                             # 3
            fwd[:2],                                         # 2
            d.qvel[3:6],                                     # 3 ang vel
            [com_v[0], -com_v[1], com_v[2]],                 # 3 CoM vel (fwd = -Y -> +)
            [float(d.qpos[2]) - NOMINAL_CHEST_Z],            # 1
            [(-lf[1]) - self._x0 * 0.0, lf[2], (-rf[1]) - self._x0 * 0.0, rf[2]],  # foot fwd/z (rel start not needed)
            [lf[0] - rf[0], -(lf[1] - rf[1])],               # inter-foot
            [float(bs.l_contact), float(bs.r_contact),
             lnf / BODY_WEIGHT_N, rnf / BODY_WEIGHT_N],      # 4
            legpos, legvel,                                  # 14 + 14
            clk,                                             # 2
            [self.speed_tgt, -com_v[1] - self.speed_tgt],    # 2 target + error
            self._prev_action,                              # 14
        ]
        if self.stride_mode:
            parts.append([self.stride_tgt, self._stride_ema])   # 2  (bigger-steps mode only)
        return np.concatenate([np.atleast_1d(np.asarray(p, np.float32)) for p in parts])

    def _obs(self):
        o = self._build_obs().astype(np.float32)
        if self._obs_noise > 0.0:
            o = o + self.np_random.normal(0.0, self._obs_noise, size=o.shape).astype(np.float32)
        return o

    # ------------------------------------------------ step
    def step(self, action):
        m, d = self.model, self.data
        a = np.asarray(action, np.float32).clip(-1.0, 1.0)
        a_cmd = a
        if self._act_delay > 0:                              # sim-to-real actuation latency
            self._act_buf.append(a.copy())
            a_cmd = self._act_buf.pop(0)
        residual = a_cmd * ACT_SCALE
        dt = m.opt.timestep

        # scheduled random torso disturbance (robust mode)
        if self.robust > 0.0 and self._t >= self._next_push:
            ang = float(self.np_random.uniform(0, 2 * np.pi))
            mag = self.robust * float(self.np_random.uniform(8.0, 22.0))
            self._push = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
            self._push_until = self._t + 1                   # ~1 control step (~25 ms)
            self._next_push = self._t + int(self.np_random.integers(70, 200))

        r = 0.0
        fell = False
        step_bonus = 0.0
        assist_cost = 0.0
        chat0 = self._chatter
        for _ in range(FRAME_SKIP):
            if self._t <= self._push_until:
                d.xfrc_applied[CHEST_BODY, :3] = self._push
            elif d.xfrc_applied[CHEST_BODY, 0] or d.xfrc_applied[CHEST_BODY, 1]:
                d.xfrc_applied[CHEST_BODY, :3] = 0.0
            d.ctrl[:15] = self._compose_ctrl(residual)
            mujoco.mj_step(m, d)
            self._vsync()
            self._phase = (self._phase + 2.0 * np.pi * self.gait_hz * dt) % (2.0 * np.pi)
            both_air = True
            for side, bid in (("L", self._b_lf), ("R", self._b_rf)):
                nf = _foot_normal_force(m, d, side)
                clr = self._sole_clear(side)
                both_air = both_air and (nf < 4.0)
                # contact-chatter counter (HYSTERESIS so it counts real make/break
                # transitions, not force ringing at the boundary): a clean gait
                # does ~2 transitions per step; more = the foot buzzing.
                pv = self._lc_prev if side == "L" else self._rc_prev
                cs = pv
                if not pv and nf > 10.0:
                    cs = True
                elif pv and nf < 2.0:
                    cs = False
                if cs != pv:
                    self._chatter += 1
                if side == "L":
                    self._lc_prev = cs
                else:
                    self._rc_prev = cs
                # debounced per-foot swing state machine -- immune to the rapid
                # make/break contact chatter that corrupts a naive lift/land test.
                self._contact_run[side] = self._contact_run[side] + 1 if nf > 12.0 else 0
                gen_air = nf < 3.0 and clr > 0.014
                self._swing_run[side] = self._swing_run[side] + 1 if gen_air else 0
                if self._foot_state[side] == "stance":
                    if self._swing_run[side] >= 6:               # 6 ms genuinely airborne -> lifted
                        self._foot_state[side] = "swing"
                        self._peak_clr[side] = clr
                        self._swing_air_ms[side] = 0.0
                else:                                            # swing
                    self._peak_clr[side] = max(self._peak_clr[side], clr)
                    if gen_air:
                        self._swing_air_ms[side] += dt * 1000.0
                    if self._contact_run[side] >= 4:             # 4 ms solid contact -> planted
                        self._foot_state[side] = "stance"
                        fwd = -_foot_xy_z(m, d, side)[1]
                        if self._last_land_fwd[side] is not None:
                            step_len = fwd - self._last_land_fwd[side]
                            genuine = (self._peak_clr[side] > 0.016
                                       and self._swing_air_ms[side] > 18.0
                                       and step_len > 0.03)
                            if genuine:
                                tnow = self._t * FRAME_SKIP * dt
                                iv = tnow - self._last_step_t[side]
                                self._last_step_t[side] = tnow
                                if self.stride_mode:
                                    # TWO gaussians that must BOTH be satisfied ->
                                    # forces big AND slow steps (can't farm the
                                    # stride bonus by taking many quick short ones,
                                    # because the interval gaussian then misses).
                                    el = (step_len - self.stride_tgt) / 0.06
                                    iv_tgt = self.stride_tgt / max(self.speed_tgt, 0.06)
                                    ei = (iv - iv_tgt) / (0.45 * iv_tgt)
                                    g_len = float(np.exp(-el * el))
                                    g_iv = float(np.exp(-ei * ei))
                                    step_bonus += 7.0 * g_len * g_iv          # joint peak
                                    step_bonus += 2.0 * g_len                 # some credit for length alone
                                    if iv < 0.7 * iv_tgt:                     # stepping too fast
                                        step_bonus += -6.0 * (0.7 * iv_tgt - iv) / (0.7 * iv_tgt)
                                    fs = float(np.hypot(d.cvel[bid][3], d.cvel[bid][4]))
                                    step_bonus += 1.0 * float(np.clip(1.0 - fs / 0.6, -0.5, 1.0))
                                else:
                                    step_bonus += 3.2 * float(np.clip((step_len - 0.02) / 0.09, -0.5, 1.0))
                                step_bonus += 1.0 * (1.0 - abs(self._swing_air_ms[side] / 1000.0
                                                               - AIR_TIME_TGT) / AIR_TIME_TGT)
                                step_bonus += 1.0 * float(np.clip(
                                    (self._peak_clr[side] - 0.014) / 0.030, -0.5, 1.0))
                                self._step_ct += 1
                                self._strides.append(float(step_len))
                                self._stride_ema = 0.72 * self._stride_ema + 0.28 * float(step_len)
                            else:
                                step_bonus += -1.5           # scrape / pivot / micro-step
                        self._last_land_fwd[side] = fwd
            if both_air:
                self._flight_run += FRAME_SKIP
                self._flight_ms += FRAME_SKIP
            else:
                self._flight_run = 0
            if self._fallen() or self._flight_run > 80:           # 80 ms of continuous flight == a hop
                fell = True
                break

        bs = sample_balance(m, d)
        vfwd = -float(bs.com_vel[1])
        vlat = float(bs.com_vel[0])
        tilt = self._tilt()
        w = self.data.xmat[CHEST_BODY].reshape(3, 3) @ self.data.qvel[3:6]
        ang_speed = float(np.hypot(w[0], w[1]))
        vz = float(bs.com_vel[2])
        tgt = max(self.speed_tgt, 0.1)

        # ---------------- reward ----------------
        # Design: FORWARD PROGRESS is the largest term, so a *surviving walk*
        # scores far above a *surviving stand*.  Alive is still clearly positive
        # (surviving matters, and a fall forfeits all remaining reward + a big
        # penalty), but not so large that standing still beats slow walking.
        r += 2.0            # ALIVE (small -- forward progress must dominate)
        # FORWARD PROGRESS -- the dominant term, but heavily fenced against the
        # "barrel forward, fall every time" farm:
        #   * GATED ON BEING UPRIGHT (gate -> 0 by ~23 deg tilt),
        #   * NO reward at all for exceeding the target speed (hard clip at 1.0x),
        #   * a STEEP overspeed penalty from just above target,
        #   * the fall penalty scales up the earlier the fall (see below).
        upright_gate = float(np.clip(1.15 - tilt / 20.0, 0.0, 1.0))
        prog_gate = 1.0
        if self.stride_mode:
            # forward-progress credit RAMPS with recent stride: ~0.15x at 0.65x of
            # target, full at target, flat above (no penalty for a longer step).
            # Makes stride a first-class objective -- a fast tiny-step shuffle at
            # target speed is heavily throttled, so the policy must commit to the
            # long step it can already do in exploration.
            prog_gate = float(np.clip(
                (self._stride_ema - 0.65 * self.stride_tgt) / (0.35 * self.stride_tgt),
                0.15, 1.0))
        r += 6.5 * upright_gate * prog_gate * float(np.clip(vfwd / tgt, 0.0, 1.0))
        r += -7.0 * max(0.0, vfwd - 1.06 * tgt)               # steep: no barrelling
        r += -1.0 * max(0.0, -vfwd)                           # never reward going backward
        # DIVE guard: fast forward CoM + forward pitch == a lunge, not a step.
        if vfwd > 1.10 * tgt and float(bs.fwd_lean_deg) > 20.0:
            fell = True
            self._dived = True
        # stability -- keep modest so it can't out-vote the progress incentive
        r += -0.9 * float(np.clip(tilt / 18.0, 0.0, 1.8))
        r += -0.35 * ang_speed
        r += -0.5 * abs(vlat)
        r += -0.4 * abs(vz)                                   # no CoM bounce (anti-hop signature)
        r += -0.3 * abs(float(bs.yaw_deg)) / 20.0
        r += -0.05 * float(np.mean((a - self._prev_action) ** 2))
        r += -0.015 * float(np.mean(a ** 2))
        r += -0.0010 * float(np.mean(d.actuator_force[:15] ** 2))
        r += -0.10 * (assist_cost / FRAME_SKIP)               # wean off the training wheels
        # contact chatter -- penalise the running rate above a clean-gait baseline
        # (~0.06 transitions / control-step).  Small always; ramps with robustness.
        chat_ps = self._chatter / max(self._t, 1)
        r += -(2.5 + 4.0 * self.robust) * max(0.0, chat_ps - 0.06)
        r += -(0.4 + 1.2 * self.robust) * max(0, (self._chatter - chat0) - 3)   # burst spike
        if self.robust > 0.0:
            r += -0.05 * self.robust * float(np.mean(np.abs(d.qacc[6:20])))   # jerk-ish, smooth
            r += -0.020 * self.robust * float(np.mean(a ** 2))               # extra effort penalty

        if self.stride_mode:
            # running cadence pressure: at a fixed speed, holding cadence at/below
            # (2*speed/stride_tgt) forces the stride to be the thing that grows.
            elapsed = max(self._t * FRAME_SKIP * m.opt.timestep, 0.6)
            cad_now = self._step_ct / elapsed
            cad_tgt = 2.0 * self.speed_tgt / max(self.stride_tgt, 0.06)
            r += -2.2 * max(0.0, cad_now - 1.12 * cad_tgt)
            r += 0.4 * float(np.clip(1.0 - abs(cad_now - cad_tgt) / cad_tgt, -1.0, 1.0))

        # ---- STRICT alternating gait ----
        lc = _foot_normal_force(m, d, "L") > 6.0
        rc = _foot_normal_force(m, d, "R") > 6.0
        ph_l = (self._phase / (2.0 * np.pi)) % 1.0
        ph_r = (self._phase / (2.0 * np.pi) + 0.5) % 1.0
        want_l, want_r = ph_l < DUTY, ph_r < DUTY
        r += 0.30 * (1.0 if want_l == lc else -1.0)           # periodic contact (sets cadence)
        r += 0.30 * (1.0 if want_r == rc else -1.0)
        if not lc and not rc:
            r += -3.0                                          # both airborne == hop
        single_now = lc != rc
        if want_l != want_r:                                   # clock -> single support
            r += 0.6 if single_now else -0.6
        elif lc and rc:                                        # brief hand-off
            r += 0.15
        # planted-foot slip
        for side, bid in (("L", self._b_lf), ("R", self._b_rf)):
            if _foot_normal_force(m, d, side) > 20.0:
                sv = float(np.hypot(d.cvel[bid][3], d.cvel[bid][4]))
                r += -0.4 * min(1.0, sv / 0.2)
        # swing-foot clearance (a step, not a scuff) -- TRUE sole clearance
        for side in ("L", "R"):
            if _foot_normal_force(m, d, side) < 4.0:
                fz = self._sole_clear(side)
                r += 0.25 * float(np.clip(fz / 0.030, -0.6, 1.0))

        r += step_bonus                                        # completed-step quality
        self._prev_action = a.copy()
        self._t += 1

        if fell:
            # fall penalty scales UP the earlier it happens: -50 at timeout,
            # -95 at t=0.  Makes "sprint 1 s then wipe out" strictly worse than
            # any slower gait that stays up longer.
            frac_left = 1.0 - self._t / EP_STEPS
            return (self._obs(), r - 50.0 - 45.0 * frac_left, True, False,
                    self._info(fell=True, vfwd=vfwd))
        if self._t >= EP_STEPS:
            return self._obs(), r, False, True, self._info(fell=False, vfwd=vfwd)
        return self._obs(), r, False, False, self._info(fell=False, vfwd=vfwd)

    def _sole_clear(self, side):
        gid, loc = self._sole_loc[side]
        d = self.data
        rot = d.geom_xmat[gid].reshape(3, 3)
        wz = loc @ rot.T[:, 2] + d.geom_xpos[gid][2]
        return float(wz.min() - FLOOR_Z)

    def _tilt(self):
        R = self.data.xmat[CHEST_BODY].reshape(3, 3)
        return float(np.degrees(np.arccos(min(1.0, max(-1.0, (R @ self._up_local)[2])))))

    def _fallen(self):
        return (self._tilt() > FALL_TILT_DEG
                or self.data.qpos[2] < NOMINAL_CHEST_Z - FALL_CHEST_DROP)

    def _info(self, fell, vfwd):
        d = self.data
        dist = (self._x0 - float(d.subtree_com[CHEST_BODY][1]))
        return dict(fell=bool(fell), vfwd=float(vfwd), dist=float(dist),
                    t=int(self._t), speed_tgt=float(self.speed_tgt),
                    tilt=float(self._tilt()), genuine_steps=int(self._step_ct),
                    assist=float(self.assist), dived=bool(self._dived),
                    stride_tgt=float(self.stride_tgt), stride_ema=float(self._stride_ema),
                    mean_stride=float(np.mean(self._strides)) if self._strides else 0.0,
                    robust=float(self.robust),
                    chatter_ps=float(self._chatter) / max(self._t, 1),
                    flight_frac=float(self._flight_ms) / max(self._t * FRAME_SKIP, 1))

    # ------------------------------------------------ render
    def _vsync(self):
        if self.render_mode != "human":
            return
        import time
        if getattr(self, "_viewer", None) is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.azimuth = 90
            self._viewer.cam.elevation = -10
            self._viewer.cam.distance = 2.4
            self._t_last = time.perf_counter()
        if not self._viewer.is_running():
            raise KeyboardInterrupt
        self._viewer.cam.lookat[:] = [0.0, float(self.data.xpos[CHEST_BODY][1]), 0.9]
        self._viewer.sync()
        slp = self.model.opt.timestep * self._view_slow - (time.perf_counter() - self._t_last)
        if slp > 0:
            time.sleep(slp)
        self._t_last = time.perf_counter()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 320, 260)
            self._cam = mujoco.MjvCamera()
            self._cam.azimuth = 90
            self._cam.elevation = -10
            self._cam.distance = 2.6
        p = self.data.xpos[CHEST_BODY]
        self._cam.lookat[:] = [0.0, float(p[1]), 0.9]
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None


# ============================================================ CLI
def _load_policy(run_dir, model=DEFAULT_MODEL):
    import json
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    hp = json.load(open(f"{run_dir}/hparams.json"))
    tmp = DummyVecEnv([lambda: BipedLocomotionEnv(model_path=model)])
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


def _smoke(n=10, render=None, policy=None, model=DEFAULT_MODEL, speed=SPEED_TGT_DEFAULT):
    pol = _load_policy(policy, model) if policy else None
    env = BipedLocomotionEnv(model_path=model, speed_tgt=speed,
                             render_mode=("rgb_array" if render else None))
    frames = []
    dists, vs, ts = [], [], []
    for i in range(n):
        o, _ = env.reset(seed=100 + i)
        done = False
        info = {}
        ep = []
        vv = []
        while not done:
            if pol:
                import torch
                with torch.no_grad():
                    a, _ = pol[0].predict(pol[1](o), deterministic=True)
            else:
                a = np.zeros(14, np.float32)
            o, r, term, trunc, info = env.step(a)
            done = term or trunc
            vv.append(info["vfwd"])
            if render and i < 4:
                ep.append(env.render())
        dists.append(info["dist"]); vs.append(float(np.mean(vv[10:]))); ts.append(info["t"])
        if render and i < 4:
            frames.append(ep)
        print(f"  ep {i:2d}  dist {info['dist']:+.2f} m  mean vfwd {np.mean(vv[10:]):+.2f} m/s  "
              f"{'FELL@' + str(info['t']) if info['fell'] else 'survived ' + str(info['t'])}")
    print(f"\n  {'POLICY ' + policy if policy else 'ZERO-ACTION reference'}  ({n} eps, speed_tgt {speed}):")
    print(f"    median dist {np.median(dists):+.2f} m   median vfwd {np.median(vs):+.2f} m/s   "
          f"median steps survived {int(np.median(ts))}/{EP_STEPS}")
    if render and frames:
        import imageio.v2 as imageio
        H = max(len(f) for f in frames)
        tiles = []
        for t in range(H):
            row = [f[min(t, len(f) - 1)] for f in frames]
            tiles.append(np.hstack(row[:2]) if len(row) >= 2 else row[0]
                         if len(row) == 1 else np.hstack(row))
        imageio.mimsave(render, tiles, fps=40, macro_block_size=1)
        print(f"    wrote {render}")
    env.close()


def _watch(policy, model, speed, n=6, pol_file=None, vn_file=None):
    """Live MuJoCo viewer -- one episode at a time, close the window to advance."""
    import json
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import pickle
    rd = policy
    hp = json.load(open(f"{rd}/hparams.json"))
    pf = pol_file or (f"{rd}/policy_best.pth" if os.path.exists(f"{rd}/policy_best.pth") else f"{rd}/policy.pth")
    vf = vn_file or (f"{rd}/vecnormalize_best.pkl" if os.path.exists(f"{rd}/vecnormalize_best.pkl")
                     else f"{rd}/vecnormalize.pkl")
    tmp = DummyVecEnv([lambda: BipedLocomotionEnv(model_path=model, speed_tgt=speed)])
    mm = PPO("MlpPolicy", tmp, device="cpu",
             policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(pf, map_location="cpu", weights_only=True))
    mm.policy.eval()
    vn = pickle.load(open(vf, "rb"))
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    print(f"  {pf}  |  close the viewer window to advance to the next run")
    env = BipedLocomotionEnv(model_path=model, speed_tgt=speed, render_mode="human")
    for i in range(n):
        o, _ = env.reset(seed=200 + i)
        done = False
        info = {}
        try:
            while not done:
                obs = np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
                with torch.no_grad():
                    a, _ = mm.predict(obs, deterministic=True)
                o, r, term, trunc, info = env.step(a)
                done = term or trunc
        except KeyboardInterrupt:
            break
        print(f"  run {i}: dist {info.get('dist', 0):+.2f} m  "
              f"{'FELL' if info.get('fell') else 'survived'} @ {info.get('t', 0)}")
    env.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--watch", action="store_true", help="live MuJoCo viewer")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--render", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--pol-file", default=None)
    ap.add_argument("--vn-file", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--speed", type=float, default=SPEED_TGT_DEFAULT)
    a = ap.parse_args(argv)
    if a.watch:
        _watch(a.policy, a.model, a.speed, a.n, a.pol_file, a.vn_file)
    elif a.smoke:
        _smoke(a.n, a.render, a.policy, a.model, a.speed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
