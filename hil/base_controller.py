"""The sim2real_v1 IMU-only base controller -- reproduces
biped_sim2real_env.BipedSim2RealEnv._compose_ctrl (imu_ctrl=True branch).

Given the policy residual + IMU + FK, returns the 15-D joint POSITION command
that the sim would send to `d.ctrl`.  In LOG-ONLY HIL this is recorded (it is
"what the motors would be told"); it is NOT sent while MOTORS_ENABLED is False.

Validated against the live env by hil/validate_stack.py (< 1e-4).

  u = clip( ctrl0
            + CPG_reference(phase)                 [gain 0.75]
            - K_att @ [ orient_err - WALK_LEAN*e_x ,  gyro - qvel0 ]
            + frontal_CoP(foot_FK, v_est, contact)
            + scatter( policy_action * ACT_SCALE ) ,
            ctrlrange )
  u[neck] = 0
"""
from __future__ import annotations

import numpy as np

from hil import spec

# ---- constants copied verbatim from biped_locomotion_env.py -------------------
NECK = 0
ARM = dict(L=dict(sh=1, el=2), R=dict(sh=3, el=4))
LEG = dict(L=dict(hr=5, hp=6, kn=7, ap=8, ar=9),
           R=dict(hr=10, hp=11, kn=12, ap=13, ar=14))
CPG_GAIN = 0.75
GAIT_HZ = 0.80
DUTY = 0.64
HIP_AMP = 0.24
KNEE_SWING = 0.66
KNEE_STANCE = 0.12
ANK_CLEAR = 0.18
ANK_PUSH = 0.22
ARM_AMP = 0.5
ELB_FLEX = 0.30
LAT_ROLL = 0.20
LAT_HR = 0.12
WALK_LEAN = 0.04
STEP_FWD = 0.12
HP_FWD = dict(L=-1.0, R=1.0)
KN_FLEX = dict(L=1.0, R=-1.0)
AP_DORSI = dict(L=-1.0, R=1.0)
SH_FWD = dict(L=-1.0, R=1.0)
# frontal stabiliser gains (biped_sim2real_env)
FR_KX, FR_KV = 3.2, 0.9
FR_AR_MAX = 0.28
FR_HR_KX, FR_HR_KV, FR_HR_MAX = 2.6, 0.5, 0.34


def _smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * t))


class BaseController:
    def __init__(self):
        c = spec.load_consts()
        self.K_att = c["K_att"]            # 15 x 6
        self.qpos0 = c["qpos0"]            # 22
        self.qvel0 = c["qvel0"]            # 21
        self.ctrl0 = c["ctrl0"]           # 15  (all zero for this model)
        self.ctrlrange = c["ctrlrange"]   # 15 x 2
        self.act_scale = spec.ACT_SCALE   # 14  (POLICY_JOINT_ORDER)
        self._q0 = self.qpos0
        try:
            import mujoco
            self._subquat = mujoco.mju_subQuat
        except Exception:                 # tiny fallback if mujoco unavailable
            self._subquat = _subquat_np

    # -- CPG leg reference (== biped_locomotion_env._leg_ref, stride_mode off) --
    def _leg_ref(self, side, ph):
        hp0 = self._q0[7 + LEG[side]["hp"]]
        kn0 = self._q0[7 + LEG[side]["kn"]]
        ap0 = self._q0[7 + LEG[side]["ap"]]
        if ph < DUTY:
            s = ph / DUTY
            hip = HP_FWD[side] * HIP_AMP * (0.5 * np.cos(np.pi * s))
            knee = KN_FLEX[side] * (KNEE_STANCE + 0.08 * np.sin(np.pi * s))
            push = _smooth((s - 0.70) / 0.30) if s > 0.70 else 0.0
            ank = AP_DORSI[side] * (0.03 - (ANK_PUSH + 0.03) * push)
        else:
            s = (ph - DUTY) / (1.0 - DUTY)
            hip = HP_FWD[side] * (HIP_AMP * (-0.5 + _smooth(s)) + STEP_FWD * _smooth(s))
            knee = KN_FLEX[side] * (KNEE_STANCE + KNEE_SWING * np.sin(np.pi * s))
            ank = AP_DORSI[side] * (ANK_CLEAR * np.sin(np.pi * s))
        return hp0 + hip, kn0 + knee, ap0 + ank

    # -- full compose ---------------------------------------------------------
    def compose(self, action14, phase, imu_quat_wxyz, gyro_body,
                foot_L_rel, foot_R_rel, v_est, contact_L, contact_R,
                foot_load_L=None, foot_load_R=None):
        """
        action14   : (14,) policy output in [-1,1], POLICY_JOINT_ORDER
        phase      : gait clock, rad
        imu_quat   : (4,) chest orientation w,x,y,z  (body->world)
        gyro_body  : (3,) body-frame angular velocity, rad/s
        foot_*_rel : (3,) foot position in chest BODY frame (FK)  [hil.kinematics]
        v_est      : (3,) leg-odometry base velocity, chest BODY frame
        contact_*  : bool   -- contact switch (~>6 N)
        foot_load_*: float N  -- foot normal force IF the robot has force/pressure
                     sensing.  The sim's frontal stabiliser gates on load > 12 N;
                     with only a contact switch we fall back to `contact_*`, which
                     makes the single-support transitions slightly different from
                     sim (fine for LOG-ONLY; matters for the closed-loop phase).
        returns    : (15,) joint position command  (MuJoCo ctrl order); [neck]=0
        """
        u = np.array(self.ctrl0, float)

        # ---- CPG reference ----
        g = CPG_GAIN
        lat = np.sin(phase)
        for side in ("L", "R"):
            ph = (phase / (2.0 * np.pi) + (0.0 if side == "L" else 0.5)) % 1.0
            hp, kn, ap = self._leg_ref(side, ph)
            sgn = 1.0 if side == "L" else -1.0
            u[LEG[side]["hp"]] += g * (hp - u[LEG[side]["hp"]])
            u[LEG[side]["kn"]] += g * (kn - u[LEG[side]["kn"]])
            u[LEG[side]["ap"]] += g * (ap - u[LEG[side]["ap"]])
            u[LEG[side]["ar"]] += g * sgn * LAT_ROLL * lat
            u[LEG[side]["hr"]] += g * sgn * LAT_HR * lat
            oph = (phase / (2.0 * np.pi) + (0.5 if side == "L" else 0.0)) % 1.0
            u[ARM[side]["sh"]] += g * SH_FWD[side] * ARM_AMP * np.sin(2.0 * np.pi * oph)
            u[ARM[side]["el"]] += g * (1.0 if side == "L" else -1.0) * ELB_FLEX

        # ---- attitude LQR (orientation + angular-rate error only) ----
        dq_rot = np.zeros(3)
        self._subquat(dq_rot, np.asarray(imu_quat_wxyz, float), self.qpos0[3:7])
        dq_rot[0] -= WALK_LEAN                      # forward-pitch bias
        dw = np.asarray(gyro_body, float) - self.qvel0[3:6]
        u = u - (self.K_att @ np.concatenate([dq_rot, dw]))

        # ---- frontal CoP stabiliser (FK + leg odometry) ----
        # sim gates on foot normal force > 12 N; use that if available, else the
        # contact switch.
        if foot_load_L is not None and foot_load_R is not None:
            loaded = {"L": float(foot_load_L) > 12.0, "R": float(foot_load_R) > 12.0}
        else:
            loaded = {"L": bool(contact_L), "R": bool(contact_R)}
        fx, n = 0.0, 0
        for side, fr in (("L", foot_L_rel), ("R", foot_R_rel)):
            if loaded[side]:
                fx += float(fr[0])
                n += 1
        err = -(fx / n) if n else 0.0
        vx = float(v_est[0])
        ar = float(np.clip(-(FR_KX * err + FR_KV * vx) / 6.0, -FR_AR_MAX, FR_AR_MAX))
        u[LEG["L"]["ar"]] += ar
        u[LEG["R"]["ar"]] += ar
        hr = float(np.clip(-(FR_HR_KX * err + FR_HR_KV * vx), -FR_HR_MAX, FR_HR_MAX))
        for side in ("L", "R"):
            if loaded[side]:
                u[LEG[side]["hr"]] += hr

        # ---- policy residual (scatter POLICY_JOINT_ORDER -> ctrl order) ----
        residual = np.asarray(action14, float).clip(-1, 1) * self.act_scale
        for k, ci in enumerate(spec.ACT_CTRL):
            u[ci] += residual[k]

        u[NECK] = 0.0
        return np.clip(u, self.ctrlrange[:, 0], self.ctrlrange[:, 1])


def _subquat_np(res, qa, qb):
    """mju_subQuat fallback: res = rotational difference from qb to qa (in qa frame)."""
    w0, x0, y0, z0 = qb[0], -qb[1], -qb[2], -qb[3]          # qb conjugate
    w1, x1, y1, z1 = qa
    dw = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    dx = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    dy = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    dz = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    v = np.array([dx, dy, dz])
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        res[:] = 0.0
        return
    ang = 2.0 * np.arctan2(nv, dw)
    if ang > np.pi:
        ang -= 2.0 * np.pi
    res[:] = v / nv * ang
