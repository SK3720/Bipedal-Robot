"""Build the 217-D sim2real_v1 observation from real sensor readings.

Mirrors biped_sim2real_env._sensor_frame / _realistic_obs EXACTLY (validated to
< 1e-4 by hil/validate_stack.py with the sim as the sensor source).

Inputs per control step (all already in the CHEST BODY frame -- see README for the
IMU mounting requirement):
  imu_quat_wxyz : (4,) chest orientation, body->world, unit, [w,x,y,z]
  imu_gyro      : (3,) body-frame angular velocity, rad/s   (raw gyro, bias-removed)
  imu_accel     : (3,) body-frame specific force, m/s^2      (raw accel; ~[0,9.81,0] at rest)
  joint_pos     : (14,) POLICY_JOINT_ORDER, rad              (encoders)
  joint_vel     : (14,) POLICY_JOINT_ORDER, rad/s            (encoders)
  contact_L/R   : bool                                       (foot contact switches)

Call `reset(...)` once (fills the 4-frame history with the first frame), then
`step(...)` every control step.  `prev_action` is the LAST policy output (clipped
to [-1,1], POLICY_JOINT_ORDER); pass it in each step.
"""
from __future__ import annotations

import numpy as np

from hil import spec
from hil.kinematics import Kinematics


def quat_to_mat(q_wxyz):
    w, x, y, z = q_wxyz
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])


UP_LOCAL = np.array([0.0, 1.0, 0.0])     # chest local +Y  (== biped_locomotion_env._up_local)
FWD_LOCAL = np.array([0.0, 0.0, -1.0])   # chest local -Z  (== _fwd_local)


class ObservationBuilder:
    def __init__(self, control_dt=spec.CONTROL_DT):
        self.dt = float(control_dt)
        self.kin = Kinematics()
        self._hist = None
        self._phase = 0.0
        self._last_frame = None

    # -- lifecycle -----------------------------------------------------------
    def reset(self, imu_quat_wxyz, imu_gyro, imu_accel, joint_pos, joint_vel,
              contact_L, contact_R, phase0=0.0):
        self._phase = float(phase0) % (2.0 * np.pi)
        self.kin.prime(joint_pos)
        f = self._frame(imu_quat_wxyz, imu_gyro, imu_accel, joint_pos, joint_vel,
                        contact_L, contact_R, advance_kin=False)
        self._hist = [f.copy() for _ in range(spec.HIST)]
        self._last_frame = f
        return self._assemble(np.zeros(14))

    # -- per-step ----------------------------------------------------------
    def step(self, imu_quat_wxyz, imu_gyro, imu_accel, joint_pos, joint_vel,
             contact_L, contact_R, prev_action):
        self._phase = (self._phase + spec.PHASE_RATE * self.dt) % (2.0 * np.pi)
        f = self._frame(imu_quat_wxyz, imu_gyro, imu_accel, joint_pos, joint_vel,
                        contact_L, contact_R, advance_kin=True)
        self._hist.append(f)
        self._hist.pop(0)
        self._last_frame = f
        return self._assemble(np.asarray(prev_action, float).clip(-1, 1))

    # -- internals -------------------------------------------------------
    def _frame(self, q, gyro, accel, jp, jv, cL, cR, advance_kin):
        q = np.asarray(q, float)
        R = quat_to_mat(q)                       # body -> world
        up = R @ UP_LOCAL
        fwd = R @ FWD_LOCAL
        gyro_obs = R.T @ np.asarray(gyro, float)          # == biped_sim2real_env: R.T @ omega_body
        accel_obs = np.asarray(accel, float)             # == R.T @ (a_world + g) already
        jp = np.asarray(jp, float)
        jv = np.asarray(jv, float)
        if advance_kin:
            self.kin.update(jp, bool(cL), bool(cR), self.dt)
        return np.concatenate([
            up,                    # 3
            fwd[:2],               # 2
            gyro_obs,              # 3
            accel_obs,             # 3
            jp, jv,                # 28
            [float(bool(cL)), float(bool(cR))],   # 2
            self.kin.v_est,        # 3
            self.kin.foot_L_rel,   # 3
            self.kin.foot_R_rel,   # 3
        ]).astype(np.float32)

    def _assemble(self, prev_action):
        extra = np.concatenate([
            [np.sin(self._phase), np.cos(self._phase)],
            [spec.SPEED_TGT],
            prev_action,
        ]).astype(np.float32)
        return np.concatenate(self._hist + [extra]).astype(np.float32)

    @property
    def phase(self):
        return self._phase
