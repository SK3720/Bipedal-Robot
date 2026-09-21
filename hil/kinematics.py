"""Forward kinematics + leg odometry -- from joint encoders only.

Both quantities the observation needs but a real robot cannot measure directly:

  * FOOT-IN-BASE-FRAME  (`foot_L_rel`, `foot_R_rel`) -- pure FK: joint angles ->
    foot pose relative to the chest.  Reuses the exact MuJoCo kinematic model so
    the link geometry matches the sim bit-for-bit.

  * LEG-ODOMETRY BASE VELOCITY (`v_est`) -- a stance foot is ~stationary on the
    ground, so   v_base_body ~ -d/dt( foot_in_base_frame )   averaged over the
    feet reporting contact; decays toward zero when both feet are airborne.
    This is exactly what biped_sim2real_env._sensor_frame computes.
"""
from __future__ import annotations

import numpy as np
import mujoco

from hil import spec


class Kinematics:
    """Stateful: call `update(joint_pos, joint_vel, contact, dt)` once per control
    step, then read `.foot_L_rel`, `.foot_R_rel`, `.v_est`."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(spec.MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.b_chest = self.model.body("Chest").id
        self.b_lf = self.model.body("L_foot").id
        self.b_rf = self.model.body("R_foot").id
        # map POLICY_JOINT_ORDER -> qpos address (7 + mujoco_ctrl_index)
        self.qadr = np.array([7 + i for i in spec.ACT_CTRL])
        self._prev_foot_rel = {"L": np.zeros(3), "R": np.zeros(3)}
        self._v_est = np.zeros(3)
        self.foot_L_rel = np.zeros(3)
        self.foot_R_rel = np.zeros(3)
        self.v_est = np.zeros(3)
        self._primed = False

    def _fk(self, joint_pos):
        d = self.data
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0                       # identity base quaternion
        d.qpos[self.qadr] = joint_pos         # 14 policy-ordered joint angles
        mujoco.mj_kinematics(self.model, d)   # cheap: no dynamics
        # chest frame == base body frame; xmat[b_chest] is identity here
        base = d.xpos[self.b_chest]
        R = d.xmat[self.b_chest].reshape(3, 3)
        fl = R.T @ (d.xpos[self.b_lf] - base)
        fr = R.T @ (d.xpos[self.b_rf] - base)
        return fl, fr

    def prime(self, joint_pos):
        """Call once at startup with the initial encoder reading so the first
        odometry finite-difference is zero, not a spike."""
        fl, fr = self._fk(np.asarray(joint_pos, float))
        self._prev_foot_rel["L"] = fl
        self._prev_foot_rel["R"] = fr
        self.foot_L_rel, self.foot_R_rel = fl, fr
        self._v_est[:] = 0.0
        self.v_est[:] = 0.0
        self._primed = True

    def update(self, joint_pos, contact_L, contact_R, dt):
        jp = np.asarray(joint_pos, float)
        if not self._primed:
            self.prime(jp)
            return
        fl, fr = self._fk(jp)
        ests = []
        if contact_L:
            ests.append(-(fl - self._prev_foot_rel["L"]) / dt)
        if contact_R:
            ests.append(-(fr - self._prev_foot_rel["R"]) / dt)
        if ests:
            self._v_est = np.mean(ests, axis=0)
        else:
            self._v_est = self._v_est * 0.6          # decay when airborne
        self._prev_foot_rel["L"] = fl
        self._prev_foot_rel["R"] = fr
        self.foot_L_rel, self.foot_R_rel = fl, fr
        self.v_est = self._v_est.copy()
