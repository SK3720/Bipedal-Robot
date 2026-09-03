"""RL environment for forward-push RECOVERY STEPPING, built on the LQR baseline.

WHAT WAS FIXED vs the base biped_env (the user's question about obs limits):
  * Observation 34 -> 66: added torso LINEAR + ANGULAR velocity (the base obs had
    neither - a balance policy was doing position-only feedback), per-foot contact
    + normalised load, CoM position + velocity relative to the feet, chest-height
    error, previous action, and time-since-push.
  * Control at 200 Hz (frame skip 5); the base env queried the policy every 1 ms.
  * Termination / "fell" uses the REAL chest up-axis tilt, not the yaw-conflated
    _quat_tilt_rad the base env uses.
  * Forward-biased (sagittal) push with a magnitude curriculum.
  * A torso-stabilising LQR base:  ctrl = clip(lqr_torso(state) + action*SCALE),
    so a zero policy is ~the torso-LQR (holds ~145 N in place) and the policy only
    has to learn the stepping residual.

RESULT OF 6 PPO RUNS (v1..v6, up to 3 M steps, warm-start / BC / shaping / curriculum
variants):  the policy reliably learns CLEAN in-place recovery for the pushes the
LQR base already handles, and NEVER learns a recovery step.  Every attempt to
breadcrumb toward a step (reach / anti-slide / swing rewards, behaviour cloning
from a scripted step) instead produced a bad attractor - the policy flings a foot
forward and face-plants even for LQR-recoverable pushes.  This matches the 5
hand-tuning passes: the recovery STEP is not achievable with this model
(+/-2.3 N.m position servos that must both balance and step; 40 mm asymmetric
stance -> ~zero single-support lateral margin; ~0.4 s window).

Does not modify robot.xml / biped_env.py / the golden experiments.
"""

from __future__ import annotations

import numpy as np
import mujoco

from biped_env import (
    BipedalWalkEnv, CHEST_Z_CONTACT, DEFAULT_POSE, PUSH_DURATION_STEPS,
    STANDING_QUAT,
)
from gymnasium import spaces
from standing_balance_lqr import StandingLQR

FLOOR_Z = 1.0
G = 9.81
FORWARD_DIR_RAD = -np.pi / 2.0        # world -Y

FRAME_SKIP = 5                        # -> 200 Hz control (more bandwidth for a step)
STAND_PHYS_STEPS = 400                # ~0.40 s standing before the push
STAND_JITTER_PHYS = 120
MAX_PHYS_STEPS = 4000                 # 4.0 s episodes
RESIDUAL_SCALE = 1.00                 # rad; policy residual (abs_action=False)
ABS_ACT_SCALE = 0.90                  # rad; policy full target offset from DEFAULT_POSE (abs_action=True)

# reward weights
W_ALIVE = 1.0
W_UPRIGHT = 2.0
W_HEIGHT = 6.0
W_COM_SPEED = 0.6
W_DOUBLE_SUPPORT = 0.4
W_POSTURE = 0.15
W_ACTION = 0.003
W_ACTION_RATE = 0.015
FALL_PENALTY = 20.0
# --- shaping DISABLED ---
# Every attempt to breadcrumb toward a step (reach/antislide/swing bonuses,
# behaviour cloning) either did nothing or created a bad attractor that broke
# the in-place recovery the LQR base already provides (policy face-planted at
# 125 N where the base LQR holds 145 N).  v6 reward is pure survival + a light
# side-lean penalty; the policy is free to use anything (incl. the arms) to stay
# up, and we simply measure how far it pushes the in-place ceiling.
W_REACH = 0.0
W_ANTISLIDE = 0.0
W_SIDE = 1.5
COMV_FWD_ACTIVE = 0.20

# termination
FALL_UPAXIS_RAD = np.deg2rad(50.0)   # chest up-axis this far from vertical => down
FALL_HEIGHT_DROP = 0.20              # m below nominal chest height

# push curriculum
PUSH_MIN_START = 115.0
PUSH_MAX_START = 145.0
PUSH_MAX_LIMIT = 280.0
PUSH_CURRIC_STEP = 7.0
PUSH_CURRIC_WINDOW = 60
PUSH_CURRIC_SURVIVE = 0.86
PUSH_FWD_SPREAD_RAD = np.deg2rad(15.0)


class RecoveryEnv(BipedalWalkEnv):
    def __init__(self, seed: int | None = None, abs_action: bool = False):
        super().__init__()
        self.abs_action = abs_action
        # LQR base controller (built once; deterministic for this model/pose)
        self._lqr = StandingLQR(self.model, self.data, verbose=False)
        self.disturbance_enabled = True

        # cached ids / constants for the hot path
        self._g_lfoot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "L_foot_collision")
        self._g_rfoot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "R_foot_collision")
        self._b_lfoot = self.model.body("L_foot").id
        self._b_rfoot = self.model.body("R_foot").id
        self._K = np.ascontiguousarray(self._lqr.K)
        self._ctrl0 = self._lqr.ctrl0.copy()
        self._qpos0 = self._lqr.qpos0.copy()
        self._qvel0 = self._lqr.qvel0.copy()
        self._up_local = np.array([0.0, 1.0, 0.0])
        self._fwd_local = np.array([0.0, 0.0, -1.0])
        self._cache_step = -1
        self._c_com = self._c_comv = self._c_feet = None
        self._c_lc = self._c_lnf = self._c_rc = self._c_rnf = None
        self._prev_sw_y = None

        # curriculum state
        self.cur_push_min = PUSH_MIN_START
        self.cur_push_max = PUSH_MAX_START
        self._curric_results: list[bool] = []
        self.curriculum_level = 0

        self._phys_step = 0
        self._push_phys = STAND_PHYS_STEPS
        self._prev_action = np.zeros(15, dtype=np.float32)
        self._t_since_push = 1.0
        self._peak_upaxis = 0.0
        self._prev_sw_y = None

        self.obs_dim = self._obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    # ---------------- helpers ----------------
    def _chest_axes(self):
        R = self.data.xmat[self.chest_body_id].reshape(3, 3)
        return R @ self._up_local, R @ self._fwd_local, R

    def _upaxis_tilt(self):
        up = self.data.xmat[self.chest_body_id].reshape(3, 3) @ self._up_local
        return float(np.arccos(min(1.0, max(-1.0, up[2]))))

    def _refresh(self):
        """Compute the per-step quantities used by both obs and reward, once."""
        if self._cache_step == self._phys_step:
            return
        self._cache_step = self._phys_step
        mujoco.mj_subtreeVel(self.model, self.data)
        self._c_com = self.data.subtree_com[1].copy()
        self._c_comv = self.data.subtree_linvel[1].copy()
        self._c_feet = (self.data.xpos[self._b_lfoot][:2].copy(),
                        self.data.xpos[self._b_rfoot][:2].copy())
        lnf = rnf = 0.0
        lc = rc = False
        w = np.zeros(6)
        con = self.data.contact
        for ci in range(self.data.ncon):
            g1 = con[ci].geom1
            g2 = con[ci].geom2
            is_l = g1 == self._g_lfoot or g2 == self._g_lfoot
            is_r = g1 == self._g_rfoot or g2 == self._g_rfoot
            if not (is_l or is_r):
                continue
            mujoco.mj_contactForce(self.model, self.data, ci, w)
            f = max(0.0, float(w[0]))
            if is_l:
                lc = True
                lnf += f
            else:
                rc = True
                rnf += f
        self._c_lc, self._c_lnf, self._c_rc, self._c_rnf = lc, lnf, rc, rnf

    def _foot_state(self, side):
        self._refresh()
        return ((self._c_lc, self._c_lnf) if side == "L" else (self._c_rc, self._c_rnf))

    def _com(self):
        self._refresh()
        return self._c_com, self._c_comv

    def _feet_xy(self):
        self._refresh()
        return self._c_feet

    def _lqr_ctrl(self, full=False):
        """full=True: the real standing LQR (1000 Hz, recovers <= ~140 N).
        full=False: TORSO-ONLY base for the residual policy - zeros BOTH the
        joint-position and joint-velocity feedback so it neither pulls the legs
        back to the nominal pose nor damps the policy's stepping motion; it only
        stabilises the floating base."""
        nv = self.model.nv
        dq = np.zeros(nv)
        mujoco.mj_differentiatePos(self.model, dq, 1.0, self._qpos0, self.data.qpos)
        dv = self.data.qvel - self._qvel0
        if not full:
            dq[6:21] = 0.0
            dv[6:21] = 0.0
        u = self._ctrl0 - self._K @ np.concatenate([dq, dv])
        return u.clip(self.ctrl_low, self.ctrl_high, out=u)

    def lqr_action(self):
        """The full standing LQR expressed in this env's ABSOLUTE action units."""
        u = self._lqr_ctrl(full=True)
        return np.clip((u - DEFAULT_POSE) / ABS_ACT_SCALE, -1.0, 1.0).astype(np.float32)

    def scripted_step_action(self):
        """A crude scripted recovery-step RESIDUAL for behaviour cloning. Does a
        weight shift + hip swing timed off t_since_push, swing leg = lighter foot
        at push landing.  Zero before the push."""
        from scripted_step_policy import scripted_residual_rad
        t = float(self._t_since_push)
        # only demonstrate a step for pushes the base LQR cannot hold in place
        if t < 0.0 or self.push_magnitude < 143.0:
            return np.zeros(15, dtype=np.float32)
        if getattr(self, "_bc_swing", None) is None:
            _, lnf = self._foot_state("L")
            _, rnf = self._foot_state("R")
            self._bc_swing = "L" if lnf <= rnf else "R"
        r = scripted_residual_rad(t, self._bc_swing) / RESIDUAL_SCALE
        return np.clip(r, -1.0, 1.0).astype(np.float32)

    # ---------------- obs ----------------
    def _obs(self):
        up, fwd, R = self._chest_axes()
        ang = self.data.qvel[3:6].copy()
        lin = self.data.qvel[0:3].copy()
        com, comv = self._com()
        lxy, rxy = self._feet_xy()
        mid = 0.5 * (lxy + rxy)
        com_rel = com[:2] - mid
        lc, lnf = self._foot_state("L")
        rc, rnf = self._foot_state("R")
        bw = 2.86 * G
        h_err = float(self.data.qpos[2] - CHEST_Z_CONTACT)
        jpos = self.data.qpos[7:22].copy()
        jvel = self.data.qvel[6:21].copy()
        obs = np.concatenate([
            up, fwd[:2],                       # 5  torso orientation
            ang, lin,                          # 6  torso ang + lin velocity
            [h_err],                           # 1
            com_rel, comv[:2],                 # 4  CoM pos/vel relative to mid-feet
            [float(lc), float(rc), lnf / bw, rnf / bw],   # 4  contacts + load
            jpos, jvel,                        # 30 joints
            self._prev_action,                 # 15
            [np.clip(self._t_since_push, 0.0, 3.0)],      # 1
        ]).astype(np.float32)
        return obs

    def _get_obs(self):
        return self._obs()

    # ---------------- reward ----------------
    def _reward(self, action, fell):
        up, _, _ = self._chest_axes()
        com, comv = self._com()
        lc, lnf = self._foot_state("L")
        rc, rnf = self._foot_state("R")
        lxy, rxy = self._feet_xy()
        post_push = self._t_since_push > 0.0 and self._phys_step > self._push_phys + PUSH_DURATION_STEPS
        com_vfwd = -float(comv[1])
        com_fwd = -float(com[1])
        front_foot_fwd = max(-lxy[1], -rxy[1])
        pushing_fwd = post_push and com_vfwd > COMV_FWD_ACTIVE

        r_alive = W_ALIVE
        r_upright = W_UPRIGHT * max(0.0, up[2]) ** 2
        h_drop = max(0.0, CHEST_Z_CONTACT - float(self.data.qpos[2]))
        r_height = -W_HEIGHT * h_drop ** 2
        # side lean = chest up-axis tipped along world X (lateral)
        side = float(up[0])
        r_side = -W_SIDE * side ** 2

        com_speed = float(np.hypot(comv[0], comv[1]))
        # ask for the CoM to come to rest only well after the step should be done
        r_comspeed = -W_COM_SPEED * com_speed ** 2 if self._t_since_push > 1.3 else 0.0

        # DS bonus only once the step should have finished
        r_ds = W_DOUBLE_SUPPORT if (self._t_since_push > 1.5 and lc and rc) else 0.0

        # optional step shaping (W_REACH / W_ANTISLIDE default 0 - see module
        # docstring: every non-zero setting created a bad face-planting attractor)
        r_reach = r_antislide = 0.0
        if W_REACH and pushing_fwd:
            r_reach = W_REACH * float(np.clip(front_foot_fwd - com_fwd - 0.03, -0.05, 0.15))
        if W_ANTISLIDE and pushing_fwd and lc and rc:
            r_antislide = -W_ANTISLIDE * (com_vfwd - COMV_FWD_ACTIVE) ** 2

        jdev = self.data.qpos[7:22] - np.concatenate([np.zeros(5), DEFAULT_POSE[5:]])
        r_posture = -W_POSTURE * float(np.mean(jdev[5:] ** 2)) if self._t_since_push > 1.8 else 0.0

        r_act = -W_ACTION * float(np.sum(action ** 2))
        r_actrate = -W_ACTION_RATE * float(np.sum((action - self._prev_action) ** 2))

        r = (r_alive + r_upright + r_height + r_side + r_comspeed + r_ds
             + r_reach + r_antislide + r_posture + r_act + r_actrate)
        if fell:
            r -= FALL_PENALTY
        info = dict(r_alive=r_alive, r_upright=r_upright, r_height=r_height,
                    r_side=r_side, r_comspeed=r_comspeed, r_ds=r_ds, r_reach=r_reach,
                    r_antislide=r_antislide, com_speed=com_speed, com_vfwd=com_vfwd,
                    upaxis_tilt=self._upaxis_tilt())
        return r, info

    # ---------------- gym API ----------------
    def _schedule_forward_push(self):
        self._push_phys = STAND_PHYS_STEPS + int(self.np_random.integers(0, STAND_JITTER_PHYS + 1))
        mag = float(self.np_random.uniform(self.cur_push_min, self.cur_push_max))
        ang = FORWARD_DIR_RAD + float(self.np_random.uniform(-PUSH_FWD_SPREAD_RAD, PUSH_FWD_SPREAD_RAD))
        self.push_magnitude = mag
        self.push_direction_rad = ang
        self.push_force_xy = mag * np.array([np.cos(ang), np.sin(ang)])
        self.push_step_start = self._push_phys

    def reset(self, seed=None, options=None):
        options = options or {}
        super(BipedalWalkEnv, self).reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
        self.data.qpos[3:7] = STANDING_QUAT
        self.data.qpos[7:22] = DEFAULT_POSE
        self.data.qvel[:] = 0.0
        self.data.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_forward(self.model, self.data)
        for _ in range(300):
            self.data.ctrl[:15] = self._lqr_ctrl()
            mujoco.mj_step(self.model, self.data)

        self.initial_xy = self.data.qpos[0:2].copy()
        self.initial_chest_z = self.data.qpos[2]
        self._phys_step = 0
        self._prev_action = np.zeros(15, dtype=np.float32)
        self._t_since_push = -1.0
        self._peak_upaxis = 0.0
        self._prev_sw_y = None
        self._bc_swing = None
        self.data.xfrc_applied[self.chest_body_id, :] = 0.0

        if "push_magnitude" in options:
            self.push_magnitude = float(options["push_magnitude"])
            ang = float(options.get("push_direction_rad", FORWARD_DIR_RAD))
            self.push_direction_rad = ang
            self.push_force_xy = self.push_magnitude * np.array([np.cos(ang), np.sin(ang)])
            self._push_phys = int(options.get("push_step_start", STAND_PHYS_STEPS))
            self.push_step_start = self._push_phys
        elif options.get("enable_push", True):
            self._schedule_forward_push()
        else:
            self._push_phys = MAX_PHYS_STEPS + 1
            self.push_step_start = self._push_phys
            self.push_force_xy = np.zeros(2)

        return self._obs(), {"push_magnitude": self.push_magnitude,
                             "push_direction_rad": self.push_direction_rad}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        if self.abs_action:
            target = DEFAULT_POSE + action * ABS_ACT_SCALE
        else:
            residual = action * RESIDUAL_SCALE
        for _ in range(FRAME_SKIP):
            self.data.xfrc_applied[self.chest_body_id, :] = 0.0
            if self._push_phys <= self._phys_step < self._push_phys + PUSH_DURATION_STEPS:
                self.data.xfrc_applied[self.chest_body_id, 0:2] = self.push_force_xy
            if self.abs_action:
                ctrl = np.clip(target, self.ctrl_low, self.ctrl_high)
            else:
                ctrl = np.clip(self._lqr_ctrl() + residual, self.ctrl_low, self.ctrl_high)
            self.data.ctrl[:15] = ctrl
            mujoco.mj_step(self.model, self.data)
            self._phys_step += 1

        if self._phys_step >= self._push_phys + PUSH_DURATION_STEPS:
            self._t_since_push = (self._phys_step - (self._push_phys + PUSH_DURATION_STEPS)) / 1000.0
        upaxis = self._upaxis_tilt()
        self._peak_upaxis = max(self._peak_upaxis, upaxis)

        fell = (upaxis > FALL_UPAXIS_RAD
                or self.data.qpos[2] < CHEST_Z_CONTACT - FALL_HEIGHT_DROP
                or not np.isfinite(self.data.qpos).all())
        reward, rinfo = self._reward(action, fell)

        terminated = bool(fell)
        truncated = self._phys_step >= MAX_PHYS_STEPS
        obs = self._obs()
        self._prev_action = action.copy()

        info = {
            "push_magnitude": self.push_magnitude,
            "t_since_push": self._t_since_push,
            "upaxis_tilt": upaxis,
            "chest_z": float(self.data.qpos[2]),
            "is_fallen": fell,
            **rinfo,
        }
        if terminated or truncated:
            survived = not fell and self._t_since_push > 1.0
            info["episode_survived"] = survived
        return obs, float(reward), terminated, truncated, info

    # ---------------- curriculum ----------------
    def record_episode(self, survived: bool) -> bool:
        self._curric_results.append(bool(survived))
        if len(self._curric_results) > PUSH_CURRIC_WINDOW:
            self._curric_results.pop(0)
        if len(self._curric_results) < PUSH_CURRIC_WINDOW:
            return False
        if np.mean(self._curric_results) >= PUSH_CURRIC_SURVIVE:
            new_max = min(self.cur_push_max + PUSH_CURRIC_STEP, PUSH_MAX_LIMIT)
            if new_max > self.cur_push_max:
                self.cur_push_max = new_max
                self.cur_push_min = min(self.cur_push_min + PUSH_CURRIC_STEP * 0.5,
                                        PUSH_MAX_LIMIT - 40)
                self.curriculum_level += 1
                self._curric_results.clear()
                return True
        return False
