import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

# Standing pose: CAD Y-up frame compensated by 90 deg rotation about world X.
STANDING_QUAT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
# Foot collision geoms sit at chest_z - CHEST_FOOT_OFFSET when joints are at default.
CHEST_Z_CONTACT = 1.26
CHEST_FOOT_OFFSET = 0.26
DEFAULT_POSE = np.zeros(15, dtype=np.float64)
ACTION_SCALE = np.deg2rad(15.0)
SETTLE_STEPS = 500

# Balance reward coefficients.
W_UPRIGHT = 2.0
W_ANGVEL = 0.1
W_DRIFT = 0.5
W_ALIVE = 0.1
W_ACTION = 0.01
W_ACTION_RATE = 0.05
W_ACTION_WARMSTART = 0.05
W_PRE_PUSH_TILT = 1.0
W_RECOVERY = 1.0
RECOVERY_TILT_MIN = 0.04
RECOVERY_IMPROVEMENT_DEADBAND = 0.0005
RECOVERY_IMPROVEMENT_CLIP = 0.02

# Termination thresholds.
MAX_TILT_RAD = 0.9  # ~52 degrees from nominal standing orientation.
MIN_HEIGHT_MARGIN = 0.15  # meters below settled standing chest height.
MAX_EPISODE_STEPS = 10_000  # 10 s at 1 ms physics timestep.

# Disturbance injection (one horizontal chest impulse per episode).
PUSH_STAND_STEPS = 1_000  # ~1 s standing after reset before earliest push.
PUSH_TIME_JITTER_STEPS = 500  # random delay up to 0.5 s after stand period.
PUSH_DURATION_STEPS = 5
PUSH_FORCE_MIN = 60.0
PUSH_FORCE_MAX_INITIAL = 100.0
PUSH_FORCE_MAX_LIMIT = 180.0
PUSH_FORCE_CURRICULUM_STEP = 10.0
PUSH_CURRICULUM_SURVIVAL_THRESHOLD = 0.8
PUSH_CURRICULUM_WINDOW = 20
PUSH_DIRECTION_UNIFORM = "uniform"
PUSH_DIRECTION_WIDE = "wide"
PUSH_DIRECTION_SAGITTAL = "sagittal"
PUSH_SAGITTAL_SPREAD_RAD = np.deg2rad(30.0)


class BipedalWalkEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.num_actions = 15

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )

        self.obs_dim = 34

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.model = mujoco.MjModel.from_xml_path("robot/robot.xml")

        self.data = mujoco.MjData(self.model)

        self.chest_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Chest"
        )
        self.ctrl_low = self.model.actuator_ctrlrange[:15, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:15, 1].copy()
        self.nominal_gravity_body = self._compute_nominal_gravity_body()
        self.nominal_chest_up_body = np.array([0.0, 1.0, 0.0])
        self.world_up = np.array([0.0, 0.0, 1.0])

        self.initial_xy = np.zeros(2, dtype=np.float64)
        self.initial_chest_z = CHEST_Z_CONTACT
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
        self.step_count = 0

        self.push_force_min = PUSH_FORCE_MIN
        self.push_force_max = PUSH_FORCE_MAX_INITIAL
        self.curriculum_level = 0
        self._curriculum_episode_results = []

        self.push_step_start = PUSH_STAND_STEPS
        self.push_magnitude = 0.0
        self.push_direction_rad = 0.0
        self.push_force_xy = np.zeros(2, dtype=np.float64)
        self.disturbance_enabled = True
        self.push_direction_mode = PUSH_DIRECTION_UNIFORM
        self.push_sagittal_spread_rad = PUSH_SAGITTAL_SPREAD_RAD
        self.w_action = W_ACTION
        self.w_action_rate = W_ACTION_RATE
        self.w_pre_push_tilt = 0.0
        self.w_recovery = 0.0
        self.prev_quat_tilt = 0.0

    def configure_warmstart_reward(self):
        """Enable stronger action and pre-push stability shaping."""
        self.w_action = W_ACTION_WARMSTART
        self.w_pre_push_tilt = W_PRE_PUSH_TILT

    def configure_recovery_reward(self):
        """Enable warm-start shaping plus post-push recovery reward."""
        self.configure_warmstart_reward()
        self.w_recovery = W_RECOVERY

    def _sample_push_direction_rad(self, mode=None, spread_rad=None):
        mode = mode or self.push_direction_mode
        spread_rad = (
            self.push_sagittal_spread_rad if spread_rad is None else spread_rad
        )

        if mode in (PUSH_DIRECTION_UNIFORM, PUSH_DIRECTION_WIDE):
            return float(self.np_random.uniform(0.0, 2.0 * np.pi))

        if mode == PUSH_DIRECTION_SAGITTAL:
            axis = 0.0 if self.np_random.random() < 0.5 else np.pi
            noise = float(self.np_random.uniform(-spread_rad, spread_rad))
            return axis + noise

        raise ValueError(f"Unknown push direction mode: {mode!r}")

    def _compute_nominal_gravity_body(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
        self.data.qpos[3:7] = STANDING_QUAT
        self.data.qpos[7:22] = DEFAULT_POSE
        mujoco.mj_forward(self.model, self.data)
        chest_rot = self.data.xmat[self.chest_body_id].reshape(3, 3)
        return chest_rot.T @ np.array([0.0, 0.0, -1.0])

    def _chest_gravity_body(self):
        chest_rot = self.data.xmat[self.chest_body_id].reshape(3, 3)
        return chest_rot.T @ np.array([0.0, 0.0, -1.0])

    def _chest_up_alignment(self):
        chest_rot = self.data.xmat[self.chest_body_id].reshape(3, 3)
        chest_up_world = chest_rot @ self.nominal_chest_up_body
        return float(np.dot(chest_up_world, self.world_up))

    def _quat_tilt_rad(self):
        quat = self.data.qpos[3:7]
        cos_half_angle = np.clip(
            abs(np.dot(quat, STANDING_QUAT)), 0.0, 1.0
        )
        return 2.0 * np.arccos(cos_half_angle)

    def _compute_reward(self, action):
        gravity_body = self._chest_gravity_body()
        gravity_error = np.linalg.norm(gravity_body - self.nominal_gravity_body)
        up_alignment = np.clip(self._chest_up_alignment(), 0.0, 1.0)
        upright_reward = W_UPRIGHT * up_alignment**2

        chest_angvel = self.data.qvel[3:6]
        angvel_penalty = -W_ANGVEL * np.sum(np.square(chest_angvel))

        xy_drift = self.data.qpos[0:2] - self.initial_xy
        drift_penalty = -W_DRIFT * np.sum(np.square(xy_drift))

        alive_reward = W_ALIVE

        action_cost = -self.w_action * np.sum(np.square(action))

        action_delta = action - self.prev_action
        action_rate_cost = -self.w_action_rate * np.sum(np.square(action_delta))

        pre_push_tilt_cost = 0.0
        if self.w_pre_push_tilt > 0.0 and self.step_count <= self.push_step_start:
            tilt = self._quat_tilt_rad()
            pre_push_tilt_cost = -self.w_pre_push_tilt * (tilt**2)

        current_tilt = self._quat_tilt_rad()
        recovery_reward = 0.0
        tilt_improvement = 0.0
        post_push = self.step_count > self.push_step_start + PUSH_DURATION_STEPS
        if self.w_recovery > 0.0 and post_push and current_tilt > RECOVERY_TILT_MIN:
            tilt_improvement = self.prev_quat_tilt - current_tilt
            if abs(tilt_improvement) > RECOVERY_IMPROVEMENT_DEADBAND:
                clipped_improvement = float(
                    np.clip(
                        tilt_improvement,
                        -RECOVERY_IMPROVEMENT_CLIP,
                        RECOVERY_IMPROVEMENT_CLIP,
                    )
                )
                recovery_reward = self.w_recovery * clipped_improvement

        reward = (
            upright_reward
            + angvel_penalty
            + drift_penalty
            + alive_reward
            + action_cost
            + action_rate_cost
            + pre_push_tilt_cost
            + recovery_reward
        )

        reward_info = {
            "reward_upright": upright_reward,
            "reward_angvel": angvel_penalty,
            "reward_drift": drift_penalty,
            "reward_alive": alive_reward,
            "reward_action": action_cost,
            "reward_action_rate": action_rate_cost,
            "reward_pre_push_tilt": pre_push_tilt_cost,
            "reward_recovery": recovery_reward,
            "tilt_improvement": tilt_improvement,
            "post_push_recovery_active": post_push,
            "gravity_error": gravity_error,
            "up_alignment": up_alignment,
            "quat_tilt_rad": current_tilt,
        }
        return reward, reward_info

    def set_push_force_max(self, push_force_max):
        self.push_force_max = float(
            np.clip(push_force_max, self.push_force_min, PUSH_FORCE_MAX_LIMIT)
        )

    def update_curriculum(self, survived):
        """Record episode outcome and raise push cap after sustained recovery."""
        self._curriculum_episode_results.append(bool(survived))
        if len(self._curriculum_episode_results) > PUSH_CURRICULUM_WINDOW:
            self._curriculum_episode_results.pop(0)

        if len(self._curriculum_episode_results) < PUSH_CURRICULUM_WINDOW:
            return False

        survival_rate = np.mean(self._curriculum_episode_results)
        if survival_rate >= PUSH_CURRICULUM_SURVIVAL_THRESHOLD:
            new_max = min(
                self.push_force_max + PUSH_FORCE_CURRICULUM_STEP,
                PUSH_FORCE_MAX_LIMIT,
            )
            if new_max > self.push_force_max:
                self.push_force_max = new_max
                self.curriculum_level += 1
                self._curriculum_episode_results.clear()
                return True
        return False

    def _schedule_push(self, options):
        options = options or {}

        if not options.get("enable_push", self.disturbance_enabled):
            self.push_step_start = MAX_EPISODE_STEPS + 1
            self.push_magnitude = 0.0
            self.push_direction_rad = 0.0
            self.push_force_xy[:] = 0.0
            return

        if "push_step_start" in options:
            self.push_step_start = int(options["push_step_start"])
        else:
            jitter = self.np_random.integers(0, PUSH_TIME_JITTER_STEPS + 1)
            self.push_step_start = PUSH_STAND_STEPS + int(jitter)

        if "push_direction_rad" in options:
            self.push_direction_rad = float(options["push_direction_rad"])
        else:
            direction_mode = options.get(
                "push_direction_mode", self.push_direction_mode
            )
            spread_rad = options.get(
                "push_sagittal_spread_rad", self.push_sagittal_spread_rad
            )
            self.push_direction_rad = self._sample_push_direction_rad(
                mode=direction_mode,
                spread_rad=spread_rad,
            )

        if "push_magnitude" in options:
            self.push_magnitude = float(options["push_magnitude"])
        else:
            self.push_magnitude = float(
                self.np_random.uniform(self.push_force_min, self.push_force_max)
            )

        direction = np.array(
            [
                np.cos(self.push_direction_rad),
                np.sin(self.push_direction_rad),
            ],
            dtype=np.float64,
        )
        self.push_force_xy = self.push_magnitude * direction

    def _apply_scheduled_push(self):
        self.data.xfrc_applied[self.chest_body_id, :] = 0.0

        push_end = self.push_step_start + PUSH_DURATION_STEPS
        if self.push_step_start <= self.step_count < push_end:
            self.data.xfrc_applied[self.chest_body_id, 0] = self.push_force_xy[0]
            self.data.xfrc_applied[self.chest_body_id, 1] = self.push_force_xy[1]
            return True
        return False

    def _disturbance_info(self, disturbance_active):
        return {
            "disturbance_active": disturbance_active,
            "push_step_start": self.push_step_start,
            "push_duration_steps": PUSH_DURATION_STEPS,
            "push_magnitude": self.push_magnitude,
            "push_direction_rad": self.push_direction_rad,
            "push_force_xy": self.push_force_xy.copy(),
            "push_force_min": self.push_force_min,
            "push_force_max": self.push_force_max,
            "curriculum_level": self.curriculum_level,
        }

    def _action_to_ctrl(self, action):
        action = np.asarray(action, dtype=np.float64)
        target = DEFAULT_POSE + action * ACTION_SCALE
        return np.clip(target, self.ctrl_low, self.ctrl_high)

    def _is_fallen(self):
        chest_z = self.data.qpos[2]
        return (
            self._quat_tilt_rad() > MAX_TILT_RAD
            or chest_z < self.initial_chest_z - MIN_HEIGHT_MARGIN
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        disturbance_active = self._apply_scheduled_push()
        commanded_ctrl = self._action_to_ctrl(action)
        self.data.ctrl[:15] = commanded_ctrl

        mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        reward, reward_info = self._compute_reward(action)

        terminated = self._is_fallen()
        truncated = self.step_count >= MAX_EPISODE_STEPS

        observation = self._get_obs()

        info = {
            "chest_z": self.data.qpos[2],
            "chest_xy_displacement": float(
                np.linalg.norm(self.data.qpos[0:2] - self.initial_xy)
            ),
            "chest_angvel": self.data.qvel[3:6].copy(),
            "commanded_ctrl": commanded_ctrl.copy(),
            "is_fallen": terminated,
            **reward_info,
            **self._disturbance_info(disturbance_active),
        }

        self.prev_action = action.copy()
        self.prev_quat_tilt = reward_info["quat_tilt_rad"]

        return (observation, reward, terminated, truncated, info)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
        self.data.qpos[3:7] = STANDING_QUAT
        self.data.qpos[7:22] = DEFAULT_POSE
        self.data.qvel[:] = 0.0
        self.data.ctrl[:15] = DEFAULT_POSE

        mujoco.mj_forward(self.model, self.data)

        for _ in range(SETTLE_STEPS):
            self.data.ctrl[:15] = DEFAULT_POSE
            mujoco.mj_step(self.model, self.data)

        self.initial_xy = self.data.qpos[0:2].copy()
        self.initial_chest_z = self.data.qpos[2]
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
        self.step_count = 0
        self.data.xfrc_applied[self.chest_body_id, :] = 0.0

        if "push_force_max" in options:
            self.set_push_force_max(options["push_force_max"])

        self._schedule_push(options)
        self.prev_quat_tilt = self._quat_tilt_rad()

        return self._get_obs(), self._disturbance_info(disturbance_active=False)

    def _get_obs(self):

        torso_quat = self.data.qpos[3:7]

        joint_angles = self.data.qpos[7:22]

        joint_velocities = self.data.qvel[6:21]

        obs = np.concatenate([torso_quat, joint_angles, joint_velocities])

        return obs.astype(np.float32)
