import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


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

    def step(self, action):

        x_pos_before = self.data.qpos[0]

        max_radian_limit = 1.57

        self.data.ctrl[:15] = action * max_radian_limit

        mujoco.mj_step(self.model, self.data)

        x_pos_after = self.data.qpos[0]

        dt = self.model.opt.timestep

        forward_velocity = (x_pos_after - x_pos_before) / dt

        # -------------------------
        # REWARD
        # -------------------------

        forward_reward = forward_velocity * 1.25

        z_height = self.data.qpos[2]

        is_healthy = z_height > 1.0

        healthy_reward = 5.0 if is_healthy else 0.0

        ctrl_cost = 0.1 * np.sum(np.square(action))

        posture_penalty = 0.5 * (
            np.square(self.data.qpos[4]) + np.square(self.data.qpos[5])
        )

        reward = forward_reward + healthy_reward - ctrl_cost - posture_penalty

        # -------------------------
        # TERMINATION
        # -------------------------

        terminated = not is_healthy
        truncated = False

        observation = self._get_obs()

        info = {
            "forward_velocity": forward_velocity,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return (observation, reward, terminated, truncated, info)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Starting position
        self.data.qpos[3:7] = [1, 0, 0, 0]
        # Rotate the ENTIRE robot 90 degrees around X
        angle = np.pi / 2

        self.data.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]

        # -------------------------
        # LEFT LEG
        # -------------------------

        # self.data.qpos[12] = 0.0
        # self.data.qpos[13] = 1.5
        # self.data.qpos[14] = 0.0
        # self.data.qpos[15] = 0.0
        # self.data.qpos[16] = 0.0

        # # -------------------------
        # # RIGHT LEG
        # # -------------------------

        # self.data.qpos[17] = 0.0
        # self.data.qpos[18] = -1.5
        # self.data.qpos[19] = 0.0
        # self.data.qpos[20] = 0.0
        # self.data.qpos[21] = 0.0

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def _get_obs(self):

        torso_quat = self.data.qpos[3:7]

        joint_angles = self.data.qpos[7:22]

        joint_velocities = self.data.qvel[6:21]

        obs = np.concatenate([torso_quat, joint_angles, joint_velocities])

        return obs.astype(np.float32)
