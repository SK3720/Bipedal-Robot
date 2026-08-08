import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class BipedalWalkEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.num_actions = 15  # total number of servos
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_actions,), 
            dtype=np.float32
        )

        # Observation Space (34 sensor values)
        self.obs_dim = 34
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )

        self.model = mujoco.MjModel.from_xml_path("robot/robot.xml")
        self.data = mujoco.MjData(self.model)

    def step(self, action):
        # 1. Record X-position before the step
        x_pos_before = self.data.qpos[0]
        
        # 2. Scale and apply actions
        max_radian_limit = 1.57
        self.data.ctrl[:15] = action * max_radian_limit
        mujoco.mj_step(self.model, self.data)

        # 3. Record X-position after the step to calculate velocity
        x_pos_after = self.data.qpos[0]
        dt = self.model.opt.timestep
        forward_velocity = (x_pos_after - x_pos_before) / dt

        # --- REWARD CALCULATION ---
        
        # 1. Forward Progress
        forward_reward = forward_velocity * 1.25 
        
        # 2. Survival Bonus (Check if torso height is above 1.0 meters)
        z_height = self.data.qpos[2]
        is_healthy = z_height > 1.0
        healthy_reward = 5.0 if is_healthy else 0.0
        
        # 3. Control Cost (Sum of squared actions)
        ctrl_cost = 0.1 * np.sum(np.square(action))
        
        # 4. Posture Penalty (Assuming qpos[4] and qpos[5] are pitch/roll)
        posture_penalty = 0.5 * (np.square(self.data.qpos[4]) + np.square(self.data.qpos[5]))

        # Total Reward
        reward = forward_reward + healthy_reward - ctrl_cost - posture_penalty
        
        # --- TERMINATION ---
        
        # End the episode immediately if the robot falls below the height limit
        terminated = not is_healthy 
        truncated = False
        
        observation = self._get_obs()
        info = {
            "forward_velocity": forward_velocity,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward
        }
        
        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the physics simulation to its initial default state
        mujoco.mj_resetData(self.model, self.data)
        
        # Drop the robot from a safe starting height (e.g., 0.5 meters)
        self.data.qpos[0:3] = [0, 0, 0.5]
        self.data.qpos[3:7] = [1, 0, 0, 0] # Upright quaternion
        
        # Calculate the starting physics state
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_obs()
        info = {}
        
        return observation, info


    def _get_obs(self):
        # 1. Torso orientation (quaternion: indices 3, 4, 5, 6)
        torso_quat = self.data.qpos[3:7]
        
        # 2. Servo angles (skip the first 7 torso values)
        joint_angles = self.data.qpos[7:22]
        
        # 3. Servo velocities (skip the first 6 torso velocity values)
        joint_velocities = self.data.qvel[6:21]
        
        # Concatenate everything into a single 1D array for the neural network
        obs = np.concatenate([
            torso_quat,
            joint_angles,
            joint_velocities
        ])
        
        return obs.astype(np.float32)