import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from biped_env import BipedalWalkEnv

# 1. Instantiate your custom environment
env = BipedalWalkEnv()

# 2. Validate the environment
# This built-in utility checks your observation and action spaces 
# to ensure they comply with Gymnasium standards before training starts.
check_env(env)

# 3. Initialize the PPO agent
# "MlpPolicy" tells SB3 to use a standard Multi-Layer Perceptron neural network.
# verbose=1 prints the training statistics to your terminal.
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_biped_tensorboard/")

# 4. Train the agent
print("Beginning training loop...")
# 1,000,000 steps is a standard starting point for bipedal locomotion
model.learn(total_timesteps=1_000_000)

# 5. Save the trained policy
model.save("ppo_biped_walking_model")
print("Training complete and model saved!")