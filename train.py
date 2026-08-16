from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from biped_env import BipedalWalkEnv

env = BipedalWalkEnv()

check_env(env)

model = PPO("MlpPolicy", env, verbose=1)

print("Beginning training loop...")

model.learn(total_timesteps=100000)

model.save("ppo_biped_walking_model")

print("Training complete and model saved!")
