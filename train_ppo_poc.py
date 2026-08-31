"""Short PPO proof-of-concept training for disturbance recovery."""

import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_env import BipedalWalkEnv

POC_PUSH_FORCE_MIN = 80.0
POC_PUSH_FORCE_MAX = 120.0
TOTAL_TIMESTEPS = 250_000
N_STEPS = 1024
ENT_COEF = 0.01
SAVE_DIR = "ppo_poc_delta"


def make_env():
    env = BipedalWalkEnv()
    env.push_force_min = POC_PUSH_FORCE_MIN
    env.set_push_force_max(POC_PUSH_FORCE_MAX)
    return Monitor(env, filename=os.path.join(SAVE_DIR, "monitor.csv"))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=N_STEPS,
        batch_size=64,
        n_epochs=10,
        ent_coef=ENT_COEF,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=SAVE_DIR,
        name_prefix="ppo_poc_checkpoint",
    )

    print(
        f"Training delta-action PPO POC for {TOTAL_TIMESTEPS} steps "
        f"(push range {POC_PUSH_FORCE_MIN}-{POC_PUSH_FORCE_MAX} N, no curriculum)."
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

    model_path = os.path.join(SAVE_DIR, "ppo_poc_model")
    vecnorm_path = os.path.join(SAVE_DIR, "vecnormalize.pkl")
    model.save(model_path)
    vec_env.save(vecnorm_path)

    policy_path = os.path.join(SAVE_DIR, "policy.pth")
    import torch

    torch.save(model.policy.state_dict(), policy_path)

    print(f"Saved model to {model_path}.zip")
    print(f"Saved policy weights to {policy_path}")
    print(f"Saved VecNormalize stats to {vecnorm_path}")


if __name__ == "__main__":
    main()
