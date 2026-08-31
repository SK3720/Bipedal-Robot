"""PPO with warm-start and post-push recovery reward shaping."""

import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_env import (
    BipedalWalkEnv,
    PUSH_DIRECTION_SAGITTAL,
    PUSH_SAGITTAL_SPREAD_RAD,
    W_ACTION_WARMSTART,
    W_PRE_PUSH_TILT,
    W_RECOVERY,
)
from warmstart_utils import INITIAL_LOG_STD, init_zero_policy

POC_PUSH_FORCE_MIN = 80.0
POC_PUSH_FORCE_MAX = 120.0
TOTAL_TIMESTEPS = 250_000
N_STEPS = 1024
ENT_COEF = 0.01
SAVE_DIR = "ppo_poc_recovery"


def make_env():
    env = BipedalWalkEnv()
    env.push_force_min = POC_PUSH_FORCE_MIN
    env.set_push_force_max(POC_PUSH_FORCE_MAX)
    env.push_direction_mode = PUSH_DIRECTION_SAGITTAL
    env.push_sagittal_spread_rad = PUSH_SAGITTAL_SPREAD_RAD
    env.configure_recovery_reward()
    return Monitor(env, filename=os.path.join(SAVE_DIR, "monitor.csv"))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
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

    init_zero_policy(model)
    print(
        "Warm-start initialized: action_net weights/bias = 0, "
        f"log_std = {INITIAL_LOG_STD} (std ~= {torch.exp(model.policy.log_std).mean().item():.3f})"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=SAVE_DIR,
        name_prefix="ppo_poc_checkpoint",
    )

    spread_deg = PUSH_SAGITTAL_SPREAD_RAD * 180.0 / 3.141592653589793
    print(
        f"Training recovery-reward PPO for {TOTAL_TIMESTEPS} steps "
        f"(push {POC_PUSH_FORCE_MIN}-{POC_PUSH_FORCE_MAX} N, "
        f"sagittal +/-{spread_deg:.0f} deg, "
        f"W_ACTION={W_ACTION_WARMSTART}, W_PRE_PUSH_TILT={W_PRE_PUSH_TILT}, "
        f"W_RECOVERY={W_RECOVERY}, norm_reward=False)."
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

    model_path = os.path.join(SAVE_DIR, "ppo_poc_model")
    vecnorm_path = os.path.join(SAVE_DIR, "vecnormalize.pkl")
    model.save(model_path)
    vec_env.save(vecnorm_path)

    policy_path = os.path.join(SAVE_DIR, "policy.pth")
    torch.save(model.policy.state_dict(), policy_path)

    print(f"Saved model to {model_path}.zip")
    print(f"Saved policy weights to {policy_path}")
    print(f"Saved VecNormalize stats to {vecnorm_path}")


if __name__ == "__main__":
    main()
