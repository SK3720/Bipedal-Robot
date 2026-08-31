"""Utilities for zero-policy warm-start PPO training."""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

INITIAL_LOG_STD = -2.0  # std ~= 0.135 for moderate initial exploration


def init_zero_policy(model: PPO, log_std: float = INITIAL_LOG_STD) -> None:
    """Initialize the actor mean to zero while keeping learning enabled.

    Sets the final actor linear layer weights and bias to zero so the
    deterministic policy outputs action ~= 0 (target ~= DEFAULT_POSE).
    Reduces initial exploration by setting a smaller action log-std.
    """
    nn.init.zeros_(model.policy.action_net.weight)
    nn.init.zeros_(model.policy.action_net.bias)
    model.policy.log_std.data.fill_(log_std)


def verify_zero_policy(model: PPO, env, seeds=(0, 1, 2)):
    """Check that initialized policy is near-zero and stable under zero-mean."""
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        det_action, _ = model.predict(obs, deterministic=True)
        det_norm = float(np.linalg.norm(det_action))

        sto_norms = []
        for _ in range(10):
            sto_action, _ = model.predict(obs, deterministic=False)
            sto_norms.append(float(np.linalg.norm(sto_action)))

        results.append(
            {
                "seed": seed,
                "det_norm": det_norm,
                "sto_norm_mean": float(np.mean(sto_norms)),
                "sto_norm_max": float(np.max(sto_norms)),
            }
        )
    return results


def compare_zero_policy_to_baseline(model: PPO, env, steps=3000, seed=0):
    """Compare warm-started deterministic policy against true zero actions."""
    obs, _ = env.reset(seed=seed, options={"enable_push": False})

    policy_max_tilt = 0.0
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        policy_max_tilt = max(policy_max_tilt, info["quat_tilt_rad"])
        if terminated or truncated:
            break

    env.reset(seed=seed, options={"enable_push": False})
    zero_max_tilt = 0.0
    zero_action = np.zeros(env.num_actions, dtype=np.float32)
    for _ in range(steps):
        obs, _, terminated, truncated, info = env.step(zero_action)
        zero_max_tilt = max(zero_max_tilt, info["quat_tilt_rad"])
        if terminated or truncated:
            break

    return {
        "policy_max_tilt": policy_max_tilt,
        "zero_max_tilt": zero_max_tilt,
        "policy_steps": steps,
    }
