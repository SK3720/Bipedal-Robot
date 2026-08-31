"""Verify zero-policy warm-start initialization before training."""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_env import BipedalWalkEnv
from warmstart_utils import (
    INITIAL_LOG_STD,
    compare_zero_policy_to_baseline,
    init_zero_policy,
    verify_zero_policy,
)


def make_env():
    env = BipedalWalkEnv()
    env.configure_warmstart_reward()
    return env


def main():
    print("=== Zero-policy warm-start verification ===\n")

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu")

    print("Before warm start:")
    before = verify_zero_policy(model, make_env())
    for row in before:
        print(
            f"  seed {row['seed']}: |action| det={row['det_norm']:.4f}, "
            f"sto mean={row['sto_norm_mean']:.3f}, max={row['sto_norm_max']:.3f}"
        )

    init_zero_policy(model)
    print(f"\nAfter warm start (log_std={INITIAL_LOG_STD}, std~{np.exp(INITIAL_LOG_STD):.3f}):")
    after = verify_zero_policy(model, make_env())
    for row in after:
        print(
            f"  seed {row['seed']}: |action| det={row['det_norm']:.4f}, "
            f"sto mean={row['sto_norm_mean']:.3f}, max={row['sto_norm_max']:.3f}"
        )

    baseline = compare_zero_policy_to_baseline(model, make_env())
    print(
        "\nStability comparison (3000 steps, no push):\n"
        f"  warm-start policy max tilt: {baseline['policy_max_tilt']:.4f} rad\n"
        f"  zero-action max tilt:       {baseline['zero_max_tilt']:.4f} rad"
    )

    det_ok = all(row["det_norm"] < 1e-5 for row in after)
    sto_ok = all(row["sto_norm_max"] < 1.0 for row in after)
    tilt_ok = baseline["policy_max_tilt"] < 0.1
    ok = det_ok and sto_ok and tilt_ok
    print(f"\n=== Overall: {'PASS' if ok else 'FAIL'} ===")


if __name__ == "__main__":
    main()
