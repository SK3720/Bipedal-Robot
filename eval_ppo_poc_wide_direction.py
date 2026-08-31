"""Evaluate wide-direction recovery PPO vs sagittal recovery and zero baseline."""

import io
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_env import BipedalWalkEnv, PUSH_DURATION_STEPS, PUSH_STAND_STEPS

WIDE_SAVE_DIR = "ppo_poc_wide_direction"
RECOVERY_SAVE_DIR = "ppo_poc_recovery"
EPISODES_PER_CASE = 10

EVAL_CASES = [
    {"name": "100N +X", "push_magnitude": 100.0, "push_direction_rad": 0.0},
    {"name": "120N +X", "push_magnitude": 120.0, "push_direction_rad": 0.0},
    {"name": "150N +X", "push_magnitude": 150.0, "push_direction_rad": 0.0},
    {"name": "100N -X", "push_magnitude": 100.0, "push_direction_rad": np.pi},
    {"name": "120N -X", "push_magnitude": 120.0, "push_direction_rad": np.pi},
    {"name": "150N -X", "push_magnitude": 150.0, "push_direction_rad": np.pi},
    {"name": "120N 45deg", "push_magnitude": 120.0, "push_direction_rad": np.pi / 4},
    {"name": "120N 135deg", "push_magnitude": 120.0, "push_direction_rad": 3 * np.pi / 4},
    {"name": "120N 225deg", "push_magnitude": 120.0, "push_direction_rad": 5 * np.pi / 4},
    {"name": "120N 315deg", "push_magnitude": 120.0, "push_direction_rad": 7 * np.pi / 4},
    {"name": "120N +Y", "push_magnitude": 120.0, "push_direction_rad": np.pi / 2},
    {"name": "120N -Y", "push_magnitude": 120.0, "push_direction_rad": 3 * np.pi / 2},
]

POST_PUSH_WINDOW = 50


@dataclass
class EpisodeMetrics:
    survived: bool
    episode_length: int
    fell_before_push: bool
    max_pre_push_tilt: float
    tilt_at_push: float
    peak_post_push_tilt: float
    final_tilt: float
    mean_action_mag: float
    peak_action_mag: float
    mean_pre_push_action: float
    mean_post_push_action: float
    post_push_action_surge: bool


def make_env():
    env = BipedalWalkEnv()
    env.push_force_min = 80.0
    env.set_push_force_max(200.0)
    return env


def load_trained_policy(save_dir):
    model_path = os.path.join(save_dir, "ppo_poc_model")
    vecnorm_path = os.path.join(save_dir, "vecnormalize.pkl")
    policy_path = os.path.join(save_dir, "policy.pth")
    model_zip = model_path + ".zip"

    if not os.path.exists(model_zip) and not os.path.exists(policy_path):
        return None

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    policy_env = make_env()
    model = PPO("MlpPolicy", policy_env, device="cpu")

    if os.path.exists(policy_path):
        policy_state = torch.load(policy_path, map_location="cpu", weights_only=True)
    else:
        with zipfile.ZipFile(model_zip) as archive:
            policy_state = torch.load(
                io.BytesIO(archive.read("policy.pth")),
                map_location="cpu",
                weights_only=True,
            )
    model.policy.load_state_dict(policy_state)

    def policy(raw_obs):
        obs = vec_env.normalize_obs(np.array([raw_obs], dtype=np.float32))
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action[0], dtype=np.float32)

    return policy


def run_episode(env, policy_fn, case, seed):
    options = {
        "push_magnitude": case["push_magnitude"],
        "push_direction_rad": case["push_direction_rad"],
        "push_step_start": PUSH_STAND_STEPS,
    }
    obs, _ = env.reset(seed=seed, options=options)

    max_pre_push_tilt = 0.0
    tilt_at_push = None
    peak_post_push_tilt = 0.0
    pre_push_actions = []
    post_push_actions = []
    all_action_mags = []
    step = 0
    final_tilt = 0.0
    terminated = False
    truncated = False

    while True:
        action = policy_fn(obs)
        action_mag = float(np.linalg.norm(action))
        all_action_mags.append(action_mag)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if not np.isfinite(obs).all() or not np.isfinite(reward):
            break

        tilt = info["quat_tilt_rad"]
        final_tilt = tilt

        if step < PUSH_STAND_STEPS:
            max_pre_push_tilt = max(max_pre_push_tilt, tilt)
            pre_push_actions.append(action_mag)
        elif step == PUSH_STAND_STEPS:
            tilt_at_push = tilt
            pre_push_actions.append(action_mag)
        else:
            peak_post_push_tilt = max(peak_post_push_tilt, tilt)
            post_push_actions.append(action_mag)

        if terminated or truncated:
            break

    survived = not terminated
    fell_before_push = step <= PUSH_STAND_STEPS and terminated
    mean_pre = float(np.mean(pre_push_actions)) if pre_push_actions else 0.0
    mean_post = float(np.mean(post_push_actions)) if post_push_actions else 0.0
    post_push_action_surge = (
        mean_post > mean_pre + 0.05 and len(post_push_actions) >= POST_PUSH_WINDOW
    )

    return EpisodeMetrics(
        survived=survived,
        episode_length=step,
        fell_before_push=fell_before_push,
        max_pre_push_tilt=max_pre_push_tilt,
        tilt_at_push=tilt_at_push if tilt_at_push is not None else final_tilt,
        peak_post_push_tilt=peak_post_push_tilt,
        final_tilt=final_tilt,
        mean_action_mag=float(np.mean(all_action_mags)) if all_action_mags else 0.0,
        peak_action_mag=float(np.max(all_action_mags)) if all_action_mags else 0.0,
        mean_pre_push_action=mean_pre,
        mean_post_push_action=mean_post,
        post_push_action_surge=post_push_action_surge,
    )


def zero_policy(_obs):
    return np.zeros(15, dtype=np.float32)


def summarize(label, metrics_list):
    print(
        f"{label:16s} | survive {np.mean([m.survived for m in metrics_list]):5.1%} | "
        f"pre_fail {np.mean([m.fell_before_push for m in metrics_list]):5.1%} | "
        f"len {np.mean([m.episode_length for m in metrics_list]):6.0f} | "
        f"pre_tilt {np.mean([m.max_pre_push_tilt for m in metrics_list]):5.3f} | "
        f"at_push {np.mean([m.tilt_at_push for m in metrics_list]):5.3f} | "
        f"peak {np.mean([m.peak_post_push_tilt for m in metrics_list]):5.3f} | "
        f"final {np.mean([m.final_tilt for m in metrics_list]):5.3f} | "
        f"|a| {np.mean([m.mean_action_mag for m in metrics_list]):4.2f}/"
        f"{np.mean([m.peak_action_mag for m in metrics_list]):4.2f} | "
        f"post_surge {np.mean([m.post_push_action_surge for m in metrics_list]):5.1%}"
    )


def main():
    print("=== Wide-direction recovery evaluation ===\n")

    wide_policy = load_trained_policy(WIDE_SAVE_DIR)
    recovery_policy = load_trained_policy(RECOVERY_SAVE_DIR)

    if wide_policy is None:
        print(f"Missing model in {WIDE_SAVE_DIR}. Run train_ppo_poc_wide_direction.py first.")
        return

    for case in EVAL_CASES:
        print(f"\n{case['name']}")
        policies = [
            ("zero", zero_policy),
            ("prev_recovery", recovery_policy),
            ("wide_dir", wide_policy),
        ]
        for label, policy_fn in policies:
            if policy_fn is None:
                continue
            metrics = [
                run_episode(make_env(), policy_fn, case, seed=ep)
                for ep in range(EPISODES_PER_CASE)
            ]
            summarize(label, metrics)

    print("\n=== Comparison summary ===")
    print("Focus: does wide_dir improve lateral/diagonal vs prev_recovery without")
    print("breaking sagittal performance or pre-push stability?")


if __name__ == "__main__":
    main()
