"""Evaluate sagittal-trained PPO vs zero/random on fixed disturbances."""

import io
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_env import BipedalWalkEnv, PUSH_DURATION_STEPS, PUSH_STAND_STEPS

POC_PUSH_FORCE_MIN = 80.0
POC_PUSH_FORCE_MAX = 120.0
SAGITTAL_SAVE_DIR = "ppo_poc_sagittal"
DELTA_SAVE_DIR = "ppo_poc_delta"
EPISODES_PER_CASE = 5

EVAL_CASES = [
    {"name": "100N +X", "push_magnitude": 100.0, "push_direction_rad": 0.0},
    {"name": "100N -X", "push_magnitude": 100.0, "push_direction_rad": np.pi},
    {"name": "120N +X", "push_magnitude": 120.0, "push_direction_rad": 0.0},
    {"name": "120N -X", "push_magnitude": 120.0, "push_direction_rad": np.pi},
    {"name": "100N +Y", "push_magnitude": 100.0, "push_direction_rad": np.pi / 2},
    {"name": "100N diag45", "push_magnitude": 100.0, "push_direction_rad": np.pi / 4},
]

POST_PUSH_SAMPLE_STEP = PUSH_STAND_STEPS + PUSH_DURATION_STEPS


@dataclass
class EpisodeMetrics:
    survived: bool
    episode_length: int
    termination_reason: str
    max_tilt_rad: float
    max_pre_push_tilt: float
    tilt_after_push_rad: float
    max_post_push_tilt: float
    final_tilt_rad: float
    fell_before_push: bool
    recovered: bool
    nan_detected: bool


def make_env():
    env = BipedalWalkEnv()
    env.push_force_min = POC_PUSH_FORCE_MIN
    env.set_push_force_max(POC_PUSH_FORCE_MAX)
    return env


def run_episode(env, policy_fn, case, seed):
    options = {
        "push_magnitude": case["push_magnitude"],
        "push_direction_rad": case["push_direction_rad"],
        "push_step_start": PUSH_STAND_STEPS,
    }
    obs, _ = env.reset(seed=seed, options=options)

    max_tilt = 0.0
    max_pre_push_tilt = 0.0
    max_post_push_tilt = 0.0
    tilt_after_push = None
    min_post_push_alignment = 1.0
    nan_detected = False
    step = 0
    terminated = False
    truncated = False
    termination_reason = "running"
    final_tilt = 0.0

    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if not np.isfinite(obs).all() or not np.isfinite(reward):
            nan_detected = True
            termination_reason = "nan"
            break

        tilt = info["quat_tilt_rad"]
        max_tilt = max(max_tilt, tilt)
        final_tilt = tilt

        if step <= PUSH_STAND_STEPS:
            max_pre_push_tilt = max(max_pre_push_tilt, tilt)
        else:
            max_post_push_tilt = max(max_post_push_tilt, tilt)
            min_post_push_alignment = min(
                min_post_push_alignment, info["up_alignment"]
            )

        if step == POST_PUSH_SAMPLE_STEP:
            tilt_after_push = tilt

        if terminated:
            termination_reason = "fall"
            break
        if truncated:
            termination_reason = "timeout"
            break

    survived = not terminated and not nan_detected
    fell_before_push = step <= PUSH_STAND_STEPS and (terminated or nan_detected)
    recovered = (
        survived
        and tilt_after_push is not None
        and max_post_push_tilt < 0.5
        and final_tilt < 0.3
        and min_post_push_alignment > 0.9
    )

    return EpisodeMetrics(
        survived=survived,
        episode_length=step,
        termination_reason=termination_reason,
        max_tilt_rad=max_tilt,
        max_pre_push_tilt=max_pre_push_tilt,
        tilt_after_push_rad=tilt_after_push if tilt_after_push is not None else max_tilt,
        max_post_push_tilt=max_post_push_tilt,
        final_tilt_rad=final_tilt,
        fell_before_push=fell_before_push,
        recovered=recovered,
        nan_detected=nan_detected,
    )


def random_policy(_obs):
    return np.random.uniform(-1.0, 1.0, 15).astype(np.float32)


def zero_policy(_obs):
    return np.zeros(15, dtype=np.float32)


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


def summarize(label, all_metrics):
    survived = np.mean([m.survived for m in all_metrics])
    recovered = np.mean([m.recovered for m in all_metrics])
    pre_push_fail = np.mean([m.fell_before_push for m in all_metrics])
    length = np.mean([m.episode_length for m in all_metrics])
    max_tilt = np.mean([m.max_tilt_rad for m in all_metrics])
    pre_tilt = np.mean([m.max_pre_push_tilt for m in all_metrics])
    tilt_after = np.mean([m.tilt_after_push_rad for m in all_metrics])
    post_tilt = np.mean([m.max_post_push_tilt for m in all_metrics])
    final_tilt = np.mean([m.final_tilt_rad for m in all_metrics])
    nan_rate = np.mean([m.nan_detected for m in all_metrics])

    print(
        f"{label:18s} | survive {survived:5.1%} | recover {recovered:5.1%} | "
        f"pre_fail {pre_push_fail:5.1%} | len {length:6.0f} | "
        f"max_tilt {max_tilt:5.3f} | after_push {tilt_after:5.3f} | "
        f"post_max {post_tilt:5.3f} | final {final_tilt:5.3f} | "
        f"pre_tilt {pre_tilt:5.3f} | nan {nan_rate:5.1%}"
    )


def main():
    print("=== Sagittal PPO evaluation (fixed disturbances) ===\n")

    sagittal_policy = load_trained_policy(SAGITTAL_SAVE_DIR)
    delta_policy = load_trained_policy(DELTA_SAVE_DIR)

    if sagittal_policy is None:
        print(f"Missing model in {SAGITTAL_SAVE_DIR}. Run train_ppo_poc_sagittal.py first.")
        return

    for case in EVAL_CASES:
        print(f"\n{case['name']}")
        policies = [
            ("zero", zero_policy),
            ("random", random_policy),
            ("prev_delta", delta_policy),
            ("sagittal", sagittal_policy),
        ]

        for label, policy_fn in policies:
            if policy_fn is None:
                continue
            metrics = [
                run_episode(make_env(), policy_fn, case, seed=ep)
                for ep in range(EPISODES_PER_CASE)
            ]
            summarize(label, metrics)


if __name__ == "__main__":
    main()
