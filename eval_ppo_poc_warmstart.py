"""Evaluate warm-start PPO vs zero/random/previous delta policies."""

import io
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_env import BipedalWalkEnv, PUSH_DURATION_STEPS, PUSH_STAND_STEPS

POC_PUSH_FORCE_MIN = 80.0
POC_PUSH_FORCE_MAX = 120.0
WARMSTART_SAVE_DIR = "ppo_poc_warmstart"
DELTA_SAVE_DIR = "ppo_poc_delta"
EPISODES_PER_CASE = 5
DIAGNOSTIC_CASE = {"name": "120N +X", "push_magnitude": 120.0, "push_direction_rad": 0.0}

EVAL_CASES = [
    {"name": "100N +X", "push_magnitude": 100.0, "push_direction_rad": 0.0},
    {"name": "120N +X", "push_magnitude": 120.0, "push_direction_rad": 0.0},
    {"name": "100N -X", "push_magnitude": 100.0, "push_direction_rad": np.pi},
    {"name": "120N -X", "push_magnitude": 120.0, "push_direction_rad": np.pi},
]

POST_PUSH_SAMPLE_STEP = PUSH_STAND_STEPS + PUSH_DURATION_STEPS


@dataclass
class EpisodeMetrics:
    survived: bool
    episode_length: int
    termination_reason: str
    max_tilt_rad: float
    max_pre_push_tilt: float
    tilt_at_push_rad: float
    tilt_after_push_rad: float
    max_post_push_tilt: float
    final_tilt_rad: float
    max_angvel: float
    mean_action_mag: float
    max_action_mag: float
    mean_pre_push_action: float
    max_pre_push_action: float
    mean_post_push_action: float
    max_post_push_action: float
    fell_before_push: bool
    recovered: bool
    nan_detected: bool
    action_trace: list = field(default_factory=list)
    tilt_trace: list = field(default_factory=list)


def make_env():
    env = BipedalWalkEnv()
    env.push_force_min = POC_PUSH_FORCE_MIN
    env.set_push_force_max(POC_PUSH_FORCE_MAX)
    return env


def run_episode(env, policy_fn, case, seed, collect_trace=False):
    options = {
        "push_magnitude": case["push_magnitude"],
        "push_direction_rad": case["push_direction_rad"],
        "push_step_start": PUSH_STAND_STEPS,
    }
    obs, _ = env.reset(seed=seed, options=options)

    max_tilt = 0.0
    max_pre_push_tilt = 0.0
    max_post_push_tilt = 0.0
    max_angvel = 0.0
    tilt_at_push = None
    tilt_after_push = None
    min_post_push_alignment = 1.0
    nan_detected = False
    action_mags = []
    pre_push_actions = []
    post_push_actions = []
    action_trace = []
    tilt_trace = []
    step = 0
    terminated = False
    truncated = False
    termination_reason = "running"
    final_tilt = 0.0

    while True:
        action = policy_fn(obs)
        action_mag = float(np.linalg.norm(action))
        action_mags.append(action_mag)

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if not np.isfinite(obs).all() or not np.isfinite(reward):
            nan_detected = True
            termination_reason = "nan"
            break

        tilt = info["quat_tilt_rad"]
        angvel = float(np.linalg.norm(info["chest_angvel"]))
        max_tilt = max(max_tilt, tilt)
        max_angvel = max(max_angvel, angvel)
        final_tilt = tilt

        if collect_trace:
            action_trace.append(action_mag)
            tilt_trace.append(tilt)

        if step < PUSH_STAND_STEPS:
            max_pre_push_tilt = max(max_pre_push_tilt, tilt)
            pre_push_actions.append(action_mag)
        elif step == PUSH_STAND_STEPS:
            tilt_at_push = tilt
            pre_push_actions.append(action_mag)
        else:
            max_post_push_tilt = max(max_post_push_tilt, tilt)
            post_push_actions.append(action_mag)
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
        tilt_at_push_rad=tilt_at_push if tilt_at_push is not None else max_tilt,
        tilt_after_push_rad=tilt_after_push if tilt_after_push is not None else max_tilt,
        max_post_push_tilt=max_post_push_tilt,
        final_tilt_rad=final_tilt,
        max_angvel=max_angvel,
        mean_action_mag=float(np.mean(action_mags)) if action_mags else 0.0,
        max_action_mag=float(np.max(action_mags)) if action_mags else 0.0,
        mean_pre_push_action=float(np.mean(pre_push_actions)) if pre_push_actions else 0.0,
        max_pre_push_action=float(np.max(pre_push_actions)) if pre_push_actions else 0.0,
        mean_post_push_action=float(np.mean(post_push_actions)) if post_push_actions else 0.0,
        max_post_push_action=float(np.max(post_push_actions)) if post_push_actions else 0.0,
        fell_before_push=fell_before_push,
        recovered=recovered,
        nan_detected=nan_detected,
        action_trace=action_trace,
        tilt_trace=tilt_trace,
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
    pre_fail = np.mean([m.fell_before_push for m in all_metrics])
    length = np.mean([m.episode_length for m in all_metrics])
    pre_tilt = np.mean([m.max_pre_push_tilt for m in all_metrics])
    tilt_push = np.mean([m.tilt_at_push_rad for m in all_metrics])
    tilt_after = np.mean([m.tilt_after_push_rad for m in all_metrics])
    post_max = np.mean([m.max_post_push_tilt for m in all_metrics])
    final_tilt = np.mean([m.final_tilt_rad for m in all_metrics])
    max_angvel = np.mean([m.max_angvel for m in all_metrics])
    mean_a = np.mean([m.mean_action_mag for m in all_metrics])
    max_a = np.mean([m.max_action_mag for m in all_metrics])
    pre_a = np.mean([m.mean_pre_push_action for m in all_metrics])
    post_a = np.mean([m.mean_post_push_action for m in all_metrics])

    print(
        f"{label:18s} | survive {survived:5.1%} | recover {recovered:5.1%} | "
        f"pre_fail {pre_fail:5.1%} | len {length:6.0f} | "
        f"pre_tilt {pre_tilt:5.3f} | at_push {tilt_push:5.3f} | "
        f"after {tilt_after:5.3f} | post_max {post_max:5.3f} | "
        f"final {final_tilt:5.3f} | angvel {max_angvel:5.3f} | "
        f"|a| {mean_a:4.2f}/{max_a:4.2f} | pre {pre_a:4.2f} post {post_a:4.2f}"
    )


def print_diagnostics(label, metrics: EpisodeMetrics, case_name):
    print(f"\n--- Diagnostics: {label} on {case_name} (seed 0) ---")
    print(
        f"Before push: mean |action|={metrics.mean_pre_push_action:.3f}, "
        f"max |action|={metrics.max_pre_push_action:.3f}, "
        f"max tilt={metrics.max_pre_push_tilt:.3f}"
    )
    print(
        f"At push:     tilt={metrics.tilt_at_push_rad:.3f}"
    )
    print(
        f"After push:  tilt@+5steps={metrics.tilt_after_push_rad:.3f}, "
        f"peak tilt={metrics.max_post_push_tilt:.3f}, "
        f"peak |action|={metrics.max_post_push_action:.3f}, "
        f"mean |action|={metrics.mean_post_push_action:.3f}"
    )
    print(f"Final:       tilt={metrics.final_tilt_rad:.3f}, len={metrics.episode_length}")

    if metrics.action_trace:
        push_idx = PUSH_STAND_STEPS
        pre_slice = metrics.action_trace[max(0, push_idx - 50) : push_idx]
        post_slice = metrics.action_trace[push_idx : push_idx + 100]
        if pre_slice and post_slice:
            print(
                f"Action surge after push: pre50 mean={np.mean(pre_slice):.3f}, "
                f"post100 mean={np.mean(post_slice):.3f}, "
                f"post100 max={np.max(post_slice):.3f}"
            )


def main():
    print("=== Warm-start PPO evaluation ===\n")

    warmstart_policy = load_trained_policy(WARMSTART_SAVE_DIR)
    delta_policy = load_trained_policy(DELTA_SAVE_DIR)

    if warmstart_policy is None:
        print(f"Missing model in {WARMSTART_SAVE_DIR}. Run train_ppo_poc_warmstart.py first.")
        return

    for case in EVAL_CASES:
        print(f"\n{case['name']}")
        policies = [
            ("zero", zero_policy),
            ("random", random_policy),
            ("prev_delta", delta_policy),
            ("warmstart", warmstart_policy),
        ]
        for label, policy_fn in policies:
            if policy_fn is None:
                continue
            metrics = [
                run_episode(make_env(), policy_fn, case, seed=ep)
                for ep in range(EPISODES_PER_CASE)
            ]
            summarize(label, metrics)

    print("\n=== Representative diagnostics ===")
    for label, policy_fn in [
        ("zero", zero_policy),
        ("prev_delta", delta_policy),
        ("warmstart", warmstart_policy),
    ]:
        if policy_fn is None:
            continue
        diag = run_episode(
            make_env(),
            policy_fn,
            DIAGNOSTIC_CASE,
            seed=0,
            collect_trace=True,
        )
        print_diagnostics(label, diag, DIAGNOSTIC_CASE["name"])


if __name__ == "__main__":
    main()
