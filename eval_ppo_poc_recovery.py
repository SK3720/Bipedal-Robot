"""Evaluate recovery-reward PPO vs zero/delta/warmstart baselines."""

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
RECOVERY_SAVE_DIR = "ppo_poc_recovery"
DELTA_SAVE_DIR = "ppo_poc_delta"
WARMSTART_SAVE_DIR = "ppo_poc_warmstart"
EPISODES_PER_CASE = 5
DIAGNOSTIC_CASE = {"name": "120N +X", "push_magnitude": 120.0, "push_direction_rad": 0.0}

EVAL_CASES = [
    {"name": "100N +X", "push_magnitude": 100.0, "push_direction_rad": 0.0},
    {"name": "120N +X", "push_magnitude": 120.0, "push_direction_rad": 0.0},
    {"name": "100N -X", "push_magnitude": 100.0, "push_direction_rad": np.pi},
    {"name": "120N -X", "push_magnitude": 120.0, "push_direction_rad": np.pi},
]

POST_PUSH_OFFSETS = [100, 200, 500]


@dataclass
class EpisodeMetrics:
    survived: bool
    episode_length: int
    termination_reason: str
    max_pre_push_tilt: float
    tilt_at_push: float
    peak_post_push_tilt: float
    tilt_at_offsets: dict
    final_tilt: float
    max_angvel: float
    mean_pre_push_action: float
    mean_post_push_action: float
    max_action_mag: float
    total_recovery_reward: float
    fell_before_push: bool
    recovered: bool
    nan_detected: bool
    tilt_trace: list = field(default_factory=list)
    action_trace: list = field(default_factory=list)
    recovery_trace: list = field(default_factory=list)


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

    max_pre_push_tilt = 0.0
    tilt_at_push = None
    peak_post_push_tilt = 0.0
    tilt_at_offsets = {}
    max_angvel = 0.0
    pre_push_actions = []
    post_push_actions = []
    all_action_mags = []
    total_recovery_reward = 0.0
    tilt_trace = []
    action_trace = []
    recovery_trace = []
    step = 0
    terminated = False
    truncated = False
    termination_reason = "running"
    final_tilt = 0.0
    min_post_push_alignment = 1.0

    while True:
        action = policy_fn(obs)
        action_mag = float(np.linalg.norm(action))
        all_action_mags.append(action_mag)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if not np.isfinite(obs).all() or not np.isfinite(reward):
            termination_reason = "nan"
            break

        tilt = info["quat_tilt_rad"]
        angvel = float(np.linalg.norm(info["chest_angvel"]))
        final_tilt = tilt
        max_angvel = max(max_angvel, angvel)
        total_recovery_reward += info.get("reward_recovery", 0.0)

        if collect_trace:
            tilt_trace.append(tilt)
            action_trace.append(action_mag)
            recovery_trace.append(info.get("reward_recovery", 0.0))

        if step < PUSH_STAND_STEPS:
            max_pre_push_tilt = max(max_pre_push_tilt, tilt)
            pre_push_actions.append(action_mag)
        elif step == PUSH_STAND_STEPS:
            tilt_at_push = tilt
            pre_push_actions.append(action_mag)
        else:
            peak_post_push_tilt = max(peak_post_push_tilt, tilt)
            post_push_actions.append(action_mag)
            min_post_push_alignment = min(
                min_post_push_alignment, info["up_alignment"]
            )

        for offset in POST_PUSH_OFFSETS:
            target_step = PUSH_STAND_STEPS + offset
            if step == target_step:
                tilt_at_offsets[offset] = tilt

        if terminated:
            termination_reason = "fall"
            break
        if truncated:
            termination_reason = "timeout"
            break

    survived = termination_reason != "fall" and termination_reason != "nan"
    nan_detected = termination_reason == "nan"
    fell_before_push = step <= PUSH_STAND_STEPS and not survived
    recovered = (
        survived
        and peak_post_push_tilt < 0.5
        and final_tilt < 0.3
        and min_post_push_alignment > 0.9
    )

    return EpisodeMetrics(
        survived=survived,
        episode_length=step,
        termination_reason=termination_reason,
        max_pre_push_tilt=max_pre_push_tilt,
        tilt_at_push=tilt_at_push if tilt_at_push is not None else final_tilt,
        peak_post_push_tilt=peak_post_push_tilt,
        tilt_at_offsets=tilt_at_offsets,
        final_tilt=final_tilt,
        max_angvel=max_angvel,
        mean_pre_push_action=float(np.mean(pre_push_actions)) if pre_push_actions else 0.0,
        mean_post_push_action=float(np.mean(post_push_actions)) if post_push_actions else 0.0,
        max_action_mag=float(np.max(all_action_mags)) if all_action_mags else 0.0,
        total_recovery_reward=total_recovery_reward,
        fell_before_push=fell_before_push,
        recovered=recovered,
        nan_detected=nan_detected,
        tilt_trace=tilt_trace,
        action_trace=action_trace,
        recovery_trace=recovery_trace,
    )


def zero_policy(_obs):
    return np.zeros(15, dtype=np.float32)


def random_policy(_obs):
    return np.random.uniform(-1.0, 1.0, 15).astype(np.float32)


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


def summarize(label, metrics_list):
    survived = np.mean([m.survived for m in metrics_list])
    recovered = np.mean([m.recovered for m in metrics_list])
    pre_fail = np.mean([m.fell_before_push for m in metrics_list])
    length = np.mean([m.episode_length for m in metrics_list])
    pre_tilt = np.mean([m.max_pre_push_tilt for m in metrics_list])
    at_push = np.mean([m.tilt_at_push for m in metrics_list])
    peak = np.mean([m.peak_post_push_tilt for m in metrics_list])
    final = np.mean([m.final_tilt for m in metrics_list])
    angvel = np.mean([m.max_angvel for m in metrics_list])
    pre_a = np.mean([m.mean_pre_push_action for m in metrics_list])
    post_a = np.mean([m.mean_post_push_action for m in metrics_list])
    max_a = np.mean([m.max_action_mag for m in metrics_list])
    rec_r = np.mean([m.total_recovery_reward for m in metrics_list])

    t100 = np.mean([m.tilt_at_offsets.get(100, np.nan) for m in metrics_list])
    t200 = np.mean([m.tilt_at_offsets.get(200, np.nan) for m in metrics_list])
    t500 = np.mean([m.tilt_at_offsets.get(500, np.nan) for m in metrics_list])

    print(
        f"{label:18s} | survive {survived:5.1%} | recover {recovered:5.1%} | "
        f"pre_fail {pre_fail:5.1%} | len {length:6.0f} | "
        f"pre_tilt {pre_tilt:5.3f} | at_push {at_push:5.3f} | "
        f"peak {peak:5.3f} | t+100 {t100:5.3f} | t+200 {t200:5.3f} | "
        f"t+500 {t500:5.3f} | final {final:5.3f} | angvel {angvel:5.3f} | "
        f"pre_a {pre_a:4.2f} post_a {post_a:4.2f} max_a {max_a:4.2f} | "
        f"rec_r {rec_r:6.2f}"
    )


def print_timeline(label, metrics: EpisodeMetrics):
    push = PUSH_STAND_STEPS
    print(f"\n--- Timeline: {label} on 120N +X (seed 0) ---")

    def window_mean(trace, start, end):
        if not trace or end <= start:
            return float("nan")
        return float(np.mean(trace[start:end]))

    def window_max(trace, start, end):
        if not trace or end <= start:
            return float("nan")
        return float(np.max(trace[start:end]))

    pre_tilt = window_max(metrics.tilt_trace, push - 100, push)
    pre_action = window_mean(metrics.action_trace, push - 100, push)
    at_push_tilt = metrics.tilt_trace[push - 1] if len(metrics.tilt_trace) >= push else float("nan")
    at_push_action = metrics.action_trace[push - 1] if len(metrics.action_trace) >= push else float("nan")

    post_end = min(len(metrics.tilt_trace), push + 300)
    post_peak_tilt = window_max(metrics.tilt_trace, push, post_end)
    post_peak_action = window_max(metrics.action_trace, push, post_end)
    post_recovery = float(np.sum(metrics.recovery_trace[push:post_end])) if post_end > push else 0.0

    print(
        f"BEFORE PUSH (steps {push-100}-{push}): "
        f"max tilt={pre_tilt:.3f}, mean |action|={pre_action:.3f}"
    )
    print(
        f"AT PUSH (step {push}): tilt={at_push_tilt:.3f}, |action|={at_push_action:.3f}"
    )
    print(
        f"AFTER PUSH (steps {push+1}-{post_end}): peak tilt={post_peak_tilt:.3f}, "
        f"peak |action|={post_peak_action:.3f}, recovery_reward_sum={post_recovery:.4f}"
    )

    for offset in [5, 50, 100, 200, 500]:
        idx = push + offset - 1
        if idx < len(metrics.tilt_trace):
            print(
                f"  step {push+offset}: tilt={metrics.tilt_trace[idx]:.3f}, "
                f"|action|={metrics.action_trace[idx]:.3f}, "
                f"rec_r={metrics.recovery_trace[idx]:.6f}"
            )

    decreasing = False
    if post_end > push + 20:
        early = np.mean(metrics.tilt_trace[push:push + 50])
        late = np.mean(metrics.tilt_trace[push + 100:push + 200]) if post_end > push + 200 else float("nan")
        decreasing = late < early
    print(f"Tilt decreasing after response (mean early vs late): {decreasing}")


def main():
    print("=== Recovery-reward PPO evaluation ===\n")

    recovery_policy = load_trained_policy(RECOVERY_SAVE_DIR)
    delta_policy = load_trained_policy(DELTA_SAVE_DIR)
    warmstart_policy = load_trained_policy(WARMSTART_SAVE_DIR)

    if recovery_policy is None:
        print(f"Missing model in {RECOVERY_SAVE_DIR}. Run train_ppo_poc_recovery.py first.")
        return

    for case in EVAL_CASES:
        print(f"\n{case['name']}")
        policies = [
            ("zero", zero_policy),
            ("prev_delta", delta_policy),
            ("prev_warmstart", warmstart_policy),
            ("recovery", recovery_policy),
        ]
        for label, policy_fn in policies:
            if policy_fn is None:
                continue
            metrics = [
                run_episode(make_env(), policy_fn, case, seed=ep)
                for ep in range(EPISODES_PER_CASE)
            ]
            summarize(label, metrics)

    print("\n=== Representative timeline (120N +X) ===")
    for label, policy_fn in [
        ("zero", zero_policy),
        ("prev_delta", delta_policy),
        ("prev_warmstart", warmstart_policy),
        ("recovery", recovery_policy),
    ]:
        if policy_fn is None:
            continue
        diag = run_episode(
            make_env(), policy_fn, DIAGNOSTIC_CASE, seed=0, collect_trace=True
        )
        print_timeline(label, diag)


if __name__ == "__main__":
    main()
