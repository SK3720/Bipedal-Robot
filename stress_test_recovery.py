"""Stress-test the trained recovery policy against fixed push magnitudes/directions."""

import io
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_env import BipedalWalkEnv, PUSH_STAND_STEPS

RECOVERY_SAVE_DIR = "ppo_poc_recovery"
SAGITTAL_MAGNITUDES = [100, 120, 140, 150, 160, 180]
SAGITTAL_SEEDS = 10
DIRECTION_MAGNITUDE = 120.0
DIRECTION_DEGREES = [0, 30, 45, 60, 90, 135, 180, 225, 270, 315]
DIRECTION_SEEDS = 5


@dataclass
class TrialResult:
    force_n: float
    direction_deg: float
    direction_label: str
    seed: int
    survived: bool
    termination: str
    episode_length: int
    max_pre_push_tilt: float
    tilt_at_push: float
    peak_post_push_tilt: float
    final_tilt: float
    max_angvel: float
    mean_action_mag: float
    peak_action_mag: float
    failed_before_push: bool


def make_env():
    env = BipedalWalkEnv()
    env.push_force_min = 80.0
    env.set_push_force_max(200.0)
    return env


def load_recovery_policy():
    save_dir = RECOVERY_SAVE_DIR
    model_path = os.path.join(save_dir, "ppo_poc_model")
    vecnorm_path = os.path.join(save_dir, "vecnormalize.pkl")
    policy_path = os.path.join(save_dir, "policy.pth")
    model_zip = model_path + ".zip"

    if not os.path.exists(model_zip) and not os.path.exists(policy_path):
        raise FileNotFoundError(f"Missing recovery policy in {save_dir}")

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


def direction_label(degrees):
    if degrees == 0:
        return "+X"
    if degrees == 180:
        return "-X"
    return f"{degrees:.0f}deg"


def run_trial(policy_fn, force_n, direction_rad, seed):
    env = make_env()
    options = {
        "push_magnitude": float(force_n),
        "push_direction_rad": float(direction_rad),
        "push_step_start": PUSH_STAND_STEPS,
    }
    obs, _ = env.reset(seed=seed, options=options)

    max_pre_push_tilt = 0.0
    tilt_at_push = None
    peak_post_push_tilt = 0.0
    max_angvel = 0.0
    action_mags = []
    step = 0
    termination = "running"
    final_tilt = 0.0

    while True:
        action = policy_fn(obs)
        action_mag = float(np.linalg.norm(action))
        action_mags.append(action_mag)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if not np.isfinite(obs).all() or not np.isfinite(reward):
            termination = "nan"
            break

        tilt = info["quat_tilt_rad"]
        final_tilt = tilt
        max_angvel = max(max_angvel, float(np.linalg.norm(info["chest_angvel"])))

        if step < PUSH_STAND_STEPS:
            max_pre_push_tilt = max(max_pre_push_tilt, tilt)
        elif step == PUSH_STAND_STEPS:
            tilt_at_push = tilt
        else:
            peak_post_push_tilt = max(peak_post_push_tilt, tilt)

        if terminated:
            termination = "fall"
            break
        if truncated:
            termination = "timeout"
            break

    survived = termination == "timeout"
    failed_before_push = step <= PUSH_STAND_STEPS and not survived
    direction_deg = float(np.degrees(direction_rad) % 360.0)

    return TrialResult(
        force_n=force_n,
        direction_deg=direction_deg,
        direction_label=direction_label(direction_deg),
        seed=seed,
        survived=survived,
        termination=termination,
        episode_length=step,
        max_pre_push_tilt=max_pre_push_tilt,
        tilt_at_push=tilt_at_push if tilt_at_push is not None else final_tilt,
        peak_post_push_tilt=peak_post_push_tilt,
        final_tilt=final_tilt,
        max_angvel=max_angvel,
        mean_action_mag=float(np.mean(action_mags)) if action_mags else 0.0,
        peak_action_mag=float(np.max(action_mags)) if action_mags else 0.0,
        failed_before_push=failed_before_push,
    )


def aggregate_rows(results):
    rows = []
    grouped = {}
    for result in results:
        key = (result.force_n, result.direction_label)
        grouped.setdefault(key, []).append(result)

    for (force_n, direction_label), trials in sorted(grouped.items()):
        rows.append(
            {
                "force": force_n,
                "direction": direction_label,
                "survival": np.mean([t.survived for t in trials]),
                "peak_tilt": np.mean([max(t.peak_post_push_tilt, t.final_tilt) for t in trials]),
                "final_tilt": np.mean([t.final_tilt for t in trials]),
                "episode_length": np.mean([t.episode_length for t in trials]),
                "pre_push_fail": np.mean([t.failed_before_push for t in trials]),
                "trials": trials,
            }
        )
    return rows


def print_table(title, rows):
    print(f"\n{title}")
    print(
        f"{'Force':>6} | {'Direction':>8} | {'Survival':>8} | "
        f"{'Peak Tilt':>9} | {'Final Tilt':>10} | {'Ep Len':>8}"
    )
    print("-" * 72)
    for row in rows:
        print(
            f"{row['force']:6.0f} | {row['direction']:>8} | "
            f"{row['survival']:8.1%} | {row['peak_tilt']:9.3f} | "
            f"{row['final_tilt']:10.3f} | {row['episode_length']:8.0f}"
        )


def print_detail(title, results):
    print(f"\n{title} (per-seed detail)")
    print(
        f"{'Force':>6} {'Dir':>8} {'Seed':>4} | {'Surv':>5} | {'Term':>7} | "
        f"{'Len':>5} | {'PreTilt':>7} | {'AtPush':>7} | {'Peak':>7} | "
        f"{'Final':>7} | {'AngVel':>7} | {'|a|':>6} | {'Peak|a|':>7} | "
        f"{'PreFail':>7}"
    )
    print("-" * 110)
    for r in results:
        print(
            f"{r.force_n:6.0f} {r.direction_label:>8} {r.seed:4d} | "
            f"{str(r.survived):>5} | {r.termination:>7} | {r.episode_length:5d} | "
            f"{r.max_pre_push_tilt:7.3f} | {r.tilt_at_push:7.3f} | "
            f"{r.peak_post_push_tilt:7.3f} | {r.final_tilt:7.3f} | "
            f"{r.max_angvel:7.3f} | {r.mean_action_mag:6.2f} | "
            f"{r.peak_action_mag:7.2f} | {str(r.failed_before_push):>7}"
        )


def analyze_sagittal(rows):
    print("\n=== Sagittal stress analysis ===")
    reliable = []
    for row in rows:
        if row["survival"] >= 1.0:
            reliable.append(row["force"])
    if reliable:
        print(f"Strongest reliably survived sagittal force (100%): {max(reliable):.0f} N")
    else:
        print("No sagittal case achieved 100% survival in this sweep.")

    boundary = None
    for row in rows:
        if row["survival"] < 1.0:
            boundary = row["force"]
            break
    if boundary is not None:
        print(f"First sagittal magnitude below 100% survival: {boundary:.0f} N")
    else:
        print("All tested sagittal magnitudes achieved 100% survival.")


def analyze_directions(rows):
    print("\n=== Direction stress analysis (120 N) ===")
    hardest = sorted(rows, key=lambda r: (r["survival"], -r["peak_tilt"]))[:3]
    print("Hardest directions:")
    for row in hardest:
        print(
            f"  {row['direction']:>8}: survival {row['survival']:.1%}, "
            f"peak tilt {row['peak_tilt']:.3f}, final tilt {row['final_tilt']:.3f}"
        )

    lateral = [r for r in rows if r["direction"] not in ("+X", "-X")]
    sagittal = [r for r in rows if r["direction"] in ("+X", "-X")]
    if lateral and sagittal:
        lat_surv = np.mean([r["survival"] for r in lateral])
        sag_surv = np.mean([r["survival"] for r in sagittal])
        print(
            f"Sagittal mean survival (+X/-X): {sag_surv:.1%}; "
            f"non-sagittal mean survival: {lat_surv:.1%}"
        )


def stepping_assessment(sagittal_rows, direction_rows):
    print("\n=== Stepping necessity assessment ===")
    high_force_fail = any(
        row["force"] >= 160 and row["survival"] < 1.0 for row in sagittal_rows
    )
    lateral_fail = any(
        row["survival"] < 1.0
        for row in direction_rows
        if row["direction"] not in ("+X", "-X")
    )
    if high_force_fail:
        print(
            "Failures appear at higher sagittal magnitudes — likely balance-limit "
            "exceeded before stepping would help."
        )
    elif lateral_fail:
        print(
            "Failures are direction-dependent — lateral pushes may need explicit "
            "lateral recovery training before stepping."
        )
    else:
        print(
            "Policy survived all tested magnitudes/directions — stepping not yet "
            "required within this test envelope."
        )


def main():
    print("=== Recovery policy stress test ===\n")
    policy = load_recovery_policy()

    sagittal_results = []
    for magnitude in SAGITTAL_MAGNITUDES:
        for direction_rad in (0.0, np.pi):
            for seed in range(SAGITTAL_SEEDS):
                sagittal_results.append(
                    run_trial(policy, magnitude, direction_rad, seed)
                )

    direction_results = []
    for degrees in DIRECTION_DEGREES:
        direction_rad = np.deg2rad(degrees)
        for seed in range(DIRECTION_SEEDS):
            direction_results.append(
                run_trial(policy, DIRECTION_MAGNITUDE, direction_rad, seed)
            )

    sagittal_rows = aggregate_rows(sagittal_results)
    direction_rows = aggregate_rows(direction_results)

    print_detail("Sagittal magnitude sweep", sagittal_results)
    print_table("Sagittal summary", sagittal_rows)

    print_detail("Direction sweep @ 120 N", direction_results)
    print_table("Direction summary @ 120 N", direction_rows)

    analyze_sagittal(sagittal_rows)
    analyze_directions(direction_rows)
    stepping_assessment(sagittal_rows, direction_rows)


if __name__ == "__main__":
    main()
