"""Sanity checks for delta-action mapping in BipedalWalkEnv."""

import numpy as np

from biped_env import (
    ACTION_SCALE,
    BipedalWalkEnv,
    DEFAULT_POSE,
    PUSH_STAND_STEPS,
)

JOINT_NAMES = [
    "Chest_neck",
    "Chest_L_shoulder",
    "L_arm_L_elbow",
    "Chest_R_shoulder",
    "R_arm_R_elbow",
    "Chest_L_hip_roll",
    "L_hip_L_hip_pitch",
    "L_leg_L_knee",
    "L_shin_L_ankle_pitch",
    "L_ankle_L_ankle_roll",
    "Chest_R_hip_roll",
    "R_hip_R_hip_pitch",
    "R_leg_R_knee",
    "R_shin_R_ankle_pitch",
    "R_ankle_R_ankle_roll",
]


def test_action_mapping():
    env = BipedalWalkEnv()
    expected_delta = ACTION_SCALE

    zero_ctrl = env._action_to_ctrl(np.zeros(15))
    assert np.allclose(zero_ctrl, DEFAULT_POSE), (
        f"zero action should map to DEFAULT_POSE, got {zero_ctrl}"
    )

    plus_ctrl = env._action_to_ctrl(np.ones(15))
    minus_ctrl = env._action_to_ctrl(-np.ones(15))

    print("=== Action mapping (+/-1) ===")
    for i, name in enumerate(JOINT_NAMES):
        low, high = env.ctrl_low[i], env.ctrl_high[i]
        target_plus = DEFAULT_POSE[i] + expected_delta
        target_minus = DEFAULT_POSE[i] - expected_delta
        clipped_plus = np.clip(target_plus, low, high)
        clipped_minus = np.clip(target_minus, low, high)
        ok_plus = np.isclose(plus_ctrl[i], clipped_plus)
        ok_minus = np.isclose(minus_ctrl[i], clipped_minus)
        print(
            f"  {name:24s} | +1 -> {np.degrees(plus_ctrl[i]):6.2f} deg "
            f"(expect {np.degrees(clipped_plus):6.2f}) | "
            f"-1 -> {np.degrees(minus_ctrl[i]):6.2f} deg "
            f"(expect {np.degrees(clipped_minus):6.2f}) | "
            f"ok={ok_plus and ok_minus}"
        )
        assert ok_plus and ok_minus, f"mapping failed for joint {name}"

    print("  PASS: action mapping matches DEFAULT_POSE + action * ACTION_SCALE")


def test_zero_action_stability(duration_steps=5000):
    env = BipedalWalkEnv()
    obs, _ = env.reset(seed=0, options={"enable_push": False})
    zero_action = np.zeros(15, dtype=np.float32)

    max_tilt = 0.0
    nan_detected = False
    for step in range(duration_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        if not np.isfinite(obs).all():
            nan_detected = True
            break
        max_tilt = max(max_tilt, info["quat_tilt_rad"])
        if terminated:
            print(f"  FAIL: zero action fell at step {step + 1}")
            return False

    print(
        f"=== Zero-action stability ({duration_steps} steps, no push) ===\n"
        f"  max_tilt={max_tilt:.4f} rad | nan={nan_detected} | survived=True"
    )
    return not nan_detected


def test_small_random_actions(episodes=3, duration_steps=3000):
    env = BipedalWalkEnv()
    print("=== Small random actions [-0.2, 0.2] ===")

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep, options={"enable_push": False})
        max_tilt = 0.0
        nan_detected = False
        terminated = False

        for _ in range(duration_steps):
            action = np.random.uniform(-0.2, 0.2, 15).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(obs).all():
                nan_detected = True
                break
            max_tilt = max(max_tilt, info["quat_tilt_rad"])
            if terminated:
                break

        print(
            f"  ep {ep}: max_tilt={max_tilt:.4f} | "
            f"terminated={terminated} | nan={nan_detected}"
        )
        if nan_detected or terminated:
            return False

    print("  PASS: small random actions remained stable")
    return True


def test_disturbance_timing():
    env = BipedalWalkEnv()
    env.push_force_min = 80.0
    env.set_push_force_max(120.0)

    options = {
        "push_magnitude": 100.0,
        "push_direction_rad": 0.0,
        "push_step_start": PUSH_STAND_STEPS,
    }
    obs, info = env.reset(seed=0, options=options)
    zero_action = np.zeros(15, dtype=np.float32)

    push_seen = False
    push_step = None
    nan_detected = False

    for step in range(PUSH_STAND_STEPS + 20):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        if not np.isfinite(obs).all():
            nan_detected = True
            break
        if info["disturbance_active"] and not push_seen:
            push_seen = True
            push_step = step + 1

    ok = (
        push_seen
        and push_step == PUSH_STAND_STEPS + 1
        and np.isclose(info["push_magnitude"], 100.0)
        and not nan_detected
    )
    print(
        "=== Disturbance timing ===\n"
        f"  push_seen={push_seen} at step {push_step} "
        f"(expected {PUSH_STAND_STEPS + 1}) | "
        f"magnitude={info['push_magnitude']:.1f} N | nan={nan_detected} | ok={ok}"
    )
    return ok


def main():
    print("=== Delta-action sanity tests ===\n")
    test_action_mapping()
    print()
    ok_zero = test_zero_action_stability()
    print()
    ok_random = test_small_random_actions()
    print()
    ok_push = test_disturbance_timing()
    print()

    all_ok = ok_zero and ok_random and ok_push
    print(f"=== Overall: {'PASS' if all_ok else 'FAIL'} ===")


if __name__ == "__main__":
    main()
