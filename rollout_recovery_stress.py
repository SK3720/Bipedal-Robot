"""Visual rollout for recovery-policy stress cases with slow-motion around push."""

import argparse
import ctypes
import io
import os
import sys
import time
import zipfile
from ctypes import wintypes

import mujoco.viewer
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_env import BipedalWalkEnv, PUSH_STAND_STEPS

SAVE_DIR = "ppo_poc_recovery"

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0

# Edit this default or pass --case on the command line, e.g. "160N +X".
DEFAULT_CASE = "160N +X"

CASES = {
    "100N +X": {"push_magnitude": 100.0, "push_direction_rad": 0.0},
    "120N +X": {"push_magnitude": 120.0, "push_direction_rad": 0.0},
    "140N +X": {"push_magnitude": 140.0, "push_direction_rad": 0.0},
    "150N +X": {"push_magnitude": 150.0, "push_direction_rad": 0.0},
    "160N +X": {"push_magnitude": 160.0, "push_direction_rad": 0.0},
    "180N +X": {"push_magnitude": 180.0, "push_direction_rad": 0.0},
    "100N -X": {"push_magnitude": 100.0, "push_direction_rad": np.pi},
    "120N -X": {"push_magnitude": 120.0, "push_direction_rad": np.pi},
    "140N -X": {"push_magnitude": 140.0, "push_direction_rad": np.pi},
    "150N -X": {"push_magnitude": 150.0, "push_direction_rad": np.pi},
    "160N -X": {"push_magnitude": 160.0, "push_direction_rad": np.pi},
    "180N -X": {"push_magnitude": 180.0, "push_direction_rad": np.pi},
    "120N 0deg": {"push_magnitude": 120.0, "push_direction_rad": 0.0},
    "120N 30deg": {"push_magnitude": 120.0, "push_direction_rad": np.deg2rad(30.0)},
    "120N 45deg": {"push_magnitude": 120.0, "push_direction_rad": np.deg2rad(45.0)},
    "120N 60deg": {"push_magnitude": 120.0, "push_direction_rad": np.deg2rad(60.0)},
    "120N 90deg": {"push_magnitude": 120.0, "push_direction_rad": np.deg2rad(90.0)},
    "120N 135deg": {"push_magnitude": 120.0, "push_direction_rad": np.deg2rad(135.0)},
    "120N 180deg": {"push_magnitude": 120.0, "push_direction_rad": np.pi},
    "120N 225deg": {"push_magnitude": 120.0, "push_direction_rad": np.deg2rad(225.0)},
    "120N 270deg": {"push_magnitude": 120.0, "push_direction_rad": np.deg2rad(270.0)},
    "120N 315deg": {"push_magnitude": 120.0, "push_direction_rad": np.deg2rad(315.0)},
}

SLOW_START = PUSH_STAND_STEPS - 50
SLOW_END = PUSH_STAND_STEPS + 150
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.015


def make_env():
    env = BipedalWalkEnv()
    env.push_force_min = 80.0
    env.set_push_force_max(200.0)
    return env


def load_trained_policy():
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(os.path.join(SAVE_DIR, "vecnormalize.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    policy_env = make_env()
    model = PPO("MlpPolicy", policy_env, device="cpu")
    policy_path = os.path.join(SAVE_DIR, "policy.pth")
    if os.path.exists(policy_path):
        policy_state = torch.load(policy_path, map_location="cpu", weights_only=True)
    else:
        with zipfile.ZipFile(os.path.join(SAVE_DIR, "ppo_poc_model.zip")) as archive:
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


def _find_mujoco_viewer_hwnd(
    title_prefix=VIEWER_TITLE_PREFIX, timeout_s=VIEWER_POSITION_TIMEOUT_S
):
    if sys.platform != "win32":
        return None

    user32 = ctypes.windll.user32
    end_time = time.time() + timeout_s
    while time.time() < end_time:
        matches = []

        def enum_callback(hwnd, _lparam):
            if user32.IsWindowVisible(hwnd):
                title_length = user32.GetWindowTextLengthW(hwnd)
                if title_length > 0:
                    buffer = ctypes.create_unicode_buffer(title_length + 1)
                    user32.GetWindowTextW(hwnd, buffer, title_length + 1)
                    if buffer.value.startswith(title_prefix):
                        matches.append(hwnd)
            return True

        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(
            enum_callback
        )
        user32.EnumWindows(enum_proc, 0)
        if matches:
            return matches[0]
        time.sleep(0.05)
    return None


def configure_viewer_window(
    width=VIEWER_WIDTH,
    height=VIEWER_HEIGHT,
    title_prefix=VIEWER_TITLE_PREFIX,
    timeout_s=VIEWER_POSITION_TIMEOUT_S,
):
    if sys.platform != "win32":
        return

    hwnd = _find_mujoco_viewer_hwnd(title_prefix=title_prefix, timeout_s=timeout_s)
    if hwnd is None:
        print(
            f"Warning: could not find viewer window titled '{title_prefix}*' "
            f"within {timeout_s:.1f}s."
        )
        return

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    user32 = ctypes.windll.user32
    monitor_info = MONITORINFO()
    monitor_info.cbSize = ctypes.sizeof(MONITORINFO)
    monitor = user32.MonitorFromWindow(hwnd, 1)
    user32.GetMonitorInfoW(monitor, ctypes.byref(monitor_info))
    work_area = monitor_info.rcWork
    x = work_area.left + (work_area.right - work_area.left - width) // 2
    y = work_area.top + (work_area.bottom - work_area.top - height) // 2
    SWP_NOZORDER = 0x0004
    user32.SetWindowPos(hwnd, 0, x, y, width, height, SWP_NOZORDER)


def rollout(case_name, case, seed=0):
    env = make_env()
    options = {
        "push_magnitude": case["push_magnitude"],
        "push_direction_rad": case["push_direction_rad"],
        "push_step_start": PUSH_STAND_STEPS,
    }
    policy = load_trained_policy()
    obs, reset_info = env.reset(seed=seed, options=options)

    print(
        f"\nRecovery rollout: {case_name} | push {reset_info['push_magnitude']:.1f} N "
        f"@ {np.degrees(reset_info['push_direction_rad']):.1f} deg | seed {seed}"
    )
    print(
        f"Slow motion active for steps {SLOW_START}-{SLOW_END} "
        f"({SLOW_SLEEP_S:.3f}s/step vs {NORMAL_SLEEP_S:.3f}s/step)"
    )

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        configure_viewer_window()
        viewer.cam.lookat[:] = [0, 0, 1.1]
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 115
        viewer.cam.elevation = -20

        step = 0
        while viewer.is_running():
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if step in {
                PUSH_STAND_STEPS - 10,
                PUSH_STAND_STEPS,
                PUSH_STAND_STEPS + 5,
                PUSH_STAND_STEPS + 50,
                PUSH_STAND_STEPS + 100,
            }:
                print(
                    f"  step {step:4d} | tilt {info['quat_tilt_rad']:.3f} | "
                    f"|action| {np.linalg.norm(action):.3f} | "
                    f"angvel {np.linalg.norm(info['chest_angvel']):.3f} | "
                    f"push={info['disturbance_active']}"
                )

            viewer.sync()
            sleep_s = SLOW_SLEEP_S if SLOW_START <= step <= SLOW_END else NORMAL_SLEEP_S
            time.sleep(sleep_s)

            if terminated or truncated:
                print(
                    f"  done at step {step} | survived={not terminated} | "
                    f"final_tilt={info['quat_tilt_rad']:.3f}"
                )
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Recovery policy stress rollout")
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE,
        help=f"Case name (default: {DEFAULT_CASE}). Options: {', '.join(CASES)}",
    )
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    parser.add_argument(
        "--list-cases", action="store_true", help="List available case names and exit"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_cases:
        print("Available cases:")
        for name in CASES:
            print(f"  {name}")
        return

    if args.case not in CASES:
        raise SystemExit(
            f"Unknown case '{args.case}'. Use --list-cases to see valid names."
        )

    rollout(args.case, CASES[args.case], seed=args.seed)


if __name__ == "__main__":
    main()
