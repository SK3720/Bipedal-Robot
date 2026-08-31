"""Sanity check for sagittal-biased push direction sampling."""

import numpy as np

from biped_env import (
    BipedalWalkEnv,
    PUSH_DIRECTION_SAGITTAL,
    PUSH_SAGITTAL_SPREAD_RAD,
)

EPISODES = 1000
LATERAL_THRESHOLD_DEG = 60.0


def main():
    env = BipedalWalkEnv()
    env.push_force_min = 80.0
    env.set_push_force_max(120.0)
    env.push_direction_mode = PUSH_DIRECTION_SAGITTAL
    env.push_sagittal_spread_rad = PUSH_SAGITTAL_SPREAD_RAD

    directions_deg = []
    for seed in range(EPISODES):
        _, info = env.reset(seed=seed)
        directions_deg.append(np.degrees(info["push_direction_rad"]) % 360.0)

    directions_deg = np.array(directions_deg)
    lateral = np.logical_or(
        (directions_deg > LATERAL_THRESHOLD_DEG)
        & (directions_deg < 180.0 - LATERAL_THRESHOLD_DEG),
        (directions_deg > 180.0 + LATERAL_THRESHOLD_DEG)
        & (directions_deg < 360.0 - LATERAL_THRESHOLD_DEG),
    )
    sagittal = ~lateral

    plus_x = np.logical_or(
        directions_deg <= PUSH_SAGITTAL_SPREAD_RAD * 180.0 / np.pi,
        directions_deg >= 360.0 - PUSH_SAGITTAL_SPREAD_RAD * 180.0 / np.pi,
    )
    minus_x = np.abs(directions_deg - 180.0) <= (
        PUSH_SAGITTAL_SPREAD_RAD * 180.0 / np.pi
    )

    print("=== Sagittal push direction sanity test ===\n")
    print(f"Episodes sampled: {EPISODES}")
    print(
        f"Spread: +/-{np.degrees(PUSH_SAGITTAL_SPREAD_RAD):.1f} deg around +X / -X"
    )
    print(f"Sagittal fraction (|angle| from +/-X > {LATERAL_THRESHOLD_DEG} deg): "
          f"{sagittal.mean():.1%}")
    print(f"+X cluster fraction: {plus_x.mean():.1%}")
    print(f"-X cluster fraction: {minus_x.mean():.1%}")
    print(f"Lateral fraction: {lateral.mean():.1%}")
    print(f"Direction range: [{directions_deg.min():.1f}, {directions_deg.max():.1f}] deg")
    print(f"Mean direction: {directions_deg.mean():.1f} deg")

    ok = sagittal.mean() >= 0.95
    print(f"\n=== Overall: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    main()
