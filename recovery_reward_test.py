"""Quick check that recovery reward magnitudes are reasonable vs upright reward."""

import numpy as np

from biped_env import BipedalWalkEnv, PUSH_STAND_STEPS, W_RECOVERY

CASE = {
    "push_magnitude": 120.0,
    "push_direction_rad": 0.0,
    "push_step_start": PUSH_STAND_STEPS,
}


def run_episode(use_recovery, steps=3000):
    env = BipedalWalkEnv()
    if use_recovery:
        env.configure_recovery_reward()
    options = CASE
    obs, _ = env.reset(seed=0, options=options)
    totals = {
        "upright": 0.0,
        "recovery": 0.0,
        "total": 0.0,
        "recovery_steps": 0,
    }
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(np.zeros(15, dtype=np.float32))
        totals["upright"] += info["reward_upright"]
        totals["recovery"] += info.get("reward_recovery", 0.0)
        totals["total"] += reward
        if info.get("reward_recovery", 0.0) != 0.0:
            totals["recovery_steps"] += 1
        if terminated or truncated:
            break
    return totals


def main():
    print("=== Recovery reward magnitude sanity ===\n")
    without = run_episode(use_recovery=False)
    with_rec = run_episode(use_recovery=True)

    for label, totals in [("without recovery", without), ("with recovery", with_rec)]:
        print(label)
        print(f"  upright sum:   {totals['upright']:.2f}")
        print(f"  recovery sum:  {totals['recovery']:.4f}")
        print(f"  total reward:  {totals['total']:.2f}")
        print(f"  recovery active steps: {totals['recovery_steps']}")
        if totals["total"] != 0:
            frac = abs(totals["recovery"]) / abs(totals["total"])
            print(f"  |recovery|/|total|: {frac:.3%}")
        print()

    print(f"W_RECOVERY = {W_RECOVERY}")


if __name__ == "__main__":
    main()
