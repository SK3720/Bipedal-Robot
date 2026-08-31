"""Diagnostic tests for chest push disturbances (Phase 3)."""

import numpy as np

from biped_env import BipedalWalkEnv, PUSH_STAND_STEPS


def run_episode(
    push_magnitude,
    push_direction_rad=0.0,
    push_step_start=PUSH_STAND_STEPS,
    post_push_steps=4000,
):
  """Run one episode with a fixed push and zero policy action."""
  env = BipedalWalkEnv()
  zero_action = np.zeros(15, dtype=np.float32)

  obs, reset_info = env.reset(
      seed=0,
      options={
          "push_magnitude": push_magnitude,
          "push_direction_rad": push_direction_rad,
          "push_step_start": push_step_start,
      },
  )
  assert np.isfinite(obs).all()

  records = []
  terminated = False
  truncated = False
  nan_detected = False

  total_steps = push_step_start + 20 + post_push_steps
  for step in range(total_steps):
    obs, reward, terminated, truncated, info = env.step(zero_action)
    if not np.isfinite(obs).all() or not np.isfinite(reward):
      nan_detected = True
      break

    if (
        step == push_step_start - 1
        or step == push_step_start
        or step == push_step_start + 4
        or step == push_step_start + 50
        or step == push_step_start + 200
        or step == push_step_start + 1000
        or terminated
        or truncated
        or step == total_steps - 1
    ):
      records.append(
          {
              "step": step,
              "tilt_rad": info["quat_tilt_rad"],
              "angvel_norm": float(np.linalg.norm(info["chest_angvel"])),
              "xy_disp": info["chest_xy_displacement"],
              "up_alignment": info["up_alignment"],
              "reward": reward,
              "terminated": terminated,
              "truncated": truncated,
              "disturbance_active": info["disturbance_active"],
          }
      )

    if terminated or truncated:
      break

  peak_tilt = max(r["tilt_rad"] for r in records)
  peak_angvel = max(r["angvel_norm"] for r in records)
  peak_xy = max(r["xy_disp"] for r in records)
  min_alignment = min(r["up_alignment"] for r in records)
  survived = not terminated

  return {
      "push_magnitude": push_magnitude,
      "push_direction_rad": push_direction_rad,
      "survived": survived,
      "terminated": terminated,
      "nan_detected": nan_detected,
      "peak_tilt_rad": peak_tilt,
      "peak_angvel": peak_angvel,
      "peak_xy_disp": peak_xy,
      "min_up_alignment": min_alignment,
      "records": records,
      "reset_info": reset_info,
  }


def classify_result(result):
  if result["nan_detected"]:
    return "nan"
  if not result["survived"]:
    return "unrecoverable"
  if result["peak_tilt_rad"] < 0.08:
    return "negligible"
  if result["peak_tilt_rad"] < 0.35:
    return "recoverable"
  if result["peak_tilt_rad"] < 0.75:
    return "difficult"
  return "recoverable"


def print_result(result):
  direction_deg = np.degrees(result["push_direction_rad"])
  label = classify_result(result)
  print(
      f"  {result['push_magnitude']:5.1f} N @ {direction_deg:6.1f} deg -> "
      f"{label:14s} | peak tilt {result['peak_tilt_rad']:.3f} rad, "
      f"peak angvel {result['peak_angvel']:.3f}, "
      f"peak xy {result['peak_xy_disp']:.4f} m, "
      f"min align {result['min_up_alignment']:.3f}, "
      f"survived={result['survived']}"
  )


def main():
  print("=== Disturbance diagnostic (zero-action recovery) ===\n")

  print("1) Known push (+X, 100 N) timeline")
  result = run_episode(push_magnitude=100.0, push_direction_rad=0.0)
  assert not result["nan_detected"]
  for row in result["records"]:
    print(
        f"  step {row['step']:4d} | tilt {row['tilt_rad']:.4f} | "
        f"angvel {row['angvel_norm']:.4f} | xy {row['xy_disp']:.5f} | "
        f"align {row['up_alignment']:.4f} | reward {row['reward']:.4f} | "
        f"push={row['disturbance_active']} | term={row['terminated']}"
    )

  print("\n2) Direction sweep at 120 N")
  for direction_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
    result = run_episode(
        push_magnitude=120.0,
        push_direction_rad=np.radians(direction_deg),
    )
    print_result(result)

  print("\n3) Magnitude sweep (+X direction)")
  magnitudes = [0, 40, 60, 80, 100, 120, 140, 150, 160, 180, 200, 250]
  buckets = {
      "negligible": [],
      "recoverable": [],
      "difficult": [],
      "unrecoverable": [],
      "nan": [],
  }
  for mag in magnitudes:
    result = run_episode(push_magnitude=mag, push_direction_rad=0.0)
    label = classify_result(result)
    buckets[label].append(mag)
    print_result(result)

  print("\n=== Measured magnitude ranges (+X, zero action) ===")
  print(f"  negligible:     {buckets['negligible']}")
  print(f"  recoverable:    {buckets['recoverable']}")
  print(f"  difficult:      {buckets['difficult']}")
  print(f"  unrecoverable:  {buckets['unrecoverable']}")
  print(f"  nan:            {buckets['nan']}")

  print("\n4) Randomized push parameters (10 episodes)")
  env = BipedalWalkEnv()
  zero_action = np.zeros(15, dtype=np.float32)
  for episode in range(10):
    obs, reset_info = env.reset(seed=episode)
    terminated = False
    nan_detected = False
    peak_tilt = 0.0
    while not terminated:
      obs, reward, terminated, truncated, info = env.step(zero_action)
      if not np.isfinite(obs).all():
        nan_detected = True
        break
      peak_tilt = max(peak_tilt, info["quat_tilt_rad"])
      if terminated or truncated:
        print(
            f"  ep {episode}: mag={reset_info['push_magnitude']:.2f} N, "
            f"dir={np.degrees(reset_info['push_direction_rad']):.1f} deg, "
            f"start={reset_info['push_step_start']}, "
            f"peak_tilt={peak_tilt:.3f}, survived={not terminated}, nan={nan_detected}"
        )
        break


if __name__ == "__main__":
  main()
