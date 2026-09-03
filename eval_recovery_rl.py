"""Evaluate the recovery controller (trained policy, or the torso-LQR base).

  python eval_recovery_rl.py --zero                       # torso-LQR ceiling sweep
  python eval_recovery_rl.py --zero --push 140 --slow     # WATCH the ~145 N in-place recovery
  python eval_recovery_rl.py --run runs/recovery_v6       # a trained policy sweep
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from recovery_env import RecoveryEnv
from recovery_metrics import _foot_xy_z, sample_balance


def load_policy(run_dir, ckpt=None):
    """PPO.load chokes on its own nested zip on this torch build - rebuild the
    policy and load the separately-saved state_dict instead."""
    import io
    import zipfile
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO("MlpPolicy", DummyVecEnv([lambda: RecoveryEnv()]), device="cpu",
                policy_kwargs=dict(net_arch=[256, 256]))
    src = ckpt or os.path.join(run_dir, "policy.pth")
    if src.endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    else:
        sd = torch.load(src, map_location="cpu", weights_only=True)
    model.policy.load_state_dict(sd)
    model.policy.eval()

    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    vn = None
    if os.path.exists(vn_path):
        vn = VecNormalize.load(vn_path, DummyVecEnv([lambda: RecoveryEnv()]))
        vn.training = False
        vn.norm_reward = False

    def act(obs, deterministic=True):
        o = np.asarray(obs, dtype=np.float32)
        if vn is not None:
            o = vn.normalize_obs(o)
        a, _ = model.predict(o, deterministic=deterministic)
        return np.asarray(a, dtype=np.float32)
    return act


def rollout(env, act, push_n, seed=0, viewer=None, slow=False):
    import time
    obs, _ = env.reset(seed=seed, options={"push_magnitude": push_n})
    com0 = env._com()[0].copy()
    lf0 = _foot_xy_z(env.model, env.data, "L")[1]
    rf0 = _foot_xy_z(env.model, env.data, "R")[1]
    peak_up = 0.0
    l_air = r_air = 0          # consecutive frames a foot is off the ground post-push
    l_air_max = r_air_max = 0
    min_z = 9.0
    k = 0
    for k in range(600):
        a = act(obs)
        obs, r, term, trunc, info = env.step(a)
        peak_up = max(peak_up, info["upaxis_tilt"])
        min_z = min(min_z, info["chest_z"])
        lc, lnf = env._foot_state("L"); rc, rnf = env._foot_state("R")
        post = info["t_since_push"] > 0
        l_air = l_air + 1 if (post and lnf < 2.0) else 0
        r_air = r_air + 1 if (post and rnf < 2.0) else 0
        l_air_max = max(l_air_max, l_air); r_air_max = max(r_air_max, r_air)
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(0.02 if slow else 0.004)
        if term or trunc:
            break
    bs = sample_balance(env.model, env.data)
    com = env._com()[0]
    lf = _foot_xy_z(env.model, env.data, "L")[1]
    rf = _foot_xy_z(env.model, env.data, "R")[1]
    lc, _ = env._foot_state("L"); rc, _ = env._foot_state("R")
    settled = (not term and abs(bs.fwd_lean_deg) < 12 and abs(bs.side_lean_deg) < 12
               and lc and rc and bs.com_speed_horiz < 0.15
               and env.data.qpos[2] > 1.26 - 0.06)
    # a genuine step = one foot off the ground >= 8 frames (80 ms) while the
    # other stays down, and that foot ends up displaced
    STEP_FRAMES = 8
    stepped_L = l_air_max >= STEP_FRAMES and r_air_max < STEP_FRAMES
    stepped_R = r_air_max >= STEP_FRAMES and l_air_max < STEP_FRAMES
    both = l_air_max >= STEP_FRAMES and r_air_max >= STEP_FRAMES
    return dict(
        push_n=push_n, steps=k + 1, fell=term, peak_up_deg=np.degrees(peak_up),
        min_z=min_z, com_fwd_mm=-(com[1] - com0[1]) * 1000.0,
        lfoot_fwd_mm=-(lf - lf0) * 1000.0, rfoot_fwd_mm=-(rf - rf0) * 1000.0,
        l_air_ms=l_air_max * 10, r_air_ms=r_air_max * 10,
        stepped=(stepped_L or stepped_R or both),
        which=("L" if stepped_L else "R" if stepped_R else "both" if both else "-"),
        end_fwd_lean=bs.fwd_lean_deg, end_side_lean=bs.side_lean_deg,
        end_com_speed=bs.com_speed_horiz, end_ds=(lc and rc), settled=settled,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/recovery_v6")
    p.add_argument("--ckpt", default=None, help="path to a checkpoint .zip (else policy.pth)")
    p.add_argument("--push", type=float, default=None)
    p.add_argument("--sweep", default="110,130,145,155,165,180,200")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--zero", action="store_true",
                   help="use zero action (= the torso-LQR base, recovers <= ~145 N in place)")
    a = p.parse_args()
    if a.zero:
        act = lambda obs, deterministic=True: np.zeros(15, dtype=np.float32)
    else:
        act = load_policy(a.run, ckpt=a.ckpt)

    if a.push is not None and not a.headless:
        import mujoco.viewer
        env = RecoveryEnv()
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            v.cam.lookat[:] = [0.0, -0.1, 1.05]; v.cam.distance = 2.2
            v.cam.azimuth = 90; v.cam.elevation = -8
            r = rollout(env, act, a.push, seed=0, viewer=v, slow=a.slow)
            print(r)
            print("\n  close viewer to exit")
            import time
            while v.is_running():
                v.sync(); time.sleep(0.02)
        return

    env = RecoveryEnv()
    mags = [a.push] if a.push else [float(x) for x in a.sweep.split(",")]
    print(f"{'push':>6} {'survive':>8} {'settled':>8} {'stepped':>8} {'air(ms)':>8} {'peakUp':>7} "
          f"{'CoM_fwd':>8} {'foot_fwd':>9} {'endLean(f/s)':>13}")
    for mag in mags:
        rs = [rollout(env, act, mag, seed=s) for s in range(a.seeds)]
        surv = np.mean([not r["fell"] for r in rs])
        setl = np.mean([r["settled"] for r in rs])
        stp = np.mean([r["stepped"] for r in rs])
        air = np.mean([max(r["l_air_ms"], r["r_air_ms"]) for r in rs])
        pu = np.mean([r["peak_up_deg"] for r in rs])
        cf = np.mean([r["com_fwd_mm"] for r in rs])
        ff = np.mean([max(r["lfoot_fwd_mm"], r["rfoot_fwd_mm"]) for r in rs])
        fl = np.mean([r["end_fwd_lean"] for r in rs])
        sl = np.mean([r["end_side_lean"] for r in rs])
        print(f"{mag:6.0f} {surv:8.0%} {setl:8.0%} {stp:8.0%} {air:8.0f} {pu:7.0f} "
              f"{cf:8.0f} {ff:9.0f} {fl:+6.0f}/{sl:+5.0f}")


if __name__ == "__main__":
    main()
