"""Evaluate a saved locomotion checkpoint and render a clean single-episode clip.

  python eval_loco_ckpt.py --pol runs/loco_w2/policy_at_9p4M_s3p544.pth \
        --vn runs/loco_w2/vecnormalize_at_9p4M.pkl --speed 0.30 --n 40 --render out.mp4
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_locomotion_env import BipedLocomotionEnv, DEFAULT_MODEL, EP_STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol", required=True)
    ap.add_argument("--vn", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--hparams", default="runs/loco_w2/hparams.json")
    ap.add_argument("--speed", type=float, default=0.30)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--render", default=None)
    a = ap.parse_args()

    hp = json.load(open(a.hparams))
    tmp = DummyVecEnv([lambda: BipedLocomotionEnv(model_path=a.model, speed_tgt=a.speed)])
    vn = VecNormalize.load(a.vn, tmp)
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    mm = PPO("MlpPolicy", tmp, device="cpu",
             policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(a.pol, map_location="cpu", weights_only=True))
    mm.policy.eval()
    norm = lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)

    env = BipedLocomotionEnv(model_path=a.model, speed_tgt=a.speed,
                             render_mode=("rgb_array" if a.render else None))
    surv, fell, dists, vs, gens, flts = [], [], [], [], [], []
    best_ep_frames, best_ep_dist = None, -1
    for i in range(a.n):
        o, _ = env.reset(seed=4000 + i)
        done = False
        info = {}
        vv = []
        frames = []
        while not done:
            with torch.no_grad():
                act, _ = mm.predict(norm(o), deterministic=True)
            o, r, term, trunc, info = env.step(act)
            done = term or trunc
            vv.append(info["vfwd"])
            if a.render:
                frames.append(env.render())
        surv.append(info["t"] / EP_STEPS)
        fell.append(int(info["fell"]))
        dists.append(info["dist"])
        vs.append(float(np.mean(vv[15:])) if len(vv) > 25 else float(np.mean(vv)))
        gens.append(info["genuine_steps"])
        flts.append(info["flight_frac"])
        if a.render and not info["fell"] and info["dist"] > best_ep_dist:
            best_ep_dist, best_ep_frames = info["dist"], frames
    print(f"n={a.n} speed_tgt={a.speed}")
    print(f"  survival     {np.mean(surv)*100:.0f}%   fell {np.mean(fell)*100:.0f}%")
    print(f"  mean vfwd    {np.mean(vs):+.3f} m/s  ({np.mean(vs)/a.speed*100:.0f}% of target)")
    print(f"  mean dist    {np.mean(dists):+.2f} m/ep")
    print(f"  genuine steps {np.mean(gens):.1f}/ep   flight {np.mean(flts):.3f}")
    if a.render and best_ep_frames:
        import imageio.v2 as imageio
        imageio.mimsave(a.render, best_ep_frames, fps=40, macro_block_size=1)
        print(f"  wrote {a.render}  ({len(best_ep_frames)} frames, dist {best_ep_dist:.2f} m)")
    env.close()


if __name__ == "__main__":
    main()
