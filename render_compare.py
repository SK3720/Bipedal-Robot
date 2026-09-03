"""Side-by-side render: whip w11-best vs cp s3-best at a chosen push magnitude."""
import argparse, sys
import numpy as np, torch, imageio.v2 as imageio
from biped_walk_env import BipedWalkEnv, _load_policy

ap = argparse.ArgumentParser()
ap.add_argument("--push", type=float, default=150)
ap.add_argument("--out", default="compare.mp4")
ap.add_argument("--seed", type=int, default=1200)
a = ap.parse_args()

def rollout(run, variant, band, push, seed):
    pol = _load_policy(run, band)
    env = BipedWalkEnv(push_band=(push, push), variant=variant, max_steps=6,
                       render_mode="rgb_array")
    o, _ = env.reset(seed=seed, options={"push_n": float(push), "push_dir_rad": -np.pi/2})
    frames, done, info = [], False, {}
    while not done:
        with torch.no_grad():
            act, _ = pol[0].predict(pol[1](o), deterministic=True)
        o, r, t1, t2, info = env.step(act)
        done = t1 or t2
        frames.append(env.render())
    env.close()
    tag = "RECOVERED" if info.get("success") else ("FELL" if info.get("fell") else "?")
    return frames, f"{tag} ({info.get('n_steps',0)} steps)"

fw, tw = rollout("runs/walk_w11", "whip", (126, 144), a.push, a.seed)
fs, ts = rollout("runs/walk_s3", "cp", (125, 158), a.push, a.seed)
print(f"whip: {tw}   cp: {ts}")
H = max(len(fw), len(fs))
fw += [fw[-1]] * (H - len(fw))
fs += [fs[-1]] * (H - len(fs))
grid = [np.hstack([fw[t], fs[t]]) for t in range(H)]
imageio.mimsave(a.out, grid, fps=40, macro_block_size=1)
print(f"wrote {a.out}   (left = whip w11-best, right = cp s3-best, push {a.push:.0f} N)")
