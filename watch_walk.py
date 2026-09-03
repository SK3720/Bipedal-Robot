"""Watch the trained reactive-recovery policy in the MuJoCo interactive viewer.

    python watch_walk.py                         # w5-best, random pushes 124-146 N
    python watch_walk.py --push 140              # every episode a 140 N straight push
    python watch_walk.py --push 138 --dir 10     # 138 N, 10 deg off-axis
    python watch_walk.py --slow 2 --n 20         # 2x slow motion, 20 episodes
    python watch_walk.py --policy runs/walk_w5 --band 118,150

Close the viewer window (or Ctrl-C) to stop.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from biped_walk_env import BipedWalkEnv, _load_policy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="runs/walk_s5", help="run dir (prefers policy_best.pth)")
    ap.add_argument("--band", default="150,200", help="random push magnitude range, N")
    ap.add_argument("--push", type=float, default=None, help="fixed push magnitude, N (overrides --band)")
    ap.add_argument("--dir", type=float, default=0.0, help="push direction, deg off straight-forward (+ = toward robot's left)")
    ap.add_argument("--slow", type=float, default=1.0, help="slow-motion factor (2 = half speed)")
    ap.add_argument("--n", type=int, default=50, help="episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause", type=float, default=1.2, help="seconds to hold on the final pose")
    ap.add_argument("--model", default="robot/_exp_hands_2x_feet_2x.xml", help="robot xml")
    ap.add_argument("--variant", default="cp", help="'cp' or 'whip'")
    a = ap.parse_args(argv)

    band = tuple(float(x) for x in a.band.split(","))
    model, norm = _load_policy(a.policy, band)

    env = BipedWalkEnv(model_path=a.model, push_band=band, dir_spread_deg=0.0,
                       render_mode="human", variant=a.variant, max_steps=6)
    env._view_slow = a.slow

    print(f"\nwatching {a.policy}  |  "
          f"{'push %.0f N' % a.push if a.push else 'push %g-%g N' % band}  "
          f"dir {a.dir:+g} deg  |  slow x{a.slow}\n")

    try:
        for i in range(a.n):
            opts = None
            if a.push is not None:
                opts = {"push_n": a.push, "push_dir_rad": -np.pi / 2 + np.radians(a.dir)}
            obs, _ = env.reset(seed=a.seed + i, options=opts)
            done = False
            info = {}
            while not done:
                import torch
                with torch.no_grad():
                    act, _ = model.predict(norm(obs), deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc
            tag = ("RECOVERED" if info.get("success") else
                   ("FELL" if info.get("fell") else info.get("done_reason", "?")))
            print(f"  ep {i:2d}  push {info.get('push_n', 0):6.1f} N  "
                  f"{info.get('n_steps', 0)} step(s)  ->  {tag}"
                  f"{'  (flat double-support)' if info.get('flat_ok') else ''}  "
                  f"peak tilt {info.get('peak_up', 0):.0f} deg,  end speed {info.get('end_spd', 0):.2f} m/s")
            time.sleep(a.pause)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
