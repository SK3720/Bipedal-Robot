"""Regenerate runs/sim2real_v1/base_controller_consts.npz from the live env.

Run this if biped_sim2real_env.py / the model / the StandingLQR ever changes.
Does NOT touch the policy or any checkpoint.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biped_sim2real_env import BipedSim2RealEnv          # noqa: E402
from hil import spec                                     # noqa: E402


def main():
    env = BipedSim2RealEnv(privileged_obs=False, robust=0.0)
    m, s = env.model, env._stand
    nv = m.nv
    cols = [3, 4, 5, nv + 3, nv + 4, nv + 5]
    out = dict(
        K_att=s.K[:, cols].copy(),           # 15x6  [dq_rx,dq_ry,dq_rz, dw_x,dw_y,dw_z]
        K_full=s.K.copy(),                   # 15x42 (reference)
        qpos0=s.qpos0.copy(),
        qvel0=s.qvel0.copy(),
        ctrl0=s.ctrl0.copy(),
        ctrlrange=m.actuator_ctrlrange[:15].copy(),
        act_scale=spec.ACT_SCALE.copy(),
    )
    np.savez(spec.CONSTS_FILE, **out)
    print(f"wrote {spec.CONSTS_FILE}")
    for k, v in out.items():
        print(f"  {k:12s} {np.asarray(v).shape}")


if __name__ == "__main__":
    main()
