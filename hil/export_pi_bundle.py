"""Export the sim2real_v1 policy + obs-normalisation to a plain .npz so the
Raspberry Pi needs only numpy + mujoco (no torch, no stable-baselines3).

Run ONCE on the PC (needs torch + SB3, which the training env already has):

    python -m hil.export_pi_bundle

Writes runs/sim2real_v1/hil_policy.npz.  Read-only w.r.t. the checkpoints.
`hil.policy.Policy` prefers this file when present.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hil import spec  # noqa: E402


def main():
    import torch
    sd = torch.load(spec.POLICY_FILE, map_location="cpu", weights_only=True)
    g = lambda k: sd[k].cpu().numpy().astype(np.float64)
    with open(spec.VECNORM_FILE, "rb") as f:
        vn = pickle.load(f)
    out = dict(
        w0=g("mlp_extractor.policy_net.0.weight"), b0=g("mlp_extractor.policy_net.0.bias"),
        w1=g("mlp_extractor.policy_net.2.weight"), b1=g("mlp_extractor.policy_net.2.bias"),
        wa=g("action_net.weight"), ba=g("action_net.bias"),
        mean=np.asarray(vn.obs_rms.mean, np.float64),
        var=np.asarray(vn.obs_rms.var, np.float64),
    )
    np.savez(spec.HIL_POLICY_FILE, **out)
    print(f"wrote {spec.HIL_POLICY_FILE}")
    for k, v in out.items():
        print(f"  {k:5s} {v.shape}")


if __name__ == "__main__":
    main()
