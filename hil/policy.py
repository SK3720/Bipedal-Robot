"""Load and run the sim2real_v1 policy -- no gym env, no MuJoCo needed.

The policy is a plain 2-layer MLP (SB3 PPO MlpPolicy, net_arch [256,256]).  We
reimplement the forward pass in pure numpy so inference has no torch/SB3
dependency on the target and latency is trivially measurable.  `validate_stack.py`
checks it matches the SB3 policy bit-for-bit.
"""
from __future__ import annotations

import numpy as np

from hil import spec


def _load_mlp_weights():
    """Pull the MLP weights out of the SB3 state_dict once, cache as numpy."""
    import torch
    sd = torch.load(spec.POLICY_FILE, map_location="cpu", weights_only=True)
    g = lambda k: sd[k].cpu().numpy().astype(np.float64)
    return dict(
        w0=g("mlp_extractor.policy_net.0.weight"), b0=g("mlp_extractor.policy_net.0.bias"),
        w1=g("mlp_extractor.policy_net.2.weight"), b1=g("mlp_extractor.policy_net.2.bias"),
        wa=g("action_net.weight"),                 ba=g("action_net.bias"),
    )


class Policy:
    def __init__(self):
        import os
        if os.path.exists(spec.HIL_POLICY_FILE):        # Pi path: numpy only
            z = np.load(spec.HIL_POLICY_FILE)
            self.mean, self.var = z["mean"], z["var"]
            self.W = {k: z[k] for k in ("w0", "b0", "w1", "b1", "wa", "ba")}
            self.source = "npz"
        else:                                           # PC path: needs torch + SB3
            self.mean, self.var = spec.load_norm()      # 217-D each
            self.W = _load_mlp_weights()
            self.source = "pth"
        self.std = np.sqrt(self.var + spec.NORM_EPS)
        self.n_in = self.W["w0"].shape[1]
        assert self.n_in == spec.OBS_DIM, (self.n_in, spec.OBS_DIM)

    def normalize(self, obs):
        return np.clip((np.asarray(obs, np.float64) - self.mean) / self.std,
                       -spec.NORM_CLIP, spec.NORM_CLIP)

    def act(self, obs):
        """obs: raw 217-D.  Returns the deterministic 14-D action in [-1,1]
        (POLICY_JOINT_ORDER)."""
        x = self.normalize(obs)
        W = self.W
        h = np.tanh(W["w0"] @ x + W["b0"])
        h = np.tanh(W["w1"] @ h + W["b1"])
        a = W["wa"] @ h + W["ba"]                # SB3 DiagGaussian mean (no squashing)
        return np.clip(a, -1.0, 1.0).astype(np.float64)
