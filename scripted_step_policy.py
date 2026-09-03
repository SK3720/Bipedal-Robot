"""A crude scripted 'recovery step' expressed as a RESIDUAL for RecoveryEnv.

Not meant to work well on its own - it exists to give an RL policy a rough
motor skill to imitate (behaviour cloning warm-start), which RL then fine-tunes.
The residual is timed off `t_since_push` (obs element -1, un-normalised) and the
swing leg is the one that is lighter when the push lands.
"""

from __future__ import annotations

import numpy as np

from recovery_env import RESIDUAL_SCALE

# ctrl indices
L = dict(hip_roll=5, hip=6, knee=7, ankle=8, ankle_roll=9)
R = dict(hip_roll=10, hip=11, knee=12, ankle=13, ankle_roll=14)
AR_L, AR_R = 9, 14
FWD_HIP_SIGN = {"L": -1.0, "R": +1.0}
# ankle-roll bias that unloads that foot (verified: -0.12 unloads R)
AR_UNLOAD_SIGN = {"L": +1.0, "R": -1.0}


def _seg(t, a, b):
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    x = (t - a) / (b - a)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def scripted_residual_rad(t_since_push: float, swing: str) -> np.ndarray:
    """Residual in RADIANS (before dividing by RESIDUAL_SCALE)."""
    r = np.zeros(15, dtype=np.float32)
    if t_since_push < 0.0:
        return r
    t = t_since_push
    sw = R if swing == "R" else L
    st = L if swing == "R" else R
    hf = FWD_HIP_SIGN[swing]
    ks = -1.0

    # 0 - 0.30 s : ankle-roll weight shift onto the stance foot + swing hip-roll
    p_shift = _seg(t, 0.0, 0.30)
    ar = AR_UNLOAD_SIGN[swing] * 0.15 * p_shift
    r[AR_L] += ar
    r[AR_R] += ar
    r[sw["hip_roll"]] += (-0.13 if swing == "R" else 0.13) * p_shift

    # 0.18 - 0.55 s : swing the hip forward, knee flexes on a bump
    p_sw = _seg(t, 0.18, 0.55)
    p_bump = np.sin(np.pi * np.clip((t - 0.18) / 0.55, 0.0, 1.0))
    r[sw["hip"]] += hf * 0.55 * p_sw
    r[sw["knee"]] += ks * (0.06 + 0.30 * p_bump)
    r[sw["ankle"]] += -hf * 0.18 * p_bump

    # 0.5 - 1.0 s : plant - knee straightens, hip eases, stance knee bends a little
    p_pl = _seg(t, 0.5, 1.0)
    r[sw["knee"]] += ks * (0.10 * p_pl)
    r[sw["hip"]] += -hf * 0.12 * p_pl
    r[st["knee"]] += ks * 0.18 * p_pl

    # after ~1.1 s : relax back toward the LQR
    p_rel = _seg(t, 1.1, 1.8)
    r *= (1.0 - 0.85 * p_rel)
    return r


class ScriptedStepPolicy:
    def __init__(self, swing: str | None = None):
        self.swing = swing
        self._chosen = None

    def reset(self):
        self._chosen = self.swing

    def __call__(self, env, obs):
        t = float(env._t_since_push)
        if self._chosen is None:
            # pick the lighter foot at push landing
            _, lnf = env._foot_state("L")
            _, rnf = env._foot_state("R")
            self._chosen = "L" if lnf <= rnf else "R"
        res = scripted_residual_rad(t, self._chosen) / RESIDUAL_SCALE
        return np.clip(res, -1.0, 1.0).astype(np.float32)
