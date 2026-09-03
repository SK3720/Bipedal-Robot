"""Read-only: render the trained Stage-1 recovery policy for visual inspection.

Loads runs/recovery_s1/policy.pth + vecnormalize.pkl (NOT modified), runs the
deterministic policy across the push distribution, renders each episode, and
writes:
  * recovery_montage.mp4   - N-episode grid, full episode incl. settle
  * recovery_hero_*.mp4    - a few single episodes at higher resolution

Touches nothing except the output mp4s.  robot/robot.xml untouched.

    python render_recovery.py                      # 9-ep montage + 3 hero clips
    python render_recovery.py --n 12 --grid 4x3
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from biped_recovery_env import BipedRecoveryEnv, DEFAULT_MODEL

RUN = "runs/recovery_s1"


def load_policy(run_dir):
    hp = json.load(open(f"{run_dir}/hparams.json"))
    tmp = DummyVecEnv([lambda: BipedRecoveryEnv(push_band=(130., 140.))])
    vn = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", tmp)
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    m = PPO("MlpPolicy", tmp, device="cpu",
            policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    m.policy.load_state_dict(torch.load(f"{run_dir}/policy.pth", map_location="cpu",
                                        weights_only=True))
    m.policy.eval()
    return m, (lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32))


class RenderEnv(BipedRecoveryEnv):
    """same env, but render() uses a framed side/3-4 view and captures a frame
    every physics step of the WHOLE episode (incl. the auto-run transfer+settle)."""

    def __init__(self, *a, res=(480, 640), **k):
        self._res = res
        super().__init__(*a, render_mode="rgb_array", **k)
        self._frames = []
        self._cam = mujoco.MjvCamera()
        self._cam.azimuth = 128
        self._cam.elevation = -12
        self._cam.distance = 1.9
        self._cam.lookat[:] = [0.0, -0.35, 0.95]

    def _grab(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self._res[0], self._res[1])
        self._renderer.update_scene(self.data, camera=self._cam)
        self._frames.append(self._renderer.render())

    def reset(self, *a, **k):
        out = super().reset(*a, **k)
        self._frames = []
        self._grab()
        return out

    def step(self, action):
        m, d = self.model, self.data
        # replicate the parent step but grab a frame every physics substep
        act = np.asarray(action, np.float32).clip(-1.0, 1.0)
        residual = act * np.array([0.35, 0.35, 0.20])
        self._ep_step += 1
        r_shape = 0.0
        fell = False
        entered_transfer = False
        for _ in range(5):
            if self._phase == "swing":
                self._w = min(1.0, self._sk / 186)
            elif self._phase == "descend":
                self._w = 1.0 + self._sk / 56
            u = self._scaffold_ctrl(self._phase, self._sk, self._w, residual)
            d.ctrl[:15] = u
            mujoco.mj_step(m, d)
            self._sk += 1
            self._grab()
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                fell = True
                break
            bs = self._balance()
            from recovery_metrics import _foot_normal_force
            sw_nf = _foot_normal_force(m, d, "R")
            if self._phase == "swing":
                if not self._foot_lifted and sw_nf < 3.0:
                    self._foot_lifted = True
                if sw_nf < 3.0:
                    self._unloaded_streak += 1
                if self._sk >= 186:
                    self._phase = "descend"
                    self._sk = 0
            elif self._phase == "descend":
                genuine = (self._foot_lifted and bs.r_contact and sw_nf > 12.0)
                self._plant_streak = self._plant_streak + 1 if genuine else 0
                if self._plant_streak >= 10 or self._sk >= 520:
                    self._snapshot_touchdown(bs, sw_nf)
                    self._phase = "transfer"
                    self._sk = 0
                    entered_transfer = True
                    break
        self._prev_action = act.copy()
        if fell:
            return self._obs(), -20.0, True, False, self._info(fell=True)
        if entered_transfer:
            tr, info = self._auto_finish_render()
            return self._obs(), r_shape + tr, True, False, info
        if self._ep_step > 400:
            return self._obs(), r_shape, False, True, self._info(fell=False)
        return self._obs(), r_shape + 0.3, False, False, self._info(fell=False)

    def _auto_finish_render(self):
        m, d = self.model, self.data
        from recovery_metrics import sample_balance, NOMINAL_CHEST_Z
        for _ in range(150):
            d.ctrl[:15] = self._scaffold_ctrl("transfer", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            self._grab()
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                return -20.0, self._info(fell=True)
        self._phase = "settle"
        self._sk = 0
        for _ in range(650):
            d.ctrl[:15] = self._scaffold_ctrl("settle", self._sk, 0.0, np.zeros(3))
            mujoco.mj_step(m, d)
            self._sk += 1
            self._grab()
            self._peak_up = max(self._peak_up, self._cheap_tilt())
            if self._fallen():
                return -20.0, self._info(fell=True)
        b = sample_balance(m, d)
        ds = b.l_contact and b.r_contact
        stable = (ds and b.up_tilt_deg < 8.0 and abs(b.side_lean_deg) < 8.0
                  and b.com_speed_horiz < 0.10 and b.chest_z > NOMINAL_CHEST_Z - 0.10
                  and self._peak_up <= 35.0)
        success = stable and self._touchdown is not None and self._touchdown["planted"] \
            and self._touchdown["sep_mm"] >= 15.0
        info = self._info(fell=False)
        info["success"] = bool(success)
        info["end_sep_mm"] = float(self._touchdown["sep_mm"]) if self._touchdown else 0.0
        return (8.0 if success else 0.0), info


def _tile(clips, rows, cols, pad=4):
    H = max(len(c) for c in clips)
    fh, fw = clips[0][0].shape[:2]
    out = []
    for t in range(H):
        cells = []
        for c in clips:
            cells.append(c[min(t, len(c) - 1)])
        while len(cells) < rows * cols:
            cells.append(np.zeros_like(cells[0]))
        grid_rows = []
        for r in range(rows):
            row = cells[r * cols:(r + 1) * cols]
            row = [np.pad(im, ((pad, pad), (pad, pad), (0, 0)), constant_values=20) for im in row]
            grid_rows.append(np.hstack(row))
        out.append(np.vstack(grid_rows))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--grid", default="3x3")
    ap.add_argument("--band", default="124,143")
    ap.add_argument("--spread", type=float, default=12.0)
    ap.add_argument("--seed0", type=int, default=90000)
    a = ap.parse_args(argv)
    rows, cols = (int(x) for x in a.grid.split("x"))
    band = tuple(float(x) for x in a.band.split(","))

    model, norm = load_policy(a.run)
    import imageio.v2 as imageio

    # montage: N episodes at modest res
    env = RenderEnv(push_band=band, dir_spread_deg=a.spread, res=(300, 380))
    clips = []
    results = []
    for i in range(a.n):
        o, _ = env.reset(seed=a.seed0 + i)
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                act, _ = model.predict(norm(o), deterministic=True)
            o, r, term, trunc, info = env.step(act)
            done = term or trunc
        clips.append(env._frames)
        results.append((round(info["push_n"], 1), bool(info.get("success")),
                        round(info.get("end_sep_mm", 0), 1), round(info.get("peak_up", 0), 1)))
        print(f"  ep {i}: push {results[-1][0]:6.1f} N  "
              f"{'RECOVER' if results[-1][1] else 'fail   '}  "
              f"step {results[-1][2]:5.1f} mm  peak-tilt {results[-1][3]:4.1f} deg  "
              f"({len(env._frames)} frames)")
    tiles = _tile(clips, rows, cols)
    imageio.mimsave("recovery_montage.mp4", tiles, fps=60, macro_block_size=1)
    n_ok = sum(1 for _, ok, _, _ in results if ok)
    print(f"\n  wrote recovery_montage.mp4  ({n_ok}/{a.n} recover, "
          f"{sum(len(c) for c in clips)} total frames)")

    # 3 hero clips at higher res spanning the push range
    hero_env = RenderEnv(push_band=band, dir_spread_deg=a.spread, res=(470, 620))
    for tag, pn in (("low", 126.0), ("mid", 134.0), ("high", 142.0)):
        o, _ = hero_env.reset(options={"push_n": pn})
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                act, _ = model.predict(norm(o), deterministic=True)
            o, r, term, trunc, info = hero_env.step(act)
            done = term or trunc
        imageio.mimsave(f"recovery_hero_{tag}.mp4", hero_env._frames, fps=60, macro_block_size=1)
        print(f"  wrote recovery_hero_{tag}.mp4  (push {pn} N, "
              f"{'RECOVER' if info.get('success') else 'fail'}, {len(hero_env._frames)} frames)")


if __name__ == "__main__":
    main(sys.argv[1:])
