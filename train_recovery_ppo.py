"""PPO training for the RL push-recovery step (biped_recovery_env), Stage 1.

Policy learns a bounded residual on the swing-leg (R) joint trajectory, on top of
the frontal-LIPM + StandingLQR scaffold, on the 2x-hand morphology.  Zero action
== the CMA-optimised fixed trajectory.

Built-in band curriculum: starts on the easy distribution (push 130-140 N, no
direction spread), widens each time the eval success clears a threshold.

  python train_recovery_ppo.py --steps 3000000 --envs 8 --tag s1
  python train_recovery_ppo.py --resume runs/recovery_s1 --steps 2000000

Checkpoints: policy.pth (raw state_dict) + vecnormalize.pkl + hparams.json.
model.zip is written best-effort but NOT used for resume (it corrupts on this
torch/sb3 build).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from biped_recovery_env import BipedRecoveryEnv, DEFAULT_MODEL

INFO_KW = ("success", "peak_up", "end_sep_mm", "push_n")

# band curriculum: (push_lo, push_hi, dir_spread_deg); advance on eval success
CURRICULUM = [
    (130.0, 140.0, 0.0),
    (128.0, 141.0, 4.0),
    (126.0, 142.0, 8.0),
    (124.0, 143.0, 12.0),
]
ADVANCE_AT = 0.85


def make_env(rank, band, spread, model_path, log_dir):
    def _thunk():
        env = BipedRecoveryEnv(model_path=model_path, push_band=band,
                               dir_spread_deg=spread, seed=1000 + rank)
        return Monitor(env, os.path.join(log_dir, f"mon_{rank}.csv"),
                       info_keywords=INFO_KW)
    return _thunk


def build_policy(venv, hp):
    model = PPO("MlpPolicy", venv, verbose=0, device="cpu",
                n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
                learning_rate=hp["lr"], ent_coef=hp["ent"],
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    return model


class EvalCallback(BaseCallback):
    def __init__(self, every, run_dir, model_path, n_eval=24, verbose=1):
        super().__init__(verbose)
        self.every = every
        self.run_dir = run_dir
        self.model_path = model_path
        self.n_eval = n_eval
        self._next = every
        self._env = None
        self.last_sr = 0.0
        self.fresh = False        # set True right after an eval, consumed by curriculum

    def _init(self):
        # eval on the CURRENT curriculum band + one fixed wide band for tracking
        b = self.training_env.get_attr("push_band")[0]
        s = float(np.degrees(self.training_env.get_attr("dir_spread")[0]))
        self._env = BipedRecoveryEnv(model_path=self.model_path, push_band=b,
                                     dir_spread_deg=s, render_mode="rgb_array")
        self._wide = BipedRecoveryEnv(model_path=self.model_path,
                                      push_band=(124.0, 143.0), dir_spread_deg=12.0)

    def _run(self, env, n, collect_frames=0):
        vn = self.model.get_vec_normalize_env()
        succ = 0
        seps, peaks, pushes = [], [], []
        frames = []
        for i in range(n):
            obs, _ = env.reset(seed=7000 + i)
            done = False
            info = {}
            ep_f = []
            while not done:
                o = vn.normalize_obs(obs) if vn is not None else obs
                act, _ = self.model.predict(o, deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc
                if i < collect_frames:
                    ep_f.append(env.render())
            succ += int(info.get("success", False))
            seps.append(info.get("end_sep_mm", 0.0))
            peaks.append(info.get("peak_up", 99.0))
            pushes.append((info.get("push_n", 0.0), int(info.get("success", False))))
            if i < collect_frames:
                frames.append(ep_f)
        return succ / n, seps, peaks, pushes, frames

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        if self._env is None:
            self._init()
        else:
            b = self.training_env.get_attr("push_band")[0]
            s = float(np.degrees(self.training_env.get_attr("dir_spread")[0]))
            self._env.set_task(b, s)

        sr, seps, peaks, pushes, frames = self._run(self._env, self.n_eval, collect_frames=6)
        wsr, _, _, wpush, _ = self._run(self._wide, 24)
        self.last_sr = sr
        self.fresh = True
        self.logger.record("eval/success_rate", sr)
        self.logger.record("eval/wide_success_rate", wsr)
        self.logger.record("eval/median_sep_mm", float(np.median(seps)))
        self.logger.record("eval/median_peak_up", float(np.median(peaks)))
        if self.verbose:
            b = self.training_env.get_attr("push_band")[0]
            s = float(np.degrees(self.training_env.get_attr("dir_spread")[0]))
            bins = {}
            for pn, ok in pushes + wpush:
                k = int(pn // 3 * 3)
                bins.setdefault(k, [0, 0])
                bins[k][0] += ok
                bins[k][1] += 1
            bstr = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(bins.items()))
            print(f"  [eval @ {self.num_timesteps}] band {b} +-{s:.0f}deg  "
                  f"success {sr:.2f}  wide {wsr:.2f}  medSep {np.median(seps):.0f}  "
                  f"medPeakUp {np.median(peaks):.1f}")
            print(f"      per-push {bstr}")
        self._save(frames)
        return True

    def _save(self, frames):
        torch.save(self.model.policy.state_dict(),
                   os.path.join(self.run_dir, "policy.pth"))
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(os.path.join(self.run_dir, "vecnormalize.pkl"))
        try:
            import imageio.v2 as imageio
            H = max(len(f) for f in frames)
            tiles = []
            for t in range(H):
                row = [f[min(t, len(f) - 1)] for f in frames]
                grid = (np.vstack([np.hstack(row[:3]), np.hstack(row[3:6])])
                        if len(row) >= 6 else np.hstack(row))
                tiles.append(grid)
            imageio.mimsave(os.path.join(self.run_dir, f"eval_{self.num_timesteps}.mp4"),
                            tiles, fps=40, macro_block_size=1)
        except Exception as e:
            print(f"  montage skipped: {e}")


class BandCurriculum(BaseCallback):
    def __init__(self, eval_cb, run_dir, start_level=0, verbose=1):
        super().__init__(verbose)
        self.eval_cb = eval_cb
        self.run_dir = run_dir
        self.level = start_level
        self.streak = 0

    def _on_step(self):
        if not self.eval_cb.fresh:
            return True
        self.eval_cb.fresh = False
        if self.level >= len(CURRICULUM) - 1:
            return True
        self.streak = self.streak + 1 if self.eval_cb.last_sr >= ADVANCE_AT else 0
        if self.streak >= 2:                       # two consecutive good evals
            self.streak = 0
            self.level += 1
            lo, hi, sp = CURRICULUM[self.level]
            self.training_env.env_method("set_task", (lo, hi), sp)
            hp = json.load(open(os.path.join(self.run_dir, "hparams.json")))
            hp["curriculum_level"] = self.level
            json.dump(hp, open(os.path.join(self.run_dir, "hparams.json"), "w"), indent=2)
            print(f"  >> CURRICULUM advance -> level {self.level}: push {lo}-{hi} N, +-{sp} deg")
        return True


class Progress(BaseCallback):
    def __init__(self, every=25000):
        super().__init__(0)
        self.every = every
        self._next = every
        self._t0 = time.time()

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        buf = self.model.ep_info_buffer
        if buf:
            mr = np.mean([e["r"] for e in buf])
            ml = np.mean([e["l"] for e in buf])
            ms = np.mean([e.get("success", 0) for e in buf])
            fps = self.num_timesteps / (time.time() - self._t0)
            print(f"  t={self.num_timesteps:>9}  ep_rew={mr:7.2f}  ep_len={ml:5.1f}  "
                  f"train_success~{ms:.2f}  {fps:.0f} steps/s")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--tag", default="s1")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--resume", default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent", type=float, default=0.005)
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--no-curriculum", action="store_true")
    a = p.parse_args()

    run_dir = a.resume or os.path.join("runs", f"recovery_{a.tag}")
    os.makedirs(run_dir, exist_ok=True)
    hp_path = os.path.join(run_dir, "hparams.json")

    if a.resume and os.path.exists(hp_path):
        hp = json.load(open(hp_path))
    else:
        hp = dict(n_steps=512, batch_size=1024, n_epochs=6, lr=a.lr, ent=a.ent,
                  net_arch=[128, 128], log_std=-1.6, curriculum_level=0)
        json.dump(hp, open(hp_path, "w"), indent=2)

    lvl = hp.get("curriculum_level", 0)
    lo, hi, sp = CURRICULUM[lvl] if not a.no_curriculum else CURRICULUM[-1]
    print(f"start: curriculum level {lvl} -> push {lo}-{hi} N, +-{sp} deg")

    venv = SubprocVecEnv([make_env(i, (lo, hi), sp, a.model, run_dir)
                          for i in range(a.envs)])
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    if a.resume and os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = True
        venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = build_policy(venv, hp)
    pol_path = os.path.join(run_dir, "policy.pth")
    if a.resume and os.path.exists(pol_path):
        model.policy.load_state_dict(torch.load(pol_path, map_location="cpu", weights_only=True))
        print(f"resumed policy from {pol_path} (Adam state reset)")
    else:
        with torch.no_grad():
            model.policy.action_net.weight.mul_(0.0)
            model.policy.action_net.bias.mul_(0.0)
        print("new run: zero-init action head -> starts at the fixed trajectory")

    ecb = EvalCallback(a.eval_every, run_dir, a.model, n_eval=24)
    cbs = [Progress(25_000), ecb]
    if not a.no_curriculum:
        cbs.append(BandCurriculum(ecb, run_dir, start_level=lvl))

    model.learn(total_timesteps=a.steps, callback=cbs, progress_bar=False,
                reset_num_timesteps=not bool(a.resume))

    torch.save(model.policy.state_dict(), pol_path)
    venv.save(vn_path)
    try:
        model.save(os.path.join(run_dir, "model"))
    except Exception:
        pass
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
