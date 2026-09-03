"""PPO fine-tuning for the 2-step milestone (biped_2step_env.Biped2StepEnv).

Warm-starts from the frozen Stage-1 checkpoint (runs/recovery_s1): same policy
architecture and same 66-d observation, so policy.pth loads directly and the
Stage-1 VecNormalize obs statistics carry over (reward normalisation is reset —
the 2-step episode reward has a different scale).

The policy is used in a canonical "swing = R" frame for BOTH steps; step 2 runs
the scaffold mirrored and the env reflects obs/'action around that.  So the
same 3-d swing-leg residual drives step 1 (identity) and step 2 (L/R mirror).

  python train_2step_ppo.py --steps 200000 --envs 8 --tag a1_smoke
  python train_2step_ppo.py --resume runs/twostep_a1 --steps 800000

Checkpoints: policy.pth + vecnormalize.pkl + hparams.json  (model.zip best-effort,
not used for resume — it corrupts on this torch/sb3 build).
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

from biped_2step_env import Biped2StepEnv, DEFAULT_MODEL

S1_RUN = "runs/recovery_s1"
INFO_KW = ("success", "peak_up", "end_sep_mm", "end_stagger_mm", "push_n",
           "planted1", "planted2", "step1_only", "sole_pitch_max", "sole_roll_max", "com_support")

# 2-step band curriculum: (push_lo, push_hi, dir_spread_deg).  Step 1 is already
# solved by the frozen policy across 124-143 N; start NARROW so step 1 is in its
# sweet spot and the policy can focus on step 2, then widen.
CURRICULUM = [
    (132.0, 138.0, 0.0),
    (130.0, 140.0, 0.0),
    (128.0, 141.0, 6.0),
    (126.0, 142.0, 10.0),
    (124.0, 143.0, 12.0),
]
ADVANCE_AT = 0.72          # 2-step is harder than Stage 1's 0.85 bar


def make_env(rank, band, spread, model_path, log_dir):
    def _thunk():
        env = Biped2StepEnv(model_path=model_path, push_band=band,
                            dir_spread_deg=spread, seed=4000 + rank)
        return Monitor(env, os.path.join(log_dir, f"mon_{rank}.csv"),
                       info_keywords=INFO_KW)
    return _thunk


def build_policy(venv, hp):
    return PPO("MlpPolicy", venv, verbose=0, device="cpu",
               n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
               learning_rate=hp["lr"], ent_coef=hp["ent"],
               gamma=0.99, gae_lambda=0.95, clip_range=0.2,
               policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))


class EvalCallback(BaseCallback):
    def __init__(self, every, run_dir, model_path, n_eval=24, verbose=1):
        super().__init__(verbose)
        self.every = every
        self.run_dir = run_dir
        self.model_path = model_path
        self.n_eval = n_eval
        self._next = every
        self._env = self._wide = None
        self.last_sr = 0.0
        self.fresh = False

    def _init(self):
        b = self.training_env.get_attr("push_band")[0]
        s = float(np.degrees(self.training_env.get_attr("dir_spread")[0]))
        self._env = Biped2StepEnv(model_path=self.model_path, push_band=b,
                                  dir_spread_deg=s, render_mode="rgb_array")
        self._wide = Biped2StepEnv(model_path=self.model_path,
                                   push_band=(124.0, 143.0), dir_spread_deg=12.0)

    def _run(self, env, n, collect=0):
        vn = self.model.get_vec_normalize_env()
        succ = p1 = p2 = s1o = fell = 0
        sep2s, stags, peaks, sprm, srrm, csup = [], [], [], [], [], []
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
                if i < collect:
                    ep_f.append(env.render())
            succ += int(info.get("success", False))
            p1 += int(info.get("planted1", False))
            p2 += int(info.get("planted2", False))
            s1o += int(info.get("step1_only", False))
            fell += int(info.get("fell", False))
            sep2s.append(info.get("end_sep_mm", 0.0))
            stags.append(info.get("end_stagger_mm", 0.0))
            peaks.append(info.get("peak_up", 99.0))
            sprm.append(info.get("sole_pitch_max", 99.0))
            srrm.append(info.get("sole_roll_max", 99.0))
            csup.append(info.get("com_support", 0.0))
            if i < collect:
                frames.append(ep_f)
        return dict(sr=succ / n, p1=p1 / n, p2=p2 / n, s1o=s1o / n, fell=fell / n,
                    sep2=float(np.median(sep2s)), stag=float(np.median(stags)),
                    peak=float(np.median(peaks)), sprm=float(np.median(sprm)),
                    srrm=float(np.median(srrm)), csup=float(np.median(csup)), frames=frames)

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

        r = self._run(self._env, self.n_eval, collect=6)
        w = self._run(self._wide, 20)
        self.last_sr = r["sr"]
        self.fresh = True
        for k in ("sr", "p1", "p2", "s1o", "fell", "sep2", "stag", "peak", "sprm", "srrm", "csup"):
            self.logger.record(f"eval/{k}", r[k])
        self.logger.record("eval/wide_sr", w["sr"])
        self.logger.record("eval/wide_p2", w["p2"])
        if self.verbose:
            b = self.training_env.get_attr("push_band")[0]
            s = float(np.degrees(self.training_env.get_attr("dir_spread")[0]))
            print(f"  [eval @ {self.num_timesteps}] band {b} +-{s:.0f}  "
                  f"SUCC {r['sr']:.2f} (wide {w['sr']:.2f})  p1 {r['p1']:.2f} p2 {r['p2']:.2f} "
                  f"s1only {r['s1o']:.2f} fell {r['fell']:.2f}")
            print(f"      sep2 {r['sep2']:+.0f}mm  stag {r['stag']:+.0f}mm  peakUp {r['peak']:.1f}  "
                  f"soleP_max {r['sprm']:.0f}  soleR_max {r['srrm']:.0f}  comSup {r['csup']:.2f}")
        self._save(r["frames"])
        return True

    def _save(self, frames):
        torch.save(self.model.policy.state_dict(), os.path.join(self.run_dir, "policy.pth"))
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
        if self.streak >= 2:
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
            ms = np.mean([e.get("success", 0) for e in buf])
            m2 = np.mean([e.get("planted2", 0) for e in buf])
            fps = self.num_timesteps / (time.time() - self._t0)
            print(f"  t={self.num_timesteps:>9}  ep_rew={mr:7.2f}  train_succ~{ms:.2f}  "
                  f"plant2~{m2:.2f}  {fps:.0f} steps/s")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=800_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--tag", default="a1")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--resume", default=None)
    p.add_argument("--warm", default=S1_RUN, help="checkpoint dir to warm-start policy+obs-norm from")
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--ent", type=float, default=0.004)
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--no-curriculum", action="store_true")
    a = p.parse_args()

    run_dir = a.resume or os.path.join("runs", f"twostep_{a.tag}")
    os.makedirs(run_dir, exist_ok=True)
    hp_path = os.path.join(run_dir, "hparams.json")
    if a.resume and os.path.exists(hp_path):
        hp = json.load(open(hp_path))
    else:
        hp = dict(n_steps=512, batch_size=1024, n_epochs=6, lr=a.lr, ent=a.ent,
                  net_arch=[128, 128], log_std=-1.6, curriculum_level=0)
        json.dump(hp, open(hp_path, "w"), indent=2)

    lvl = hp.get("curriculum_level", 0)
    lo, hi, sp = CURRICULUM[-1] if a.no_curriculum else CURRICULUM[lvl]
    print(f"start: curriculum level {lvl} -> push {lo}-{hi} N, +-{sp} deg   (run {run_dir})")

    venv = SubprocVecEnv([make_env(i, (lo, hi), sp, a.model, run_dir) for i in range(a.envs)])
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    if a.resume and os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = True
        venv.norm_reward = True
    elif a.warm and os.path.exists(os.path.join(a.warm, "vecnormalize.pkl")):
        venv = VecNormalize.load(os.path.join(a.warm, "vecnormalize.pkl"), venv)
        venv.training = True
        venv.norm_reward = True
        venv.ret_rms = type(venv.ret_rms)(shape=venv.ret_rms.mean.shape)   # reset reward norm
        print(f"warm obs-norm from {a.warm}/vecnormalize.pkl (reward norm reset)")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = build_policy(venv, hp)
    pol_path = os.path.join(run_dir, "policy.pth")
    src = pol_path if (a.resume and os.path.exists(pol_path)) else \
        (os.path.join(a.warm, "policy.pth") if a.warm else None)
    if src and os.path.exists(src):
        model.policy.load_state_dict(torch.load(src, map_location="cpu", weights_only=True))
        print(f"loaded policy weights from {src}  (Adam state fresh)")
    else:
        print("no warm-start policy found — training from scratch")

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
