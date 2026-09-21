"""PPO for BIGGER STEPS -- biped_locomotion_env with stride_mode=True.

Starts from the w3 walking policy and lengthens the stride: a slower reference
cadence, a stride-length target the reward PEAKS on (not a fixed cap), a cadence
cap that forbids satisfying the speed target by shuffling, and a soft-landing
term (the swing foot must be actively decelerated before touchdown -- needed for
long steps, per PLOS One 2025).  Speed target is held fixed while the stride
curriculum grows, so cadence must drop.

All the w3 anti-loophole fences carry over (upright-gated progress, dive guard,
time-scaled fall penalty, flight termination, true-sole-clearance step gate).

  python train_bigstep_ppo.py --steps 16000000 --envs 8 --tag b1 --warm runs/loco_w3
  python train_bigstep_ppo.py --resume runs/bigstep_b1 --steps 8000000
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

from biped_locomotion_env import (BipedLocomotionEnv, DEFAULT_MODEL, EP_STEPS,
                                  BIGSTEP_SPEED_TGT, STRIDE_REF)

INFO_KW = ("fell", "vfwd", "dist", "t", "speed_tgt", "flight_frac",
           "genuine_steps", "dived", "stride_tgt", "mean_stride", "stride_ema")

STRIDE_CURRICULUM = [0.17, 0.19, 0.21]
ADVANCE_STRIDE_FRAC = 0.90     # mean_stride / stride_tgt to advance
SPEED = BIGSTEP_SPEED_TGT


def make_env(rank, stride_tgt, model_path, log_dir):
    def _thunk():
        env = BipedLocomotionEnv(model_path=model_path, speed_tgt=SPEED,
                                 stride_mode=True, seed=7300 + rank)
        env.set_task(stride_tgt=stride_tgt)
        return Monitor(env, os.path.join(log_dir, f"mon_{rank}.csv"), info_keywords=INFO_KW)
    return _thunk


def build_policy(venv, hp):
    return PPO("MlpPolicy", venv, verbose=0, device="cpu",
               n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
               learning_rate=hp["lr"], ent_coef=hp["ent"],
               gamma=0.995, gae_lambda=0.95, clip_range=0.2, target_kl=0.035,
               policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))


def warm_load(model, warm_dir):
    """Load the w3 walking policy, padding the first MLP layer 68 -> 69 inputs
    (the extra input is stride_tgt; its column starts at zero)."""
    wp = os.path.join(warm_dir, "policy_best.pth")
    wp = wp if os.path.exists(wp) else os.path.join(warm_dir, "policy.pth")
    src = torch.load(wp, map_location="cpu", weights_only=True)
    dst = model.policy.state_dict()
    loaded, padded, skipped = 0, 0, 0
    for k, v in dst.items():
        if k not in src:
            skipped += 1
            continue
        if src[k].shape == v.shape:
            dst[k] = src[k]
            loaded += 1
        elif v.dim() == 2 and src[k].dim() == 2 and v.shape[0] == src[k].shape[0] \
                and v.shape[1] > src[k].shape[1]:
            w = v.clone()
            w[:, :src[k].shape[1]] = src[k]
            w[:, src[k].shape[1]:] = 0.0
            dst[k] = w
            padded += 1
        else:
            skipped += 1
    model.policy.load_state_dict(dst)
    print(f"warm from {wp}: {loaded} copied, {padded} padded (+obs dims), {skipped} skipped")


class EvalCB(BaseCallback):
    def __init__(self, every, run_dir, model_path, n_eval=16, verbose=1):
        super().__init__(verbose)
        self.every, self.run_dir, self.model_path, self.n_eval = every, run_dir, model_path, n_eval
        self._next = every
        self._env = None
        self.fresh = False
        self.adv_ok = False
        self._best = -1e9

    def _init(self):
        self._env = BipedLocomotionEnv(model_path=self.model_path, speed_tgt=SPEED,
                                       stride_mode=True, render_mode="rgb_array")
        self._env.set_task(stride_tgt=self.training_env.get_attr("stride_tgt")[0])

    def _run(self, n, collect=0):
        vn = self.model.get_vec_normalize_env()
        st = self._env.stride_tgt
        vs, surv, fell, gen, flt, div, strd = [], [], [], [], [], [], []
        frames = []
        for i in range(n):
            obs, _ = self._env.reset(seed=9300 + i)
            done = False
            info = {}
            ep = []
            vv = []
            while not done:
                o = vn.normalize_obs(obs) if vn is not None else obs
                act, _ = self.model.predict(o, deterministic=True)
                obs, r, term, trunc, info = self._env.step(act)
                done = term or trunc
                vv.append(info["vfwd"])
                if i < collect:
                    ep.append(self._env.render())
            vs.append(float(np.mean(vv[15:])) if len(vv) > 25 else float(np.mean(vv)))
            surv.append(info["t"] / EP_STEPS)
            fell.append(int(info["fell"]))
            gen.append(info["genuine_steps"])
            flt.append(info["flight_frac"])
            div.append(int(info["dived"]))
            strd.append(info["mean_stride"])
            if i < collect:
                frames.append(ep)
        cad = float(np.mean(gen)) / (EP_STEPS * 5e-3)
        return dict(v=float(np.mean(vs)), surv=float(np.mean(surv)), fell=float(np.mean(fell)),
                    gen=float(np.mean(gen)), flt=float(np.mean(flt)), dived=float(np.mean(div)),
                    stride=float(np.mean(strd)), cadence=cad, stride_tgt=st, frames=frames)

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        if self._env is None:
            self._init()
        else:
            self._env.set_task(stride_tgt=self.training_env.get_attr("stride_tgt")[0])
        r = self._run(self.n_eval, collect=4)
        self.fresh = True
        self.adv_ok = (r["stride"] >= ADVANCE_STRIDE_FRAC * r["stride_tgt"]
                       and r["surv"] >= 0.85 and r["dived"] <= 0.06 and r["fell"] <= 0.20)
        for k in ("v", "surv", "fell", "gen", "flt", "dived", "stride", "cadence"):
            self.logger.record(f"eval/{k}", r[k])
        if self.verbose:
            print(f"  [eval @ {self.num_timesteps}] stride_tgt {r['stride_tgt']:.3f}  "
                  f"mean_stride {r['stride']:.3f} m  cadence {r['cadence']:.1f}/s  "
                  f"vfwd {r['v']:+.2f}  survive {r['surv']*100:.0f}%  fell {r['fell']*100:.0f}%  "
                  f"genuine {r['gen']:.1f}  flight {r['flt']:.2f}  dived {r['dived']*100:.0f}%")
        # score: LONGER stride is better up to target (no overshoot penalty),
        # stay up, keep speed, no loopholes
        strd_track = float(np.clip(r["stride"] / max(r["stride_tgt"], 0.05), 0.0, 1.0))
        spd_track = 1.0 - min(1.0, abs(r["v"] - SPEED) / SPEED)
        score = (1.6 * strd_track + 1.0 * r["surv"] + 0.7 * spd_track
                 - 2.0 * r["fell"] - 3.0 * max(0.0, r["flt"] - 0.12) - 1.5 * r["dived"])
        self._save(r["frames"], score)
        return True

    def _save(self, frames, score):
        torch.save(self.model.policy.state_dict(), os.path.join(self.run_dir, "policy.pth"))
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(os.path.join(self.run_dir, "vecnormalize.pkl"))
        if score >= self._best:
            self._best = score
            torch.save(self.model.policy.state_dict(), os.path.join(self.run_dir, "policy_best.pth"))
            if vn is not None:
                vn.save(os.path.join(self.run_dir, "vecnormalize_best.pkl"))
            with open(os.path.join(self.run_dir, "best.txt"), "w") as f:
                f.write(f"step {self.num_timesteps}  score {score:.3f}\n")
        if (self.num_timesteps // self.every) % 5 == 0:      # periodic milestone ckpt
            torch.save(self.model.policy.state_dict(),
                       os.path.join(self.run_dir, f"ckpt_{self.num_timesteps}.pth"))
            vn = self.model.get_vec_normalize_env()
            if vn is not None:
                vn.save(os.path.join(self.run_dir, f"ckpt_{self.num_timesteps}_vn.pkl"))
        try:
            import imageio.v2 as imageio
            H = max(len(f) for f in frames)
            tiles = [np.hstack([f[min(t, len(f) - 1)] for f in frames][:2]) for t in range(H)]
            imageio.mimsave(os.path.join(self.run_dir, f"eval_{self.num_timesteps}.mp4"),
                            tiles, fps=40, macro_block_size=1)
        except Exception as e:
            print(f"  montage skipped: {e}")


class Curric(BaseCallback):
    def __init__(self, eval_cb, run_dir, start=0, verbose=1):
        super().__init__(verbose)
        self.eval_cb, self.run_dir, self.level, self.streak = eval_cb, run_dir, start, 0

    def _on_step(self):
        if not self.eval_cb.fresh:
            return True
        self.eval_cb.fresh = False
        if self.level >= len(STRIDE_CURRICULUM) - 1:
            return True
        self.streak = self.streak + 1 if self.eval_cb.adv_ok else 0
        if self.streak >= 2:
            self.streak = 0
            self.level += 1
            st = STRIDE_CURRICULUM[self.level]
            self.training_env.env_method("set_task", None, None, None, st)
            hp = json.load(open(os.path.join(self.run_dir, "hparams.json")))
            hp["curriculum_level"] = self.level
            json.dump(hp, open(os.path.join(self.run_dir, "hparams.json"), "w"), indent=2)
            print(f"  >> STRIDE CURRICULUM -> level {self.level}: target {st} m")
        return True


class Progress(BaseCallback):
    def __init__(self, every=25000):
        super().__init__(0)
        self.every, self._next, self._t0 = every, every, time.time()

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        buf = self.model.ep_info_buffer
        if buf:
            mr = np.mean([e["r"] for e in buf])
            ms = np.mean([e.get("mean_stride", 0) for e in buf])
            mg = np.mean([e.get("genuine_steps", 0) for e in buf])
            mf = np.mean([e.get("fell", 0) for e in buf])
            mdv = np.mean([e.get("dived", 0) for e in buf])
            fps = self.num_timesteps / (time.time() - self._t0)
            print(f"  t={self.num_timesteps:>9}  ep_rew={mr:7.1f}  stride~{ms:.3f}  "
                  f"gen~{mg:.1f}  fell~{mf:.2f}  dive~{mdv:.2f}  {fps:.0f}/s")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=16_000_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--tag", default="b1")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--resume", default=None)
    p.add_argument("--warm", default=None, help="walking run dir to warm-start from (e.g. runs/loco_w3)")
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--ent", type=float, default=0.010)
    p.add_argument("--eval-every", type=int, default=250_000)
    a = p.parse_args()

    run_dir = a.resume or os.path.join("runs", f"bigstep_{a.tag}")
    os.makedirs(run_dir, exist_ok=True)
    hp_path = os.path.join(run_dir, "hparams.json")
    if a.resume and os.path.exists(hp_path):
        hp = json.load(open(hp_path))
    else:
        hp = dict(n_steps=2048, batch_size=4096, n_epochs=5, lr=a.lr, ent=a.ent,
                  net_arch=[256, 256], log_std=-2.0, curriculum_level=0)
        json.dump(hp, open(hp_path, "w"), indent=2)

    lvl = hp.get("curriculum_level", 0)
    st = STRIDE_CURRICULUM[lvl]
    print(f"start: stride curriculum {lvl} -> target {st} m  (speed fixed {SPEED} m/s)  ({run_dir})")

    venv = SubprocVecEnv([make_env(i, st, a.model, run_dir) for i in range(a.envs)])
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    if a.resume and os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = True
        venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)
        if a.warm:
            # seed obs normalisation from the walking run, padding the extra
            # stride_tgt dim so the warm policy sees in-distribution inputs.
            wv = os.path.join(a.warm, "vecnormalize_best.pkl")
            wv = wv if os.path.exists(wv) else os.path.join(a.warm, "vecnormalize.pkl")
            if os.path.exists(wv):
                import pickle
                with open(wv, "rb") as f:
                    src = pickle.load(f)           # a VecNormalize instance (no venv attached on unpickle)
                n = src.obs_rms.mean.shape[0]
                venv.obs_rms.mean[:n] = src.obs_rms.mean
                venv.obs_rms.var[:n] = src.obs_rms.var
                venv.obs_rms.mean[n:] = st          # stride_tgt + stride_ema columns
                venv.obs_rms.var[n:] = 0.003
                venv.obs_rms.count = src.obs_rms.count
                venv.ret_rms = src.ret_rms
                print(f"seeded VecNormalize from {wv} ({n} dims + {venv.obs_rms.mean.shape[0]-n} padded)")

    model = build_policy(venv, hp)
    pol_path = os.path.join(run_dir, "policy.pth")
    if a.resume and os.path.exists(pol_path):
        model.policy.load_state_dict(torch.load(pol_path, map_location="cpu", weights_only=True))
        print(f"resumed policy from {pol_path}")
    elif a.warm:
        warm_load(model, a.warm)
    else:
        with torch.no_grad():
            model.policy.action_net.weight.mul_(0.0)
            model.policy.action_net.bias.mul_(0.0)
        print("new run: zero-init action head -> starts at the (bigger-stride) CPG reference")

    ecb = EvalCB(a.eval_every, run_dir, a.model, n_eval=16)
    model.learn(total_timesteps=a.steps,
                callback=[Progress(25_000), ecb, Curric(ecb, run_dir, start=lvl)],
                progress_bar=False, reset_num_timesteps=not bool(a.resume))
    torch.save(model.policy.state_dict(), pol_path)
    venv.save(vn_path)
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
