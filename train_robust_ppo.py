"""Sim-to-real HARDENING of the loco_w3 walk -- biped_locomotion_env robust mode.

Warm-starts from runs/loco_w3 and continues training under a domain-randomisation
+ disturbance + smoothness curriculum:
  robust in {0.3, 0.5, 0.7, 1.0} scales
    - friction / joint-damping / servo-gain / link-mass randomisation,
    - 0-3 control-step action latency + gaussian observation noise,
    - random torso pushes (8-22 N, ~25 ms, every 1.5-4 s),
    - a contact-chatter penalty + jerk/effort penalties -> a hardware-friendly gait.
Speed target fixed at 0.30 m/s.  All the w3 anti-loophole fences carry over.

  python train_robust_ppo.py --steps 14000000 --envs 8 --tag r1 --warm runs/loco_w3
  python train_robust_ppo.py --resume runs/robust_r1 --steps 6000000
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

from biped_locomotion_env import BipedLocomotionEnv, DEFAULT_MODEL, EP_STEPS

SPEED = 0.30
ROBUST_CURRICULUM = [0.3, 0.5, 0.7, 1.0]
INFO_KW = ("fell", "vfwd", "dist", "t", "flight_frac", "genuine_steps", "dived",
           "robust", "chatter_ps")


def make_env(rank, robust, model_path, log_dir):
    def _thunk():
        env = BipedLocomotionEnv(model_path=model_path, speed_tgt=SPEED,
                                 robust=robust, seed=7700 + rank)
        return Monitor(env, os.path.join(log_dir, f"mon_{rank}.csv"), info_keywords=INFO_KW)
    return _thunk


def build_policy(venv, hp):
    return PPO("MlpPolicy", venv, verbose=0, device="cpu",
               n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
               learning_rate=hp["lr"], ent_coef=hp["ent"],
               gamma=0.995, gae_lambda=0.95, clip_range=0.2, target_kl=0.035,
               policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))


class EvalCB(BaseCallback):
    def __init__(self, every, run_dir, model_path, n_eval=20, verbose=1):
        super().__init__(verbose)
        self.every, self.run_dir, self.model_path, self.n_eval = every, run_dir, model_path, n_eval
        self._next = every
        self._env = None
        self.fresh = False
        self.adv_ok = False
        self._best = -1e9

    def _init(self):
        rb = self.training_env.get_attr("robust")[0]
        self._env = BipedLocomotionEnv(model_path=self.model_path, speed_tgt=SPEED,
                                       robust=rb, render_mode="rgb_array")

    def _run(self, n, collect=0):
        vn = self.model.get_vec_normalize_env()
        rb = self._env.robust
        vs, surv, fell, gen, flt, div, cha = [], [], [], [], [], [], []
        frames = []
        for i in range(n):
            obs, _ = self._env.reset(seed=9700 + i)
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
            cha.append(info["chatter_ps"])
            if i < collect:
                frames.append(ep)
        return dict(v=float(np.mean(vs)), surv=float(np.mean(surv)), fell=float(np.mean(fell)),
                    gen=float(np.mean(gen)), flt=float(np.mean(flt)), dived=float(np.mean(div)),
                    chatter=float(np.mean(cha)), robust=rb, frames=frames)

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        if self._env is None:
            self._init()
        else:
            self._env.set_task(robust=self.training_env.get_attr("robust")[0])
        r = self._run(self.n_eval, collect=4)
        self.fresh = True
        self.adv_ok = (r["surv"] >= 0.85 and r["fell"] <= 0.30 and r["dived"] <= 0.06)
        for k in ("v", "surv", "fell", "gen", "flt", "dived", "chatter"):
            self.logger.record(f"eval/{k}", r[k])
        if self.verbose:
            print(f"  [eval @ {self.num_timesteps}] robust {r['robust']:.2f}  vfwd {r['v']:+.2f}  "
                  f"survive {r['surv']*100:.0f}%  fell {r['fell']*100:.0f}%  genuine {r['gen']:.1f}  "
                  f"flight {r['flt']:.2f}  dived {r['dived']*100:.0f}%  chatter {r['chatter']:.2f}/st")
        spd_track = 1.0 - min(1.0, abs(r["v"] - SPEED) / SPEED)
        score = (1.6 * r["surv"] + 0.8 * spd_track + 0.6 * r["robust"]
                 - 2.0 * r["fell"] - 3.0 * max(0.0, r["flt"] - 0.12) - 1.5 * r["dived"]
                 - 0.8 * max(0.0, r["chatter"] - 0.15))
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
        if (self.num_timesteps // self.every) % 5 == 0:
            torch.save(self.model.policy.state_dict(),
                       os.path.join(self.run_dir, f"ckpt_{self.num_timesteps}.pth"))
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
        if self.level >= len(ROBUST_CURRICULUM) - 1:
            return True
        self.streak = self.streak + 1 if self.eval_cb.adv_ok else 0
        if self.streak >= 2:
            self.streak = 0
            self.level += 1
            rb = ROBUST_CURRICULUM[self.level]
            self.training_env.env_method("set_task", None, None, None, None, rb)
            hp = json.load(open(os.path.join(self.run_dir, "hparams.json")))
            hp["curriculum_level"] = self.level
            json.dump(hp, open(os.path.join(self.run_dir, "hparams.json"), "w"), indent=2)
            print(f"  >> ROBUST CURRICULUM -> level {self.level}: robust {rb}")
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
            mf = np.mean([e.get("fell", 0) for e in buf])
            mc = np.mean([e.get("chatter_ps", 0) for e in buf])
            mg = np.mean([e.get("genuine_steps", 0) for e in buf])
            fps = self.num_timesteps / (time.time() - self._t0)
            print(f"  t={self.num_timesteps:>9}  ep_rew={mr:7.1f}  gen~{mg:.1f}  fell~{mf:.2f}  "
                  f"chatter~{mc:.2f}  {fps:.0f}/s")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=14_000_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--tag", default="r1")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--resume", default=None)
    p.add_argument("--warm", default=None)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--ent", type=float, default=0.004)
    p.add_argument("--eval-every", type=int, default=250_000)
    a = p.parse_args()

    run_dir = a.resume or os.path.join("runs", f"robust_{a.tag}")
    os.makedirs(run_dir, exist_ok=True)
    hp_path = os.path.join(run_dir, "hparams.json")
    if a.resume and os.path.exists(hp_path):
        hp = json.load(open(hp_path))
    else:
        hp = dict(n_steps=2048, batch_size=4096, n_epochs=5, lr=a.lr, ent=a.ent,
                  net_arch=[256, 256], log_std=-2.0, curriculum_level=0)
        json.dump(hp, open(hp_path, "w"), indent=2)

    lvl = hp.get("curriculum_level", 0)
    rb = ROBUST_CURRICULUM[lvl]
    print(f"start: robust curriculum {lvl} -> robust {rb}  (speed {SPEED})  ({run_dir})")

    venv = SubprocVecEnv([make_env(i, rb, a.model, run_dir) for i in range(a.envs)])
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    if a.resume and os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = True
        venv.norm_reward = True
    elif a.warm and os.path.exists(os.path.join(a.warm, "vecnormalize_best.pkl")):
        venv = VecNormalize.load(os.path.join(a.warm, "vecnormalize_best.pkl"), venv)
        venv.training = True
        venv.norm_reward = True
        print(f"warm VecNormalize from {a.warm}")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)

    model = build_policy(venv, hp)
    pol_path = os.path.join(run_dir, "policy.pth")
    if a.resume and os.path.exists(pol_path):
        model.policy.load_state_dict(torch.load(pol_path, map_location="cpu", weights_only=True))
        print(f"resumed policy from {pol_path}")
    elif a.warm:
        wp = os.path.join(a.warm, "policy_best.pth")
        wp = wp if os.path.exists(wp) else os.path.join(a.warm, "policy.pth")
        model.policy.load_state_dict(torch.load(wp, map_location="cpu", weights_only=True))
        print(f"warm-started policy from {wp}")

    ecb = EvalCB(a.eval_every, run_dir, a.model, n_eval=20)
    model.learn(total_timesteps=a.steps,
                callback=[Progress(25_000), ecb, Curric(ecb, run_dir, start=lvl)],
                progress_bar=False, reset_num_timesteps=not bool(a.resume))
    torch.save(model.policy.state_dict(), pol_path)
    venv.save(vn_path)
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
