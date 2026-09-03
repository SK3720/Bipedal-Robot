"""PPO for LARGER / state-dependent sequential push recovery -- biped_walk_env
variant="cp" (capture-point forward-arc swing: step SIZE tracks disturbance).

Same policy/obs/action as train_walk_ppo; the env base differs (uncapped, adaptive
swing, non-retracting descend, capture-scaled forward reach, commit-aware reward).
Warm-starts from the whip deliverable runs/walk_w11 (torso/frontal balance skills
transfer) with a fresh optimizer + a curriculum that extends push magnitude up.

  python train_stumble_ppo.py --steps 2000000 --envs 8 --tag s1 --warm runs/walk_w11
  python train_stumble_ppo.py --resume runs/stumble_s1 --steps 1000000

Checkpoints: policy.pth + policy_best.pth + vecnormalize*.pkl + hparams.json.
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

from biped_walk_env import BipedWalkEnv, DEFAULT_MODEL

INFO_KW = ("success", "flat_ok", "neutral_ok", "n_steps", "peak_up", "fell",
           "end_spd", "end_capt_fwd_mm", "end_fa_sep_mm", "push_n", "timed_out")

# curriculum: (push_lo, push_hi, dir_spread_deg).  The 2x-foot model
# (_exp_hands_2x_feet_2x.xml) enlarges the fore-aft support polygon so the
# capture point stays inside through the recovery -- the torso-pitch runaway that
# capped the baseline / duck-foot runs at ~150 N is gone.  s3-cp (trained on
# baseline feet) already recovers 150-190 N on this model with 2 steps and 0
# falls, so start there and climb to ~225 N.
CURRICULUM = [
    (148.0, 172.0, 0.0),
    (146.0, 188.0, 3.0),
    (143.0, 202.0, 6.0),
    (140.0, 216.0, 9.0),
    (138.0, 230.0, 12.0),
]
ADVANCE_AT = 0.80


def make_env(rank, band, spread, model_path, log_dir):
    def _thunk():
        env = BipedWalkEnv(model_path=model_path, push_band=band,
                           dir_spread_deg=spread, seed=7000 + rank,
                           variant="cp", max_steps=6)
        return Monitor(env, os.path.join(log_dir, f"mon_{rank}.csv"), info_keywords=INFO_KW)
    return _thunk


def build_policy(venv, hp):
    return PPO("MlpPolicy", venv, verbose=0, device="cpu",
               n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
               learning_rate=hp["lr"], ent_coef=hp["ent"],
               gamma=0.99, gae_lambda=0.95, clip_range=0.2, target_kl=0.025,
               policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))


class EvalCB(BaseCallback):
    def __init__(self, every, run_dir, model_path, n_eval=30, verbose=1):
        super().__init__(verbose)
        self.every, self.run_dir, self.model_path, self.n_eval = every, run_dir, model_path, n_eval
        self._next = every
        self._env = self._wide = None
        self.last_sr = 0.0
        self.fresh = False

    def _init(self):
        b = self.training_env.get_attr("push_band")[0]
        s = float(np.degrees(self.training_env.get_attr("dir_spread")[0]))
        self._env = BipedWalkEnv(model_path=self.model_path, push_band=b,
                                 dir_spread_deg=s, render_mode="rgb_array",
                                 variant="cp", max_steps=6)
        self._wide = BipedWalkEnv(model_path=self.model_path, push_band=(150.0, 220.0),
                                  dir_spread_deg=12.0, variant="cp", max_steps=6)

    def _run(self, env, n, collect=0):
        vn = self.model.get_vec_normalize_env()
        succ = flat = neut = fell = tmo = 0
        steps, ups, spds = [], [], []
        frames = []
        for i in range(n):
            obs, _ = env.reset(seed=9000 + i)
            done = False
            info = {}
            ep = []
            while not done:
                o = vn.normalize_obs(obs) if vn is not None else obs
                act, _ = self.model.predict(o, deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc
                if i < collect:
                    ep.append(env.render())
            succ += int(info.get("success", False))
            flat += int(info.get("flat_ok", False))
            neut += int(info.get("neutral_ok", False))
            fell += int(info.get("fell", False))
            tmo += int(info.get("timed_out", False))
            steps.append(info.get("n_steps", 0))
            ups.append(info.get("peak_up", 99))
            spds.append(info.get("end_spd", 9.9))
            if i < collect:
                frames.append(ep)
        return dict(sr=succ / n, flat=flat / n, neut=neut / n, fell=fell / n, tmo=tmo / n,
                    steps=float(np.mean(steps)), up=float(np.median(ups)),
                    spd=float(np.median(spds)),
                    dist=np.bincount(steps, minlength=8).tolist(), frames=frames)

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
        w = self._run(self._wide, 24)
        self.last_sr = r["sr"]
        self.fresh = True
        for k in ("sr", "flat", "fell", "tmo", "steps", "up", "spd"):
            self.logger.record(f"eval/{k}", r[k])
        self.logger.record("eval/wide_sr", w["sr"])
        self.logger.record("eval/wide_fell", w["fell"])
        self.logger.record("eval/neutral", r["neut"])
        if self.verbose:
            b = self.training_env.get_attr("push_band")[0]
            s = float(np.degrees(self.training_env.get_attr("dir_spread")[0]))
            print(f"  [eval @ {self.num_timesteps}] band {b} +-{s:.0f}  "
                  f"SUCC {r['sr']:.2f} (flat {r['flat']:.2f}, neutral {r['neut']:.2f}, wide {w['sr']:.2f})  "
                  f"fell {r['fell']:.2f}  timeout {r['tmo']:.2f}")
            print(f"      steps~{r['steps']:.1f} dist {r['dist'][1:]}  peakUp {r['up']:.1f}  endSpd {r['spd']:.2f}")
        # best = training-band success + wide-band success + bonuses for flat and
        # neutral-stance finishes
        score = (0.38 * r["sr"] + 0.38 * w["sr"] + 0.12 * r["flat"]
                 + 0.12 * r["neut"] - 0.6 * r["fell"])
        self._save(r["frames"], score=score)
        return True

    def _save(self, frames, score=None):
        torch.save(self.model.policy.state_dict(), os.path.join(self.run_dir, "policy.pth"))
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(os.path.join(self.run_dir, "vecnormalize.pkl"))
        # keep the best-by-eval checkpoint separately (policy.pth gets overwritten)
        if score is not None and score >= getattr(self, "_best_score", -1):
            self._best_score = score
            torch.save(self.model.policy.state_dict(), os.path.join(self.run_dir, "policy_best.pth"))
            if vn is not None:
                vn.save(os.path.join(self.run_dir, "vecnormalize_best.pkl"))
            with open(os.path.join(self.run_dir, "best.txt"), "w") as f:
                f.write(f"step {self.num_timesteps}  score {score:.3f}\n")
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


class Curric(BaseCallback):
    def __init__(self, eval_cb, run_dir, start=0, verbose=1):
        super().__init__(verbose)
        self.eval_cb, self.run_dir, self.level, self.streak = eval_cb, run_dir, start, 0

    def _on_step(self):
        if not self.eval_cb.fresh:
            return True
        self.eval_cb.fresh = False
        if self.level >= len(CURRICULUM) - 1:
            return True
        self.streak = self.streak + 1 if self.eval_cb.last_sr >= ADVANCE_AT else 0
        if self.streak >= 3:
            self.streak = 0
            self.level += 1
            lo, hi, sp = CURRICULUM[self.level]
            self.training_env.env_method("set_task", (lo, hi), sp)
            hp = json.load(open(os.path.join(self.run_dir, "hparams.json")))
            hp["curriculum_level"] = self.level
            json.dump(hp, open(os.path.join(self.run_dir, "hparams.json"), "w"), indent=2)
            print(f"  >> CURRICULUM -> level {self.level}: push {lo}-{hi} N, +-{sp} deg")
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
            ms = np.mean([e.get("success", 0) for e in buf])
            mf = np.mean([e.get("fell", 0) for e in buf])
            mn = np.mean([e.get("n_steps", 0) for e in buf])
            fps = self.num_timesteps / (time.time() - self._t0)
            print(f"  t={self.num_timesteps:>9}  ep_rew={mr:7.2f}  succ~{ms:.2f}  fell~{mf:.2f}  "
                  f"steps~{mn:.1f}  {fps:.0f}/s")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--tag", default="w1")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--resume", default=None)
    p.add_argument("--warm", default=None, help="dir to warm-start policy weights from (fresh optimizer/curriculum)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent", type=float, default=0.0015)
    p.add_argument("--eval-every", type=int, default=150_000)
    p.add_argument("--no-curriculum", action="store_true")
    a = p.parse_args()

    run_dir = a.resume or os.path.join("runs", f"walk_{a.tag}")
    os.makedirs(run_dir, exist_ok=True)
    hp_path = os.path.join(run_dir, "hparams.json")
    if a.resume and os.path.exists(hp_path):
        hp = json.load(open(hp_path))
    else:
        hp = dict(n_steps=1024, batch_size=2048, n_epochs=6, lr=a.lr, ent=a.ent,
                  net_arch=[256, 256], log_std=-2.9, curriculum_level=0)
        json.dump(hp, open(hp_path, "w"), indent=2)

    lvl = hp.get("curriculum_level", 0)
    lo, hi, sp = CURRICULUM[-1] if a.no_curriculum else CURRICULUM[lvl]
    print(f"start: curriculum {lvl} -> push {lo}-{hi} N, +-{sp} deg  ({run_dir})")

    venv = SubprocVecEnv([make_env(i, (lo, hi), sp, a.model, run_dir) for i in range(a.envs)])
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
        print(f"resumed policy from {pol_path}")
    elif a.warm:
        wp = os.path.join(a.warm, "policy_best.pth")
        wp = wp if os.path.exists(wp) else os.path.join(a.warm, "policy.pth")
        model.policy.load_state_dict(torch.load(wp, map_location="cpu", weights_only=True))
        print(f"warm-started policy weights from {wp} (fresh optimizer + curriculum)")
    else:
        with torch.no_grad():
            model.policy.action_net.weight.mul_(0.0)
            model.policy.action_net.bias.mul_(0.0)
        print("new run: zero-init action head -> starts at the reference gait")

    ecb = EvalCB(a.eval_every, run_dir, a.model, n_eval=30)
    cbs = [Progress(25_000), ecb]
    if not a.no_curriculum:
        cbs.append(Curric(ecb, run_dir, start=lvl))

    model.learn(total_timesteps=a.steps, callback=cbs, progress_bar=False,
                reset_num_timesteps=not bool(a.resume))

    torch.save(model.policy.state_dict(), pol_path)
    venv.save(vn_path)
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
