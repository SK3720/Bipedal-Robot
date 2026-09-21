"""PPO for forward WALKING -- biped_locomotion_env.

Policy: 14-joint position residual (10 legs + 4 arms) on a CPG gait reference +
attitude-only LQR torso hold.  Zero-init action head -> starts at the stable
marching-in-place reference; RL learns the forward-driving corrections.

Curriculum raises the forward-speed target.

  python train_loco_ppo.py --steps 8000000 --envs 8 --tag x1
  python train_loco_ppo.py --resume runs/loco_x1 --steps 4000000

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

from biped_locomotion_env import BipedLocomotionEnv, DEFAULT_MODEL, EP_STEPS

INFO_KW = ("fell", "vfwd", "dist", "t", "tilt", "speed_tgt", "flight_frac",
           "genuine_steps", "dived")

# curriculum: forward-speed target (m/s).  Medium speed + stability is the goal --
# NOT max speed -- so it tops out at a comfortable walk.
CURRICULUM = [0.12, 0.18, 0.24, 0.30]
ADVANCE_AT = 0.78          # fraction of target speed reached (eval) to advance


def make_env(rank, speed_tgt, model_path, log_dir):
    def _thunk():
        env = BipedLocomotionEnv(model_path=model_path, speed_tgt=speed_tgt, seed=7000 + rank)
        return Monitor(env, os.path.join(log_dir, f"mon_{rank}.csv"), info_keywords=INFO_KW)
    return _thunk


def build_policy(venv, hp):
    return PPO("MlpPolicy", venv, verbose=0, device="cpu",
               n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
               learning_rate=hp["lr"], ent_coef=hp["ent"],
               gamma=0.995, gae_lambda=0.95, clip_range=0.2, target_kl=0.03,
               policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))


class EvalCB(BaseCallback):
    def __init__(self, every, run_dir, model_path, n_eval=16, verbose=1):
        super().__init__(verbose)
        self.every, self.run_dir, self.model_path, self.n_eval = every, run_dir, model_path, n_eval
        self._next = every
        self._env = None
        self.last_frac = 0.0
        self.fresh = False
        self._best = -1e9

    def _init(self):
        st = self.training_env.get_attr("speed_tgt")[0]
        self._env = BipedLocomotionEnv(model_path=self.model_path, speed_tgt=st,
                                       render_mode="rgb_array")

    def _run(self, n, collect=0):
        vn = self.model.get_vec_normalize_env()
        st = self._env.speed_tgt
        vs, dists, surv, fells, gens, flts, divs = [], [], [], [], [], [], []
        frames = []
        for i in range(n):
            obs, _ = self._env.reset(seed=9000 + i)
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
            vs.append(float(np.mean(vv[15:])) if len(vv) > 20 else float(np.mean(vv)))
            dists.append(info["dist"])
            surv.append(info["t"] / EP_STEPS)
            fells.append(int(info["fell"]))
            gens.append(info.get("genuine_steps", 0))
            flts.append(info.get("flight_frac", 0.0))
            divs.append(int(info.get("dived", False)))
            if i < collect:
                frames.append(ep)
        return dict(v=float(np.mean(vs)), dist=float(np.mean(dists)),
                    surv=float(np.mean(surv)), fell=float(np.mean(fells)),
                    gen=float(np.mean(gens)), flt=float(np.mean(flts)),
                    dived=float(np.mean(divs)),
                    frac=float(np.mean(vs)) / max(st, 0.1), frames=frames)

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        if self._env is None:
            self._init()
        else:
            self._env.set_task(speed_tgt=self.training_env.get_attr("speed_tgt")[0])
        r = self._run(self.n_eval, collect=4)
        # frac used for curriculum advance: speed tracking CAPPED at target (no
        # credit for overspeed) AND gated on real survival + no diving.
        capped_frac = min(1.0, r["v"] / max(self._env.speed_tgt, 0.1))
        self.last_frac = capped_frac if (r["surv"] >= 0.65 and r["dived"] <= 0.10) else 0.0
        self.last_surv = r["surv"]
        self.fresh = True
        for k in ("v", "dist", "surv", "fell", "frac", "gen", "flt", "dived"):
            self.logger.record(f"eval/{k}", r[k])
        if self.verbose:
            st = self._env.speed_tgt
            print(f"  [eval @ {self.num_timesteps}] tgt {st:.2f} m/s  "
                  f"vfwd {r['v']:+.2f} ({r['frac']*100:.0f}% of tgt)  dist {r['dist']:+.2f} m  "
                  f"survive {r['surv']*100:.0f}%  fell {r['fell']*100:.0f}%  "
                  f"genuine {r['gen']:.1f}  flight {r['flt']:.2f}  dived {r['dived']*100:.0f}%")
        # best = a genuine forward WALK: survive + track speed + real alternating
        # steps (not marching in place), no bilateral flight, no falls, no dive-and-die.
        spd_track = 1.0 - min(1.0, abs(r["v"] - self._env.speed_tgt) / max(self._env.speed_tgt, 0.1))
        score = (1.3 * r["surv"] + 1.3 * spd_track + 0.14 * min(r["gen"], 14)
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
        try:
            import imageio.v2 as imageio
            H = max(len(f) for f in frames)
            tiles = []
            for t in range(H):
                row = [f[min(t, len(f) - 1)] for f in frames]
                tiles.append(np.hstack(row[:2]) if len(row) >= 2 else row[0])
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
        self.streak = self.streak + 1 if self.eval_cb.last_frac >= ADVANCE_AT else 0
        if self.streak >= 3:
            self.streak = 0
            self.level += 1
            st = CURRICULUM[self.level]
            self.training_env.env_method("set_task", st)
            hp = json.load(open(os.path.join(self.run_dir, "hparams.json")))
            hp["curriculum_level"] = self.level
            json.dump(hp, open(os.path.join(self.run_dir, "hparams.json"), "w"), indent=2)
            print(f"  >> CURRICULUM -> level {self.level}: speed target {st} m/s")
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
            mv = np.mean([e.get("vfwd", 0) for e in buf])
            md = np.mean([e.get("dist", 0) for e in buf])
            mf = np.mean([e.get("fell", 0) for e in buf])
            mg = np.mean([e.get("genuine_steps", 0) for e in buf])
            mdv = np.mean([e.get("dived", 0) for e in buf])
            fps = self.num_timesteps / (time.time() - self._t0)
            print(f"  t={self.num_timesteps:>9}  ep_rew={mr:7.1f}  vfwd~{mv:+.2f}  dist~{md:+.2f}  "
                  f"gen~{mg:.1f}  fell~{mf:.2f}  dive~{mdv:.2f}  {fps:.0f}/s")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8_000_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--tag", default="x1")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--resume", default=None)
    p.add_argument("--warm", default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent", type=float, default=0.004)
    p.add_argument("--eval-every", type=int, default=150_000)
    p.add_argument("--no-curriculum", action="store_true")
    a = p.parse_args()

    run_dir = a.resume or os.path.join("runs", f"loco_{a.tag}")
    os.makedirs(run_dir, exist_ok=True)
    hp_path = os.path.join(run_dir, "hparams.json")
    if a.resume and os.path.exists(hp_path):
        hp = json.load(open(hp_path))
    else:
        hp = dict(n_steps=2048, batch_size=4096, n_epochs=5, lr=a.lr, ent=a.ent,
                  net_arch=[256, 256], log_std=-1.8, curriculum_level=0)
        json.dump(hp, open(hp_path, "w"), indent=2)

    lvl = hp.get("curriculum_level", 0)
    st = CURRICULUM[-1] if a.no_curriculum else CURRICULUM[lvl]
    print(f"start: curriculum {lvl} -> speed target {st} m/s  ({run_dir})")

    venv = SubprocVecEnv([make_env(i, st, a.model, run_dir) for i in range(a.envs)])
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    if a.resume and os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = True
        venv.norm_reward = True
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
        print(f"warm-started from {wp}")
    else:
        with torch.no_grad():
            model.policy.action_net.weight.mul_(0.0)
            model.policy.action_net.bias.mul_(0.0)
        print("new run: zero-init action head -> starts at the CPG reference")

    ecb = EvalCB(a.eval_every, run_dir, a.model, n_eval=16)
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
