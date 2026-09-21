"""PPO for the SIM-TO-REAL walk -- biped_sim2real_env (IMU + encoder obs).

Fresh train (obs shape and base controller both differ from loco_w3, so no
warm-start): 14-DOF residual on the CPG + IMU-only attitude/roll base.  Speed
curriculum 0.12 -> 0.30 m/s.  Mild domain randomisation throughout; a phase-2
run (`--robust 0.4 --resume ...`) hardens once the gait is solid.

  python train_sim2real_ppo.py --steps 24000000 --envs 8 --tag v1
  python train_sim2real_ppo.py --resume runs/sim2real_v1 --robust 0.4 --steps 10000000
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

from biped_sim2real_env import BipedSim2RealEnv, DEFAULT_MODEL, S2R_ROBUST
from biped_locomotion_env import EP_STEPS

INFO_KW = ("fell", "vfwd", "dist", "t", "flight_frac", "genuine_steps", "dived",
           "chatter_ps", "robust")
CURRICULUM = [0.12, 0.18, 0.24, 0.30]
ADVANCE_AT = 0.78


def make_env(rank, speed_tgt, robust, model_path, log_dir):
    def _thunk():
        env = BipedSim2RealEnv(model_path=model_path, speed_tgt=speed_tgt,
                               robust=robust, seed=7800 + rank)
        return Monitor(env, os.path.join(log_dir, f"mon_{rank}.csv"), info_keywords=INFO_KW)
    return _thunk


def build_policy(venv, hp):
    return PPO("MlpPolicy", venv, verbose=0, device="cpu",
               n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
               learning_rate=hp["lr"], ent_coef=hp["ent"],
               gamma=0.995, gae_lambda=0.95, clip_range=0.2, target_kl=0.035,
               policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))


class EvalCB(BaseCallback):
    def __init__(self, every, run_dir, model_path, robust, n_eval=16, verbose=1):
        super().__init__(verbose)
        self.every, self.run_dir, self.model_path = every, run_dir, model_path
        self.robust, self.n_eval = robust, n_eval
        self._next = every
        self._env = None
        self.last_frac = 0.0
        self.fresh = False
        self._best = -1e9

    def _init(self):
        st = self.training_env.get_attr("speed_tgt")[0]
        self._env = BipedSim2RealEnv(model_path=self.model_path, speed_tgt=st,
                                     robust=self.robust, render_mode="rgb_array")

    def _run(self, n, collect=0):
        vn = self.model.get_vec_normalize_env()
        st = self._env.speed_tgt
        vs, surv, fell, gen, flt, div, cha = [], [], [], [], [], [], []
        frames = []
        for i in range(n):
            obs, _ = self._env.reset(seed=9800 + i)
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
                    chatter=float(np.mean(cha)),
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
        capped = min(1.0, r["v"] / max(self._env.speed_tgt, 0.1))
        self.last_frac = capped if (r["surv"] >= 0.6 and r["dived"] <= 0.1) else 0.0
        self.fresh = True
        for k in ("v", "surv", "fell", "frac", "gen", "flt", "dived", "chatter"):
            self.logger.record(f"eval/{k}", r[k])
        if self.verbose:
            print(f"  [eval @ {self.num_timesteps}] tgt {self._env.speed_tgt:.2f}  "
                  f"vfwd {r['v']:+.2f} ({r['frac']*100:.0f}%)  survive {r['surv']*100:.0f}%  "
                  f"fell {r['fell']*100:.0f}%  genuine {r['gen']:.1f}  flight {r['flt']:.2f}  "
                  f"dived {r['dived']*100:.0f}%  chatter {r['chatter']:.2f}")
        spd = 1.0 - min(1.0, abs(r["v"] - self._env.speed_tgt) / max(self._env.speed_tgt, 0.1))
        score = (1.4 * r["surv"] + 1.1 * spd + 0.10 * min(r["gen"], 14)
                 - 2.0 * r["fell"] - 3.0 * max(0.0, r["flt"] - 0.12) - 1.5 * r["dived"]
                 - 0.6 * max(0.0, r["chatter"] - 0.15))
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
            print(f"  >> CURRICULUM -> level {self.level}: speed {st} m/s")
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
            mg = np.mean([e.get("genuine_steps", 0) for e in buf])
            mf = np.mean([e.get("fell", 0) for e in buf])
            mc = np.mean([e.get("chatter_ps", 0) for e in buf])
            fps = self.num_timesteps / (time.time() - self._t0)
            print(f"  t={self.num_timesteps:>9}  ep_rew={mr:7.1f}  vfwd~{mv:+.2f}  gen~{mg:.1f}  "
                  f"fell~{mf:.2f}  chatter~{mc:.2f}  {fps:.0f}/s")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=24_000_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--tag", default="v1")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--resume", default=None)
    p.add_argument("--robust", type=float, default=S2R_ROBUST)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent", type=float, default=0.005)
    p.add_argument("--logstd", type=float, default=None, help="reset policy log_std on resume")
    p.add_argument("--eval-every", type=int, default=250_000)
    p.add_argument("--no-curriculum", action="store_true")
    a = p.parse_args()

    run_dir = a.resume or os.path.join("runs", f"sim2real_{a.tag}")
    os.makedirs(run_dir, exist_ok=True)
    hp_path = os.path.join(run_dir, "hparams.json")
    if a.resume and os.path.exists(hp_path):
        hp = json.load(open(hp_path))
    else:
        hp = dict(n_steps=2048, batch_size=4096, n_epochs=5, lr=a.lr, ent=a.ent,
                  net_arch=[256, 256], log_std=-1.9, curriculum_level=0)
        json.dump(hp, open(hp_path, "w"), indent=2)

    lvl = hp.get("curriculum_level", 0)
    st = CURRICULUM[-1] if a.no_curriculum else CURRICULUM[lvl]
    print(f"start: curriculum {lvl} -> speed {st} m/s   robust {a.robust}   ({run_dir})")

    venv = SubprocVecEnv([make_env(i, st, a.robust, a.model, run_dir) for i in range(a.envs)])
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
        if a.logstd is not None:
            with torch.no_grad():
                model.policy.log_std.fill_(float(a.logstd))
            print(f"  reset log_std -> {a.logstd}")
        print(f"resumed policy from {pol_path}")
    else:
        with torch.no_grad():
            model.policy.action_net.weight.mul_(0.0)
            model.policy.action_net.bias.mul_(0.0)
        print("new run: zero-init action head -> starts at the CPG + IMU-PD base")

    ecb = EvalCB(a.eval_every, run_dir, a.model, a.robust, n_eval=16)
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
