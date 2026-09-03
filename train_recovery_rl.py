"""PPO training for forward-push recovery stepping on top of the LQR base.

  ctrl = clip( torso-stabilising LQR(state) + policy_residual )   (see recovery_env)

Zero-init policy head  ->  training starts at the LQR baseline (~140 N in place).
The policy has to learn the stepping residual for the harder pushes, which the
forward-push magnitude curriculum ramps up as survival stays high.

Usage:
  python train_recovery_rl.py --steps 600000 --envs 8 --tag v1
  python train_recovery_rl.py --steps 400000 --resume runs/recovery_v1/model.zip
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from recovery_env import RecoveryEnv
from warmstart_utils import init_zero_policy


def make_env(rank: int, log_dir: str):
    def _thunk():
        env = RecoveryEnv(seed=rank)
        return Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
    return _thunk


def behavior_clone(model, venv, n_steps: int, epochs: int = 6, verbose=True):
    """Warm-start the policy by imitating env.scripted_step_action() (a crude
    weight-shift + hip-swing).  Gives the policy a rough stepping motor skill for
    RL to fine-tune instead of having to discover it from scratch."""
    import torch
    obs_buf, act_buf = [], []
    obs = venv.reset()
    for _ in range(n_steps):
        acts = np.asarray(venv.env_method("scripted_step_action"), dtype=np.float32)
        obs_buf.append(np.asarray(obs, dtype=np.float32))
        act_buf.append(acts)
        obs, _, dones, _ = venv.step(acts)
    X = torch.as_tensor(np.concatenate(obs_buf), dtype=torch.float32)
    Y = torch.as_tensor(np.concatenate(act_buf), dtype=torch.float32)
    frac_step = float((np.abs(Y.numpy()) > 0.05).any(axis=1).mean())
    if verbose:
        print(f"[BC] {len(X)} samples, {frac_step:.0%} with a non-trivial scripted action")
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            feats = model.policy.extract_features(X[idx])
            latent_pi, _ = model.policy.mlp_extractor(feats)
            mean = model.policy.action_net(latent_pi)
            loss = ((mean - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if verbose:
            print(f"[BC] epoch {ep}  mse={tot / n:.4f}")
    venv.reset()


class CurriculumCallback(BaseCallback):
    """Advance each sub-env's forward-push curriculum from its own episode outcomes."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep = 0

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][i]
            survived = bool(info.get("episode_survived", False))
            self._ep += 1
            try:
                bumped = self.training_env.env_method("record_episode", survived, indices=[i])[0]
            except Exception:
                bumped = False
            if bumped and self.verbose:
                lvl = self.training_env.get_attr("curriculum_level", indices=[i])[0]
                pmax = self.training_env.get_attr("cur_push_max", indices=[i])[0]
                print(f"[curriculum] env{i} -> level {lvl}, push_max {pmax:.0f} N")
        return True


class SaveVecNormCallback(BaseCallback):
    """CheckpointCallback doesn't save VecNormalize; do it so mid-run eval works."""
    def __init__(self, every, path, verbose=0):
        super().__init__(verbose)
        self.every = every
        self.path = path
        self._next = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            try:
                self.training_env.save(os.path.join(self.path, "vecnormalize.pkl"))
                import torch as _t
                _t.save(self.model.policy.state_dict(),
                        os.path.join(self.path, "policy.pth"))
            except Exception:
                pass
        return True


class ProgressCallback(BaseCallback):
    def __init__(self, every=20000, verbose=1):
        super().__init__(verbose)
        self.every = every
        self._next = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            pmax = np.mean(self.training_env.get_attr("cur_push_max"))
            lvl = np.mean(self.training_env.get_attr("curriculum_level"))
            ep_rew = self.model.ep_info_buffer
            mr = np.mean([e["r"] for e in ep_rew]) if ep_rew else float("nan")
            ml = np.mean([e["l"] for e in ep_rew]) if ep_rew else float("nan")
            print(f"  t={self.num_timesteps:>8}  ep_rew={mr:7.1f}  ep_len={ml:6.1f}  "
                  f"push_max~{pmax:5.1f}N  curric_lvl~{lvl:.1f}")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=600_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--tag", default="v1")
    p.add_argument("--resume", default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--logstd", type=float, default=-1.5)
    p.add_argument("--bc", type=int, default=0, help="BC pretrain steps per env (0=off)")
    args = p.parse_args()

    run_dir = os.path.join("runs", f"recovery_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)

    venv = SubprocVecEnv([make_env(i, run_dir) for i in range(args.envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    if args.resume:
        vn = args.resume.replace("model.zip", "vecnormalize.pkl")
        if os.path.exists(vn):
            venv = VecNormalize.load(vn, SubprocVecEnv([make_env(i, run_dir) for i in range(args.envs)]))
            venv.norm_reward = True

    if args.resume and os.path.exists(args.resume):
        model = PPO.load(args.resume, env=venv, device="cpu")
        model.learning_rate = args.lr
        print(f"resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy", venv, verbose=0, device="cpu",
            n_steps=1024, batch_size=2048, n_epochs=8,
            learning_rate=args.lr, ent_coef=args.ent,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
        init_zero_policy(model, log_std=args.logstd)
        print(f"zero-init policy head (log_std={args.logstd}) -> starts at the LQR baseline")
        if args.bc > 0:
            behavior_clone(model, venv, n_steps=args.bc)
            model.policy.log_std.data.fill_(args.logstd)  # BC touched all params

    cbs = [
        CurriculumCallback(verbose=1),
        ProgressCallback(every=25_000),
        SaveVecNormCallback(every=100_000, path=run_dir),
        CheckpointCallback(save_freq=max(100_000 // args.envs, 1),
                           save_path=run_dir, name_prefix="ckpt"),
    ]
    model.learn(total_timesteps=args.steps, callback=cbs, progress_bar=False)

    model.save(os.path.join(run_dir, "model"))
    venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
    torch.save(model.policy.state_dict(), os.path.join(run_dir, "policy.pth"))
    print(f"saved -> {run_dir}/model.zip")


if __name__ == "__main__":
    main()
