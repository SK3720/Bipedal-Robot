"""Teacher -> student distillation (with DAgger) for the sim-to-real walk.

Cold-start RL under the realistic (IMU + encoder + history) observation does not
converge, and RL fine-tuning a distilled student collapses it (fresh value fn +
advantage noise knock it out of the basin -- a known failure mode in this repo).
So the whole pipeline is supervised:

  1. teacher (loco_w3 + rebuilt IMU-only base controller) drives; record
     (realistic_obs -> teacher_action); behaviour-clone student v0.
  2. DAgger rounds: the STUDENT drives (visiting its own mistake/recovery
     states); the teacher labels every visited realistic_obs; aggregate; retrain.
  3. ~1/3 of every rollout runs under mild domain randomisation so the final
     student is dynamics-robust without any RL.

  python distill_s2r.py --init 350 --dagger 3 --dagger-roll 150 --epochs 60 \
                        --out runs/sim2real_v1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd

from biped_sim2real_env import BipedSim2RealEnv, DEFAULT_MODEL
from biped_locomotion_env import EP_STEPS


def _w3_norm(w3_dir):
    v = pickle.load(open(f"{w3_dir}/vecnormalize_best.pkl", "rb"))
    m, s = v.obs_rms.mean, np.sqrt(v.obs_rms.var + 1e-8)
    return lambda o: np.clip((o - m) / s, -10, 10).astype(np.float32)


def load_w3(w3_dir, model):
    hp = json.load(open(f"{w3_dir}/hparams.json"))
    tmp = DummyVecEnv([lambda: BipedSim2RealEnv(model_path=model, privileged_obs=True,
                                               imu_ctrl=True, robust=0.0)])
    mm = PPO("MlpPolicy", tmp, device="cpu",
             policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(f"{w3_dir}/policy_best.pth", map_location="cpu",
                                         weights_only=True))
    mm.policy.eval()
    return mm, _w3_norm(w3_dir)


def rollouts(mm_teacher, tnorm, model, n, speed, seed0, student=None, snorm=None,
             beta_student=1.0, act_noise=0.10):
    """Collect (realistic_obs, teacher_action).  `student` (if given) drives with
    prob beta_student, else the teacher drives (DAgger mixing)."""
    X, Y, surv = [], [], []
    rng = np.random.default_rng(seed0 * 7 + 1)
    for r in range(n):
        rb = 0.15 if (r % 3 == 0) else 0.0
        env = BipedSim2RealEnv(model_path=model, speed_tgt=speed,
                               privileged_obs=True, imu_ctrl=True, robust=rb)
        po, _ = env.reset(seed=seed0 + r)
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                ta, _ = mm_teacher.predict(tnorm(po), deterministic=True)
            ro = env._last_realistic.copy()
            X.append(ro)
            Y.append(ta.astype(np.float32))
            if student is not None and rng.random() < beta_student:
                with torch.no_grad():
                    exe = student.policy.forward(
                        torch.as_tensor(snorm(ro))[None], deterministic=True)[0].numpy()[0]
                exe = np.clip(exe, -1, 1).astype(np.float32)
            else:
                exe = ta.astype(np.float32)
            if rng.random() < 0.20:
                exe = np.clip(exe + rng.normal(0, act_noise, 14).astype(np.float32), -1, 1)
            po, _, term, trunc, info = env.step(exe)
            done = term or trunc
        surv.append(info["t"] / EP_STEPS)
        env.close()
    return np.array(X, np.float32), np.array(Y, np.float32), float(np.mean(surv))


def bc_train(student, Xn, Y, epochs, lr=1e-3):
    Xt, Yt = torch.as_tensor(Xn), torch.as_tensor(Y)
    opt = torch.optim.Adam(student.policy.parameters(), lr=lr)
    N = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, 1024):
            idx = perm[i:i + 1024]
            pred = student.policy.forward(Xt[idx], deterministic=True)[0]
            loss = torch.mean((pred - Yt[idx]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:2d}  MSE {tot / N:.4f}")


def eval_student(student, snorm, model, speed, n=24, robust=0.0):
    sv, ds, ge, fl = [], [], [], []
    for s in range(n):
        env = BipedSim2RealEnv(model_path=model, speed_tgt=speed, privileged_obs=False,
                               imu_ctrl=True, robust=robust)
        o, _ = env.reset(seed=5000 + s)
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                a = student.policy.forward(torch.as_tensor(snorm(o))[None],
                                           deterministic=True)[0].numpy()[0]
            o, _, term, trunc, info = env.step(np.clip(a, -1, 1))
            done = term or trunc
        sv.append(info["t"] / EP_STEPS); ds.append(info["dist"])
        ge.append(info["genuine_steps"]); fl.append(int(info["fell"]))
        env.close()
    return np.mean(sv), np.mean(ds), np.mean(ge), sum(fl), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w3", default="runs/loco_w3")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--init", type=int, default=350)
    ap.add_argument("--dagger", type=int, default=3)
    ap.add_argument("--dagger-roll", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--speed", type=float, default=0.30)
    ap.add_argument("--out", default="runs/sim2real_v1")
    ap.add_argument("--net", type=int, nargs="+", default=[256, 256])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    mm, tnorm = load_w3(a.w3, a.model)

    print(f"[init] {a.init} teacher rollouts ...")
    X, Y, ts = rollouts(mm, tnorm, a.model, a.init, a.speed, seed0=0)
    print(f"  dataset {X.shape}  teacher survive {ts*100:.0f}%")

    hp = dict(n_steps=2048, batch_size=4096, n_epochs=5, lr=3e-4, ent=0.004,
              net_arch=list(a.net), log_std=-2.0, curriculum_level=3)
    json.dump(hp, open(os.path.join(a.out, "hparams.json"), "w"), indent=2)
    senv = DummyVecEnv([lambda: BipedSim2RealEnv(model_path=a.model, speed_tgt=a.speed,
                                                privileged_obs=False, imu_ctrl=True, robust=0.0)])
    student = PPO("MlpPolicy", senv, device="cpu",
                  policy_kwargs=dict(net_arch=list(a.net), log_std_init=hp["log_std"]))

    def fit_and_save(X, Y, tag):
        rms = RunningMeanStd(shape=(X.shape[1],))
        rms.update(X)
        Xn = np.clip((X - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10).astype(np.float32)
        snorm = lambda o: np.clip((o - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10).astype(np.float32)
        print(f"  BC on {len(X)} samples")
        bc_train(student, Xn, Y, a.epochs)
        torch.save(student.policy.state_dict(), os.path.join(a.out, "policy.pth"))
        torch.save(student.policy.state_dict(), os.path.join(a.out, "policy_best.pth"))
        vn = VecNormalize(senv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)
        vn.obs_rms = rms
        vn.save(os.path.join(a.out, "vecnormalize.pkl"))
        vn.save(os.path.join(a.out, "vecnormalize_best.pkl"))
        s0 = eval_student(student, snorm, a.model, a.speed, robust=0.0)
        s2 = eval_student(student, snorm, a.model, a.speed, robust=0.25)
        print(f"  [{tag}] student  robust0: survive {s0[0]*100:.0f}% dist {s0[1]:.2f} "
              f"gen {s0[2]:.0f} fell {s0[3]}/{s0[4]}   |  robust.25: survive {s2[0]*100:.0f}% "
              f"fell {s2[3]}/{s2[4]}")
        return rms, snorm, s0

    rms, snorm, best = fit_and_save(X, Y, "init")

    for it in range(a.dagger):
        beta = min(1.0, 0.5 + 0.25 * it)          # student drives more each round
        print(f"[dagger {it+1}/{a.dagger}]  student beta={beta:.2f}  {a.dagger_roll} rollouts ...")
        dx, dy, ss = rollouts(mm, tnorm, a.model, a.dagger_roll, a.speed,
                              seed0=1000 + 500 * it, student=student, snorm=snorm,
                              beta_student=beta)
        print(f"  rollout survive {ss*100:.0f}%   +{len(dx)} samples")
        X = np.concatenate([X, dx])
        Y = np.concatenate([Y, dy])
        rms, snorm, s0 = fit_and_save(X, Y, f"dagger{it+1}")

    with open(os.path.join(a.out, "best.txt"), "w") as f:
        f.write(f"distilled+DAgger  robust0 survive {best[0]*100:.0f}%\n")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
