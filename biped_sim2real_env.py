"""SIM-TO-REAL walking env -- IMU + joint-encoder observation only.

Derived from biped_locomotion_env (the loco_w3 recipe).  The changes are exactly
what a hardware transfer needs:

  OBSERVATION  -- rebuilt so every element maps 1:1 to a real driver output:
    * chest gravity vector + heading      (IMU orientation / accelerometer)
    * chest angular velocity, body frame  (gyro)
    * chest specific force, body frame     (accelerometer, incl. gravity reaction)
    * 14 actuated joint positions + vels   (encoders)
    * 2 foot-contact booleans              (contact switches)
    * base linear velocity estimate        (LEG ODOMETRY: -d/dt of the stance
                                            foot position in the base frame, from
                                            FK -- exactly what a real robot runs)
    * foot positions in the base frame     (FK from encoders)
    * gait clock, speed command, prev act  (internal)
    ...noised to the real sensors' characteristics, and STACKED over the last
    HIST frames so the policy can infer velocity from the temporal signal.
    DROPPED (privileged): true CoM velocity, absolute base height, world-frame
    foot positions, contact-force magnitudes.

  BASE CONTROLLER -- reduced to what runs from IMU + encoders:
    * CPG reference (internal clock)                       -- unchanged
    * attitude hold: the StandingLQR with the HEIGHT terms masked out, so only
      orientation + angular-rate errors (both IMU-observable) drive it
    * frontal stabiliser: chest-roll + roll-rate PD on ankle-roll / loaded-leg
      hip-roll (was a CoM-lateral + contact-force CoP law -- privileged)

  DOMAIN RANDOMISATION -- moderate, fixed (the maximal curriculum broke
    runs/robust_r1): friction / mass / gain / damping spread + 1-2 step latency
    + random pushes (via the parent `robust` machinery at 0.5) plus IMU bias and
    encoder noise applied here.

  python biped_sim2real_env.py --smoke                 # zero-action base
  python biped_sim2real_env.py --smoke --policy runs/sim2real_v1
  python biped_sim2real_env.py --watch  --policy runs/sim2real_v1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import mujoco
from gymnasium import spaces

from biped_locomotion_env import (BipedLocomotionEnv, DEFAULT_MODEL, CHEST_BODY,
                                  ACT_CTRL, LEG, NECK, CPG_GAIN, WALK_LEAN,
                                  SPEED_TGT_DEFAULT, G, _foot_normal_force)

HIST = 4                      # sensor frames stacked into the observation
S2R_ROBUST = 0.15            # DR strength while first learning the IMU-obs walk;
                             # a phase-2 run ramps this up once the gait is solid

# real-sensor noise (1-sigma), applied every control step
IMU_TILT_NOISE = 0.015       # rad  on the gravity-vector direction
GYRO_NOISE = 0.02           # rad/s
GYRO_BIAS = 0.03            # rad/s, constant per episode
ACC_NOISE = 0.35           # m/s^2
ENC_POS_NOISE = 0.004      # rad
ENC_VEL_NOISE = 0.06       # rad/s
V_EST_NOISE = 0.03        # m/s  on the leg-odometry velocity estimate
CONTACT_DROPOUT = 0.03     # prob a contact switch reads wrong

# frontal stabiliser -- w3's CoP law, rebuilt from DEPLOYABLE inputs:
#   w3:  err = com_x - (load-weighted foot world-x);   ar = -(KX*err + KV*com_vx)
#   s2r: err = -(mean base-frame lateral offset of the contacting feet)   [FK]
#        com_vx  ->  v_est[x]   [leg odometry]
# gains match biped_locomotion_env's FR_* so the behaviour is ~identical.
FR_KX, FR_KV = 3.2, 0.9
FR_AR_MAX = 0.28
FR_HR_KX, FR_HR_KV, FR_HR_MAX = 2.6, 0.5, 0.34


class BipedSim2RealEnv(BipedLocomotionEnv):
    def __init__(self, model_path=DEFAULT_MODEL, seed=None, render_mode=None,
                 speed_tgt=SPEED_TGT_DEFAULT, robust=S2R_ROBUST,
                 privileged_obs=False, imu_ctrl=True, **kw):
        # privileged_obs / imu_ctrl -- diagnostic knobs to isolate whether the
        # obs or the base controller is the bottleneck.
        self._privileged_obs = bool(privileged_obs)
        self._imu_ctrl = bool(imu_ctrl)
        # state used by _build_obs / _compose_ctrl -- must exist before the parent
        # __init__ probes the observation with _build_obs().
        self._prev_cvel = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        self._hist = None
        self._sframe_n = None
        self._prev_foot_rel = {"L": np.zeros(3), "R": np.zeros(3)}
        self._v_est = np.zeros(3)
        super().__init__(model_path=model_path, seed=None, render_mode=render_mode,
                         speed_tgt=speed_tgt, robust=robust, **kw)
        # height terms OFF in the attitude hold: only orientation (idx 3,4,5) +
        # angular rates are IMU-observable.
        if self._imu_ctrl:
            nv = self.model.nv
            self._att_mask[:] = 0.0
            for i in (3, 4, 5):
                self._att_mask[i] = 1.0
                self._att_mask[nv + i] = 1.0
        if seed is not None:
            self.reset(seed=seed)

    # -------------------------------------------------- lifecycle
    def reset(self, seed=None, options=None):
        self._hist = None
        self._prev_cvel = np.zeros(3)
        self._v_est = np.zeros(3)
        obs, info = super().reset(seed=seed, options=options)
        d = self.data
        R = d.xmat[CHEST_BODY].reshape(3, 3)
        base = d.xpos[CHEST_BODY]
        for s, bid in (("L", self._b_lf), ("R", self._b_rf)):
            self._prev_foot_rel[s] = R.T @ (d.xpos[bid] - base)
        self._gyro_bias = self.np_random.normal(0.0, GYRO_BIAS, size=3) * (self.robust > 0)
        return self._obs(), info

    # -------------------------------------------------- sensors
    def _sensor_frame(self):
        """One frame of realistic sensor data (pre-history)."""
        m, d = self.model, self.data
        R = d.xmat[CHEST_BODY].reshape(3, 3)
        rng = self.np_random
        cdt = m.opt.timestep * 5.0

        up_w, fwd_w = self._chest_axes()                 # world-frame chest axes
        up = up_w + rng.normal(0.0, IMU_TILT_NOISE, 3)
        fwd = fwd_w[:2] + rng.normal(0.0, IMU_TILT_NOISE, 2)

        gyro = R.T @ d.qvel[3:6] + self._gyro_bias + rng.normal(0.0, GYRO_NOISE, 3)

        cvel = d.cvel[CHEST_BODY][3:6].copy()            # world lin-vel of chest
        a_world = (cvel - self._prev_cvel) / cdt
        spec_force = R.T @ (a_world + np.array([0.0, 0.0, G]))   # accelerometer
        spec_force = spec_force + rng.normal(0.0, ACC_NOISE, 3)

        jp = d.qpos[self._leg_qadr] + rng.normal(0.0, ENC_POS_NOISE, 14)
        jv = d.qvel[self._leg_vadr] + rng.normal(0.0, ENC_VEL_NOISE, 14)

        lc_raw = _foot_normal_force(m, d, "L") > 6.0
        rc_raw = _foot_normal_force(m, d, "R") > 6.0
        lc = (not lc_raw) if rng.random() < CONTACT_DROPOUT else lc_raw
        rc = (not rc_raw) if rng.random() < CONTACT_DROPOUT else rc_raw

        # FK: foot positions in the base frame (deployable from encoders)
        base = d.xpos[CHEST_BODY]
        fr = {s: R.T @ (d.xpos[bid] - base) for s, bid in (("L", self._b_lf), ("R", self._b_rf))}

        # LEG ODOMETRY: a planted foot is ~stationary on the ground, so the base
        # velocity ~ -d/dt of that foot's position in the base frame.  Blend the
        # contacting feet; hold (decayed) when both are airborne.
        ests, w = [], []
        for s, con in (("L", lc_raw), ("R", rc_raw)):
            if con:
                ests.append(-(fr[s] - self._prev_foot_rel[s]) / cdt)
                w.append(1.0)
        if ests:
            self._v_est = np.average(ests, axis=0, weights=w)
        else:
            self._v_est = self._v_est * 0.6
        self._prev_foot_rel["L"] = fr["L"].copy()
        self._prev_foot_rel["R"] = fr["R"].copy()
        v_est = self._v_est + rng.normal(0.0, V_EST_NOISE, 3)

        return np.concatenate([
            up, fwd,                                     # 5
            gyro, spec_force,                            # 6
            jp, jv,                                      # 28
            [float(lc), float(rc)],                      # 2
            v_est,                                       # 3  base-vel estimate
            fr["L"], fr["R"],                            # 6  foot pos in base frame
        ]).astype(np.float32)                            # 50

    def _realistic_obs(self):
        """The IMU + encoder observation, ALWAYS (ignores privileged_obs).  Used
        by the teacher-student distillation to pair with the expert's action."""
        f = self._sensor_frame()
        if self._sframe_n is None:
            self._sframe_n = f.shape[0]
        if self._hist is None:
            self._hist = [f.copy() for _ in range(HIST)]
        else:
            self._hist.append(f)
            self._hist.pop(0)
        clk = np.array([np.sin(self._phase), np.cos(self._phase)], np.float32)
        extra = np.concatenate([clk, [self.speed_tgt], self._prev_action]).astype(np.float32)
        return np.concatenate(self._hist + [extra]).astype(np.float32)

    def _build_obs(self):
        if self._privileged_obs:
            # still advance the sensor history so _realistic_obs stays consistent
            self._last_realistic = self._realistic_obs()
            return super()._build_obs()
        return self._realistic_obs()

    def _obs(self):
        return self._build_obs().astype(np.float32)

    # -------------------------------------------------- control (IMU + encoders)
    def _compose_ctrl(self, residual, settle=False):
        if not self._imu_ctrl:
            return super()._compose_ctrl(residual, settle)
        m, d = self.model, self.data
        nv = m.nv

        # ---- CPG reference (internal clock) ----
        u = np.array(self._stand.ctrl0, float)
        if not settle and CPG_GAIN > 0.0:
            from biped_locomotion_env import (LAT_ROLL, LAT_HR, ARM, SH_FWD,
                                              ARM_AMP, ELB_FLEX)
            g = CPG_GAIN
            lat = np.sin(self._phase)
            for side in ("L", "R"):
                ph = (self._phase / (2.0 * np.pi) + (0.0 if side == "L" else 0.5)) % 1.0
                hp, kn, ap = self._leg_ref(side, ph)
                sgn = 1.0 if side == "L" else -1.0
                u[LEG[side]["hp"]] += g * (hp - u[LEG[side]["hp"]])
                u[LEG[side]["kn"]] += g * (kn - u[LEG[side]["kn"]])
                u[LEG[side]["ap"]] += g * (ap - u[LEG[side]["ap"]])
                u[LEG[side]["ar"]] += g * sgn * LAT_ROLL * lat
                u[LEG[side]["hr"]] += g * sgn * LAT_HR * lat
                oph = (self._phase / (2.0 * np.pi) + (0.5 if side == "L" else 0.0)) % 1.0
                u[ARM[side]["sh"]] += g * SH_FWD[side] * ARM_AMP * np.sin(2.0 * np.pi * oph)
                u[ARM[side]["el"]] += g * (1.0 if side == "L" else -1.0) * ELB_FLEX

        # ---- attitude hold: LQR with HEIGHT masked out (orientation + rates only,
        #      both IMU-observable) ----
        dq = np.zeros(nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, self._stand.qpos0, d.qpos)
        if not settle:
            dq[3] -= WALK_LEAN
        dx = np.concatenate([dq, d.qvel - self._stand.qvel0]) * self._att_mask
        u = u + (-(self._stand.K @ dx))

        # ---- frontal stabiliser: w3's CoP law from FK + leg-odometry inputs ----
        if not settle:
            R = d.xmat[CHEST_BODY].reshape(3, 3)
            base = d.xpos[CHEST_BODY]
            fx, nload = 0.0, 0
            loaded = {"L": False, "R": False}
            for side, bid in (("L", self._b_lf), ("R", self._b_rf)):
                if _foot_normal_force(m, d, side) > 12.0:      # contact switch (preloaded)
                    fx += float((R.T @ (d.xpos[bid] - base))[0])
                    nload += 1
                    loaded[side] = True
            err = -(fx / nload) if nload else 0.0              # CoM(≈base) lateral vs CoP
            vx = float(self._v_est[0])                         # leg-odometry lateral vel
            ar = float(np.clip(-(FR_KX * err + FR_KV * vx) / 6.0, -FR_AR_MAX, FR_AR_MAX))
            u[LEG["L"]["ar"]] += ar
            u[LEG["R"]["ar"]] += ar
            hr = float(np.clip(-(FR_HR_KX * err + FR_HR_KV * vx), -FR_HR_MAX, FR_HR_MAX))
            for side in ("L", "R"):
                if loaded[side]:
                    u[LEG[side]["hr"]] += hr

        u[ACT_CTRL] = u[ACT_CTRL] + residual
        u[NECK] = 0.0
        return np.clip(u, self.clow, self.chigh)

    # keep _prev_cvel current for the accelerometer finite-difference
    def step(self, action):
        self._prev_cvel = self.data.cvel[CHEST_BODY][3:6].copy()
        return super().step(action)


# ============================================================ CLI
def _load_policy(run_dir, model=DEFAULT_MODEL):
    import json
    import pickle
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    hp = json.load(open(f"{run_dir}/hparams.json"))
    tmp = DummyVecEnv([lambda: BipedSim2RealEnv(model_path=model)])
    pf = (f"{run_dir}/policy_best.pth" if os.path.exists(f"{run_dir}/policy_best.pth")
          else f"{run_dir}/policy.pth")
    vf = (f"{run_dir}/vecnormalize_best.pkl" if os.path.exists(f"{run_dir}/vecnormalize_best.pkl")
          else f"{run_dir}/vecnormalize.pkl")
    mm = PPO("MlpPolicy", tmp, device="cpu",
             policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(pf, map_location="cpu", weights_only=True))
    mm.policy.eval()
    mean, var = 0.0, 1.0
    if os.path.exists(vf):
        v = pickle.load(open(vf, "rb"))
        mean, var = v.obs_rms.mean, v.obs_rms.var
    print(f"  loaded {pf}")
    return mm, (lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32))


def _smoke(n=10, render=None, policy=None, model=DEFAULT_MODEL, speed=0.30):
    pol = _load_policy(policy, model) if policy else None
    env = BipedSim2RealEnv(model_path=model, speed_tgt=speed,
                           render_mode=("rgb_array" if render else None))
    print(f"  obs dim {env.observation_space.shape[0]}")
    frames = []
    sv, ds, ge = [], [], []
    for i in range(n):
        o, _ = env.reset(seed=100 + i)
        done = False
        info = {}
        ep = []
        while not done:
            if pol:
                import torch
                with torch.no_grad():
                    a, _ = pol[0].predict(pol[1](o), deterministic=True)
            else:
                a = np.zeros(14, np.float32)
            o, r, term, trunc, info = env.step(a)
            done = term or trunc
            if render and i < 4:
                ep.append(env.render())
        if render and i < 4:
            frames.append(ep)
        sv.append(info["t"] / 1400)
        ds.append(info["dist"])
        ge.append(info["genuine_steps"])
        print(f"  ep {i:2d}  survive {info['t']/1400*100:3.0f}%  dist {info['dist']:+.2f}  "
              f"genuine {info['genuine_steps']:2d}  {'FELL' if info['fell'] else ''}")
    print(f"  --- mean survive {np.mean(sv)*100:.0f}%  dist {np.mean(ds):+.2f}  "
          f"genuine {np.mean(ge):.0f}")
    if render and frames:
        import imageio.v2 as imageio
        H = max(len(f) for f in frames)
        tiles = [np.hstack([f[min(t, len(f) - 1)] for f in frames][:2]) for t in range(H)]
        imageio.mimsave(render, tiles, fps=40, macro_block_size=1)
        print(f"  wrote {render}")
    env.close()


def _watch(policy, model, speed, n=6):
    import json
    import pickle
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    hp = json.load(open(f"{policy}/hparams.json"))
    pf = (f"{policy}/policy_best.pth" if os.path.exists(f"{policy}/policy_best.pth")
          else f"{policy}/policy.pth")
    vf = (f"{policy}/vecnormalize_best.pkl" if os.path.exists(f"{policy}/vecnormalize_best.pkl")
          else f"{policy}/vecnormalize.pkl")
    tmp = DummyVecEnv([lambda: BipedSim2RealEnv(model_path=model, speed_tgt=speed)])
    mm = PPO("MlpPolicy", tmp, device="cpu",
             policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(pf, map_location="cpu", weights_only=True))
    mm.policy.eval()
    v = pickle.load(open(vf, "rb"))
    mean, var = v.obs_rms.mean, v.obs_rms.var
    env = BipedSim2RealEnv(model_path=model, speed_tgt=speed, render_mode="human")
    for i in range(n):
        o, _ = env.reset(seed=200 + i)
        done = False
        info = {}
        try:
            while not done:
                ob = np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
                with torch.no_grad():
                    a, _ = mm.predict(ob, deterministic=True)
                o, r, term, trunc, info = env.step(a)
                done = term or trunc
        except KeyboardInterrupt:
            break
        print(f"  run {i}: dist {info.get('dist', 0):+.2f}  "
              f"{'FELL' if info.get('fell') else 'ok'} @ {info.get('t', 0)}")
    env.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--render", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--speed", type=float, default=0.30)
    a = ap.parse_args(argv)
    if a.watch:
        _watch(a.policy, a.model, a.speed, a.n)
    elif a.smoke:
        _smoke(a.n, a.render, a.policy, a.model, a.speed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
