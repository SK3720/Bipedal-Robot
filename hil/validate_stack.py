"""End-to-end check: the HIL reimplementation == biped_sim2real_env.

Runs the real env with ALL sensor noise disabled and, at every control step,
rebuilds the observation / action / base command with the hil.* classes from the
same underlying state, then compares.  Run this FIRST tomorrow -- if it passes,
the software side of the HIL stack is proven and only the hardware wiring is
untested.

  python -m hil.validate_stack
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import biped_sim2real_env as S2R            # noqa: E402
from hil import spec                        # noqa: E402
from hil.observation import ObservationBuilder, quat_to_mat  # noqa: E402
from hil.base_controller import BaseController               # noqa: E402
from hil.policy import Policy                                # noqa: E402

G = 9.81


def _mat_to_quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def _sensor_components(env):
    """The raw per-frame sensor signals, computed from env.data the way a real
    (noise-free) robot would report them -- i.e. the inputs hil.* expects.
    Uses d.xmat[1] (what env._sensor_frame uses) so validation matches tightly."""
    d = env.data
    R = d.xmat[1].reshape(3, 3)                  # body 1 = Chest (as env._sensor_frame does)
    q = _mat_to_quat(R)                          # chest quat wxyz (body->world)
    gyro_body = d.qvel[3:6].copy()               # body-frame angular velocity
    cvel = d.cvel[1][3:6].copy()
    a_world = (cvel - env._prev_cvel) / (env.model.opt.timestep * 5.0)
    accel_body = R.T @ (a_world + np.array([0.0, 0.0, G]))   # specific force, body frame
    qadr = np.array([7 + i for i in spec.ACT_CTRL])
    vadr = np.array([6 + i for i in spec.ACT_CTRL])
    jp = d.qpos[qadr].copy()
    jv = d.qvel[vadr].copy()
    from biped_locomotion_env import _foot_normal_force
    fL = _foot_normal_force(env.model, d, "L")
    fR = _foot_normal_force(env.model, d, "R")
    return q, gyro_body, accel_body, jp, jv, bool(fL > 6.0), bool(fR > 6.0), fL, fR


def _exact_state_checks(env, pol_np, bc):
    """No integration: set one arbitrary fresh state and compare obs / cmd
    element-for-element against the env computing from the SAME state."""
    import mujoco
    from hil.observation import ObservationBuilder
    from hil.kinematics import Kinematics
    from biped_locomotion_env import _foot_normal_force
    d = env.data
    d.qpos[:] = env._stand.qpos0
    d.qpos[13] += 0.15; d.qpos[18] -= 0.12; d.qpos[3:7] = [0.706, 0.700, 0.05, -0.03]
    d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
    d.qvel[:] = 0.0; d.qvel[3:6] = [0.2, -0.1, 0.05]
    mujoco.mj_forward(env.model, d)
    R = d.xmat[1].reshape(3, 3); base = d.xpos[1]
    q = _mat_to_quat(R); gyro = d.qvel[3:6].copy()
    accel = R.T @ np.array([0.0, 0.0, G])
    qadr = np.array([7 + i for i in spec.ACT_CTRL]); vadr = np.array([6 + i for i in spec.ACT_CTRL])
    jp, jv = d.qpos[qadr].copy(), d.qvel[vadr].copy()
    fL, fR = _foot_normal_force(env.model, d, "L"), _foot_normal_force(env.model, d, "R")
    frL = R.T @ (d.xpos[env._b_lf] - base); frR = R.T @ (d.xpos[env._b_rf] - base)

    kin = Kinematics(); kin.prime(jp)
    fk_err = max(np.max(np.abs(kin.foot_L_rel - frL)), np.max(np.abs(kin.foot_R_rel - frR)))

    env._phase = 0.42; env._v_est[:] = [0.08, -0.03, 0.01]
    for s, bid in (("L", env._b_lf), ("R", env._b_rf)):
        env._prev_foot_rel[s] = R.T @ (d.xpos[bid] - base)
    a = np.array([0.3, -0.2, 0.1, 0.5, -0.4, 0.2, -0.1, 0.3, -0.5, 0.4, 0.1, -0.2, 0.3, -0.1])
    u_env = np.array(env._compose_ctrl(a * spec.ACT_SCALE), float)
    u_hil = bc.compose(a, env._phase, q, gyro, frL, frR, env._v_est, fL > 6, fR > 6,
                       foot_load_L=fL, foot_load_R=fR)
    cmd_err = np.max(np.abs(u_hil - u_env))

    # policy numpy vs sb3 already covered in the rolling loop; here just the mlp
    return [
        ("FK foot-in-base", fk_err, 1e-4, "hil.kinematics vs MuJoCo full-state FK"),
        ("base command",    cmd_err, 1e-6, "hil.base_controller vs env._compose_ctrl"),
    ]


def main():
    # kill every stochastic term in the env
    S2R.IMU_TILT_NOISE = S2R.GYRO_NOISE = S2R.GYRO_BIAS = 0.0
    S2R.ACC_NOISE = S2R.ENC_POS_NOISE = S2R.ENC_VEL_NOISE = 0.0
    S2R.V_EST_NOISE = 0.0
    S2R.CONTACT_DROPOUT = 0.0

    env = S2R.BipedSim2RealEnv(privileged_obs=False, imu_ctrl=True, robust=0.0,
                               speed_tgt=spec.SPEED_TGT)
    # deterministic reset from the exact standing pose, phase 0
    import mujoco
    env.reset(seed=0)
    d = env.data
    d.qpos[:] = env._stand.qpos0
    d.qvel[:] = 0.0
    mujoco.mj_forward(env.model, d)
    env._phase = 0.0
    env._hist = None
    env._prev_cvel = d.cvel[1][3:6].copy()
    R = d.xmat[1].reshape(3, 3)
    base = d.xpos[1]
    for s2, bid in (("L", env._b_lf), ("R", env._b_rf)):
        env._prev_foot_rel[s2] = R.T @ (d.xpos[bid] - base)
    env._v_est[:] = 0.0

    pol_np = Policy()
    bc = BaseController()
    ob = ObservationBuilder(control_dt=env.model.opt.timestep * 5.0)

    # sb3 reference policy
    import json
    import pickle
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    hp = json.load(open(os.path.join(spec.POLICY_DIR, "hparams.json")))
    mm = PPO("MlpPolicy", DummyVecEnv([lambda: S2R.BipedSim2RealEnv(privileged_obs=False)]),
             device="cpu", policy_kwargs=dict(net_arch=hp["net_arch"], log_std_init=hp["log_std"]))
    mm.policy.load_state_dict(torch.load(spec.POLICY_FILE, map_location="cpu", weights_only=True))
    mm.policy.eval()
    vn = pickle.load(open(spec.VECNORM_FILE, "rb"))
    sb_norm = lambda o: np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8), -10, 10)

    VE = slice(spec.SFRAME_DIM * (spec.HIST - 1) + 41,
               spec.SFRAME_DIM * (spec.HIST - 1) + 44)   # v_est in the newest frame
    ve_mask = np.ones(spec.OBS_DIM, bool)
    for h in range(spec.HIST):
        ve_mask[spec.SFRAME_DIM * h + 41: spec.SFRAME_DIM * h + 44] = False

    env_obs = env._obs()
    q, g, a, jp, jv, cl, cr, fL, fR = _sensor_components(env)
    hil_obs = ob.reset(q, g, a, jp, jv, cl, cr, phase0=env._phase)

    errs = dict(obs_novest=[], v_est=[], act_np_vs_sb=[], act_np_vs_env=[], cmd=[])
    N = 400
    for k in range(N):
        errs["obs_novest"].append(np.max(np.abs((hil_obs - env_obs)[ve_mask])))
        errs["v_est"].append(np.max(np.abs((hil_obs - env_obs)[~ve_mask])))

        a_np = pol_np.act(env_obs)
        with torch.no_grad():
            a_sb = mm.policy.forward(torch.as_tensor(sb_norm(env_obs)[None], dtype=torch.float32),
                                     deterministic=True)[0].numpy()[0]
        a_sb = np.clip(a_sb, -1, 1)
        errs["act_np_vs_sb"].append(np.max(np.abs(a_np - a_sb)))
        errs["act_np_vs_env"].append(np.max(np.abs(pol_np.act(hil_obs) - a_np)))

        residual = a_sb * spec.ACT_SCALE
        u_env = np.array(env._compose_ctrl(residual), float)
        u_hil = bc.compose(a_sb, env._phase, q, g,
                           ob.kin.foot_L_rel, ob.kin.foot_R_rel, ob.kin.v_est, cl, cr,
                           foot_load_L=fL, foot_load_R=fR)
        errs["cmd"].append(np.max(np.abs(u_hil - u_env)))

        env._prev_cvel = d.cvel[1][3:6].copy()
        o, r, term, trunc, info = env.step(a_sb)
        env_obs = o
        q, g, a, jp, jv, cl, cr, fL, fR = _sensor_components(env)
        hil_obs = ob.step(q, g, a, jp, jv, cl, cr, a_sb)
        if term or trunc:
            print(f"  (env ended at step {k}: {'FELL' if info['fell'] else 'timeout'})")
            break

    # ---- fresh-state exact checks (no MuJoCo integration staleness) ----
    exact = _exact_state_checks(env, pol_np, bc)

    print(f"\n--- exact checks (single fresh state) ---")
    ok = True
    for name, mx, tol, note in exact:
        flag = "OK " if mx <= tol else "!! "
        ok &= mx <= tol
        print(f"  {flag}{name:16s} {mx:.2e}  (tol {tol:.0e})  -- {note}")

    print(f"\n--- rolling checks ({len(errs['obs_novest'])} steps, closed-loop walk) ---")
    for name, tol, note in (
        ("obs_novest",   2e-3, "observation, all dims except v_est"),
        ("v_est",        1.2e-1, "leg-odometry vel: FK-timing gap, < ~2.5 sigma of the 0.03 train noise"),
        ("act_np_vs_sb", 1e-4, "numpy policy vs SB3 policy (same obs)"),
        ("act_np_vs_env",8e-2, "policy(hil_obs) vs policy(env_obs)"),
        ("cmd",          3e-2, "base command: <=1e-15 on a fresh state; this is MuJoCo xmat/qpos 1-substep staleness (absent on hardware)"),
    ):
        e = np.asarray(errs[name])
        mx = e.max() if len(e) else float("nan")
        flag = "OK " if mx <= tol else "!! "
        ok &= mx <= tol
        print(f"  {flag}{name:14s} max {mx:.2e}  mean {e.mean():.2e}  (tol {tol:.0e})  -- {note}")
    print("\nRESULT:", "PASS -- HIL software stack matches the sim." if ok
          else "FAIL -- see the !! rows above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
