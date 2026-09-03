import numpy as np, mujoco
from model_variants import load_model
from standing_balance_lqr import StandingLQR, settle_standing
from recovery_metrics import sample_balance, _foot_normal_force

m = load_model("wide"); d = mujoco.MjData(m)
lqr = StandingLQR(m, d, verbose=False)
d.qpos[:] = lqr.qpos0; d.qvel[:] = lqr.qvel0
mujoco.mj_forward(m, d)
print("contacts at settled wide stance:")
for ci in range(d.ncon):
    c = d.contact[ci]
    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    print(f"  {g1} <-> {g2}  dist={c.dist:+.4f}")

fxy = 120.0 * np.array([0.0, -1.0])
print("\npush 120N trace:")
for k in range(400):
    d.xfrc_applied[1, :] = 0
    if 5 <= k < 10:
        d.xfrc_applied[1, 0:2] = fxy
    dq = np.zeros(m.nv); mujoco.mj_differentiatePos(m, dq, 1.0, lqr.qpos0, d.qpos)
    u = lqr.ctrl0 - lqr.K @ np.concatenate([dq, d.qvel - lqr.qvel0])
    d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
    mujoco.mj_step(m, d)
    b = sample_balance(m, d)
    if k % 20 == 0 or k in range(210, 230):
        R = d.xmat[1].reshape(3, 3)
        up = R @ np.array([0, 1.0, 0])
        print(f"  k={k:3d} upT={b.up_tilt_deg:5.1f} fwd_lean={b.fwd_lean_deg:+6.1f} side={b.side_lean_deg:+6.1f} "
              f"chestZ={d.qpos[2]:.3f} Lnf={_foot_normal_force(m,d,'L'):.0f} Rnf={_foot_normal_force(m,d,'R'):.0f} "
              f"ncon={d.ncon}")
    if b.up_tilt_deg > 50:
        print("  FELL"); break
