"""Why does the standing LQR degrade on the wide-stance variant?"""
import numpy as np, mujoco
from model_variants import load_model
from standing_balance_lqr import StandingLQR, settle_standing, _dare
from recovery_metrics import sample_balance, _foot_normal_force


def diff(m, q0, q):
    dq = np.zeros(m.nv)
    mujoco.mj_differentiatePos(m, dq, 1.0, q0, q)
    return dq


def push_test(m, d, lqr, K, push_n):
    d.qpos[:] = lqr.qpos0; d.qvel[:] = lqr.qvel0
    mujoco.mj_forward(m, d)
    fxy = push_n * np.array([0.0, -1.0])
    for k in range(3000):
        d.xfrc_applied[1, :] = 0
        if 5 <= k < 10:
            d.xfrc_applied[1, 0:2] = fxy
        u = lqr.ctrl0 - K @ np.concatenate([diff(m, lqr.qpos0, d.qpos), d.qvel - lqr.qvel0])
        d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(m, d)
        b = sample_balance(m, d)
        if b.up_tilt_deg > 50 or d.qpos[2] < 1.06:
            return f"FELL@{k}"
    b = sample_balance(m, d)
    return f"held (upT={b.up_tilt_deg:.1f})"


for v in ("baseline", "wide", "strong"):
    m = load_model(v); d = mujoco.MjData(m)
    q0, v0 = settle_standing(m, d)
    b = sample_balance(m, d)
    print(f"\n=== {v} ===  settled chest_z={d.qpos[2]:.4f} up_tilt={b.up_tilt_deg:.2f} "
          f"Lnf={_foot_normal_force(m,d,'L'):.1f} Rnf={_foot_normal_force(m,d,'R'):.1f} com_x={b.com[0]:+.4f}")
    lqr = StandingLQR(m, d, verbose=False)
    eig = np.max(np.abs(np.linalg.eigvals(lqr.A - lqr.B @ lqr.K)))
    print(f"  baseline-Q LQR: spectral_radius={eig:.4f}  120N->{push_test(m,d,lqr,lqr.K,120)}")
    for latw, oriw in [(30.0, 400.0), (30.0, 250.0), (60.0, 250.0), (100.0, 200.0)]:
        d.qpos[:] = lqr.qpos0; d.qvel[:] = lqr.qvel0; d.ctrl[:15] = lqr.ctrl0
        mujoco.mj_forward(m, d)
        A = np.zeros((2 * m.nv, 2 * m.nv)); B = np.zeros((2 * m.nv, 15))
        mujoco.mjd_transitionFD(m, d, 1e-6, 1, A, B, None, None)
        qp = np.ones(m.nv) * 2.0; qp[0] = latw; qp[1] = 3.0; qp[2] = 40.0
        qp[3:6] = oriw; qp[6:21] = 1.0
        qv = np.ones(m.nv) * 1.0; qv[0:3] = 6.0; qv[3:6] = 25.0; qv[6:21] = 0.4
        Q = np.diag(np.concatenate([qp, qv])); Rm = np.diag(np.ones(15) * 3.0)
        K, _, _ = _dare(A, B, Q, Rm)
        print(f"    Q(latw={latw:.0f},oriw={oriw:.0f}): 120N->{push_test(m,d,lqr,K,120)}  "
              f"160N->{push_test(m,d,lqr,K,160)}")
