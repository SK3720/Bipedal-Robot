"""The 128N trace shows the ankle at ~13% torque while the CoM drifts off the
toes -> the LQR is not *trying* to arrest fore-aft CoM drift.  Sweep the
base fore-aft (world -Y = qpos/qvel index 1) position & velocity weights and
see how far the forward ceiling moves with the BASELINE model / actuators."""
import numpy as np, mujoco
from standing_balance_lqr import settle_standing, _dare
from recovery_metrics import sample_balance, NOMINAL_CHEST_Z
from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS

m = mujoco.MjModel.from_xml_path("robot/robot.xml")
d = mujoco.MjData(m)
q0, v0 = settle_standing(m, d)


def make_K(fa_pos, fa_vel, oriw=350.0):
    d.qpos[:] = q0; d.qvel[:] = v0; d.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(m, d)
    A = np.zeros((2 * m.nv, 2 * m.nv)); B = np.zeros((2 * m.nv, 15))
    mujoco.mjd_transitionFD(m, d, 1e-6, 1, A, B, None, None)
    qp = np.ones(m.nv) * 2.0; qp[0] = 3.0; qp[1] = fa_pos; qp[2] = 40.0
    qp[3:6] = oriw; qp[6:21] = 1.0
    qv = np.ones(m.nv) * 1.0; qv[0] = 6.0; qv[1] = fa_vel; qv[2] = 6.0
    qv[3:6] = 25.0; qv[6:21] = 0.4
    Q = np.diag(np.concatenate([qp, qv])); R = np.diag(np.ones(15) * 3.0)
    K, _, _ = _dare(A, B, Q, R)
    return K


def recovers(K, push_n):
    d.qpos[:] = q0; d.qvel[:] = v0; d.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(m, d)
    fxy = push_n * np.array([0.0, -1.0])
    for k in range(3500):
        d.xfrc_applied[1, :] = 0
        if 5 <= k < 5 + PUSH_DURATION_STEPS:
            d.xfrc_applied[1, 0:2] = fxy
        dq = np.zeros(m.nv); mujoco.mj_differentiatePos(m, dq, 1.0, q0, d.qpos)
        u = DEFAULT_POSE - K @ np.concatenate([dq, d.qvel - v0])
        d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(m, d)
        b = sample_balance(m, d)
        if b.up_tilt_deg > 50 or d.qpos[2] < NOMINAL_CHEST_Z - 0.20:
            return False
    b = sample_balance(m, d)
    return b.up_tilt_deg < 12 and abs(b.side_lean_deg) < 12 and b.l_contact and b.r_contact


def ceiling(K, lo=80, hi=320):
    if not recovers(K, lo):
        return 0
    while hi - lo > 4:
        mid = (lo + hi) / 2
        if recovers(K, mid):
            lo = mid
        else:
            hi = mid
    return lo


print(f"{'fa_pos':>7} {'fa_vel':>7} {'oriw':>6} {'fwd_ceiling_N':>13}")
for fa_pos in (3.0, 20.0, 60.0, 150.0, 400.0):
    for fa_vel in (1.0, 10.0, 40.0):
        c = ceiling(make_K(fa_pos, fa_vel))
        print(f"{fa_pos:>7.0f} {fa_vel:>7.0f} {350:>6.0f} {c:>13.0f}", flush=True)
