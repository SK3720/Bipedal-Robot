"""Clean lever test: lengthen ONLY the fore-aft foot dimension (mesh-local z ->
world Y), leaving width and everything else at baseline.  Measure passive and
best-retuned-LQR forward recovery ceilings, plus lateral as a no-change check."""
import numpy as np, mujoco
from pathlib import Path
from standing_balance_lqr import settle_standing, _dare
from recovery_metrics import sample_balance, NOMINAL_CHEST_Z
from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS

BASE = Path("robot/robot.xml").read_text()


def _diff(m, q0, q):
    dq = np.zeros(m.nv); mujoco.mj_differentiatePos(m, dq, 1.0, q0, q); return dq


def build(sz=1.0):
    s = BASE
    for L in ("L", "R"):
        s = s.replace(
            f'mesh name="{L}_foot" content_type="model/stl" file="meshes/{L}_foot.stl" scale="0.001 0.001 0.001"',
            f'mesh name="{L}_foot" content_type="model/stl" file="meshes/{L}_foot.stl" scale="0.001 0.001 {0.001*sz}"')
    p = Path("robot/_lf.xml"); p.write_text(s)
    try:
        return mujoco.MjModel.from_xml_path(str(p))
    finally:
        p.unlink(missing_ok=True)


def make_K(m, d, q0, v0, oriw, faw_p, faw_v):
    d.qpos[:] = q0; d.qvel[:] = v0; d.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(m, d)
    A = np.zeros((2 * m.nv, 2 * m.nv)); B = np.zeros((2 * m.nv, 15))
    mujoco.mjd_transitionFD(m, d, 1e-6, 1, A, B, None, None)
    qp = np.ones(m.nv) * 2.0; qp[0] = 3.0; qp[1] = faw_p; qp[2] = 40.0
    qp[3:6] = oriw; qp[6:21] = 1.0
    qv = np.ones(m.nv); qv[0] = 6.0; qv[1] = faw_v; qv[2] = 6.0; qv[3:6] = 25.0; qv[6:21] = 0.4
    Q = np.diag(np.concatenate([qp, qv])); R = np.diag(np.ones(15) * 3.0)
    K, _, _ = _dare(A, B, Q, R)
    return K


def recovers(m, d, ctrl_fn, push_n, direction=-np.pi / 2):
    d.qpos[:] = _Q0; d.qvel[:] = _V0; d.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(m, d)
    fxy = push_n * np.array([np.cos(direction), np.sin(direction)])
    for k in range(3500):
        d.xfrc_applied[1, :] = 0
        if 5 <= k < 5 + PUSH_DURATION_STEPS:
            d.xfrc_applied[1, 0:2] = fxy
        d.ctrl[:15] = np.clip(ctrl_fn(m, d), m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(m, d)
        b = sample_balance(m, d)
        if b.up_tilt_deg > 50 or d.qpos[2] < NOMINAL_CHEST_Z - 0.20:
            return False
    b = sample_balance(m, d)
    return b.up_tilt_deg < 12 and abs(b.side_lean_deg) < 12 and b.l_contact and b.r_contact and b.com_speed_horiz < 0.15


def ceiling(m, d, ctrl_fn, lo=40, hi=340, direction=-np.pi / 2):
    if not recovers(m, d, ctrl_fn, lo, direction):
        return 0
    while hi - lo > 5:
        mid = (lo + hi) / 2
        if recovers(m, d, ctrl_fn, mid, direction):
            lo = mid
        else:
            hi = mid
    return lo


print(f"{'sz':>5} {'foot_len_mm':>11} {'passive_fwd':>11} {'LQR_fwd_best':>12} {'LQR_lat':>8}  argmax")
for sz in (1.0, 1.3, 1.6, 2.0, 2.5):
    m = build(sz); d = mujoco.MjData(m)
    q0, v0 = settle_standing(m, d)
    globals()["_Q0"], globals()["_V0"] = q0, v0
    # foot length
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "L_foot_collision")
    mid = m.geom_dataid[gid]; va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    loc = m.mesh_vert[va:va + vn].reshape(-1, 3)
    w = (d.geom_xmat[gid].reshape(3, 3) @ loc.T).T + d.geom_xpos[gid]
    flen = (w[:, 1].max() - w[:, 1].min()) * 1000

    p_fwd = ceiling(m, d, lambda m, d: DEFAULT_POSE.copy(), lo=20, hi=260)
    best = 0.0; arg = None
    for oriw in (300.0, 450.0):
        for fp in (3.0, 300.0):
            for fv in (10.0, 40.0):
                K = make_K(m, d, q0, v0, oriw, fp, fv)
                c = ceiling(m, d, lambda m, d, K=K: DEFAULT_POSE - K @ np.concatenate([_diff(m, q0, d.qpos), d.qvel - v0]))
                if c > best:
                    best, arg = c, (oriw, fp, fv)
    Kl = make_K(m, d, q0, v0, 400.0, 3.0, 10.0)
    lat = ceiling(m, d, lambda m, d: DEFAULT_POSE - Kl @ np.concatenate([_diff(m, q0, d.qpos), d.qvel - v0]), lo=60, hi=300, direction=0.0)
    print(f"{sz:>5.1f} {flen:>11.0f} {p_fwd:>11.0f} {best:>12.0f} {lat:>8.0f}  {arg}", flush=True)
