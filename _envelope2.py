"""Fair physical-envelope test: passive (open-loop) ceiling + LQR RE-TUNED per
variant.  Adds narrow-stance and big-feet variants since the data suggests wide
stance is the wrong direction for a recovery STEP."""
import numpy as np, mujoco
from pathlib import Path
from standing_balance_lqr import settle_standing, _dare
from recovery_metrics import sample_balance, NOMINAL_CHEST_Z
from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS

def _diff(m, q0, q):
    dq = np.zeros(m.nv); mujoco.mj_differentiatePos(m, dq, 1.0, q0, q); return dq


BASE = Path("robot/robot.xml").read_text()
EXCL = '\n  <contact>\n    <exclude body1="L_hand" body2="L_hip"/>\n    <exclude body1="R_hand" body2="R_hip"/>\n  </contact>\n'


def build(strong=False, l_hip=0.0549754, r_hip=0.0148881, foot_scale=None):
    s = BASE.replace("</worldbody>", "</worldbody>" + EXCL)
    if strong:
        import re
        s = re.sub(r'<position name=[^/]*/>',
                   lambda m: m.group(0).replace('forcerange="-2.3 2.3"', 'forcerange="-6.0 6.0"')
                   if any(k in m.group(0) for k in ("hip", "knee", "ankle")) else m.group(0), s)
    s = s.replace('name="L_hip" pos="0.0549754 -0.02575 0.0452117"', f'name="L_hip" pos="{l_hip} -0.02575 0.0452117"')
    s = s.replace('name="R_hip" pos="0.0148881 -0.02575 0.0452634"', f'name="R_hip" pos="{r_hip} -0.02575 0.0452634"')
    if foot_scale is not None:
        s = s.replace('mesh name="L_foot" content_type="model/stl" file="meshes/L_foot.stl" scale="0.001 0.001 0.001"',
                      f'mesh name="L_foot" content_type="model/stl" file="meshes/L_foot.stl" scale="{0.001*foot_scale} 0.001 {0.001*foot_scale}"')
        s = s.replace('mesh name="R_foot" content_type="model/stl" file="meshes/R_foot.stl" scale="0.001 0.001 0.001"',
                      f'mesh name="R_foot" content_type="model/stl" file="meshes/R_foot.stl" scale="{0.001*foot_scale} 0.001 {0.001*foot_scale}"')
    p = Path("robot/_e2.xml"); p.write_text(s)
    try:
        return mujoco.MjModel.from_xml_path(str(p))
    finally:
        p.unlink(missing_ok=True)


VARIANTS = {
    "baseline":   dict(),
    "strong_act": dict(strong=True),
    "wide85":     dict(l_hip=0.0775, r_hip=-0.0075),
    "narrow25":   dict(l_hip=0.0475, r_hip=0.0225),
    "bigfeet1.6": dict(foot_scale=1.6),
    "strong+bigfeet": dict(strong=True, foot_scale=1.6),
    "strong+narrow": dict(strong=True, l_hip=0.0475, r_hip=0.0225),
}


def make_K(m, d, q0, v0, latw, oriw, jointvel_fb=True):
    d.qpos[:] = q0; d.qvel[:] = v0; d.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(m, d)
    A = np.zeros((2 * m.nv, 2 * m.nv)); B = np.zeros((2 * m.nv, 15))
    mujoco.mjd_transitionFD(m, d, 1e-6, 1, A, B, None, None)
    qp = np.ones(m.nv) * 2.0; qp[0] = latw; qp[1] = 3.0; qp[2] = 40.0; qp[3:6] = oriw; qp[6:21] = 1.0
    qv = np.ones(m.nv) * 1.0; qv[0:3] = 6.0; qv[3:6] = 25.0; qv[6:21] = 0.4
    Q = np.diag(np.concatenate([qp, qv])); R = np.diag(np.ones(15) * 3.0)
    K, _, _ = _dare(A, B, Q, R)
    return K


def push_recovers(m, d, ctrl_fn, push_n, direction=-np.pi / 2, settle_ctrl=None):
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0, 0, 1.26]; d.qpos[3:7] = [0.70710678, 0.70710678, 0, 0]
    d.qpos[7:22] = DEFAULT_POSE; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    sc = settle_ctrl or ctrl_fn
    for _ in range(400):
        d.ctrl[:15] = sc(m, d); mujoco.mj_step(m, d)
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
    return (b.up_tilt_deg < 12 and abs(b.side_lean_deg) < 12 and b.l_contact and b.r_contact
            and b.com_speed_horiz < 0.15)


def ceiling(m, d, ctrl_fn, lo=60, hi=300, direction=-np.pi / 2, settle_ctrl=None):
    if not push_recovers(m, d, ctrl_fn, lo, direction, settle_ctrl):
        return lo
    while hi - lo > 5:
        mid = (lo + hi) / 2
        if push_recovers(m, d, ctrl_fn, mid, direction, settle_ctrl):
            lo = mid
        else:
            hi = mid
    return lo


def main():
  print(f"{'variant':<16} {'passive_fwd':>11} {'LQR_fwd_best':>12} {'torso_fwd':>10} {'LQR_lat_best':>12}")
  for name, kw in VARIANTS.items():
    m = build(**kw); d = mujoco.MjData(m)
    q0, v0 = settle_standing(m, d)
    servo = lambda m, d: DEFAULT_POSE.copy()
    p_fwd = ceiling(m, d, servo, lo=20, hi=200)
    best_fwd = 0.0; best_lat = 0.0
    for latw in (3.0, 40.0):
        for oriw in (250.0, 400.0):
            K = make_K(m, d, q0, v0, latw, oriw)
            ctrl = lambda m, d, K=K: DEFAULT_POSE - K @ np.concatenate(
                [_diff(m, q0, d.qpos), d.qvel - v0])
            best_fwd = max(best_fwd, ceiling(m, d, ctrl))
            best_lat = max(best_lat, ceiling(m, d, ctrl, direction=0.0))
    K = make_K(m, d, q0, v0, 3.0, 400.0)
    def torso(m, d, K=K):
        dq = _diff(m, q0, d.qpos); dq[6:21] = 0
        dv = d.qvel - v0; dv[6:21] = 0
        return DEFAULT_POSE - K @ np.concatenate([dq, dv])
    t_fwd = ceiling(m, d, torso)
    print(f"{name:<16} {p_fwd:>11.0f} {best_fwd:>12.0f} {t_fwd:>10.0f} {best_lat:>12.0f}", flush=True)


if __name__ == "__main__":
    main()
