"""Separate investigation (user step): does the POSITION-SERVO actuator model
limit the forward recovery vs a direct TORQUE actuator with the same +/-2.3 N.m
cap?

Same LQR machinery (it linearises B by finite difference, so it adapts to
whichever actuator type is in the model).  Three leg-actuator models, arms/neck
left as position servos in all three:

  pos_kp30   : baseline  <position kp=30 forcerange +/-2.3>
  pos_kp8    : soft position servo (kp=8) - less spring-back toward DEFAULT_POSE
  torque     : <motor gear=1 ctrlrange -2.3..2.3>  - no built-in impedance

For 'torque' the LQR operating point ctrl0 is the inverse-dynamics hold torque
(qfrc_bias projected through the actuators), not a pose.
"""
import numpy as np, mujoco, re
from pathlib import Path
from standing_balance_lqr import settle_standing, _dare
from recovery_metrics import sample_balance, NOMINAL_CHEST_Z
from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS

BASE = Path("robot/robot.xml").read_text()
LEG_KEYS = ("hip", "knee", "ankle")


def build(mode):
    s = BASE
    def repl(m):
        line = m.group(0)
        if not any(k in line for k in LEG_KEYS):
            return line
        jn = re.search(r'joint="([^"]+)"', line).group(1)
        nm = re.search(r'name="([^"]+)"', line).group(1)
        cr = re.search(r'ctrlrange="([^"]+)"', line).group(1)
        if mode == "pos_kp30":
            return line
        if mode == "pos_kp8":
            return line.replace('kp="30"', 'kp="8"')
        if mode == "torque":
            return f'<motor name="{nm}" joint="{jn}" ctrlrange="-2.3 2.3" gear="1"/>'
    s = re.sub(r'<position name=[^/]*/>', repl, s)
    p = Path("robot/_pvt.xml"); p.write_text(s)
    try:
        return mujoco.MjModel.from_xml_path(str(p))
    finally:
        p.unlink(missing_ok=True)


def op_point(m, d, q0):
    """ctrl0 that holds q0 with zero accel, per actuator type."""
    d.qpos[:] = q0; d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    ctrl0 = np.zeros(15)
    for i in range(15):
        trntype = m.actuator_trntype[i]
        jid = m.actuator_trnid[i, 0]
        dofadr = m.jnt_dofadr[jid]
        gain = m.actuator_gainprm[i, 0]        # position: kp ; motor: 1
        bias1 = m.actuator_biasprm[i, 1]       # position: -kp ; motor: 0
        qfrc = d.qfrc_bias[dofadr]             # torque needed to hold (grav+cor)
        q = q0[m.jnt_qposadr[jid]]
        if m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED and bias1 != 0:
            # position servo: force = kp*ctrl + bias1*q  == qfrc  -> ctrl = (qfrc - bias1*q)/kp
            ctrl0[i] = (qfrc - bias1 * q) / gain
        else:
            ctrl0[i] = qfrc / gain             # motor
    return np.clip(ctrl0, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])


def make_K(m, d, q0, ctrl0, oriw=350.0, faw=60.0):
    d.qpos[:] = q0; d.qvel[:] = 0; d.ctrl[:15] = ctrl0
    mujoco.mj_forward(m, d)
    A = np.zeros((2 * m.nv, 2 * m.nv)); B = np.zeros((2 * m.nv, 15))
    mujoco.mjd_transitionFD(m, d, 1e-6, 1, A, B, None, None)
    qp = np.ones(m.nv) * 2.0; qp[0] = 3.0; qp[1] = faw; qp[2] = 40.0
    qp[3:6] = oriw; qp[6:21] = 1.0
    qv = np.ones(m.nv) * 1.0; qv[0:3] = 6.0; qv[1] = 10.0; qv[3:6] = 25.0; qv[6:21] = 0.4
    Q = np.diag(np.concatenate([qp, qv])); R = np.diag(np.ones(15) * 3.0)
    K, _, _ = _dare(A, B, Q, R)
    return K


def recovers(m, d, q0, ctrl0, K, push_n):
    d.qpos[:] = q0; d.qvel[:] = 0; d.ctrl[:15] = ctrl0
    mujoco.mj_forward(m, d)
    fxy = push_n * np.array([0.0, -1.0])
    for k in range(3500):
        d.xfrc_applied[1, :] = 0
        if 5 <= k < 5 + PUSH_DURATION_STEPS:
            d.xfrc_applied[1, 0:2] = fxy
        dq = np.zeros(m.nv); mujoco.mj_differentiatePos(m, dq, 1.0, q0, d.qpos)
        u = ctrl0 - K @ np.concatenate([dq, d.qvel])
        d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(m, d)
        b = sample_balance(m, d)
        if b.up_tilt_deg > 50 or d.qpos[2] < NOMINAL_CHEST_Z - 0.20:
            return False
    b = sample_balance(m, d)
    return b.up_tilt_deg < 12 and abs(b.side_lean_deg) < 12 and b.l_contact and b.r_contact


def ceiling(m, d, q0, ctrl0, K, lo=60, hi=320):
    if not recovers(m, d, q0, ctrl0, K, lo):
        return 0
    while hi - lo > 4:
        mid = (lo + hi) / 2
        if recovers(m, d, q0, ctrl0, K, mid):
            lo = mid
        else:
            hi = mid
    return lo


for mode in ("pos_kp30", "pos_kp8", "torque"):
    m = build(mode); d = mujoco.MjData(m)
    # settle: for position use the standard settle; for motor, hold with op-point torque
    if mode.startswith("pos"):
        q0, _ = settle_standing(m, d)
    else:
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [0, 0, 1.26]; d.qpos[3:7] = [0.70710678, 0.70710678, 0, 0]
        d.qpos[7:22] = DEFAULT_POSE
        for _ in range(50):
            mujoco.mj_forward(m, d)
            d.ctrl[:15] = op_point(m, d, d.qpos.copy())
            mujoco.mj_step(m, d)
        q0 = d.qpos.copy()
    ctrl0 = op_point(m, d, q0)
    best = 0; arg = None
    for oriw in (250.0, 400.0):
        for faw in (20.0, 80.0):
            K = make_K(m, d, q0, ctrl0, oriw, faw)
            c = ceiling(m, d, q0, ctrl0, K)
            if c > best:
                best, arg = c, (oriw, faw)
    print(f"{mode:<10} best_fwd_ceiling={best:>6.0f} N  (oriw,faw={arg})", flush=True)
