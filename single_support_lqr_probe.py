"""Build an LQR linearised about the ACTUAL leaned single-support state
(reached by stand.K + ankle-roll bias), then swing the free leg with THAT K."""
import mujoco, numpy as np
from push_step_recovery_test import _LQRAbout, _pos_error, _smooth, AR_L_IDX, AR_R_IDX, LEG_IDX, AR_SIGN
from standing_balance_lqr import _dare
from recovery_metrics import sample_balance, _foot_normal_force, _foot_xy_z
from biped_env import DEFAULT_POSE

m = mujoco.MjModel.from_xml_path("robot/robot.xml"); d = mujoco.MjData(m)
NV = m.nv
stand = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="stand", verbose=False)
swing = "R"; si = LEG_IDX[swing]
AR = AR_SIGN[swing] * 0.12


def reach_ss_state(steps=340):
    d.qpos[:] = stand.qpos0; d.qvel[:] = stand.qvel0; mujoco.mj_forward(m, d)
    for k in range(steps):
        ar = AR * min(1.0, k / 60)
        qr = stand.qpos0.copy(); cr = stand.ctrl0.copy()
        for ci in (AR_L_IDX, AR_R_IDX):
            qr[7 + ci] += ar; cr[ci] += ar
        u = cr - stand.K @ np.concatenate([_pos_error(m, qr, d.qpos), d.qvel - stand.qvel0])
        u[AR_L_IDX] = cr[AR_L_IDX]; u[AR_R_IDX] = cr[AR_R_IDX]
        d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(m, d)
    return d.qpos.copy(), d.qvel.copy(), d.ctrl[:15].copy()


q0, v0, c0 = reach_ss_state()
bs = sample_balance(m, d)
print(f"SS state reached: up_tilt={bs.up_tilt_deg:.1f} side_lean={bs.side_lean_deg:+.1f} "
      f"Rnf={_foot_normal_force(m,d,'R'):.1f} Lnf={_foot_normal_force(m,d,'L'):.1f}")

# linearise about it
d.qpos[:] = q0; d.qvel[:] = v0; d.ctrl[:15] = c0; mujoco.mj_forward(m, d)
A = np.zeros((2*NV, 2*NV)); B = np.zeros((2*NV, 15))
mujoco.mjd_transitionFD(m, d, 1e-6, 1, A, B, None, None)
qp = np.ones(NV)*2.0; qp[0:2] = 3.0; qp[2] = 40.0; qp[3:6] = 300.0; qp[6:21] = 1.0
qv = np.ones(NV)*1.0; qv[0:3] = 6.0; qv[3:6] = 20.0; qv[6:21] = 0.4
Q = np.diag(np.concatenate([qp, qv])); R = np.diag(np.ones(15)*3.0)
Kss, _, it = _dare(A, B, Q, R)
rho = np.max(np.abs(np.linalg.eigvals(A - B @ Kss)))
print(f"SS-LQR built: DARE it={it} rho={rho:.4f}")

# hold test
d.qpos[:] = q0; d.qvel[:] = v0; mujoco.mj_forward(m, d)
for k in range(800):
    u = c0 - Kss @ np.concatenate([_pos_error(m, q0, d.qpos), d.qvel - v0])
    d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15,0], m.actuator_ctrlrange[:15,1])
    mujoco.mj_step(m, d)
bs = sample_balance(m, d)
print(f"SS-LQR hold 0.8s: up_tilt={bs.up_tilt_deg:.1f} Rnf={_foot_normal_force(m,d,'R'):.1f} "
      f"{'HELD' if bs.up_tilt_deg<25 else 'FELL'}")

# swing test: slew swing hip forward under Kss, swing leg decoupled from Kss
for hip_tgt, sr, kneepk in [(0.45,140,0.25),(0.55,140,0.25),(0.45,180,0.15),(0.6,160,0.3)]:
    d.qpos[:] = q0; d.qvel[:] = v0; mujoco.mj_forward(m, d)
    y0 = _foot_xy_z(m, d, "R")[1]; fwd=clr=0; airborne=False; fell=False
    for k in range(900):
        qr = q0.copy(); cr = c0.copy()
        w = min(1.0, k/sr); s=_smooth(w); bump=np.sin(np.pi*w)
        hp = FWD_SIGN = 1.0
        cr[si["hip"]] = hip_tgt*s; cr[si["knee"]] = -(0.06+(kneepk-0.06)*bump)
        cr[si["ankle"]] = -0.20*bump
        qr[7+si["hip"]]=cr[si["hip"]]; qr[7+si["knee"]]=cr[si["knee"]]; qr[7+si["ankle"]]=cr[si["ankle"]]
        dx = np.concatenate([_pos_error(m, qr, d.qpos), d.qvel - v0])
        for ci in (si["hip"], si["knee"], si["ankle"]):
            dx[6+ci]=0; dx[NV+6+ci]=0
        u = cr - Kss @ dx
        for ci in (si["hip"], si["knee"], si["ankle"]): u[ci]=cr[ci]
        d.ctrl[:15]=np.clip(u,m.actuator_ctrlrange[:15,0],m.actuator_ctrlrange[:15,1])
        mujoco.mj_step(m,d)
        rf=_foot_xy_z(m,d,"R"); fwd=max(fwd,-(rf[1]-y0)*1000); clr=max(clr,(rf[2]-1.0)*1000)
        b=sample_balance(m,d)
        if not b.l_contact and not b.r_contact: airborne=True
        if b.up_tilt_deg>35: fell=True; break
    b=sample_balance(m,d)
    print(f"  swing hip={hip_tgt} sr={sr} kneepk={kneepk}: fwd={fwd:.0f}mm clr={clr:.0f}mm "
          f"airborne={airborne} endUpT={b.up_tilt_deg:.0f} side={b.side_lean_deg:+.0f} {'FELL' if fell else 'ok'}")
