"""At the plant instant, what does the stance actually do under different
post-plant strategies?  Runs the maneuver to plant, then for the rest:
  mode 'servo'  : hold the frozen plant pose with position servos only
  mode 'lqrfz'  : DS-LQR linearised at the frozen plant pose, hold that pose
  mode 'lqrfz_lo': same but gentle gains
Trace fwd/side lean, CoM, per-foot load.
"""
import sys
import numpy as np
import mujoco

from ilqr_recovery import RecoveryILQR, StepPlan, Weights, LEG_CTRL
from recovery_metrics import _foot_normal_force, _foot_xy_z, sample_balance
from standing_balance_lqr import _dare

MODE = sys.argv[1] if len(sys.argv) > 1 else "servo"
PUSH = float(sys.argv[2]) if len(sys.argv) > 2 else 146.0


def dslqr(s, q, u_hold, vel_w, pos_w, jw):
    A, B = s.linearize_hold(np.concatenate([q, np.zeros(s.nv)]), u_hold, 1)
    qp = np.ones(s.nv) * 2.0
    qp[0] = pos_w; qp[1] = pos_w; qp[2] = 40.0; qp[3:6] = 300.0; qp[6:21] = jw
    qv = np.ones(s.nv); qv[0] = vel_w; qv[1] = vel_w; qv[2] = 8.0
    qv[3:6] = 22.0; qv[6:21] = jw * 0.5
    Q = np.diag(np.concatenate([qp, qv])); R = np.diag(np.ones(15) * 3.0)
    K, _, _ = _dare(A, B, Q, R)
    return K


m = mujoco.MjModel.from_xml_path("robot/robot.xml")
plan = StepPlan(swing="R", push_n=PUSH, N=130, H=10)
prob = RecoveryILQR(m, plan, Weights(), verbose=False)
prob._build_ss_lqr()
s = prob.sim; d = s.d
s.set_x(prob.x0)
H = prob.H
plant_i = None
K = None
qfz = None

print(f"push {PUSH}  mode {MODE}")
for step_i in range(2200):
    if plant_i is None:
        kf = min(step_i / H, prob.N - 1e-3)
        u = prob.seed_ctrl(kf, d.qpos.copy(), d.qvel.copy())
        nfR = _foot_normal_force(m, d, "R")
        if (step_i / H) > prob.k_swing_end and nfR > 4.0:
            # confirm streak
            plant_i = step_i
            qfz = d.qpos.copy()
            uh = np.clip(qfz[7:22].copy(), s.ulo, s.uhi)
            b = sample_balance(m, d)
            print(f"PLANT @ {step_i} ms  fwdLean {b.fwd_lean_deg:+.1f} sideL {b.side_lean_deg:+.1f} "
                  f"Lnf {b.l_nf:.0f} Rnf {b.r_nf:.0f}")
            if MODE == "lqrfz":
                K = dslqr(s, qfz, uh, 60, 25, 1.5)
            elif MODE == "lqrfz_lo":
                K = dslqr(s, qfz, uh, 25, 12, 2.5)
    else:
        uh = np.clip(qfz[7:22].copy(), s.ulo, s.uhi)
        pk = step_i - plant_i
        if MODE == "servo":
            u = uh
        elif MODE == "squat":
            # drive both legs into a deep double-support squat, rolls neutral
            sw = LEG_CTRL["R"]; st = LEG_CTRL["L"]
            w = min(1.0, pk / 220.0)
            tgt = np.array(uh)
            for L in (sw, st):
                tgt[L["hip"]] = (uh[L["hip"]] if L is st else uh[L["hip"]]) * (1 - w)
                tgt[L["hip_roll"]] = uh[L["hip_roll"]] * (1 - w)
            tgt[sw["hip"]] = uh[sw["hip"]] * (1 - w) + 0.55 * w      # keep lead fwd-ish
            tgt[st["hip"]] = 0.35 * w
            tgt[sw["knee"]] = -0.85 * w + uh[sw["knee"]] * (1 - w)
            tgt[st["knee"]] = -0.85 * w + uh[st["knee"]] * (1 - w)
            tgt[sw["ankle"]] = 0.45 * w + uh[sw["ankle"]] * (1 - w)
            tgt[st["ankle"]] = 0.45 * w + uh[st["ankle"]] * (1 - w)
            tgt[9] = uh[9] * (1 - min(1.0, pk / 100.0))
            tgt[14] = uh[14] * (1 - min(1.0, pk / 100.0))
            if K is None and pk >= 220:
                qsq = d.qpos.copy(); qsq[3:7] = [0.70710678, 0.70710678, 0, 0]
                K = dslqr(s, qsq, np.clip(tgt, s.ulo, s.uhi), 40, 25, 1.5)
            u = np.clip(tgt, s.ulo, s.uhi)
            if K is not None:
                qsq = d.qpos.copy()
                dq = np.zeros(s.nv)
                mujoco.mj_differentiatePos(m, dq, 1.0, qsq, d.qpos)  # 0 - just damp vel
                u = np.clip(tgt - K @ np.concatenate([np.zeros(s.nv), d.qvel]), s.ulo, s.uhi)
        else:
            dq = np.zeros(s.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, qfz, d.qpos)
            u = np.clip(uh - K @ np.concatenate([dq, d.qvel]), s.ulo, s.uhi)
    d.ctrl[:15] = np.clip(u, s.ulo, s.uhi)
    mujoco.mj_step(m, d)

    if plant_i is not None and (step_i - plant_i) % 40 == 0:
        b = sample_balance(m, d)
        mujoco.mj_subtreeVel(m, d)
        com = d.subtree_com[1]; comv = d.subtree_linvel[1]
        print(f"  +{step_i-plant_i:4d}  fwdL {b.fwd_lean_deg:+6.1f} sideL {b.side_lean_deg:+6.1f} "
              f"CoM(x{com[0]*1000:+.0f} y{-com[1]*1000:+.0f}) v(x{comv[0]*1000:+.0f} y{-comv[1]*1000:+.0f}) "
              f"Lnf {b.l_nf:3.0f} Rnf {b.r_nf:3.0f}  chZ {b.chest_z:.3f}")
        if b.up_tilt_deg > 55:
            print("  FELL"); break
