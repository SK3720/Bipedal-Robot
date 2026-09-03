"""Trace EXACTLY what happens at the plant -> squat-catch transition in
postplant_recovery, to check whether the 'squat catch' is a legitimate,
physically-continuous recovery or an artifact (false plant / discontinuous
target / new forced configuration).

Logs every ms from 150 ms before plant to 500 ms after:
  phase, pk
  swing/stance leg: hip / knee / ankle  ACTUAL qpos  vs  COMMANDED ctrl target
  both feet: body z, sole-min z (real ground clearance), contact, normal force
  CoM, up_tilt / fwd_lean / side_lean
  |ctrl - prev_ctrl|  (command discontinuity)
"""
import numpy as np
import mujoco

from ilqr_recovery import RecoveryILQR, StepPlan, Weights, LEG_CTRL, FWD_HIP_SIGN, KNEE_FLEX_SIGN
from recovery_metrics import _foot_normal_force, _foot_xy_z, sample_balance
from postplant_recovery import _squat_targets, _squat_lqr

PUSH = 146.0
SWING = "R"
SQUAT_SETTLE = 260


def sole_min_z(m, d, side):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    mid = m.geom_dataid[gid]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    loc = m.mesh_vert[va:va + vn].reshape(-1, 3)
    w = (d.geom_xmat[gid].reshape(3, 3) @ loc.T).T + d.geom_xpos[gid]
    return w[:, 2].min()


def main():
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    plan = StepPlan(swing=SWING, push_n=PUSH, N=130, H=10)
    prob = RecoveryILQR(m, plan, Weights(), verbose=False)
    prob._build_ss_lqr()
    s = prob.sim
    d = s.d
    s.set_x(prob.x0)
    sw = LEG_CTRL[SWING]
    st = LEG_CTRL["L" if SWING == "R" else "R"]
    H = prob.H

    swq = (7 + sw["hip"], 7 + sw["knee"], 7 + sw["ankle"])   # qpos idx (ctrl idx == qpos-7 here)
    stq = (7 + st["hip"], 7 + st["knee"], 7 + st["ankle"])

    phase = "maneuver"
    plant_step = None
    K_plant = None
    q_sq_ref = None
    u_sq_ref = np.zeros(15)
    qj_freeze = None
    sq_locked = False
    prev_u = np.zeros(15)
    log = []

    for step_i in range(1600):
        b = sample_balance(m, d)
        nf = _foot_normal_force(m, d, SWING)
        if phase == "maneuver":
            kf = min(step_i / H, prob.N - 1e-3)
            u = prob.seed_ctrl(kf, d.qpos.copy(), d.qvel.copy())
        else:
            pk = step_i - plant_step
            u_sq = _squat_targets(qj_freeze, SWING, pk)
            if pk >= SQUAT_SETTLE and not sq_locked:
                q_sq_ref = d.qpos.copy(); q_sq_ref[3:7] = [0.70710678, 0.70710678, 0, 0]
                u_sq_ref = u_sq.copy()
                K_plant = _squat_lqr(s, q_sq_ref, u_sq_ref, verbose=False)
                sq_locked = True
            if not sq_locked:
                u = u_sq
            else:
                u = u_sq_ref - K_plant @ np.concatenate([np.zeros(s.nv), d.qvel])
        u = np.clip(u, s.ulo, s.uhi)
        d.ctrl[:15] = u
        mujoco.mj_step(m, d)

        # plant detection (same as postplant_recovery)
        if phase == "maneuver" and (step_i / H) > prob.k_swing_end:
            sc = getattr(b, f"{SWING.lower()}_contact")
            if (sc and nf > 4.0):
                plant_step = step_i
                qj_freeze = np.clip(d.qpos[7:22].copy(), s.ulo, s.uhi)
                phase = "postplant"

        du = float(np.linalg.norm(u - prev_u))
        prev_u = u.copy()
        pk = (step_i - plant_step) if plant_step is not None else None
        rec = dict(
            t=step_i, phase=phase[:4], pk=pk,
            swH=d.qpos[swq[0]], swK=d.qpos[swq[1]], swA=d.qpos[swq[2]],
            swHt=u[sw["hip"]], swKt=u[sw["knee"]], swAt=u[sw["ankle"]],
            stH=d.qpos[stq[0]], stK=d.qpos[stq[1]], stKt=u[st["knee"]],
            swFz=(_foot_xy_z(m, d, SWING)[2] - 1) * 1000, swSole=(sole_min_z(m, d, SWING) - 1) * 1000,
            stSole=(sole_min_z(m, d, "L" if SWING == "R" else "R") - 1) * 1000,
            swNf=_foot_normal_force(m, d, SWING), stNf=_foot_normal_force(m, d, "L" if SWING == "R" else "R"),
            swFy=-_foot_xy_z(m, d, SWING)[1] * 1000, stFy=-_foot_xy_z(m, d, "L" if SWING == "R" else "R")[1] * 1000,
            upT=b.up_tilt_deg, fwd=b.fwd_lean_deg, side=b.side_lean_deg, du=du,
        )
        log.append(rec)
        if b.up_tilt_deg > 55:
            log.append(dict(t=step_i, phase="FELL"))
            break

    p0 = plant_step
    print(f"plant detected @ {p0} ms")
    print(f"{'t':>4} {'ph':>4} {'pk':>4} | swing qpos H/K/A    swing tgt H/K/A   | "
          f"{'swSole':>6} {'swNf':>5} {'swFy':>5} | {'stSole':>6} {'stNf':>5} | "
          f"{'upT':>4} {'fwd':>5} {'side':>5} {'|du|':>5}")
    for r in log:
        if r.get("phase") == "FELL":
            print(f"  --- FELL @ {r['t']} ---"); continue
        if p0 is None or not (p0 - 160 <= r["t"] <= p0 + 520):
            continue
        if r["t"] % 4 and not (p0 <= r["t"] <= p0 + 30):
            continue
        pk = f"{r['pk']:>4}" if r["pk"] is not None else "   -"
        print(f"{r['t']:>4} {r['phase']:>4} {pk} | "
              f"{r['swH']:+5.2f}/{r['swK']:+5.2f}/{r['swA']:+5.2f}  "
              f"{r['swHt']:+5.2f}/{r['swKt']:+5.2f}/{r['swAt']:+5.2f} | "
              f"{r['swSole']:>6.0f} {r['swNf']:>5.0f} {r['swFy']:>5.0f} | "
              f"{r['stSole']:>6.0f} {r['stNf']:>5.0f} | "
              f"{r['upT']:>4.0f} {r['fwd']:>+5.0f} {r['side']:>+5.0f} {r['du']:>5.2f}")


if __name__ == "__main__":
    main()
