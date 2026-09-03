"""Diagnose the frontal-plane (lateral) topple in the 146 N one-step recovery.

Runs the postplant_recovery maneuver control and logs, each ~20 ms:
  chest side-lean, roll rate, whole-body CoM_x and v_x, the stance (L) foot
  lateral support interval, CoM_x relative to that interval, swing-foot state,
  and the roll-relevant joint angles (hip-rolls, ankle-rolls).

Goal: find WHEN the lateral margin is lost (slow drift vs. a plant event) and
WHICH mechanism drives it (weight-shift over-rotation / swing-leg reaction /
plant disturbance).
"""
import numpy as np
import mujoco

from ilqr_recovery import RecoveryILQR, StepPlan, Weights, LEG_CTRL, FWD_HIP_SIGN
from recovery_metrics import _foot_normal_force, _foot_xy_z, sample_balance

PUSH = 146.0
SWING = "R"


def foot_lat_interval(m, d, side):
    """world-X interval covered by the foot collision mesh near the floor."""
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    mid = m.geom_dataid[gid]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    loc = m.mesh_vert[va:va + vn].reshape(-1, 3)
    w = (d.geom_xmat[gid].reshape(3, 3) @ loc.T).T + d.geom_xpos[gid]
    low = w[:, 2].min()
    near = w[w[:, 2] <= low + 0.01]
    return near[:, 0].min(), near[:, 0].max()


def main():
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    plan = StepPlan(swing=SWING, push_n=PUSH, N=130, H=10)
    prob = RecoveryILQR(m, plan, Weights(), verbose=False)
    prob._build_ss_lqr()
    s = prob.sim
    d = s.d
    sw = LEG_CTRL[SWING]
    st = LEG_CTRL["L" if SWING == "R" else "R"]
    s.set_x(prob.x0)

    H = prob.H
    print(f"push {PUSH} N  swing {SWING}   k_unload={prob.k_unload} k_swing_end={prob.k_swing_end}"
          f"  (unload ends {prob.k_unload*H} ms, swing ends {prob.k_swing_end*H} ms)")
    print(f"{'t':>4} {'phase':>8} {'sideL':>6} {'rollR':>6} "
          f"{'CoMx':>6} {'CoMvx':>6} {'Lfoot[x0,x1]':>15} {'CoMx-Lc':>7} "
          f"{'Rft_x':>6} {'Rft_z':>5} {'Lnf':>4} {'Rnf':>4} "
          f"{'swHR':>5} {'stHR':>5} {'LaR':>5} {'RaR':>5}")

    plant_ms = None
    for step_i in range(1700):
        kf = min(step_i / H, prob.N - 1e-3)
        u = prob.seed_ctrl(kf, d.qpos.copy(), d.qvel.copy())
        d.ctrl[:15] = np.clip(u, s.ulo, s.uhi)
        mujoco.mj_step(m, d)

        nfR = _foot_normal_force(m, d, SWING)
        if plant_ms is None and (step_i / H) > prob.k_swing_end and nfR > 5.0:
            plant_ms = step_i

        if step_i % 20 != 0:
            continue
        b = sample_balance(m, d)
        mujoco.mj_subtreeVel(m, d)
        com = d.subtree_com[1]
        comv = d.subtree_linvel[1]
        lx0, lx1 = foot_lat_interval(m, d, "L")
        lc = 0.5 * (lx0 + lx1)
        rf = _foot_xy_z(m, d, SWING)
        # roll rate = chest angular vel about world forward (-Y) axis
        wl = d.qvel[3:6]
        ww = d.xmat[1].reshape(3, 3) @ wl
        roll_rate = ww[1]        # about world Y
        ph = ("unload" if kf <= prob.k_unload else
              "swing" if kf <= prob.k_swing_end else
              ("PLANT" if plant_ms and step_i >= plant_ms else "plantph"))
        print(f"{step_i:>4} {ph:>8} {b.side_lean_deg:>+6.1f} {roll_rate:>+6.2f} "
              f"{com[0]*1000:>+6.0f} {comv[0]*1000:>+6.0f} "
              f"[{lx0*1000:>+5.0f},{lx1*1000:>+5.0f}]  {(com[0]-lc)*1000:>+7.0f} "
              f"{rf[0]*1000:>+6.0f} {(rf[2]-1)*1000:>5.0f} {_foot_normal_force(m,d,'L'):>4.0f} {nfR:>4.0f} "
              f"{d.qpos[7+sw['hip_roll']]:>+5.2f} {d.qpos[7+st['hip_roll']]:>+5.2f} "
              f"{d.qpos[7+9]:>+5.2f} {d.qpos[7+14]:>+5.2f}")
        if b.up_tilt_deg > 55:
            print(f"  --- FELL near {step_i} ms (up_tilt {b.up_tilt_deg:.0f}, side {b.side_lean_deg:+.0f}) ---")
            break


if __name__ == "__main__":
    main()
