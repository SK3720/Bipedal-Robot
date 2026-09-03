"""Around the FIRST right-foot contact in postplant_recovery.run(push=146):
is it a genuine plant the controller fails to accept, or is the foot still not
really landing?

Traces every ms from 380 to 950:
  phase, kf, plant_step, plant_load_streak
  R foot: world x, forward(-y), body-z, SOLE-min-z (real ground clearance),
          forward velocity, vertical velocity, contact bool, normal force
  L foot: sole-z, normal force
  R-leg COMMANDED targets (hip/knee/ankle)  and  ACTUAL qpos
  whether the swing hip target is still increasing forward
  torso up_tilt / fwd_lean / side_lean
"""
import numpy as np
import mujoco

from biped_env import STANDING_QUAT as SQ
from ilqr_recovery import RecoveryILQR, StepPlan, Weights, LEG_CTRL, FWD_HIP_SIGN
from recovery_metrics import _foot_normal_force, _foot_xy_z, sample_balance
from postplant_recovery import _safe_lqr

PUSH, SWING = 146.0, "R"
sw = LEG_CTRL[SWING]
st = LEG_CTRL["L" if SWING == "R" else "R"]
SW_HIP, SW_KNEE, SW_ANK = sw["hip"], sw["knee"], sw["ankle"]


def sole_z(m, d, side):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    mid = m.geom_dataid[gid]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    loc = m.mesh_vert[va:va + vn].reshape(-1, 3)
    w = (d.geom_xmat[gid].reshape(3, 3) @ loc.T).T + d.geom_xpos[gid]
    return w[:, 2].min()


def foot_v(m, d, side):
    bid = m.body(f"{side}_foot").id
    cvel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, cvel, 0)
    return cvel[3], cvel[4], cvel[5]     # world lin x,y,z


def main():
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    plan = StepPlan(swing=SWING, push_n=PUSH, N=130, H=10)
    prob = RecoveryILQR(m, plan, Weights(), verbose=False)
    s = prob.sim
    d = s.d
    prob._build_ss_lqr()
    s.set_x(prob.x0)
    H = prob.H
    CR = m.actuator_ctrlrange[:15]

    # footfall target from the capture-point calc in RecoveryILQR.__init__
    print(f"RecoveryILQR.footfall_fwd (capture-point + margin) = "
          f"{prob.footfall_fwd*1000:.0f} mm  (computed ONCE at init, not updated)")
    print(f"seed_ctrl swing_hip_rad = {plan.swing_hip_rad}  (fixed feed-forward, "
          f"NOT from capture point)")
    print(f"k_swing_end knot {prob.k_swing_end}  -> swing ends ~{prob.k_swing_end*H} ms\n")

    phase = "maneuver"
    plant_step = None
    K_pp = None
    q_lin_ref = None
    qj_freeze = None
    plant_load_streak = 0
    T_SWING = int(prob.k_swing_end * H)
    T_PLANT = int((prob.k_swing_end + 6) * H)
    FB_CLAMP = 0.7

    first_touch = None
    log = []
    for step_i in range(1100):
        b = sample_balance(m, d)
        nf_r = _foot_normal_force(m, d, SWING)
        sc = getattr(b, f"{SWING.lower()}_contact")

        if phase == "maneuver":
            kf = min(step_i / H, prob.N - 1e-3)
            u = prob.seed_ctrl(kf, d.qpos.copy(), d.qvel.copy())
        else:
            kf = None
            if K_pp is None:
                q_lin_ref = np.concatenate([d.qpos[:7], qj_freeze]).copy()
                q_lin_ref[3:7] = SQ
                K_pp = _safe_lqr(s, q_lin_ref, qj_freeze, verbose=True)
                if K_pp is None:
                    K_pp = prob._stand_lqr.K
            dq = np.zeros(s.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, q_lin_ref, d.qpos)
            fb = np.clip(K_pp @ np.concatenate([dq, d.qvel]), -FB_CLAMP, FB_CLAMP)
            u = qj_freeze - fb
        u = np.clip(u, CR[:, 0], CR[:, 1])

        rf = _foot_xy_z(m, d, SWING)
        rvx, rvy, rvz = foot_v(m, d, SWING)
        rsole = (sole_z(m, d, SWING) - 1) * 1000
        lsole = (sole_z(m, d, "L") - 1) * 1000
        log.append(dict(
            t=step_i, ph=phase[:4], kf=kf, plant=plant_step, streak=plant_load_streak,
            rfx=rf[0] * 1000, rfwd=-rf[1] * 1000, rbz=(rf[2] - 1) * 1000, rsole=rsole,
            rvfwd=-rvy * 1000, rvz=rvz * 1000, rc=int(sc), rnf=nf_r,
            lsole=lsole, lnf=_foot_normal_force(m, d, "L"),
            hipT=u[SW_HIP], knT=u[SW_KNEE], anT=u[SW_ANK],
            hipQ=d.qpos[7 + SW_HIP], knQ=d.qpos[7 + SW_KNEE], anQ=d.qpos[7 + SW_ANK],
            upT=b.up_tilt_deg, fwd=b.fwd_lean_deg, side=b.side_lean_deg,
        ))
        if first_touch is None and sc and nf_r > 3 and step_i > 490:
            first_touch = step_i

        d.ctrl[:15] = u
        mujoco.mj_step(m, d)

        if phase == "maneuver" and step_i > T_SWING:
            plant_load_streak = plant_load_streak + 1 if (sc and nf_r > 10.0) else 0
            if plant_load_streak >= 18 or step_i > T_PLANT + 400:
                plant_step = step_i
                qj_freeze = np.clip(d.qpos[7:22].copy(), s.ulo, s.uhi)
                phase = "postplant"
        if b.up_tilt_deg > 60:
            break

    ft = first_touch
    print(f"first R-foot contact with nf>3: {ft} ms\n")
    hf = FWD_HIP_SIGN[SWING]
    lo, hi = 500, 680
    print(f"{'t':>4} {'ph':>4} {'kf':>5} {'plnt':>5} {'strk':>4} | "
          f"Rfwd  Rsole Rvfwd Rvz  Rc Rnf | Lsole Lnf | "
          f"hipT/Q  knT/Q  anT/Q | upT fwd side")
    for r in log:
        if not (lo <= r["t"] <= hi):
            continue
        if r["t"] % 3 and not (ft and ft - 6 <= r["t"] <= ft + 40):
            continue
        kf = f"{r['kf']:.1f}" if r["kf"] is not None else "  -"
        print(f"{r['t']:>4} {r['ph']:>4} {kf:>5} {str(r['plant']):>5} {r['streak']:>4} | "
              f"{r['rfwd']:>4.0f}  {r['rsole']:>4.0f}  {r['rvfwd']:>+4.0f} {r['rvz']:>+5.0f} "
              f"{r['rc']:>2} {r['rnf']:>3.0f} | {r['lsole']:>4.0f} {r['lnf']:>3.0f} | "
              f"{r['hipT']:+.2f}/{r['hipQ']:+.2f} {r['knT']:+.2f}/{r['knQ']:+.2f} "
              f"{r['anT']:+.2f}/{r['anQ']:+.2f} | {r['upT']:>3.0f} {r['fwd']:>+4.0f} {r['side']:>+4.0f}")

    if ft:
        w = [r for r in log if ft - 5 <= r["t"] <= ft + 200]
        contacts = sum(1 for r in w if r["rc"])
        loaded = sum(1 for r in w if r["rnf"] > 12)
        peaknf = max((r["rnf"] for r in w), default=0)
        hipT0 = next(r["hipT"] for r in log if r["t"] == ft)
        hipTlate = max((r["hipT"] for r in w), default=hipT0)
        print(f"\n--- {ft}..{ft+200} ms window ---")
        print(f"  frames in contact: {contacts}/{len(w)}   frames nf>12: {loaded}/{len(w)}   peak nf: {peaknf:.0f} N")
        print(f"  swing-hip TARGET at first touch: {hipT0:+.2f} rad ; max over window: {hipTlate:+.2f} rad "
              f"({'STILL DRIVING FWD' if hipTlate > hipT0 + 0.03 else 'not increasing'})")
        f0 = next(r["rfwd"] for r in log if r["t"] == ft)
        fmax = max(r["rfwd"] for r in w)
        print(f"  R foot forward pos at first touch: {f0:.0f} mm ; max in window: {fmax:.0f} mm "
              f"(+{fmax-f0:.0f} mm after contact)")


if __name__ == "__main__":
    main()
