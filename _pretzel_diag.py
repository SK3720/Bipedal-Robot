"""Frame-by-frame trace of `postplant_recovery.run(push=146)` around the moment
BOTH feet leave the ground and the legs contort ('pretzel').

Replicates the exact control loop of postplant_recovery.run and logs, each ms:
  phase / any LQR build / any reference reset
  commanded ctrl targets  vs  actual joint qpos  vs  ctrlrange (saturation)
  actuator forces (torque limit = +/-2.3)
  L/R foot: sole-z (ground clearance), contact, normal force, vertical velocity
  torso: up_tilt, fwd_lean, side_lean, roll-rate, pitch-rate

Prints a dense window centred on the first frame with NO foot contact.
"""
import numpy as np
import mujoco

from biped_env import STANDING_QUAT as STANDING_QUAT_LOCAL
from ilqr_recovery import RecoveryILQR, StepPlan, Weights, LEG_CTRL
from recovery_metrics import _foot_normal_force, _foot_xy_z, sample_balance
from postplant_recovery import _safe_lqr

PUSH, SWING = 146.0, "R"
CR_LO = None
CR_HI = None
JN = ["neck", "Lsho", "Lelb", "Rsho", "Relb",
      "LhipR", "LhipP", "Lknee", "LankP", "LankR",
      "RhipR", "RhipP", "Rknee", "RankP", "RankR"]


def sole_z(m, d, side):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    mid = m.geom_dataid[gid]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    loc = m.mesh_vert[va:va + vn].reshape(-1, 3)
    w = (d.geom_xmat[gid].reshape(3, 3) @ loc.T).T + d.geom_xpos[gid]
    return w[:, 2].min()


def foot_vz(m, d, side):
    bid = m.body(f"{side}_foot").id
    cvel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, cvel, 0)
    return cvel[5]     # world-frame linear z


def main():
    global CR_LO, CR_HI
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    plan = StepPlan(swing=SWING, push_n=PUSH, N=130, H=10)
    prob = RecoveryILQR(m, plan, Weights(), verbose=False)
    s = prob.sim
    d = s.d
    CR_LO = m.actuator_ctrlrange[:15, 0]
    CR_HI = m.actuator_ctrlrange[:15, 1]
    prob._build_ss_lqr()
    q_stand_live = d.qpos.copy()          # snapshot to detect any live reset
    s.set_x(prob.x0)
    H = prob.H

    phase = "maneuver"
    plant_step = None
    K_ds = None
    q_lin_ref = None
    u_ref0 = np.zeros(15)
    qj_freeze = None
    plant_load_streak = 0
    T_SWING = int(prob.k_swing_end * prob.H)
    T_PLANT = int((prob.k_swing_end + 6) * prob.H)

    log = []
    first_airborne = None
    d_lin_qpos_prev = s._d_lin.qpos.copy()

    for step_i in range(1400):
        b = sample_balance(m, d)
        nf_sw = _foot_normal_force(m, d, SWING)
        event = ""

        if phase == "maneuver":
            kf = min(step_i / H, prob.N - 1e-3)
            u = prob.seed_ctrl(kf, d.qpos.copy(), d.qvel.copy())
        else:
            pk = step_i - plant_step
            if K_ds is None:
                q_lin_ref = np.concatenate([d.qpos[:7], qj_freeze]).copy()
                q_lin_ref[3:7] = STANDING_QUAT_LOCAL
                K_ds = _safe_lqr(s, q_lin_ref, qj_freeze, verbose=True)
                if K_ds is None:
                    K_ds = prob._stand_lqr.K
                    event += "BUILD_LQR(fallback=standing) "
                else:
                    event += "BUILD_LQR(ok) "
            dq = np.zeros(s.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, q_lin_ref, d.qpos)
            fb = np.clip(K_ds @ np.concatenate([dq, d.qvel]), -0.7, 0.7)
            u = qj_freeze - fb

        u_raw = u.copy()
        u = np.clip(u, CR_LO, CR_HI)

        # ---- record BEFORE stepping ----
        lc = b.l_contact; rc = b.r_contact
        airborne = (not lc) and (not rc)
        w_local = d.qvel[3:6].copy()
        w_world = d.xmat[1].reshape(3, 3) @ w_local
        rec = dict(
            t=step_i, phase=phase[:4], pk=(step_i - plant_step) if plant_step is not None else None,
            event=event, airborne=airborne,
            ctrl=u.copy(), ctrl_raw=u_raw.copy(),
            qpos=d.qpos[7:22].copy(),
            afrc=d.actuator_force[:15].copy(),
            Lsole=(sole_z(m, d, "L") - 1) * 1000, Rsole=(sole_z(m, d, SWING) - 1) * 1000,
            Lnf=_foot_normal_force(m, d, "L"), Rnf=nf_sw,
            Lvz=foot_vz(m, d, "L") * 1000, Rvz=foot_vz(m, d, SWING) * 1000,
            upT=b.up_tilt_deg, fwd=b.fwd_lean_deg, side=b.side_lean_deg,
            rollrate=w_world[1], pitchrate=-w_world[0],
        )
        log.append(rec)
        if airborne and first_airborne is None and step_i > 50:
            first_airborne = step_i

        # detect any live-sim reset by linearize (paranoia)
        if not np.array_equal(s._d_lin.qpos, d_lin_qpos_prev):
            d_lin_qpos_prev = s._d_lin.qpos.copy()

        d.ctrl[:15] = u
        mujoco.mj_step(m, d)
        step_i_disp = step_i + 1

        # plant detection (identical to run())
        if phase == "maneuver" and step_i > T_SWING:
            sc = getattr(b, f"{SWING.lower()}_contact")
            plant_load_streak = plant_load_streak + 1 if (sc and nf_sw > 12.0) else 0
            force = step_i > T_PLANT + 350
            if plant_load_streak >= 20 or force:
                plant_step = step_i
                qj_freeze = np.clip(d.qpos[7:22].copy(), s.ulo, s.uhi)
    
                phase = "postplant"

        if b.up_tilt_deg > 60:
            break

    # verify linearize_hold never touched the live sim's qpos mid-run:
    print(f"[check] linearize scratch != live data object: {s._d_lin is not s.d}")

    # scan the WHOLE run for command saturation on the LEG joints (5-14)
    print("\nframes with a LEG joint commanded outside its range (|ctrl_raw-ctrl|>0.05):")
    n_sat = 0
    for r in log:
        legsat = [i for i in range(5, 15) if abs(r["ctrl_raw"][i] - r["ctrl"][i]) > 0.05]
        if legsat:
            n_sat += 1
            if n_sat <= 12:
                print(f"  t={r['t']:>4} {r['phase']} pk={r['pk']}  joints={[JN[i] for i in legsat]}  "
                      f"raw={[round(float(r['ctrl_raw'][i]),1) for i in legsat]}")
    print(f"  ... total {n_sat} such frames" + ("  (hip-roll only = benign small range)" if n_sat else ""))

    fa = first_airborne
    print(f"\nfirst both-feet-airborne frame: {fa} ms")
    lo = (fa - 60) if fa else 400
    hi = (fa + 90) if fa else 700
    print(f"\n{'t':>4} {'ph':>4} {'pk':>4} {'air':>3} event | "
          f"L/R sole(mm)  L/R nf   L/R vz(mm/s) | upT fwd  side  roll pitch(r/s) | "
          f"key ctrl targets (rad)  | key act frc (Nm, lim 2.3)")
    for r in log:
        if not (lo <= r["t"] <= hi):
            continue
        c = r["ctrl"]; q = r["qpos"]; a = r["afrc"]
        sat = "".join("!" if (abs(r["ctrl_raw"][i] - c[i]) > 1e-4) else "." for i in range(15))
        # key joints: LhipP6 Lknee7 LankP8 RhipP11 Rknee12 RankP13  LankR9 RankR14
        kc = f"LhP{c[6]:+.2f} Lkn{c[7]:+.2f} RhP{c[11]:+.2f} Rkn{c[12]:+.2f} LaR{c[9]:+.2f} RaR{c[14]:+.2f}"
        ka = f"LhP{a[6]:+.1f} Lkn{a[7]:+.1f} RhP{a[11]:+.1f} Rkn{a[12]:+.1f}"
        pk = f"{r['pk']:>4}" if r["pk"] is not None else "   -"
        print(f"{r['t']:>4} {r['phase']:>4} {pk} {int(r['airborne'])!s:>3} {r['event']:<12}| "
              f"{r['Lsole']:>5.0f}/{r['Rsole']:<5.0f} {r['Lnf']:>3.0f}/{r['Rnf']:<3.0f} "
              f"{r['Lvz']:>+5.0f}/{r['Rvz']:<+5.0f} | {r['upT']:>3.0f} {r['fwd']:>+4.0f} {r['side']:>+4.0f} "
              f"{r['rollrate']:>+4.1f} {r['pitchrate']:>+4.1f} | {kc} | {ka}  sat[{sat}]")

    # dump full ctrl vs qpos vs range at the airborne onset
    if fa:
        r = next(x for x in log if x["t"] == fa)
        print(f"\n--- full command / state at t={fa} (airborne onset) ---")
        print(f"{'joint':>7} {'ctrl':>7} {'ctrl_raw':>9} {'qpos':>7} {'range':>16} {'act_frc':>8}")
        for i in range(15):
            print(f"{JN[i]:>7} {r['ctrl'][i]:>7.3f} {r['ctrl_raw'][i]:>9.3f} {r['qpos'][i]:>7.3f} "
                  f"[{CR_LO[i]:+.2f},{CR_HI[i]:+.2f}]".rjust(16)
                  + f" {r['afrc'][i]:>8.2f}")


if __name__ == "__main__":
    main()
