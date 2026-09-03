"""Detailed step-by-step trace of push_recovery_step --push 150.

WHAT THIS REVEALED (the reason the --slow run looks wrong):
  1. The standing LQR holds the push for ~0.5 s (capture point pinned ~+18 mm),
     slowly losing; the PRE-SHIFT does not trigger until t~0.54 s (capture +47) -
     far too late.
  2. The ankle-roll weight shift, applied to the forward-PITCHED pushed state,
     OSCILLATES the load between the feet (L unloads -> R unloads -> L ...) - it
     never cleanly frees one foot.  From a symmetric standing pose the same shift
     works; the forward-pitch dynamics destabilise it.
  3. The controller switches to K_ss (single-support LQR) when the real state is
     nowhere near K_ss's linearisation point (leaning, moving, weight on the
     wrong foot).  K_ss is a fixed-point regulator - off-nominal it slams the
     LEFT (nominal stance) leg (hip +41 deg, hip-roll +38 deg) trying to load a
     foot that is actually airborne -> the left leg flails up and outward (the
     "curling" you see).
  4. The scripted right-leg swing barely moves the foot because the right foot is
     still loaded / oscillating.
  -> net: topple forward-and-sideways, both feet leave the ground.

Run:  python push_recovery_trace.py
"""
import mujoco, numpy as np
import push_recovery_step as P
from recovery_metrics import sample_balance, _foot_normal_force, _foot_xy_z, CHEST_BODY

m = mujoco.MjModel.from_xml_path("robot/robot.xml"); d = mujoco.MjData(m)
cfg = P.Cfg(swing="R")

# monkeypatch: wrap run to log. Easiest: reimplement the loop with logging.
from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS, STANDING_QUAT
from push_step_recovery_test import (_LQRAbout, _SingleSupportLQR, _pos_error, LEG_IDX,
                                     AR_L_IDX, AR_R_IDX, AR_SIGN, FWD_HIP_SIGN, KNEE_FLEX_SIGN)

stand = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="stand", verbose=False)
class _C:
    swing="R"; ankle_roll_amp_rad=cfg.ankle_roll_amp_rad
    ankle_roll_ramp=cfg.ankle_roll_ramp; ss_reach_steps=cfg.ss_reach_steps
ss = _SingleSupportLQR(m, d, stand, _C, verbose=True)
idx = LEG_IDX["R"]; sidx = LEG_IDX["L"]; sw3 = (idx["hip"], idx["knee"], idx["ankle"])
ar_full = AR_SIGN["R"] * cfg.ankle_roll_amp_rad; nv = m.nv

# qpos indices
Q = dict(Lhipp=13, Lknee=14, Lankp=15, Lhipr=12, Lankr=16,
         Rhipp=18, Rknee=19, Rankp=20, Rhipr=17, Rankr=21)

d.qpos[:] = stand.qpos0; d.qvel[:] = stand.qvel0; mujoco.mj_forward(m, d)
fxy = 150.0 * np.array([np.cos(P.FORWARD_DIR_RAD), np.sin(P.FORWARD_DIR_RAD)])
phase = "balance"
pre_streak = trig_streak = calm_streak = plant_streak = 0
preshift_k0 = swing_k0 = plant_k0 = settle_k0 = None
swing_end_hip = 0.0
last_phase = None

def _swing_targets(sk):
    hf, ks = 1.0, -1.0
    w = min(1.0, sk / cfg.swing_steps); bump = np.sin(np.pi*w)
    from push_step_recovery_test import _smooth
    return (hf*cfg.swing_hip_fwd_rad*_smooth(w),
            ks*(0.06+(cfg.swing_knee_peak_rad-0.06)*bump),
            -hf*cfg.swing_ankle_dorsi_rad*bump)

def logrow(k, tag=""):
    bs = sample_balance(m, d)
    lf = _foot_xy_z(m, d, "L"); rf = _foot_xy_z(m, d, "R")
    lnf = _foot_normal_force(m, d, "L"); rnf = _foot_normal_force(m, d, "R")
    com = d.subtree_com[CHEST_BODY]
    def dg(i): return np.degrees(d.qpos[i])
    print(f"{k:4d} {phase:8s} upT={bs.up_tilt_deg:5.1f} fl={bs.fwd_lean_deg:+5.1f} sl={bs.side_lean_deg:+5.1f} "
          f"| Lf=({lf[0]*100:+.1f},{lf[1]*100:+.1f},{(lf[2]-1)*100:+.1f}) Rf=({rf[0]*100:+.1f},{rf[1]*100:+.1f},{(rf[2]-1)*100:+.1f}) "
          f"L{int(bs.l_contact)}{lnf:4.0f} R{int(bs.r_contact)}{rnf:4.0f} "
          f"| Lhp={dg(Q['Lhipp']):+5.0f} Lkn={dg(Q['Lknee']):+5.0f} Lhr={dg(Q['Lhipr']):+4.0f} "
          f"Rhp={dg(Q['Rhipp']):+5.0f} Rkn={dg(Q['Rknee']):+5.0f} Rhr={dg(Q['Rhipr']):+4.0f} Rap={dg(Q['Rankp']):+4.0f}  {tag}")

for k in range(1600):
    d.xfrc_applied[CHEST_BODY, :] = 0.0
    if P.PUSH_AT_STEP <= k < P.PUSH_AT_STEP + PUSH_DURATION_STEPS:
        d.xfrc_applied[CHEST_BODY, 0:2] = fxy
    bs = sample_balance(m, d)
    post_push = k > P.PUSH_AT_STEP + PUSH_DURATION_STEPS
    sw_nf = _foot_normal_force(m, d, "R")

    if phase == "balance":
        u = stand.ctrl0 - stand.K @ np.concatenate([_pos_error(m, stand.qpos0, d.qpos), d.qvel - stand.qvel0])
    elif phase == "preshift":
        a = ar_full * min(1.0, (k - preshift_k0) / cfg.ankle_roll_ramp)
        qref = stand.qpos0.copy(); cref = stand.ctrl0.copy()
        for ci in (AR_L_IDX, AR_R_IDX): qref[7+ci] += a; cref[ci] += a
        u = cref - stand.K @ np.concatenate([_pos_error(m, qref, d.qpos), d.qvel - stand.qvel0])
        u[AR_L_IDX] = cref[AR_L_IDX]; u[AR_R_IDX] = cref[AR_R_IDX]
    elif phase == "swing":
        sk = k - swing_k0
        hip, knee, ankle = _swing_targets(sk); swing_end_hip = hip
        qref = ss.qpos0.copy(); cref = ss.ctrl0.copy(); qref[3:7] = STANDING_QUAT
        for ci, v in zip(sw3, (hip, knee, ankle)): qref[7+ci] = v; cref[ci] = v
        dx = np.concatenate([_pos_error(m, qref, d.qpos), d.qvel - ss.qvel0])
        dx[0]=dx[1]=0; dx[nv+0]=dx[nv+1]=0
        dx[6+idx["hip"]]=0; dx[nv+6+idx["hip"]]=0
        u = cref - ss.K @ dx
        u[idx["hip"]] = cref[idx["hip"]]
        u[AR_L_IDX]=cref[AR_L_IDX]; u[AR_R_IDX]=cref[AR_R_IDX]
    elif phase == "plant":
        pk = k - plant_k0
        hip, knee, ankle, w = P._plant_targets(cfg, pk, swing_end_hip)
        qref = ss.qpos0.copy(); cref = ss.ctrl0.copy(); qref[3:7] = STANDING_QUAT
        for ci, v in zip(sw3, (hip, knee, ankle)): qref[7+ci]=v; cref[ci]=v
        sb = KNEE_FLEX_SIGN * cfg.stance_knee_bend_rad * w
        qref[7+sidx["knee"]] = ss.qpos0[7+sidx["knee"]] + sb; cref[sidx["knee"]] = ss.ctrl0[sidx["knee"]] + sb
        dx = np.concatenate([_pos_error(m, qref, d.qpos), d.qvel - ss.qvel0])
        dx[0]=dx[1]=0; dx[nv+0]=dx[nv+1]=0
        u = cref - ss.K @ dx
        for ci in sw3: u[ci] = cref[ci]
        u[AR_L_IDX]=cref[AR_L_IDX]; u[AR_R_IDX]=cref[AR_R_IDX]
    else:
        sk = k - settle_k0
        a = ar_full * max(0.0, 1.0 - sk/cfg.settle_ankle_roll_relax_steps) * cfg.ankle_roll_plant_frac
        cref = settle_cref.copy(); qref = settle_qref.copy()
        for ci in (AR_L_IDX, AR_R_IDX): qref[7+ci]=a; cref[ci]=a
        u = cref - stand.K @ np.concatenate([_pos_error(m, qref, d.qpos), d.qvel])
        u[AR_L_IDX]=cref[AR_L_IDX]; u[AR_R_IDX]=cref[AR_R_IDX]

    d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15,0], m.actuator_ctrlrange[:15,1])
    mujoco.mj_step(m, d)

    # transitions (mirror push_recovery_step)
    if phase == "balance" and post_push and k >= cfg.preshift_min_step:
        losing = bs.capture_fwd_rel_support_mm > cfg.preshift_capture_mm and bs.com_vfwd > cfg.preshift_com_vfwd
        pre_streak = pre_streak + 1 if losing else 0
        if pre_streak >= cfg.preshift_hold: phase, preshift_k0 = "preshift", k
    elif phase == "preshift":
        div = bs.capture_fwd_rel_support_mm > cfg.trig_capture_mm and bs.com_vfwd > cfg.trig_com_vfwd
        trig_streak = trig_streak + 1 if div else 0
        calm_streak = calm_streak + 1 if bs.capture_fwd_rel_support_mm < cfg.preshift_capture_mm else 0
        ramped = (k - preshift_k0) >= cfg.ankle_roll_ramp
        if calm_streak >= 60: phase = "balance"; pre_streak = trig_streak = 0
        elif trig_streak >= cfg.trig_hold and (sw_nf < cfg.unload_nf or ramped):
            phase = "swing"; swing_k0 = k
        elif (k - preshift_k0) >= cfg.unload_cap and sw_nf < cfg.unload_nf*2:
            phase = "swing"; swing_k0 = k
    elif phase == "swing":
        if (k - swing_k0) >= cfg.swing_steps: phase = "plant"; plant_k0 = k
    elif phase == "plant":
        okp = d.contact and _foot_normal_force(m,d,"R") > cfg.plant_nf
        pc = getattr(sample_balance(m,d), "r_contact")
        plant_streak = plant_streak + 1 if (pc and _foot_normal_force(m,d,"R") > cfg.plant_nf) else 0
        if plant_streak >= cfg.plant_hold or (k - plant_k0) >= cfg.plant_cap:
            phase = "settle"; settle_k0 = k
            settle_qref = d.qpos.copy(); settle_qref[3:7] = STANDING_QUAT
            settle_cref = np.clip(d.ctrl[:15].copy(), m.actuator_ctrlrange[:15,0], m.actuator_ctrlrange[:15,1])

    if phase != last_phase:
        logrow(k, f"<<< ENTER {phase}")
        last_phase = phase
    elif k % 15 == 0:
        logrow(k)
    if sample_balance(m,d).up_tilt_deg > 70:
        logrow(k, "  (tilt>70, stop)"); break
