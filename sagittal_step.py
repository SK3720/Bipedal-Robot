"""Purely-sagittal forward-push step recovery - the SIMPLE objective.

Scope (deliberately narrow):
  * disturbance is a pure forward impulse (world -Y, perpendicular to the chest)
  * NO lateral foot placement, NO sideways step, minimal frontal-plane action
  * small/moderate push - just past the point where the standing LQR can hold
  * goal: push -> recognise a step is needed -> unload one foot -> swing it
    forward -> GENUINE sustained plant -> weight transfers onto it -> (2nd
    forward step if still diverging) -> stable double support -> standing LQR.

Why 144 N (see _sagittal_envelope.py):
  the standing LQR (standing_balance_lqr.StandingLQR) holds a forward push in
  place up to ~126 N.  From 128-143 N it is UNDER-DAMPED: it arrests the initial
  forward tip but overshoots and diverges backward over ~2 s (not a clean
  forward fall - no stepping scenario).  At >=144 N the CoM cleanly leaves the
  FRONT of the support polygon (com past the toes by ~120 mm), side-lean stays
  < 5 deg, and it falls forward in ~1.2-1.5 s.  So 144 N is the *minimum* push
  that genuinely requires a forward step; there is no gentler clean-forward-fall
  regime to find.

A foot touching the ground is NOT a successful step.  success_step() requires:
  sustained contact, meaningful peak load, the CoM actually moving toward the new
  foot, and the torso staying physically continuous (never > 45 deg) afterwards.

Reuses: StandingLQR, _LQRAbout / _SingleSupportLQR (leaned single-support K),
the ankle-roll unload, the terminal touchdown descent, recovery_metrics.

    python sagittal_step.py --slow
    python sagittal_step.py --headless --trace
    python sagittal_step.py --headless --sweep 140,144,146,150,155
Does not modify robot.xml / biped_env / any golden script.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS, STANDING_QUAT
from recovery_metrics import (
    CHEST_BODY, NOMINAL_CHEST_Z, _foot_normal_force, _foot_xy_z, sample_balance,
)
from standing_balance_lqr import StandingLQR
from push_step_recovery_test import (
    AR_L_IDX, AR_R_IDX, AR_SIGN, FWD_HIP_SIGN, KNEE_FLEX_SIGN, LEG_IDX,
    StepConfig, _LQRAbout, _SingleSupportLQR, _lqr_gain, _pos_error, _smooth,
)

FORWARD_DIR_RAD = -np.pi / 2.0
PUSH_AT = 20
CLIP_LO = None
CLIP_HI = None


@dataclass
class SagCfg:
    swing: str = "R"
    shuffle: bool = False              # True: keep the trailing foot loaded (no
                                       # real single-support phase, smaller step).
                                       # False: leaned single-support LQR swing
                                       # (bigger step, cleaner forward catch, but
                                       # a real single-support phase this robot
                                       # cannot hold laterally past ~1 s).

    # --- trigger: CoM past the toes (at 144 N the LQR provably cannot recover,
    #     so fire EARLY while the torso is still ~2 deg and nearly static) ---
    trig_com_mm: float = 0.0
    trig_com_vfwd: float = -0.02        # only require it is not actively receding
    trig_hold: int = 10
    trig_min_step: int = 40
    trig_deadline: int = 1400

    # --- preload + unload: a PROPER (but not aggressive) lateral weight shift
    #     that genuinely puts the CoM over the stance foot before the swing foot
    #     lifts - standard bipedal stepping.  Without this the swing is a real
    #     single-support phase this robot has no frontal-plane authority for. ---
    ankle_roll_amp: float = 0.06
    ankle_roll_ramp: int = 22
    unload_nf: float = 3.0
    unload_hold: int = 8
    unload_cap: int = 120
    preload_hold: int = 10            # steps after the ramp before the swing

    # --- swing: fast + low (foot skims forward, minimal lift) ---
    swing_hip_rad: float = 0.34
    swing_knee_peak_rad: float = 0.11
    swing_ankle_dorsi_rad: float = 0.10
    swing_steps: int = 68
    step_margin_mm: float = 30.0        # footfall = capture point + this
    step_fwd_min_mm: float = 65.0
    step_fwd_max_mm: float = 115.0

    # --- terminal descent (the verified touchdown phase) - fast ---
    descend_steps: int = 45
    plant_hip_retract: float = 0.10
    plant_ankle_pf: float = 0.34
    stance_knee_bend: float = 0.20
    descend_lean_keep: float = 0.40

    # --- genuine-plant detection ---
    plant_nf: float = 12.0
    plant_hold: int = 15
    descend_cap: int = 400

    # --- transfer: release the lateral lean, brake the residual forward CoM ---
    transfer_steps: int = 180
    lead_brake_ankle_rad: float = 0.24
    lead_brake_knee_rad: float = 0.18
    roll_kp: float = 1.4               # ankle-roll correction per rad of side-lean
    roll_kd: float = 0.25              # ankle-roll correction per rad/s of roll rate
    swing_roll_kp: float = 2.2         # hold the preload lean steady through swing
    swing_roll_kd: float = 0.5
    ss_fb_clip: float = 1.2            # cap on the single-support-LQR feedback correction (rad)
    stance_joint_cap: float = 0.35     # max stance-leg joint deviation from ref (rad) - no fold
    sag_kp: float = 0.9               # ankle plantarflex per rad of fwd-lean
    sag_kd: float = 0.16              # ankle plantarflex per rad/s of pitch rate
    settle_pitch_kd: float = 0.35     # post-plant angular-rate damper (fades over 260 ms)
    settle_roll_kd: float = 0.45
    settle_steps: int = 1400

    # --- second step ---
    allow_second: bool = True
    max_steps: int = 5                 # multiple small shuffle steps ("walk it out")
    restep_com_mm: float = 8.0
    restep_com_vfwd: float = 0.10
    restep_hold: int = 12
    restep_after: int = 120            # min settle steps before another step


def _clip(model, u):
    return np.clip(u, model.actuator_ctrlrange[:15, 0], model.actuator_ctrlrange[:15, 1])


def _quat_slerp(q0, q1, t):
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        th = np.arccos(np.clip(d, -1, 1))
        q = (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)
    return q / (np.linalg.norm(q) + 1e-12)


def _capture_fwd_target(bs, stance_foot_y, cfg):
    """world -y of the desired footfall = capture point + margin, clamped to a
    sane forward range relative to the current stance foot."""
    cap_fwd = bs.com_fwd + bs.com_vfwd * np.sqrt(max(bs.chest_z - 1.0, 0.05) / 9.81)
    tgt = cap_fwd + cfg.step_margin_mm / 1000.0
    stance_fwd = -stance_foot_y
    lo = stance_fwd + cfg.step_fwd_min_mm / 1000.0
    hi = stance_fwd + cfg.step_fwd_max_mm / 1000.0
    return float(np.clip(tgt, lo, hi))


def run(push_n, cfg: SagCfg, show=False, slow=False, trace=False, verbose=True):
    model = mujoco.MjModel.from_xml_path("robot/robot.xml")
    data = mujoco.MjData(model)

    stand = StandingLQR(model, data, verbose=verbose)
    scfg = StepConfig(swing=cfg.swing, ankle_roll_amp_rad=cfg.ankle_roll_amp,
                      ankle_roll_ramp=cfg.ankle_roll_ramp, ss_reach_steps=340)
    base = _LQRAbout(model, data, DEFAULT_POSE.copy(), tag="stand2", verbose=verbose)
    ss = {cfg.swing: _SingleSupportLQR(model, data, base, scfg, verbose=verbose)}

    data.qpos[:] = stand.qpos0
    data.qvel[:] = stand.qvel0
    mujoco.mj_forward(model, data)

    viewer = None
    if show:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.0, -0.3, 0.9]
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -6

    fxy = push_n * np.array([np.cos(FORWARD_DIR_RAD), np.sin(FORWARD_DIR_RAD)])

    phase = "stand"
    swing = cfg.swing
    n_steps = 0                       # steps in current sub-phase
    trig_streak = restep_streak = unload_streak = plant_streak = 0
    ar_now = 0.0                      # current ankle-roll bias
    ss_cur = ss[swing]
    swing_y0 = None
    side_at_swing = 0.0
    foot_tgt_fwd = None
    settle_qref = settle_cref = None
    lean_at_plant = 0.0

    com_fwd_0 = sample_balance(model, data).com_fwd
    steps_done = []
    peak_lead_nf = 0.0
    lead_load_steps = 0
    ever_tilt_spike = False
    plant_com_fwd = None
    result = dict(push_n=push_n, triggered=False, n_steps=0, fell=False,
                  fell_t=None, success=False, note="")

    T_END = PUSH_AT + PUSH_DURATION_STEPS + cfg.trig_deadline + 6000
    k = 0
    while k < T_END:
        data.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT <= k < PUSH_AT + PUSH_DURATION_STEPS:
            data.xfrc_applied[CHEST_BODY, 0:2] = fxy
        bs = sample_balance(model, data)
        post_push = k > PUSH_AT + PUSH_DURATION_STEPS
        idx = LEG_IDX[swing]
        sidx = LEG_IDX["R" if swing == "L" else "L"]
        ss_idx = (idx["hip"], idx["knee"], idx["ankle"])
        hf = FWD_HIP_SIGN[swing]

        # ---------------- control law by phase ----------------
        if phase in ("stand", "settle", "stand_after"):
            # feet-together StandingLQR - best-damped controller available; after
            # a small corrective step the fore-aft stagger is minor and this
            # settles the wobble far better than a frozen-pose LQR.
            u = stand.control(model, data)
            if phase in ("settle", "stand_after"):
                # once the state is far off-nominal the LQR command is garbage
                # (joints slammed to their stops); cap it near the standing pose.
                u = np.clip(u, stand.ctrl0 - 0.7, stand.ctrl0 + 0.7)
                # the step kills linear momentum but leaves ANGULAR rate at
                # touchdown; add a hard rate-only damper (cannot destabilise) on
                # the ankles for the first ~250 ms so the wobble does not grow.
                wd = data.xmat[CHEST_BODY].reshape(3, 3) @ data.qvel[3:6]
                pr, rr = -float(wd[0]), float(wd[1])
                pd = float(np.clip(cfg.settle_pitch_kd * pr, -0.6, 0.6))
                rd = float(np.clip(cfg.settle_roll_kd * rr, -0.6, 0.6))
                u = u.copy()
                u[LEG_IDX["L"]["ankle"]] += FWD_HIP_SIGN["L"] * pd
                u[LEG_IDX["R"]["ankle"]] += FWD_HIP_SIGN["R"] * pd
                u[AR_L_IDX] -= rd
                u[AR_R_IDX] -= rd
                u[LEG_IDX["L"]["hip_roll"]] -= 0.5 * rd
                u[LEG_IDX["R"]["hip_roll"]] -= 0.5 * rd

        elif phase == "unload":
            # PRELOAD: ramp the both-ankle-roll bias toward the stance side under
            # the feet-together LQR and HOLD it.  Verified (headless): held for
            # ~340 steps this reaches ~9 deg lean over the stance foot with the
            # swing foot fully unloaded - a genuine weight shift, no contortion.
            # Feed the LQR ZERO base-translation error so it does not fight the
            # forward CoM drift by folding the stance leg.
            ar_now = AR_SIGN[swing] * cfg.ankle_roll_amp * min(1.0, n_steps / cfg.ankle_roll_ramp)
            qr = stand.qpos0.copy(); cr = stand.ctrl0.copy()
            for ci in (AR_L_IDX, AR_R_IDX):
                qr[7 + ci] += ar_now; cr[ci] += ar_now
            dxp = np.concatenate([_pos_error(model, qr, data.qpos),
                                  data.qvel - stand.qvel0])
            dxp[0] = dxp[1] = 0.0
            dxp[model.nv + 0] = dxp[model.nv + 1] = 0.0
            u = cr - np.clip(stand.K @ dxp, -cfg.ss_fb_clip, cfg.ss_fb_clip)
            u[AR_L_IDX] = cr[AR_L_IDX]; u[AR_R_IDX] = cr[AR_R_IDX]

        elif phase in ("swing", "descend"):
            # SHUFFLE mode: keep the feet-together LQR as the regulator and the
            # TRAILING foot loaded (never truly single-support) - the swing foot
            # only skims ~15-20 mm off the floor.  The staggered-but-loaded
            # double stance keeps the frontal plane damped.  Non-shuffle mode
            # uses the leaned single-support LQR (bigger step, but a real
            # single-support phase this robot can't hold laterally).
            reg = stand if cfg.shuffle else ss_cur
            qref = reg.qpos0.copy(); cref = reg.ctrl0.copy()
            reg_qv0 = reg.qvel0
            if phase == "swing":
                w = min(1.0, n_steps / cfg.swing_steps)
                bump = np.sin(np.pi * w)
                hip = hf * cfg.swing_hip_rad * _smooth(w)
                knee = KNEE_FLEX_SIGN * (0.05 + (cfg.swing_knee_peak_rad - 0.05) * bump)
                ankle = -hf * cfg.swing_ankle_dorsi_rad * bump
                lean = ar_now
            else:  # descend: terminal touchdown
                w = min(1.0, n_steps / cfg.descend_steps)
                sm = _smooth(w)
                hip = hf * (cfg.swing_hip_rad - cfg.plant_hip_retract * sm)
                knee = KNEE_FLEX_SIGN * (0.05 + (cfg.swing_knee_peak_rad - 0.05) * (1 - sm)) * 0.35
                ankle = hf * (0.02 + cfg.plant_ankle_pf * sm)
                st_knee = reg.qpos0[7 + sidx["knee"]] + KNEE_FLEX_SIGN * cfg.stance_knee_bend * sm
                qref[7 + sidx["knee"]] = st_knee; cref[sidx["knee"]] = st_knee
                lean = ar_now * (1 - (1 - cfg.descend_lean_keep) * sm)
            for ci, val in zip(ss_idx, (hip, knee, ankle)):
                qref[7 + ci] = val; cref[ci] = val
            for ci in (AR_L_IDX, AR_R_IDX):
                qref[7 + ci] = lean; cref[ci] = lean
            dx = np.concatenate([_pos_error(model, qref, data.qpos), data.qvel - reg_qv0])
            for ci in ss_idx:                      # swing leg: pure feed-forward
                dx[6 + ci] = 0.0; dx[model.nv + 6 + ci] = 0.0
            # zero the lateral (x) base error; keep the forward (y) error so the
            # stance ankle helps resist the forward fall during the swing.
            dx[0] = 0.0
            dx[model.nv + 0] = 0.0
            u = cref - np.clip(reg.K @ dx, -cfg.ss_fb_clip, cfg.ss_fb_clip)
            # per-joint cap on the STANCE leg: the LQR fed the growing forward-CoM
            # error otherwise drives the stance ankle/knee to ~1.7 rad ("walking
            # the base back") - that is the leg folding at a weird angle.
            for ci in (sidx["hip"], sidx["knee"], sidx["ankle"], sidx["hip_roll"]):
                u[ci] = float(np.clip(u[ci], cref[ci] - cfg.stance_joint_cap,
                                      cref[ci] + cfg.stance_joint_cap))
            for ci in ss_idx:
                u[ci] = cref[ci]
            if phase == "descend":
                u[sidx["knee"]] = cref[sidx["knee"]]
            u[AR_L_IDX] = cref[AR_L_IDX]; u[AR_R_IDX] = cref[AR_R_IDX]
            # HOLD the preload lean steady through the swing (don't let the swing
            # leg's motion roll the robot past the stance foot).
            if not cfg.shuffle:
                rr = (data.xmat[CHEST_BODY].reshape(3, 3) @ data.qvel[3:6])[1]
                hold = -float(np.clip(cfg.swing_roll_kp
                                      * np.radians(bs.side_lean_deg - side_at_swing)
                                      + cfg.swing_roll_kd * rr, -0.30, 0.30))
                u[AR_L_IDX] += hold; u[AR_R_IDX] += hold
                u[idx["hip_roll"]] += 0.5 * hold
                u[sidx["hip_roll"]] += 0.5 * hold
            if cfg.shuffle:
                # trailing foot stays loaded -> use it to fight the forward pitch
                # the swing leg induces (both ankles plantarflex + trailing hip
                # extends against fwd-lean and pitch rate).
                pr = -(data.xmat[CHEST_BODY].reshape(3, 3) @ data.qvel[3:6])[0]
                sag = float(np.clip(cfg.sag_kp * np.radians(bs.fwd_lean_deg)
                                    + cfg.sag_kd * pr, -0.45, 0.45))
                u[sidx["ankle"]] += FWD_HIP_SIGN["R" if swing == "L" else "L"] * sag
                u[idx["ankle"]] += hf * 0.5 * sag
                u[sidx["hip"]] += FWD_HIP_SIGN["R" if swing == "L" else "L"] * 0.4 * sag

        else:  # transfer / settle / stand_after  -- hold the achieved staggered
               # stance (feet-together stand.control would fight the fore-aft
               # offset and slowly topple it)
            w = min(1.0, n_steps / cfg.transfer_steps)
            sm = _smooth(w)
            qref = settle_qref.copy(); cref = settle_cref.copy()
            # ramp the torso reference from the achieved (leaned) attitude toward
            # upright so the LQR actually rights it instead of holding the lean
            qref[3:7] = STANDING_QUAT if sm >= 1.0 else _quat_slerp(
                settle_qref[3:7], STANDING_QUAT, sm)
            # release the ankle-roll lean to neutral
            lean = lean_at_plant * (1 - sm)
            for ci in (AR_L_IDX, AR_R_IDX):
                qref[7 + ci] = lean; cref[ci] = lean
            # lead-leg brake: plantarflex ankle + small knee bend to decelerate
            # the still-forward CoM, faded in then out over the transfer
            br = np.sin(np.pi * min(1.0, w)) if w < 1 else 0.0
            cref[idx["ankle"]] = settle_cref[idx["ankle"]] + hf * cfg.lead_brake_ankle_rad * br
            qref[7 + idx["ankle"]] = cref[idx["ankle"]]
            cref[idx["knee"]] = settle_cref[idx["knee"]] + KNEE_FLEX_SIGN * cfg.lead_brake_knee_rad * br
            qref[7 + idx["knee"]] = cref[idx["knee"]]
            dx = np.concatenate([_pos_error(model, qref, data.qpos), data.qvel])
            u = cref - stand.K @ dx
            # sagittal brake: both ankles plantarflex + trailing hip extends
            # against residual fwd-lean / pitch rate (drives the CoM back)
            pr = -(data.xmat[CHEST_BODY].reshape(3, 3) @ data.qvel[3:6])[0]
            sag = float(np.clip(cfg.sag_kp * np.radians(bs.fwd_lean_deg)
                                + cfg.sag_kd * pr, -0.45, 0.45))
            u[idx["ankle"]] += hf * sag
            u[sidx["ankle"]] += FWD_HIP_SIGN["R" if swing == "L" else "L"] * sag
            u[sidx["hip"]] += FWD_HIP_SIGN["R" if swing == "L" else "L"] * 0.35 * sag
            # explicit roll damping: the staggered stance is laterally narrow and
            # the feet-together K is under-damped in roll -> a slow sideways
            # drift.  Push both ankle-rolls + a fraction of the hip-rolls against
            # measured side-lean and roll rate.
            rollrate = (data.xmat[CHEST_BODY].reshape(3, 3) @ data.qvel[3:6])[1]
            roll_corr = -float(np.clip(cfg.roll_kp * np.radians(bs.side_lean_deg)
                                       + cfg.roll_kd * rollrate, -0.30, 0.30))
            u[AR_L_IDX] = cref[AR_L_IDX] + roll_corr
            u[AR_R_IDX] = cref[AR_R_IDX] + roll_corr
            u[idx["hip_roll"]] += 0.5 * roll_corr
            u[sidx["hip_roll"]] += 0.5 * roll_corr

        data.ctrl[:15] = _clip(model, u)
        mujoco.mj_step(model, data)
        k += 1
        n_steps += 1
        bs = sample_balance(model, data)
        swing_nf = _foot_normal_force(model, data, swing)
        if phase in ("transfer", "settle", "stand_after"):
            peak_lead_nf = max(peak_lead_nf, swing_nf)
            if swing_nf > cfg.plant_nf:
                lead_load_steps += 1
        if bs.up_tilt_deg > 45:
            ever_tilt_spike = True

        # ---------------- fall check ----------------
        if bs.up_tilt_deg > 55 or bs.chest_z < NOMINAL_CHEST_Z - 0.24:
            result["fell"] = True
            result["fell_t"] = k
            if verbose:
                print(f"  FELL @ {k} ms  up_tilt {bs.up_tilt_deg:.0f}  "
                      f"fwd_lean {bs.fwd_lean_deg:+.0f}  side {bs.side_lean_deg:+.0f}")
            break

        # ---------------- phase transitions ----------------
        if phase == "stand" and post_push and n_steps > cfg.trig_min_step:
            diverging = (bs.com_fwd_rel_support_mm > cfg.trig_com_mm
                         and bs.com_vfwd > cfg.trig_com_vfwd)
            trig_streak = trig_streak + 1 if diverging else 0
            if trig_streak >= cfg.trig_hold:
                phase = "unload"; n_steps = 0
                result["triggered"] = True
                swing_y0 = _foot_xy_z(model, data, swing)[1]
                if verbose:
                    print(f"  >> STEP {len(steps_done)+1} TRIGGER @ {k} ms  swing={swing}  "
                          f"com_rel {bs.com_fwd_rel_support_mm:+.0f}mm  com_vfwd {bs.com_vfwd:.2f}  "
                          f"fwd_lean {bs.fwd_lean_deg:+.1f}")
            elif k > PUSH_AT + PUSH_DURATION_STEPS + cfg.trig_deadline:
                if verbose:
                    print(f"  no step triggered by deadline (com_rel "
                          f"{bs.com_fwd_rel_support_mm:+.0f}mm) - LQR holding or lost")
                phase = "stand_after"; n_steps = 0

        elif phase == "unload":
            unloaded = swing_nf < cfg.unload_nf
            unload_streak = unload_streak + 1 if unloaded else 0
            # require: swing foot unloaded for a good while (CoM genuinely over
            # the stance foot) AND the ankle-roll ramp finished + a settle hold
            ready = (unload_streak >= cfg.unload_hold
                     and n_steps >= cfg.ankle_roll_ramp + cfg.preload_hold)
            if ready or n_steps >= cfg.unload_cap:
                stf_y = _foot_xy_z(model, data, "R" if swing == "L" else "L")[1]
                foot_tgt_fwd = _capture_fwd_target(bs, stf_y, cfg)
                side_at_swing = bs.side_lean_deg
                phase = "swing"; n_steps = 0
                unload_streak = 0
                if verbose:
                    print(f"  >> SWING @ {k} ms  swing_nf {swing_nf:.1f}  up_tilt {bs.up_tilt_deg:.1f}  "
                          f"foot target {foot_tgt_fwd*1000 - (-swing_y0*1000):.0f} mm ahead of start")

        elif phase == "swing":
            foot = _foot_xy_z(model, data, swing)
            reached = -foot[1] >= foot_tgt_fwd
            if n_steps >= cfg.swing_steps or reached:
                phase = "descend"; n_steps = 0
                plant_streak = 0
                if verbose:
                    print(f"  >> DESCEND @ {k} ms  foot fwd {-(foot[1]-swing_y0)*1000:.0f} mm  "
                          f"clear {(foot[2]-1.0)*1000:.0f} mm  up_tilt {bs.up_tilt_deg:.1f}")

        elif phase == "descend":
            genuine = (getattr(bs, f"{swing.lower()}_contact") and swing_nf > cfg.plant_nf)
            plant_streak = plant_streak + 1 if genuine else 0
            if plant_streak >= cfg.plant_hold or n_steps >= cfg.descend_cap:
                planted = plant_streak >= cfg.plant_hold
                sf = _foot_xy_z(model, data, swing)[1]
                stf = _foot_xy_z(model, data, "R" if swing == "L" else "L")[1]
                plant_com_fwd = bs.com_fwd
                lean_at_plant = float(data.qpos[7 + AR_L_IDX])
                # KEEP the achieved (slightly lead-ward leaned) torso attitude as
                # the settle reference - the lean toward the new foot IS the
                # weight transfer we want; forcing the torso bolt-upright here
                # shoves the CoM back across onto the trailing foot and past its
                # outside edge (a slow sideways topple).
                settle_qref = data.qpos.copy()
                settle_cref = _clip(model, data.ctrl[:15].copy())
                sidx2 = LEG_IDX["R" if swing == "L" else "L"]
                settle_cref[sidx2["hip"]] = FWD_HIP_SIGN["R" if swing == "L" else "L"] * 0.08
                settle_qref[7 + sidx2["hip"]] = settle_cref[sidx2["hip"]]
                phase = "transfer"; n_steps = 0
                steps_done.append(dict(k=k, planted=planted,
                                       sep_mm=-(sf - stf) * 1000.0,
                                       residual_vfwd=bs.com_vfwd))
                if verbose:
                    print(f"  >> {'PLANT' if planted else 'DESCEND TIMEOUT (never loaded)'} @ {k} ms  "
                          f"foot sep fwd {-(sf-stf)*1000:.0f} mm  swing_nf {swing_nf:.0f}  "
                          f"residual CoM v_fwd {bs.com_vfwd:.2f}  side {bs.side_lean_deg:+.1f}")

        elif phase == "transfer":
            if n_steps >= cfg.transfer_steps:
                phase = "settle"; n_steps = 0

        elif phase == "settle":
            if n_steps >= cfg.settle_steps:
                phase = "stand_after"; n_steps = 0
            elif (cfg.allow_second and len(steps_done) < cfg.max_steps
                  and n_steps > cfg.restep_after):
                # only re-step for a GENUINE continued forward fall: capture point
                # clearly past the toes, moving forward, torso pitched forward
                # (not a backward rock or a contact-transient in the metric).
                need = (bs.capture_fwd_rel_support_mm > cfg.restep_com_mm
                        and bs.com_vfwd > cfg.restep_com_vfwd
                        and bs.fwd_lean_deg > 3.0)
                restep_streak = restep_streak + 1 if need else 0
                if restep_streak >= cfg.restep_hold:
                    swing = "R" if swing == "L" else "L"      # alternate legs
                    if swing not in ss:
                        scfg2 = StepConfig(swing=swing, ankle_roll_amp_rad=cfg.ankle_roll_amp,
                                           ankle_roll_ramp=cfg.ankle_roll_ramp, ss_reach_steps=340)
                        ss[swing] = _SingleSupportLQR(model, data, base, scfg2, verbose=verbose)
                    ss_cur = ss[swing]
                    swing_y0 = _foot_xy_z(model, data, swing)[1]
                    restep_streak = 0
                    phase = "unload"; n_steps = 0
                    if verbose:
                        print(f"  >> STEP {len(steps_done)+1}: still diverging (com_rel "
                              f"{bs.com_fwd_rel_support_mm:+.0f}mm, v {bs.com_vfwd:.2f}) - stepping {swing}")

        elif phase == "stand_after":
            if n_steps >= 800:
                break

        # ---------------- trace ----------------
        if trace and k % 25 == 0:
            q = data.qpos[7:22]
            lo = model.actuator_ctrlrange[:15, 0]; hi = model.actuator_ctrlrange[:15, 1]
            uu = data.ctrl[:15]
            sat = "".join("!" if (uu[i] <= lo[i] + 1e-3 or uu[i] >= hi[i] - 1e-3) else "."
                          for i in range(15))
            print(f"   t{k:5d} [{phase:11s}] up{bs.up_tilt_deg:5.1f} fwd{bs.fwd_lean_deg:+6.1f} "
                  f"side{bs.side_lean_deg:+6.1f} comRel{bs.com_fwd_rel_support_mm:+7.0f} "
                  f"vF{bs.com_vfwd:+5.2f} Lnf{bs.l_nf:4.0f} Rnf{bs.r_nf:4.0f} "
                  f"swZ{(_foot_xy_z(model,data,swing)[2]-1)*1000:+4.0f}  "
                  f"Lhip{q[6]:+.2f}Lkne{q[7]:+.2f}Lank{q[8]:+.2f} "
                  f"Rhip{q[11]:+.2f}Rkne{q[12]:+.2f}Rank{q[13]:+.2f} sat[{sat}]")

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(0.02 if slow else 0.0015)

    bs = sample_balance(model, data)
    ds = bs.l_contact and bs.r_contact
    com_moved = bs.com_fwd - com_fwd_0
    n_planted = sum(1 for s in steps_done if s["planted"])
    success = (result["triggered"] and not result["fell"] and n_planted >= 1
               and not ever_tilt_spike and ds
               and peak_lead_nf > 25.0 and lead_load_steps > 150
               and com_moved > 0.030
               and bs.up_tilt_deg < 12 and abs(bs.side_lean_deg) < 12
               and bs.com_speed_horiz < 0.13
               and bs.chest_z > NOMINAL_CHEST_Z - 0.12)
    result.update(n_steps=len(steps_done), n_planted=n_planted,
                  end_ds=ds, end_up_tilt=bs.up_tilt_deg, end_side=bs.side_lean_deg,
                  end_com_speed=bs.com_speed_horiz, com_moved_mm=com_moved * 1000.0,
                  peak_lead_nf=peak_lead_nf, lead_load_ms=lead_load_steps,
                  tilt_spike=ever_tilt_spike, steps=steps_done, success=success)

    if verbose:
        print(f"\n  push {push_n:.0f} N   steps={len(steps_done)} planted={n_planted}   "
              f"CoM moved fwd {com_moved*1000:+.0f} mm")
        for i, s in enumerate(steps_done):
            print(f"    step {i+1}: {'PLANT' if s['planted'] else 'no-load'} @ {s['k']} ms  "
                  f"sep {s['sep_mm']:+.0f} mm  residual v_fwd {s['residual_vfwd']:.2f}")
        print(f"  lead foot: peak load {peak_lead_nf:.0f} N   loaded {lead_load_steps} ms")
        print(f"  end: up_tilt {bs.up_tilt_deg:.1f}  side {bs.side_lean_deg:+.1f}  "
              f"DS {ds}  CoM speed {bs.com_speed_horiz:.3f}  tilt_spike {ever_tilt_spike}")
        print(f"  >>> {'SUCCESS - beautiful boring sagittal recovery' if success else ('FELL' if result['fell'] else 'FAILED - not a clean step+transfer')}")

    if viewer is not None:
        try:
            while viewer.is_running():
                viewer.sync(); time.sleep(0.02 if slow else 0.0015)
        except KeyboardInterrupt:
            pass
        viewer.close()
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--push", type=float, default=144.0)
    p.add_argument("--swing", choices=["L", "R"], default="R")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--sweep", default=None)
    p.add_argument("--no-second", action="store_true")
    a = p.parse_args(argv)
    cfg = SagCfg(swing=a.swing, allow_second=not a.no_second)

    if a.sweep:
        rows = []
        for pn in (float(x) for x in a.sweep.split(",")):
            rows.append(run(pn, cfg, show=False, trace=False, verbose=True))
        print("\n" + "=" * 70)
        for r in rows:
            print(f"  {r['push_n']:5.0f} N  trig={str(r['triggered']):>5}  "
                  f"steps={r.get('n_steps',0)} planted={r.get('n_planted',0)}  "
                  f"CoM+{r.get('com_moved_mm',0):+.0f}mm  leadNF {r.get('peak_lead_nf',0):.0f}  "
                  f"{'SUCCESS' if r['success'] else ('FELL' if r['fell'] else 'fail')}")
        return

    run(a.push, cfg, show=not a.headless, slow=a.slow, trace=a.trace, verbose=True)


if __name__ == "__main__":
    main(sys.argv[1:])
