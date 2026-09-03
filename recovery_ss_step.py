"""Push -> genuine forward RECOVERY step using the single-support LQR for the
airborne phase (not the feet-together StandingLQR).

Why this architecture
---------------------
Two dead ends were mapped first:
  * sagittal_recovery.py  - StandingLQR does 100% of balance the whole time.
    Reliable (11/11) but the "step" is a 14 mm drag: any real leg swing's
    reaction torque pitches the torso ~15 deg and the marginally-stable
    feet-together LQR cannot recover it (measured: swing_hip 0.34 ok, 0.60 falls).
  * step_primitive.py  - SS-LQR holds a real 90 mm swing (from
    single_support_step_demo) but a COMMANDED step from standing has no forward
    CoM momentum, so the weight never transfers onto the new foot and single
    support eventually gives out sideways.

This file is the intersection that should actually work: the PUSH supplies the
forward CoM momentum that the commanded primitive lacked, and the SS-LQR supplies
the single-support balance authority that StandingLQR lacked.

  stand  (StandingLQR, feet-together)         - pre-push balance + trigger (reused
                                                 verbatim from sagittal_recovery)
  ss     (_SingleSupportLQR, leaned one-foot)  - balance while the foot is airborne
         both built ONCE, offline.

  STAND -> STEP     : ankle-roll IMPULSE unloads the swing foot (pulse+release)
        -> SS_SWING : hand to ss.K; freed leg does the demo hip/knee arc; ankle
                      servo (AnkleSolver) keeps the SOLE FLAT vs the live torso
        -> SS_DESC  : ss.K; foot comes down, sole flat, contact-seeking
        -> [genuine sustained contact]
        -> TRANSFER : blend ss.K -> stand.K over ~140 ms; release the lean; new
                      leg presses, trailing ankle push-off - CoM/momentum rides
                      onto the new foot (do NOT yank the body back)
        -> SETTLE   : stand.K; 2nd forward step if still diverging

    python recovery_ss_step.py --slow
    python recovery_ss_step.py --headless --trace --push 135
    python recovery_ss_step.py --headless --sweep 124,128,132,136,140
Does not modify robot.xml / biped_env / push_step_recovery_test / any golden script.
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
    StepConfig, _LQRAbout, _SingleSupportLQR, _pos_error, _smooth,
)
from step_primitive import AnkleSolver, _sole_pitch, LEG, AR, FWD_HIP_SIGN, KNEE_FLEX_SIGN


def _lat_com_vel(m, d):
    mujoco.mj_subtreeVel(m, d)
    return float(d.subtree_linvel[CHEST_BODY][0])

FORWARD_DIR_RAD = -np.pi / 2.0
PUSH_AT = 20
AR_UNLOAD_SIGN = {"R": -1.0, "L": +1.0}


@dataclass
class Cfg:
    swing: str = "R"

    # ---- trigger (verbatim from sagittal_recovery) ----
    trig_capt_mm: float = 6.0
    trig_vfwd: float = 0.05
    trig_hold: int = 12
    trig_min_ms: int = 40
    trig_deadline_ms: int = 1200

    # ---- ankle-roll impulse to unload the swing foot ----
    imp_amp: float = 0.22
    imp_ramp_ms: int = 8
    imp_hold_ms: int = 40
    swing_start_nf: float = 6.0
    swing_start_ms: int = 90
    swing_pr_gate: float = 0.2
    swing_latest_ms: int = 340
    preshift_ms: int = 230          # min ramp of the lean under StandingLQR
    handoff_latvel: float = 0.30    # |lateral CoM vel| gate for the ss hand-off
    #   (diagnostics: ss.K slams the roll joints if handed control >~4 deg off its
    #    leaned reference; entering at side ~+7 deg removes the entry transient.
    #    ss.K then holds single support ~600-800 ms before a marginal oscillation
    #    grows - the whole swing+descend+plant MUST finish inside that window.)

    # ---- SS swing (real, demo-shaped) ----
    swing_ms: int = 105
    descend_ms: int = 55
    lean_hold_frac: float = 0.55      # fraction of the entry lean still held at plant
    swing_hip_rad: float = 0.52
    swing_knee_rad: float = 0.34
    knee_land_frac: float = 0.05
    hip_retract_frac: float = 0.38     # pull the thigh back in descend so the foot
    #                                    actually DROPS (else it hovers ~80 mm up
    #                                    out front while the torso pitches)
    sole_toe_up_mid: float = 0.09
    desc_knee_drive: float = 0.45      # knee extension while contact-seeking (w>1)
    desc_hip_drop: float = 0.30        # extra thigh retract while contact-seeking
    ankle_roll_amp: float = 0.12
    ss_reach_steps: int = 340

    # ---- contact / plant ----
    plant_nf: float = 12.0
    plant_hold: int = 8
    descend_extra_ms: int = 150     # cap total single-support time (stability window)

    # ---- transfer / frontal-plane catch ----
    transfer_ms: int = 240
    transfer_press: float = 0.24
    catch_kp: float = 2.2
    catch_kd: float = 0.28
    catch_max: float = 0.30
    catch_hiproll: float = 0.6
    catch_exit_side: float = 9.0     # hand to StandingLQR once |side| below this

    # ---- settle / restep ----
    settle_ms: int = 2400
    restep_capt_mm: float = 12.0
    restep_vfwd: float = 0.12
    restep_hold: int = 15
    restep_after_ms: int = 160
    max_steps: int = 2

    stance_cap: float = 0.35
    swing_hiproll_cap: float = 0.05
    swing_lat_bias: float = 0.0       # hip-roll offset on the swing leg during the
    #                                   swing: + brings the foot toward midline so
    #                                   it lands nearer the (laterally-leaning) CoM
    #                                   and can actually take load for the catch


def _clip(m, u):
    return np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])


def _swing_arc(cfg, phase, w, h0, k0):
    hf = FWD_HIP_SIGN[cfg.swing]
    if phase == "swing":
        s = _smooth(w)
        hip = h0 + hf * cfg.swing_hip_rad * s
        knee = k0 + KNEE_FLEX_SIGN * (0.05 + (cfg.swing_knee_rad - 0.05) * np.sin(np.pi * w))
        sole = cfg.sole_toe_up_mid * np.sin(np.pi * w)
    else:  # descend, w may exceed 1 (contact-seeking)
        s = _smooth(min(1.0, w))
        over = min(1.0, max(0.0, w - 1.0))
        hip = h0 + hf * cfg.swing_hip_rad * (
            1.0 - cfg.hip_retract_frac * s - cfg.desc_hip_drop * over)
        knee = k0 + KNEE_FLEX_SIGN * (
            cfg.knee_land_frac + 0.20 * (1.0 - s) - cfg.desc_knee_drive * over)
        sole = cfg.sole_toe_up_mid * 0.4 * (1.0 - s)
    return hip, knee, sole


def run(push_n, cfg: Cfg, show=False, slow=False, trace=False, verbose=True,
        model_path="robot/robot.xml", push_dir=None):
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    stand = StandingLQR(m, d, verbose=verbose)
    scfg = StepConfig(swing=cfg.swing, ankle_roll_amp_rad=cfg.ankle_roll_amp,
                      ss_reach_steps=cfg.ss_reach_steps)
    stand_ab = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="stand_ab", verbose=False)
    ss = _SingleSupportLQR(m, d, stand_ab, scfg, verbose=False)
    anksolve = AnkleSolver(m)

    swing = cfg.swing
    sw = LEG[swing]
    stn = LEG["L" if swing == "R" else "R"]
    hf = FWD_HIP_SIGN[swing]
    ss_idx = (sw["hip"], sw["knee"], sw["ankle"])
    lean_ss = float(ss.ctrl0[AR[swing]])
    h0 = ss.qpos0[7 + sw["hip"]]
    k0 = ss.qpos0[7 + sw["knee"]]
    a0 = ss.qpos0[7 + sw["ankle"]]

    d.qpos[:] = stand.qpos0
    d.qvel[:] = stand.qvel0
    mujoco.mj_forward(m, d)

    viewer = None
    if show:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.cam.lookat[:] = [0.0, -0.35, 1.1]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 100
        viewer.cam.elevation = -6

    if push_dir is None:
        _pd = np.array([np.cos(FORWARD_DIR_RAD), np.sin(FORWARD_DIR_RAD)])
    else:
        _pd = np.asarray(push_dir, float)
        _pd = _pd / (np.linalg.norm(_pd) + 1e-12)
    fxy = push_n * _pd

    def stand_u():
        return stand.control(m, d)

    def ss_u(hip, knee, ankle, lean):
        qref = ss.qpos0.copy()
        cref = ss.ctrl0.copy()
        for ci, v in zip(ss_idx, (hip, knee, ankle)):
            qref[7 + ci] = v
            cref[ci] = v
        # override the ankle-roll reference with the (possibly releasing) lean
        cref[AR["L"]] = DEFAULT_POSE[AR["L"]] + lean
        cref[AR["R"]] = DEFAULT_POSE[AR["R"]] + lean
        qref[7 + AR["L"]] = DEFAULT_POSE[AR["L"]] + lean
        qref[7 + AR["R"]] = DEFAULT_POSE[AR["R"]] + lean
        dx = np.concatenate([_pos_error(m, qref, d.qpos), d.qvel - ss.qvel0])
        for ci in ss_idx:
            dx[6 + ci] = 0.0
            dx[m.nv + 6 + ci] = 0.0
        dx[0] = dx[m.nv + 0] = 0.0
        u = cref - ss.K @ dx
        for ci in ss_idx:
            u[ci] = cref[ci]
        u[AR["L"]] = cref[AR["L"]]
        u[AR["R"]] = cref[AR["R"]]
        return u

    phase = "stand"
    sk = 0
    trig_streak = restep_streak = plant_streak = 0
    swing_started = False
    foot_lifted = False
    swing_t0 = 0
    an_cmd = a0
    plant_q = np.array([h0, k0, a0])
    steps = []
    com_fwd0 = sample_balance(m, d).com_fwd
    swing_y0 = swing_x0 = None
    peak_up = 0.0
    peak_fwd = peak_clear = 0.0
    result = dict(push_n=push_n, triggered=False, fell=False, fell_t=None)

    T_END = PUSH_AT + PUSH_DURATION_STEPS + cfg.trig_deadline_ms + 6000
    k = 0
    while k < T_END:
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT <= k < PUSH_AT + PUSH_DURATION_STEPS:
            d.xfrc_applied[CHEST_BODY, 0:2] = fxy
        bs = sample_balance(m, d)
        post = k > PUSH_AT + PUSH_DURATION_STEPS
        sw_nf = _foot_normal_force(m, d, swing)
        st_nf = _foot_normal_force(m, d, "L" if swing == "R" else "R")
        sp = _sole_pitch(m, d, swing)
        pr = -(d.xmat[CHEST_BODY].reshape(3, 3) @ d.qvel[3:6])[0]  # + = pitching fwd

        # ---------------- control ----------------
        if phase in ("stand", "settle"):
            u = stand_u()
            for ci in (stn["hip"], stn["knee"], stn["ankle"],
                       sw["hip"], sw["knee"], sw["ankle"]):
                u[ci] = float(np.clip(u[ci], stand.ctrl0[ci] - 0.7, stand.ctrl0[ci] + 0.7))

        elif phase == "step":
            # ankle-roll impulse to break the foot loose, then RAMP TO the ss lean
            # value and hold it (reference-biased) under StandingLQR so the robot
            # actually reaches the pose ss.K is linearised about before the swing.
            imp_end = cfg.imp_ramp_ms + cfg.imp_hold_ms
            if sk < cfg.imp_ramp_ms:
                a = AR_UNLOAD_SIGN[swing] * cfg.imp_amp * (sk / cfg.imp_ramp_ms)
            elif sk < imp_end:
                a = AR_UNLOAD_SIGN[swing] * cfg.imp_amp
            else:
                pf = min(1.0, (sk - imp_end) / max(cfg.preshift_ms, 1))
                a = (1.0 - pf) * AR_UNLOAD_SIGN[swing] * cfg.imp_amp + pf * lean_ss
            qref = stand.qpos0.copy()
            cref = stand.ctrl0.copy()
            for ci in (AR["L"], AR["R"]):
                qref[7 + ci] += a
                cref[ci] += a
            dq = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, qref, d.qpos)
            u = cref - stand.K @ np.concatenate([dq, d.qvel - stand.qvel0])
            for ci in (stn["hip"], stn["knee"], stn["ankle"],
                       sw["hip"], sw["knee"], sw["ankle"]):
                u[ci] = float(np.clip(u[ci], stand.ctrl0[ci] - 0.7, stand.ctrl0[ci] + 0.7))
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] + a
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] + a

        elif phase in ("ss_swing", "ss_desc"):
            seg = "swing" if phase == "ss_swing" else "descend"
            dur = cfg.swing_ms if seg == "swing" else cfg.descend_ms
            w = (min(1.0, sk / dur) if seg == "swing" else sk / dur)
            hip, knee, sole_tgt = _swing_arc(cfg, seg, w, h0, k0)
            an_cmd = anksolve.solve(d.qpos, swing, hip, knee, sole_tgt, prev=an_cmd)
            # bleed the ankle-roll lean back toward stance-foot centre as the swing
            # progresses: the single-support CoM starts ~6 mm inside the outer edge
            # of the 32 mm stance foot and drifts out; pulling the lean in gives
            # the marginal SS-LQR frontal-plane margin to work with.
            prog = (0.5 * w if seg == "swing" else 0.5 + 0.5 * min(1.0, w))
            lean_now = lean_ss * (1.0 - (1.0 - cfg.lean_hold_frac) * prog)
            u = np.array(ss_u(hip, knee, an_cmd, lean_now), float)
            if cfg.swing_lat_bias:
                u[sw["hip_roll"]] = (DEFAULT_POSE[sw["hip_roll"]]
                                     + cfg.swing_lat_bias * _smooth(min(1.0, w if seg == "swing" else 1.0)))

        elif phase == "transfer":
            # BOTH feet are down now.  Do NOT blend to the feet-together
            # StandingLQR yet (invalid at this lean/stagger).  Hold the landed
            # joint pose and drive both hip_roll + ankle_roll HARD to null the
            # side lean + roll rate accumulated during single support, and let
            # the CoM settle onto the new foot.  Hand to StandingLQR only once
            # the frontal plane is back under control (phase 'settle').
            rr = (d.xmat[CHEST_BODY].reshape(3, 3) @ d.qvel[3:6])[1]  # roll rate
            b = _smooth(min(1.0, sk / cfg.transfer_ms))
            u = stand_u()
            # freeze the leg joints near what they were at touchdown (+ gentle
            # load-taking on the new leg, gentle yield on the old one)
            u[sw["hip"]] = plant_q[0]
            u[sw["knee"]] = plant_q[1] + KNEE_FLEX_SIGN * 0.06 * b
            u[sw["ankle"]] = plant_q[2] + hf * 0.06 * b
            for ci in (stn["hip"], stn["knee"], stn["ankle"]):
                u[ci] = float(np.clip(u[ci], DEFAULT_POSE[ci] - cfg.stance_cap,
                                      DEFAULT_POSE[ci] + cfg.stance_cap))
            roll_corr = float(np.clip(
                cfg.catch_kp * np.radians(bs.side_lean_deg) + cfg.catch_kd * rr,
                -cfg.catch_max, cfg.catch_max))
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] - roll_corr
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] - roll_corr
            u[stn["hip_roll"]] = DEFAULT_POSE[stn["hip_roll"]] - cfg.catch_hiproll * roll_corr
            u[sw["hip_roll"]] = DEFAULT_POSE[sw["hip_roll"]] - cfg.catch_hiproll * roll_corr

        u = np.array(u, float)

        # ---- safety clamps (swing/descend only; transfer manages roll itself) ----
        if phase in ("step", "ss_swing", "ss_desc"):
            cref0 = DEFAULT_POSE
            for ci in (stn["hip"], stn["knee"], stn["ankle"], stn["hip_roll"]):
                u[ci] = float(np.clip(u[ci], cref0[ci] - cfg.stance_cap,
                                      cref0[ci] + cfg.stance_cap))
            u[sw["hip_roll"]] = float(np.clip(
                u[sw["hip_roll"]],
                cref0[sw["hip_roll"]] + cfg.swing_lat_bias - cfg.swing_hiproll_cap,
                cref0[sw["hip_roll"]] + cfg.swing_lat_bias + cfg.swing_hiproll_cap))

        u = _clip(m, u)
        d.ctrl[:15] = u
        mujoco.mj_step(m, d)
        k += 1
        if phase != "stand":
            sk += 1
        bs = sample_balance(m, d)
        sw_nf = _foot_normal_force(m, d, swing)
        f = _foot_xy_z(m, d, swing)
        if swing_y0 is not None:
            peak_fwd = max(peak_fwd, -(f[1] - swing_y0) * 1000.0)
        if phase in ("ss_swing", "ss_desc"):
            peak_clear = max(peak_clear, (f[2] - 1.032) * 1000.0)
        peak_up = max(peak_up, bs.up_tilt_deg)

        if bs.up_tilt_deg > 55 or bs.chest_z < NOMINAL_CHEST_Z - 0.24:
            result.update(fell=True, fell_t=k)
            if verbose:
                print(f"  FELL @ {k} ms  up {bs.up_tilt_deg:.0f}  fwd {bs.fwd_lean_deg:+.0f}  side {bs.side_lean_deg:+.0f}")
            break

        # ---------------- transitions ----------------
        if phase == "stand":
            if post and k > PUSH_AT + PUSH_DURATION_STEPS + cfg.trig_min_ms:
                diverging = (bs.capture_fwd_rel_support_mm > cfg.trig_capt_mm
                             and bs.com_vfwd > cfg.trig_vfwd)
                trig_streak = trig_streak + 1 if diverging else 0
                if trig_streak >= cfg.trig_hold:
                    phase = "step"
                    sk = 0
                    swing_started = foot_lifted = False
                    plant_streak = 0
                    result["triggered"] = True
                    if verbose:
                        print(f"  >> STEP {len(steps)+1} @ {k} ms  swing={swing}  "
                              f"capt {bs.capture_fwd_rel_support_mm:+.0f}mm  vfwd {bs.com_vfwd:.2f}")
                elif k > PUSH_AT + PUSH_DURATION_STEPS + cfg.trig_deadline_ms:
                    phase = "settle"
                    sk = 0
                    if verbose:
                        print("  no step needed (LQR held)")

        elif phase == "step":
            shift_min = sk >= cfg.imp_ramp_ms + cfg.imp_hold_ms + cfg.preshift_ms
            # hand to ss.K only when the weight-shift's LATERAL CoM velocity has
            # settled - a residual sideways coast toward the stance foot is what
            # the marginal single-support LQR cannot then arrest (it coasts the
            # CoM past the stance-foot edge).  Wait for the shift oscillation to
            # cross ~zero, or bail at the hard deadline.
            latv = abs(_lat_com_vel(m, d))
            shift_done = (shift_min and latv < cfg.handoff_latvel) or sk >= cfg.swing_latest_ms
            foot_light = sw_nf < cfg.swing_start_nf or sk >= cfg.swing_start_ms
            if shift_done and foot_light and (pr < cfg.swing_pr_gate or sk >= cfg.swing_latest_ms):
                phase = "ss_swing"
                swing_t0 = k
                swing_started = True
                sk = 0
                swing_y0 = _foot_xy_z(m, d, swing)[1]
                swing_x0 = _foot_xy_z(m, d, swing)[0]
                if verbose:
                    print(f"  >> SS_SWING @ {k} ms  nf {sw_nf:.1f}  up {bs.up_tilt_deg:.1f}  "
                          f"side {bs.side_lean_deg:+.1f}  pr {pr:+.2f}")

        elif phase == "ss_swing":
            if not foot_lifted and sw_nf < 3.0:
                foot_lifted = True
            if sk >= cfg.swing_ms:
                phase = "ss_desc"
                sk = 0
                if verbose:
                    print(f"  >> SS_DESC @ {k} ms  fwd {peak_fwd:.0f}mm  clr {peak_clear:.0f}mm  "
                          f"sole {np.degrees(sp):+.1f}  up {bs.up_tilt_deg:.1f}  side {bs.side_lean_deg:+.1f}")

        elif phase == "ss_desc":
            genuine = (foot_lifted and getattr(bs, f"{swing.lower()}_contact")
                       and sw_nf > cfg.plant_nf)
            plant_streak = plant_streak + 1 if genuine else 0
            planted = plant_streak >= cfg.plant_hold
            if planted or sk >= cfg.descend_ms + cfg.descend_extra_ms:
                sfx = _foot_xy_z(m, d, swing)[0]
                sf = _foot_xy_z(m, d, swing)[1]
                stf = _foot_xy_z(m, d, "L" if swing == "R" else "R")[1]
                plant_q = np.array([d.qpos[7 + sw["hip"]], d.qpos[7 + sw["knee"]],
                                    d.qpos[7 + sw["ankle"]]])
                steps.append(dict(k=k, planted=bool(planted),
                                  sep_mm=-(sf - stf) * 1000.0,
                                  lat_mm=(sfx - swing_x0) * 1000.0,
                                  vfwd=bs.com_vfwd, side=bs.side_lean_deg,
                                  sole=np.degrees(sp), nf=sw_nf))
                if verbose:
                    print(f"  >> {'PLANT' if planted else 'TIMEOUT'} @ {k} ms  "
                          f"sep fwd {-(sf-stf)*1000:.0f}mm  nf {sw_nf:.0f}  "
                          f"sole {np.degrees(sp):+.1f}  vfwd {bs.com_vfwd:.2f}  side {bs.side_lean_deg:+.1f}")
                phase = "transfer"
                sk = 0

        elif phase == "transfer":
            caught = (abs(bs.side_lean_deg) < cfg.catch_exit_side
                      and abs((d.xmat[CHEST_BODY].reshape(3, 3) @ d.qvel[3:6])[1]) < 1.0)
            if sk >= cfg.transfer_ms or (caught and sk > 60):
                phase = "settle"
                sk = 0
                restep_streak = 0
                if verbose:
                    print(f"  >> SETTLE @ {k} ms  up {bs.up_tilt_deg:.1f}  side {bs.side_lean_deg:+.1f}  "
                          f"swNF {sw_nf:.0f}  stNF {st_nf:.0f}  vfwd {bs.com_vfwd:.2f}")

        elif phase == "settle":
            if sk >= cfg.settle_ms:
                break
            if len(steps) < cfg.max_steps and sk > cfg.restep_after_ms:
                need = (bs.capture_fwd_rel_support_mm > cfg.restep_capt_mm
                        and bs.com_vfwd > cfg.restep_vfwd)
                restep_streak = restep_streak + 1 if need else 0
                if restep_streak >= cfg.restep_hold:
                    swing = "L" if swing == "R" else "R"
                    sw = LEG[swing]
                    stn = LEG["L" if swing == "R" else "R"]
                    hf = FWD_HIP_SIGN[swing]
                    ss_idx = (sw["hip"], sw["knee"], sw["ankle"])
                    # rebuild ss for the other swing leg
                    scfg2 = StepConfig(swing=swing, ankle_roll_amp_rad=cfg.ankle_roll_amp,
                                       ss_reach_steps=cfg.ss_reach_steps)
                    ss = _SingleSupportLQR(m, d, stand_ab, scfg2, verbose=False)
                    lean_ss = float(ss.ctrl0[AR[swing]])
                    h0 = ss.qpos0[7 + sw["hip"]]
                    k0 = ss.qpos0[7 + sw["knee"]]
                    a0 = ss.qpos0[7 + sw["ankle"]]
                    an_cmd = a0
                    phase = "step"
                    sk = 0
                    swing_started = foot_lifted = False
                    plant_streak = 0
                    if verbose:
                        print(f"  >> STEP {len(steps)+1} @ {k} ms  still diverging "
                              f"(capt {bs.capture_fwd_rel_support_mm:+.0f}mm) - swing {swing}")

        if trace and k % 20 == 0:
            print(f"   t{k:5d} [{phase:8s}] up{bs.up_tilt_deg:5.1f} fwd{bs.fwd_lean_deg:+6.1f} "
                  f"side{bs.side_lean_deg:+6.1f} capt{bs.capture_fwd_rel_support_mm:+6.0f} "
                  f"vF{bs.com_vfwd:+5.2f} swNF{sw_nf:4.0f} stNF{st_nf:4.0f} "
                  f"swZ{(f[2]-1.032)*1000:+4.0f} sole{np.degrees(sp):+5.1f}")

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(0.02 if slow else 0.0015)

    bs = sample_balance(m, d)
    ds = bs.l_contact and bs.r_contact
    n_planted = sum(1 for s in steps if s["planted"])
    sp = _sole_pitch(m, d, swing)
    ended_stable = (not result["fell"] and ds and bs.up_tilt_deg < 8
                    and abs(bs.side_lean_deg) < 8 and bs.com_speed_horiz < 0.09
                    and bs.chest_z > NOMINAL_CHEST_Z - 0.10)
    physically_continuous = peak_up <= 35.0
    recovered = ended_stable and physically_continuous
    success = result["triggered"] and recovered and n_planted >= 1
    final_sep_mm = -(_foot_xy_z(m, d, "R")[1] - _foot_xy_z(m, d, "L")[1]) * 1000.0
    result.update(n_steps=len(steps), n_planted=n_planted, end_ds=ds,
                  end_up=bs.up_tilt_deg, end_side=bs.side_lean_deg,
                  end_speed=bs.com_speed_horiz, end_sole_deg=np.degrees(sp),
                  com_moved_mm=(bs.com_fwd - com_fwd0) * 1000.0,
                  peak_up=peak_up, peak_fwd=peak_fwd, peak_clear=peak_clear,
                  final_sep_mm=final_sep_mm, steps=steps,
                  recovered=recovered, success=success)
    if verbose:
        print(f"\n  push {push_n:.0f} N   steps={len(steps)} planted={n_planted}   "
              f"peak fwd {peak_fwd:.0f}mm  peak clr {peak_clear:.0f}mm  "
              f"CoM fwd {result['com_moved_mm']:+.0f}mm  final sep {final_sep_mm:+.0f}mm")
        for i, s in enumerate(steps):
            print(f"    step {i+1}: {'PLANT' if s['planted'] else 'no-load'} @ {s['k']} ms  "
                  f"sep {s['sep_mm']:+.0f}mm  lat {s['lat_mm']:+.0f}mm  sole {s['sole']:+.1f}  "
                  f"vfwd {s['vfwd']:.2f}  side {s['side']:+.1f}")
        print(f"  end: up {bs.up_tilt_deg:.1f}  side {bs.side_lean_deg:+.1f}  "
              f"sole {np.degrees(sp):+.1f}  DS {ds}  speed {bs.com_speed_horiz:.3f}  peakUp {peak_up:.1f}")
        print(f"  >>> {'SUCCESS' if success else ('FELL' if result['fell'] else 'FAILED - not a clean recovery')}")
    if viewer is not None:
        try:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02 if slow else 0.0015)
        except KeyboardInterrupt:
            pass
        viewer.close()
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--push", type=float, default=135.0)
    p.add_argument("--swing", choices=["L", "R"], default="R")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--sweep", default=None)
    p.add_argument("--model", default="robot/robot.xml")
    a = p.parse_args(argv)
    cfg = Cfg(swing=a.swing)
    if a.sweep:
        rows = [run(float(x), cfg, verbose=True, model_path=a.model) for x in a.sweep.split(",")]
        print("\n" + "=" * 70)
        for r in rows:
            print(f"  {r['push_n']:5.0f} N  trig={str(r['triggered']):>5}  "
                  f"steps={r.get('n_steps',0)} planted={r.get('n_planted',0)}  "
                  f"peakFwd {r.get('peak_fwd',0):3.0f}  peakClr {r.get('peak_clear',0):3.0f}  "
                  f"sep {r.get('final_sep_mm',0):+4.0f}  peakUp {r.get('peak_up',0):4.1f}  "
                  f"{'SUCCESS' if r['success'] else ('FELL' if r['fell'] else 'fail')}")
        return
    run(a.push, cfg, show=not a.headless, slow=a.slow, trace=a.trace, model_path=a.model)


if __name__ == "__main__":
    main(sys.argv[1:])
