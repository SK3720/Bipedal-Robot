"""Forward recovery step with a DEDICATED analytical frontal-plane regulator.

Rationale (from frontal_capture_probe.py + frontal_shift_feasibility.py):
  * Under StandingLQR the weight shift reaches a RECOVERABLE single-support state
    (capture point stays ~15-35 mm inside the 66 mm stance sole, v_lat ~0.15 m/s)
    - but only after ~400-500 ms of ringing, with the foot bouncing on/off.
  * The ss.K hand-off SPIKES v_lat 0.15 -> 0.26 (capture point then exits) - a
    controller artefact, not physics.
  * In double support a ~0.08 rad both-ankle-roll bias moves the CoP ~54 mm
    (measured), so there is real frontal control authority; it just isn't being
    used deliberately.

So: drop ss.K entirely.  The frontal plane [x_com_lat, v_lat] is a 2-state linear
inverted pendulum (omega^2 = g/h ~= 36.3).  Regulate it with a 2-gain controller
whose gains come from POLE PLACEMENT on that 2x2 system (no hand tuning), acting
through a CoP command -> both-ankle-roll bias (measured static gain).  Everything
else (sagittal, posture, stance/swing leg) stays on StandingLQR with its roll
columns zeroed so the two controllers don't fight.  The forward swing is the
existing feed-forward arc + AnkleSolver.

Phases: STAND -> SHIFT (x_target ramps to the single-support point) -> SWING
(forward arc; frontal LQR holds the CoM over the stance foot) -> PLANT ->
TRANSFER (x_target eases back toward mid-stance as load moves onto the new foot)
-> SETTLE (pure StandingLQR).

    python frontal_lqr_step.py --headless --trace --push 132
    python frontal_lqr_step.py --headless --sweep 124,128,132,136,140
    python frontal_lqr_step.py --slow
Does not modify robot.xml / biped_env / sagittal_recovery / any golden script.
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
from push_step_recovery_test import StepConfig, _LQRAbout, _SingleSupportLQR, _pos_error, _smooth
from step_primitive import AnkleSolver, _sole_pitch, LEG, AR, FWD_HIP_SIGN, KNEE_FLEX_SIGN

PUSH_AT = 20
G, FLOOR_Z = 9.81, 1.0
AR_UNLOAD_SIGN = {"R": -1.0, "L": +1.0}


def _cop_x(m, d):
    num = den = 0.0
    w = np.zeros(6)
    for ci in range(d.ncon):
        c = d.contact[ci]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if "foot_collision" not in g1 and "foot_collision" not in g2:
            continue
        mujoco.mj_contactForce(m, d, ci, w)
        fn = max(0.0, float(w[0]))
        num += c.pos[0] * fn
        den += fn
    return (num / den if den > 1e-6 else np.nan), den


def _frontal_gains(omega2, p_real=-9.0, p_imag=5.0):
    """pole-placement on  xdd = omega2 (x - u),  u = CoP.
    closed loop  s^2 - (omega2*kv) s - omega2*(1+kx) = (s-p)(s-p*) = s^2 -2 Re(p) s + |p|^2
    -> kv = 2 Re(p) / omega2 ,  kx = -( |p|^2 / omega2 ) - 1
    control:  u = x_ref - kx*(x - x_ref) - kv*xdot
    """
    re = p_real
    mag2 = p_real ** 2 + p_imag ** 2
    kv = (2.0 * re) / omega2
    kx = -(mag2 / omega2) - 1.0
    return kx, kv


@dataclass
class Cfg:
    swing: str = "R"
    # trigger (same as sagittal_recovery)
    trig_capt_mm: float = 6.0
    trig_vfwd: float = 0.05
    trig_hold: int = 12
    trig_deadline_ms: int = 1200
    # impulse to break the swing foot loose
    imp_amp: float = 0.22
    imp_ramp_ms: int = 8
    imp_hold_ms: int = 40
    # frontal LIPM regulator - gentle poles (near natural omega), the ankle-roll
    # actuator saturates so aggressive gains just bang-bang and overshoot.
    front_pole_re: float = -5.0
    front_pole_im: float = 3.0
    cop_per_rad: float = 0.62         # measured: dCoP(m) per rad both-ankle-roll bias
    ar_bias_max: float = 0.14
    x_mid_mm: float = 35.0            # CoP/CoM lateral at feet-together stance
    x_ss_mm: float = 50.0            # target CoM lateral - just enough to unload the
    #                                   swing foot (measured: unloads by ~+47 mm),
    #                                   NOT the full lean.  Keeps it a REGULATION
    #                                   problem, not a big saturating move.
    cop_lo_mm: float = 5.0            # CoP command clamp (stance sole ~[+35,+100])
    cop_hi_mm: float = 96.0
    x_transfer_mm: float = 24.0      # staggered double-support polygon centre
    shift_ms: int = 200
    shift_min_ms: int = 140          # don't start the swing before this
    unload_nf: float = 6.0           # ... and not before the swing foot is this light
    # forward swing
    swing_start_nf: float = 6.0
    swing_start_ms: int = 150
    swing_pr_gate: float = 0.4        # start the swing when torso pitch-rate < this
    swing_ms: int = 165
    descend_ms: int = 80
    swing_hip_rad: float = 0.60
    swing_knee_rad: float = 0.36
    knee_land_frac: float = 0.07
    hip_retract_frac: float = 0.24
    sole_toe_up_mid: float = 0.09
    # plant / transfer
    plant_nf: float = 12.0
    plant_hold: int = 10
    step_max_ms: int = 460
    transfer_ms: int = 220
    # settle
    settle_ms: int = 2400
    max_steps: int = 2
    restep_capt_mm: float = 14.0
    restep_vfwd: float = 0.12
    restep_hold: int = 15
    restep_after_ms: int = 160
    stance_cap: float = 0.35
    swing_hiproll_cap: float = 0.06


def _swing_arc(cfg, phase, w, h0, k0):
    hf = FWD_HIP_SIGN[cfg.swing]
    if phase == "swing":
        s = _smooth(w)
        hip = h0 + hf * cfg.swing_hip_rad * s
        knee = k0 + KNEE_FLEX_SIGN * (0.05 + (cfg.swing_knee_rad - 0.05) * np.sin(np.pi * w))
        sole = cfg.sole_toe_up_mid * np.sin(np.pi * w)
    else:
        s = _smooth(min(1.0, w))
        over = min(1.0, max(0.0, w - 1.0))
        hip = h0 + hf * cfg.swing_hip_rad * (1.0 - cfg.hip_retract_frac * s - 0.30 * over)
        knee = k0 + KNEE_FLEX_SIGN * (cfg.knee_land_frac + 0.20 * (1.0 - s) - 0.45 * over)
        sole = cfg.sole_toe_up_mid * 0.4 * (1.0 - s)
    return hip, knee, sole


def run(push_n, cfg: Cfg, show=False, slow=False, trace=False, verbose=True,
        model_path="robot/robot.xml"):
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    stand = StandingLQR(m, d, verbose=verbose)
    # ss only used for the swing-leg reference pose (h0,k0,a0) - NOT its K
    ab = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="ab", verbose=False)
    ssref = _SingleSupportLQR(m, d, ab, StepConfig(swing=cfg.swing, ankle_roll_amp_rad=0.12,
                                                   ss_reach_steps=340), verbose=False)
    ank = AnkleSolver(m)
    swing = cfg.swing
    sw = LEG[swing]
    stn = LEG["L" if swing == "R" else "R"]
    hf = FWD_HIP_SIGN[swing]
    ar_unload = AR_UNLOAD_SIGN[swing]
    h0 = ssref.qpos0[7 + sw["hip"]]
    k0 = ssref.qpos0[7 + sw["knee"]]
    a0 = ssref.qpos0[7 + sw["ankle"]]

    omega2 = G / (NOMINAL_CHEST_Z - FLOOR_Z + 0.0)  # ~ g / 0.27
    kx, kv = _frontal_gains(omega2, cfg.front_pole_re, cfg.front_pole_im)
    # unload direction in world X: stance L is +X, so leaning onto L is +X;
    # swing R -> stance L -> x_target goes MORE +X.  swing L -> stance R -> -X.
    xdir = +1.0 if swing == "R" else -1.0

    d.qpos[:] = stand.qpos0
    d.qvel[:] = stand.qvel0
    mujoco.mj_forward(m, d)
    fxy = np.array([0.0, -push_n]) if True else None
    _pd = np.array([0.0, -1.0])
    fxy = push_n * _pd

    viewer = None
    if show:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.cam.lookat[:] = [0.0, -0.3, 1.1]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 100
        viewer.cam.elevation = -6

    phase = "stand"
    sk = 0
    trig_streak = plant_streak = restep_streak = 0
    swing_started = foot_lifted = False
    swing_t0 = 0
    an_cmd = a0
    plant_q = np.array([h0, k0, a0])
    x0_mid = None
    steps = []
    peak_up = 0.0
    peak_fwd = peak_clear = 0.0
    swing_x0 = swing_y0 = swing_z0 = None
    result = dict(push_n=push_n, triggered=False, fell=False)
    rows = []

    def frontal_bias(x_ref_m):
        """CoP command from the LIPM regulator -> both-ankle-roll bias."""
        bs = sample_balance(m, d)
        x = float(bs.com[0])
        v = float(bs.com_vel[0])
        cop_cmd = x_ref_m - kx * (x - x_ref_m) - kv * v
        # CoP is physically bounded to the stance sole; clamp the *command* to the
        # actual sole range (sign-flipped for a left-side stance)
        lo = cfg.cop_lo_mm / 1000.0 * xdir
        hi = cfg.cop_hi_mm / 1000.0 * xdir
        cop_cmd = float(np.clip(cop_cmd, min(lo, hi), max(lo, hi)))
        # CoP offset from where it is now -> ankle-roll bias.  +CoP(+X) needs
        # NEGATIVE both-ankle-roll (measured: bias -0.08 -> CoP +54 mm).
        bias = -(cop_cmd - x) / cfg.cop_per_rad
        return float(np.clip(bias, -cfg.ar_bias_max, cfg.ar_bias_max)), x, v, cop_cmd

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
        pr = bs.pitch_rate

        # ---------- x_target schedule (lateral CoM setpoint) ----------
        if phase in ("stand",):
            x_ref = cfg.x_mid_mm / 1000.0 * xdir
        elif phase == "shift":
            fr = _smooth(min(1.0, sk / cfg.shift_ms))
            x_ref = (cfg.x_mid_mm + fr * (cfg.x_ss_mm - cfg.x_mid_mm)) / 1000.0 * xdir
        elif phase in ("swing", "descend"):
            x_ref = cfg.x_ss_mm / 1000.0 * xdir
        elif phase == "transfer":
            # hold the single-support setpoint; the bias is faded out separately
            x_ref = cfg.x_ss_mm / 1000.0 * xdir
        elif phase == "settle":
            x_ref = cfg.x_transfer_mm / 1000.0 * xdir
        else:
            x_ref = None

        # ---------- base control ----------
        if phase == "stand":
            u = stand.control(m, d)
            bias = 0.0
        elif phase == "settle":
            u = stand.control(m, d)
            for ci in (stn["hip"], stn["knee"], stn["ankle"],
                       sw["hip"], sw["knee"], sw["ankle"]):
                u[ci] = float(np.clip(u[ci], stand.ctrl0[ci] - 0.7, stand.ctrl0[ci] + 0.7))
            # keep the frontal LQR steering the lateral CoM to the staggered-stance
            # polygon centre for the first ~700 ms, then decay to StandingLQR
            decay = max(0.0, 1.0 - sk / 700.0)
            fb, _, _, _ = frontal_bias(x_ref)
            bias = fb * decay
        else:
            # StandingLQR for sagittal + posture, roll columns zeroed
            dq = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, stand.qpos0, d.qpos)
            dx = np.concatenate([dq, d.qvel - stand.qvel0])
            for ci in (AR["L"], AR["R"], stn["hip_roll"], sw["hip_roll"]):
                dx[6 + ci] = 0.0
                dx[m.nv + 6 + ci] = 0.0
            dx[0] = dx[m.nv + 0] = 0.0        # lateral handled by the frontal LQR
            u = stand.ctrl0 - stand.K @ dx

            # ---- frontal LIPM regulator -> both-ankle-roll bias ----
            if phase == "shift" and sk < cfg.imp_ramp_ms + cfg.imp_hold_ms:
                # initial impulse to break the foot loose
                s = sk
                if s < cfg.imp_ramp_ms:
                    bias = ar_unload * cfg.imp_amp * (s / cfg.imp_ramp_ms)
                else:
                    bias = ar_unload * cfg.imp_amp
                _fx = _fv = _fc = 0.0
            else:
                bias, _fx, _fv, _fc = frontal_bias(x_ref)

            # ---- swing leg feed-forward ----
            if phase in ("swing", "descend"):
                seg = "swing" if phase == "swing" else "descend"
                dur = cfg.swing_ms if seg == "swing" else cfg.descend_ms
                w = (min(1.0, sk / dur) if seg == "swing" else sk / dur)
                hip, knee, sole_tgt = _swing_arc(cfg, seg, w, h0, k0)
                an_cmd = ank.solve(d.qpos, swing, hip, knee, sole_tgt, prev=an_cmd)
                for ci, val in zip((sw["hip"], sw["knee"], sw["ankle"]),
                                   (hip, knee, an_cmd)):
                    u[ci] = val
            elif phase == "transfer":
                b = _smooth(min(1.0, sk / cfg.transfer_ms))
                u[sw["hip"]] = plant_q[0]
                u[sw["knee"]] = plant_q[1] + KNEE_FLEX_SIGN * 0.06 * b
                u[sw["ankle"]] = plant_q[2] + hf * 0.06 * b

        u = np.array(u, float)
        # apply the frontal bias to BOTH ankle rolls
        if phase != "stand":
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] + bias
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] + bias
        if phase not in ("stand", "settle"):
            # keep the swing hip_roll near neutral (don't let anything splay it)
            u[sw["hip_roll"]] = float(np.clip(u[sw["hip_roll"]],
                                              DEFAULT_POSE[sw["hip_roll"]] - cfg.swing_hiproll_cap,
                                              DEFAULT_POSE[sw["hip_roll"]] + cfg.swing_hiproll_cap))
            for ci in (stn["hip"], stn["knee"], stn["ankle"]):
                u[ci] = float(np.clip(u[ci], DEFAULT_POSE[ci] - cfg.stance_cap,
                                      DEFAULT_POSE[ci] + cfg.stance_cap))

        u = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
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
        if phase in ("swing", "descend"):
            peak_clear = max(peak_clear, (f[2] - swing_z0) * 1000.0)
        peak_up = max(peak_up, bs.up_tilt_deg)
        cop_now, cop_load = _cop_x(m, d)
        tau_arl = float(d.actuator_force[AR["L" if swing == "R" else "R"]])

        if bs.up_tilt_deg > 55 or bs.chest_z < NOMINAL_CHEST_Z - 0.24:
            result.update(fell=True)
            if verbose:
                print(f"  FELL @ {k}  up {bs.up_tilt_deg:.0f}  fwd {bs.fwd_lean_deg:+.0f}  side {bs.side_lean_deg:+.0f}")
            break

        # ---------- transitions ----------
        if phase == "stand":
            if post and bs.capture_fwd_rel_support_mm > cfg.trig_capt_mm and bs.com_vfwd > cfg.trig_vfwd:
                trig_streak += 1
            else:
                trig_streak = 0
            if trig_streak >= cfg.trig_hold:
                phase = "shift"
                sk = 0
                result["triggered"] = True
                x0_mid = float(bs.com[0])
                if verbose:
                    print(f"  >> SHIFT @ {k}  capt {bs.capture_fwd_rel_support_mm:+.0f}  vfwd {bs.com_vfwd:.2f}")
            elif post and k > PUSH_AT + PUSH_DURATION_STEPS + cfg.trig_deadline_ms:
                phase = "settle"
                sk = 0

        elif phase == "shift":
            if not swing_started and sk >= cfg.shift_min_ms \
               and (sw_nf < cfg.unload_nf or sk >= cfg.swing_start_ms) \
               and (abs(pr) < cfg.swing_pr_gate or sk >= cfg.swing_start_ms):
                phase = "swing"
                swing_started = True
                swing_t0 = k
                sk = 0
                swing_x0 = _foot_xy_z(m, d, swing)[0]
                swing_y0 = _foot_xy_z(m, d, swing)[1]
                swing_z0 = _foot_xy_z(m, d, swing)[2]
                mujoco.mj_subtreeVel(m, d)
                if verbose:
                    print(f"  >> SWING @ {k}  swNF {sw_nf:.1f}  xCoM {bs.com[0]*1000:+.0f}  "
                          f"vLat {bs.com_vel[0]:+.3f}  side {bs.side_lean_deg:+.1f}")

        elif phase == "swing":
            if not foot_lifted and sw_nf < 3.0:
                foot_lifted = True
            if sk >= cfg.swing_ms:
                phase = "descend"
                sk = 0
                if verbose:
                    print(f"  >> DESCEND @ {k}  fwd {peak_fwd:.0f}  clr {peak_clear:.0f}  "
                          f"side {bs.side_lean_deg:+.1f}  vLat {bs.com_vel[0]:+.3f}")

        elif phase == "descend":
            genuine = (foot_lifted and getattr(bs, f"{swing.lower()}_contact") and sw_nf > cfg.plant_nf)
            plant_streak = plant_streak + 1 if genuine else 0
            if plant_streak >= cfg.plant_hold or sk >= cfg.step_max_ms:
                planted = plant_streak >= cfg.plant_hold
                sf = _foot_xy_z(m, d, swing)[1]
                stf = _foot_xy_z(m, d, "L" if swing == "R" else "R")[1]
                plant_q = np.array([d.qpos[7 + sw["hip"]], d.qpos[7 + sw["knee"]],
                                    d.qpos[7 + sw["ankle"]]])
                steps.append(dict(k=k, planted=bool(planted), sep_mm=-(sf - stf) * 1000.0,
                                  side=bs.side_lean_deg, vlat=float(bs.com_vel[0]),
                                  vfwd=bs.com_vfwd, sole=np.degrees(_sole_pitch(m, d, swing)),
                                  nf=sw_nf))
                if verbose:
                    print(f"  >> {'PLANT' if planted else 'TIMEOUT'} @ {k}  sep {-(sf-stf)*1000:+.0f}mm  "
                          f"nf {sw_nf:.0f}  side {bs.side_lean_deg:+.1f}  vLat {bs.com_vel[0]:+.3f}  "
                          f"sole {np.degrees(_sole_pitch(m,d,swing)):+.1f}")
                phase = "transfer"
                sk = 0

        elif phase == "transfer":
            settled = (st_nf > 6.0 and sw_nf > 6.0 and abs(bs.side_lean_deg) < 7.0
                       and abs(float(bs.com_vel[0])) < 0.10)
            if sk >= cfg.transfer_ms or (settled and sk > 60):
                phase = "settle"
                sk = 0
                restep_streak = 0
                if verbose:
                    print(f"  >> SETTLE @ {k}  up {bs.up_tilt_deg:.1f}  side {bs.side_lean_deg:+.1f}  "
                          f"swNF {sw_nf:.0f}  stNF {st_nf:.0f}")

        elif phase == "settle":
            if sk >= cfg.settle_ms:
                break
            if len(steps) < cfg.max_steps and sk > cfg.restep_after_ms:
                need = (bs.capture_fwd_rel_support_mm > cfg.restep_capt_mm and bs.com_vfwd > cfg.restep_vfwd)
                restep_streak = restep_streak + 1 if need else 0
                if restep_streak >= cfg.restep_hold:
                    swing = "L" if swing == "R" else "R"
                    sw = LEG[swing]; stn = LEG["L" if swing == "R" else "R"]
                    hf = FWD_HIP_SIGN[swing]; ar_unload = AR_UNLOAD_SIGN[swing]
                    xdir = +1.0 if swing == "R" else -1.0
                    h0 = ssref.qpos0[7 + sw["hip"]]; k0 = ssref.qpos0[7 + sw["knee"]]
                    a0 = ssref.qpos0[7 + sw["ankle"]]; an_cmd = a0
                    phase = "shift"; sk = 0
                    swing_started = foot_lifted = False
                    plant_streak = 0
                    if verbose:
                        print(f"  >> STEP {len(steps)+1} @ {k}  still diverging - swing {swing}")

        if trace and k % 20 == 0 and phase != "stand":
            print(f"   t{k:5d} [{phase:8s}] up{bs.up_tilt_deg:5.1f} fwd{bs.fwd_lean_deg:+6.1f} "
                  f"side{bs.side_lean_deg:+6.1f} xCoM{bs.com[0]*1000:+6.1f} vLat{bs.com_vel[0]:+.3f} "
                  f"CoP{(cop_now*1000 if not np.isnan(cop_now) else 0):+6.1f} tauAR{tau_arl:+.2f} "
                  f"bias{bias:+.3f} swNF{sw_nf:4.0f} stNF{st_nf:4.0f} swZ{(f[2]-1.032)*1000:+4.0f}")

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
    success = result["triggered"] and ended_stable and peak_up <= 35.0 and n_planted >= 1
    final_sep = -(_foot_xy_z(m, d, "R")[1] - _foot_xy_z(m, d, "L")[1]) * 1000.0
    result.update(n_steps=len(steps), n_planted=n_planted, end_ds=ds, end_up=bs.up_tilt_deg,
                  end_side=bs.side_lean_deg, end_speed=bs.com_speed_horiz,
                  end_sole_deg=np.degrees(sp), peak_up=peak_up, peak_fwd=peak_fwd,
                  peak_clear=peak_clear, final_sep_mm=final_sep, steps=steps, success=success)
    if verbose:
        print(f"\n  push {push_n:.0f} N   gains kx={kx:.2f} kv={kv:.2f}   "
              f"steps={len(steps)} planted={n_planted}   peakFwd {peak_fwd:.0f}mm "
              f"peakClr {peak_clear:.0f}mm  finalSep {final_sep:+.0f}mm")
        for i, s in enumerate(steps):
            print(f"    step {i+1}: {'PLANT' if s['planted'] else 'no-load'} @ {s['k']}  "
                  f"sep {s['sep_mm']:+.0f}mm  side {s['side']:+.1f}  vLat {s['vlat']:+.3f}  "
                  f"sole {s['sole']:+.1f}  nf {s['nf']:.0f}")
        print(f"  end: up {bs.up_tilt_deg:.1f}  side {bs.side_lean_deg:+.1f}  sole {np.degrees(sp):+.1f}  "
              f"DS {ds}  speed {bs.com_speed_horiz:.3f}  peakUp {peak_up:.1f}")
        print(f"  >>> {'SUCCESS' if success else ('FELL' if result['fell'] else 'FAILED')}")
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
    p.add_argument("--push", type=float, default=132.0)
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
                  f"planted={r.get('n_planted',0)}  peakFwd {r.get('peak_fwd',0):3.0f}  "
                  f"peakClr {r.get('peak_clear',0):3.0f}  sep {r.get('final_sep_mm',0):+4.0f}  "
                  f"peakUp {r.get('peak_up',0):4.1f}  "
                  f"{'SUCCESS' if r['success'] else ('FELL' if r['fell'] else 'fail')}")
        return
    run(a.push, cfg, show=not a.headless, slow=a.slow, trace=a.trace, model_path=a.model)


if __name__ == "__main__":
    main(sys.argv[1:])
