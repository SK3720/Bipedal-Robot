"""Trajectory-feasibility experiments for a >=60 mm forward recovery step.

Follows the analytical assessment + the CMA-ES bridge finding (a bigger fixed
swing arc collapses recovery: hip>=0.5 rad -> 0 % success).  Question: can a
*freely shaped* time-varying trajectory - swing leg alone (Stage A), then whole
body incl. arms (Stage B), then heavier hands (Stage C) - take a >=60 mm step
and still recover, under +-2.3 N.m?

Reuses frontal_lqr_step's proven pieces UNMODIFIED (StandingLQR base, frontal
LIPM regulator for the lateral plane, single-support reference pose, contact
metrics).  The only thing replaced is the swing-leg (and, Stage B+, arm/stance)
joint *trajectory*, represented as per-joint knot splines the optimizer shapes.

Instrumentation per run: whole-body + per-limb angular momentum about the global
CoM (rigorous transfer), torso pitch + pitch rate, actuator torque saturation,
foot separation, touchdown state, final stability.

    python wbtraj_opt.py --stage A                 # swing-leg only
    python wbtraj_opt.py --stage B                 # + arms + stance
    python wbtraj_opt.py --stage C                 # heavier-hands sweep
    python wbtraj_opt.py --all                     # everything + comparison table
    python wbtraj_opt.py --replay results/wb_stageB.json --push 132   # watch/trace
Does NOT modify robot.xml, frontal_lqr_step.py, or any golden controller.
Stage C writes experimental models to robot/_exp_hands_*.xml (derived, isolated).
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

print = functools.partial(print, flush=True)

import mujoco

from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS
from recovery_metrics import (
    CHEST_BODY, NOMINAL_CHEST_Z, _foot_normal_force, _foot_xy_z, sample_balance,
)
import frontal_lqr_step as F
from frontal_lqr_step import _frontal_gains, _cop_x, _smooth

PUSH_AT = 20
G, FLOOR_Z = 9.81, 1.0
AR = {"L": 9, "R": 14}
HIPR = {"L": 5, "R": 10}
LEGJ = {"L": dict(hip=6, knee=7, ankle=8), "R": dict(hip=11, knee=12, ankle=13)}
ARM_CI = (1, 2, 3, 4)          # L_shoulder, L_elbow, R_shoulder, R_elbow
BODY = dict(chest=1, l_arm=3, r_arm=5, l_hip=7, r_hip=12)
WK = np.array([0.0, 0.35, 0.70, 1.00, 1.40])     # knot phases (w=0 pinned to 0)

# ---- optimised-joint sets ---------------------------------------------------
# (ctrl_idx, ref_source, (lo,hi) delta bounds)   ref: 'ss' pose | 'def' 0 | 'lqr' add
JOINTS_A = [(12, "ss", (-1.30, 1.30)),   # R knee   (swing leg, swing='R')
            (11, "ss", (-1.20, 1.20)),   # R hip
            (13, "ss", (-0.70, 0.70))]   # R ankle
JOINTS_B = JOINTS_A + [
    (1, "def", (-1.90, 0.10)),           # L shoulder (range [-3.58, 0.17])
    (3, "def", (-0.10, 1.90)),           # R shoulder (range [-0.17, 3.58])
    (2, "def", (-1.60, 1.60)),           # L elbow
    (4, "def", (-1.60, 1.60)),           # R elbow
    (6, "lqr", (-0.45, 0.45)),           # L hip  (stance)
    (8, "lqr", (-0.45, 0.45))]           # L ankle (stance)


@dataclass
class TrajCfg:
    swing: str = "R"
    # base timing / lateral (seeded from the CMA-A optimum)
    swing_ms: int = 150
    descend_ms: int = 80
    swing_start_ms: int = 110
    shift_min_ms: int = 122
    shift_ms: int = 135
    x_ss_mm: float = 60.0
    x_transfer_mm: float = 20.0
    front_pole_re: float = -5.7
    front_pole_im: float = 2.3
    cop_per_rad: float = 0.62
    ar_bias_max: float = 0.16
    cop_lo_mm: float = 5.0
    cop_hi_mm: float = 96.0
    x_mid_mm: float = 35.0
    imp_amp: float = 0.30
    imp_ramp_ms: int = 8
    imp_hold_ms: int = 40
    trig_capt_mm: float = 6.0
    trig_vfwd: float = 0.05
    trig_hold: int = 12
    trig_deadline_ms: int = 1200
    transfer_ms: int = 150
    settle_ms: int = 2200
    plant_nf: float = 12.0
    plant_hold: int = 10
    step_max_ms: int = 520
    swing_hiproll_cap: float = 0.06
    stance_cap: float = 0.40
    # trajectory knots: {ctrl_idx: np.array([4 deltas at WK[1:]])}
    knots: dict = field(default_factory=dict)


# --------------------------------------------------------------- build (cached)
_BUILD = {}


def build(model_path):
    if model_path in _BUILD:
        return _BUILD[model_path]
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    stand = F.StandingLQR(m, d, verbose=False)
    from push_step_recovery_test import StepConfig, _LQRAbout, _SingleSupportLQR
    ab = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="ab", verbose=False)
    ssref = _SingleSupportLQR(m, d, ab, StepConfig(swing="R", ankle_roll_amp_rad=0.12,
                                                   ss_reach_steps=340), verbose=False)
    from step_primitive import AnkleSolver
    ank = AnkleSolver(m)
    _BUILD[model_path] = (m, stand, ssref, ank)
    return _BUILD[model_path]


def _knot_val(kn, w):
    """piecewise-linear over WK, pinned to 0 at w=0, held flat past WK[-1]."""
    full = np.concatenate([[0.0], kn])
    if w <= 0:
        return 0.0
    if w >= WK[-1]:
        return float(full[-1])
    j = int(np.searchsorted(WK, w)) - 1
    j = max(0, min(j, len(WK) - 2))
    t = (w - WK[j]) / (WK[j + 1] - WK[j])
    return float(full[j] * (1 - t) + full[j + 1] * t)


AXIS = {"pitch": 0, "roll": 1, "yaw": 2}


def _limb_L_about_G(m, d, b, com_G, v_G):
    """angular momentum of subtree b about the global CoM (world frame)."""
    Lb = d.subtree_angmom[b].copy()
    mb = m.body_subtreemass[b]
    cb = d.subtree_com[b].copy()
    vb = d.subtree_linvel[b].copy()
    return Lb + mb * np.cross(cb - com_G, vb - v_G)


# ------------------------------------------------------------------- run
def run_traj(push_n, tc: TrajCfg, model_path="robot/robot.xml", trace=False, viewer=False):
    m, stand, ssref, ank = build(model_path)
    d = mujoco.MjData(m)
    swing = tc.swing
    sw = LEGJ[swing]
    stn = LEGJ["L" if swing == "R" else "R"]
    sw_hip_body = BODY["r_hip"] if swing == "R" else BODY["l_hip"]
    st_hip_body = BODY["l_hip"] if swing == "R" else BODY["r_hip"]
    hf = F.FWD_HIP_SIGN[swing]
    ar_unload = F.AR_UNLOAD_SIGN[swing]
    xdir = +1.0 if swing == "R" else -1.0
    kx, kv = _frontal_gains(G / (NOMINAL_CHEST_Z - FLOOR_Z), tc.front_pole_re, tc.front_pole_im)

    # per-joint reference values
    ref = {}
    for ci, src, _ in _all_joint_specs(tc):
        if src == "ss":
            ref[ci] = float(ssref.qpos0[7 + ci])
        elif src == "def":
            ref[ci] = float(DEFAULT_POSE[ci])
        else:
            ref[ci] = None  # 'lqr' -> add to StandingLQR output
    jsrc = {ci: src for ci, src, _ in _all_joint_specs(tc)}

    d.qpos[:] = stand.qpos0
    d.qvel[:] = stand.qvel0
    mujoco.mj_forward(m, d)
    fxy = np.array([0.0, -push_n])

    vw = None
    if viewer:
        from mujoco import viewer as _mjv
        vw = _mjv.launch_passive(m, d)
        vw.cam.lookat[:] = [0, -0.3, 1.1]; vw.cam.distance = 1.7
        vw.cam.azimuth = 100; vw.cam.elevation = -6

    phase = "stand"
    sk = 0
    trig_streak = plant_streak = 0
    swing_started = foot_lifted = False
    an_cmd = float(ssref.qpos0[7 + sw["ankle"]])
    plant_q = None
    steps = []
    swing_y0 = swing_z0 = swing_x0 = None
    peak_up = peak_fwd = peak_clear = 0.0
    # instrumentation accumulators
    I = dict(Lwb_pitch=0.0, Lwb_mag=0.0, Lswing_pitch=0.0, Larm_pitch=0.0,
             Larm_pitch_signed_min=0.0, Larm_pitch_signed_max=0.0,
             pitch_rate=0.0, torso_pitch=0.0, sat_steps=0, n_steps=0,
             tau_leg_peak=0.0, tau_arm_peak=0.0, tau_ankleroll_peak=0.0,
             Lswing_signed_max=0.0, arm_cancels=0, arm_series=[], swing_series=[])
    td = {}
    result = dict(push_n=push_n, model=os.path.basename(model_path), triggered=False, fell=False)

    def frontal_bias(x_ref_m):
        bs = sample_balance(m, d)
        x = float(bs.com[0]); v = float(bs.com_vel[0])
        cop = x_ref_m - kx * (x - x_ref_m) - kv * v
        lo = tc.cop_lo_mm / 1000 * xdir; hi = tc.cop_hi_mm / 1000 * xdir
        cop = float(np.clip(cop, min(lo, hi), max(lo, hi)))
        return float(np.clip(-(cop - x) / tc.cop_per_rad, -tc.ar_bias_max, tc.ar_bias_max))

    T_END = PUSH_AT + PUSH_DURATION_STEPS + tc.trig_deadline_ms + 5500
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

        # lateral setpoint
        if phase == "stand":
            x_ref = tc.x_mid_mm / 1000 * xdir
        elif phase == "shift":
            fr = _smooth(min(1.0, sk / tc.shift_ms))
            x_ref = (tc.x_mid_mm + fr * (tc.x_ss_mm - tc.x_mid_mm)) / 1000 * xdir
        elif phase in ("swing", "descend", "transfer"):
            x_ref = tc.x_ss_mm / 1000 * xdir
        else:
            x_ref = tc.x_transfer_mm / 1000 * xdir

        # base control
        if phase in ("stand", "settle"):
            u = np.array(stand.control(m, d), float)
            if phase == "settle":
                for ci in list(sw.values()) + list(stn.values()):
                    u[ci] = np.clip(u[ci], stand.ctrl0[ci] - 0.7, stand.ctrl0[ci] + 0.7)
                bias = frontal_bias(x_ref) * max(0.0, 1.0 - sk / 700.0)
            else:
                bias = 0.0
        else:
            dq = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, stand.qpos0, d.qpos)
            dx = np.concatenate([dq, d.qvel - stand.qvel0])
            for ci in (AR["L"], AR["R"], HIPR["L"], HIPR["R"]):
                dx[6 + ci] = 0.0; dx[m.nv + 6 + ci] = 0.0
            dx[0] = dx[m.nv + 0] = 0.0
            u = np.array(stand.ctrl0 - stand.K @ dx, float)
            if phase == "shift" and sk < tc.imp_ramp_ms + tc.imp_hold_ms:
                s = sk
                bias = ar_unload * tc.imp_amp * (s / tc.imp_ramp_ms if s < tc.imp_ramp_ms else 1.0)
            else:
                bias = frontal_bias(x_ref)

            if phase in ("swing", "descend"):
                w = (min(1.0, sk / max(tc.swing_ms, 1)) if phase == "swing"
                     else 1.0 + sk / max(tc.descend_ms, 1))
                for ci, kn in tc.knots.items():
                    dval = _knot_val(kn, w)
                    if jsrc[ci] == "lqr":
                        u[ci] = u[ci] + dval
                    else:
                        u[ci] = ref[ci] + dval
                # if the ankle is NOT optimised, keep it flat via AnkleSolver
                if sw["ankle"] not in tc.knots:
                    hipv = u[sw["hip"]]; kneev = u[sw["knee"]]
                    sole_tgt = 0.09 * np.sin(np.pi * min(1.0, w)) if w < 1 else 0.0
                    an_cmd = ank.solve(d.qpos, swing, hipv, kneev, sole_tgt, prev=an_cmd)
                    u[sw["ankle"]] = an_cmd
            elif phase == "transfer":
                b = _smooth(min(1.0, sk / tc.transfer_ms))
                u[sw["hip"]] = plant_q[0]
                u[sw["knee"]] = plant_q[1] + F.KNEE_FLEX_SIGN * 0.05 * b
                u[sw["ankle"]] = plant_q[2] + hf * 0.05 * b

        if phase != "stand":
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] + bias
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] + bias
        if phase in ("swing", "descend", "transfer"):
            u[sw["hip_roll"] if "hip_roll" in sw else HIPR[swing]] = np.clip(
                u[HIPR[swing]], -tc.swing_hiproll_cap, tc.swing_hiproll_cap)
            for ci in (stn["hip"], stn["knee"], stn["ankle"]):
                if ci not in tc.knots:
                    u[ci] = np.clip(u[ci], DEFAULT_POSE[ci] - tc.stance_cap,
                                    DEFAULT_POSE[ci] + tc.stance_cap)

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
            peak_fwd = max(peak_fwd, -(f[1] - swing_y0) * 1000)
        if phase in ("swing", "descend"):
            peak_clear = max(peak_clear, (f[2] - swing_z0) * 1000)
        peak_up = max(peak_up, bs.up_tilt_deg)

        # -------- instrumentation (swing..transfer) --------
        if phase in ("swing", "descend", "transfer"):
            mujoco.mj_subtreeVel(m, d)
            comG = d.subtree_com[CHEST_BODY].copy()
            vG = d.subtree_linvel[CHEST_BODY].copy()
            Lwb = d.subtree_angmom[CHEST_BODY].copy()
            Lsw = _limb_L_about_G(m, d, sw_hip_body, comG, vG)
            Larm = (_limb_L_about_G(m, d, BODY["l_arm"], comG, vG)
                    + _limb_L_about_G(m, d, BODY["r_arm"], comG, vG))
            I["Lwb_pitch"] = max(I["Lwb_pitch"], abs(Lwb[0]))
            I["Lwb_mag"] = max(I["Lwb_mag"], np.linalg.norm(Lwb))
            I["Lswing_pitch"] = max(I["Lswing_pitch"], abs(Lsw[0]))
            I["Lswing_signed_max"] = Lsw[0] if abs(Lsw[0]) > abs(I["Lswing_signed_max"]) else I["Lswing_signed_max"]
            I["Larm_pitch"] = max(I["Larm_pitch"], abs(Larm[0]))
            I["Larm_pitch_signed_min"] = min(I["Larm_pitch_signed_min"], Larm[0])
            I["Larm_pitch_signed_max"] = max(I["Larm_pitch_signed_max"], Larm[0])
            # arm opposing the swing leg (angular momentum cancellation)?
            if Lsw[0] * Larm[0] < 0 and abs(Larm[0]) > 0.15 * abs(Lsw[0]):
                I["arm_cancels"] += 1
            I["pitch_rate"] = max(I["pitch_rate"], abs(bs.pitch_rate))
            I["torso_pitch"] = max(I["torso_pitch"], abs(bs.fwd_lean_deg))
            tau = np.abs(d.actuator_force[:15]) / 2.3
            I["tau_leg_peak"] = max(I["tau_leg_peak"], tau[[6, 7, 8, 11, 12, 13]].max())
            I["tau_arm_peak"] = max(I["tau_arm_peak"], tau[[1, 2, 3, 4]].max())
            I["tau_ankleroll_peak"] = max(I["tau_ankleroll_peak"], tau[[9, 14]].max())
            if (tau[[6, 7, 8, 11, 12, 13, 9, 14]] > 0.98).any():
                I["sat_steps"] += 1
            I["n_steps"] += 1
            if trace and I["n_steps"] % 12 == 0:
                I["arm_series"].append(round(float(Larm[0]), 3))
                I["swing_series"].append(round(float(Lsw[0]), 3))

        if bs.up_tilt_deg > 55 or bs.chest_z < NOMINAL_CHEST_Z - 0.24:
            result["fell"] = True
            break

        # transitions
        if phase == "stand":
            trig_streak = (trig_streak + 1
                           if (post and bs.capture_fwd_rel_support_mm > tc.trig_capt_mm
                               and bs.com_vfwd > tc.trig_vfwd) else 0)
            if trig_streak >= tc.trig_hold:
                phase = "shift"; sk = 0; result["triggered"] = True
            elif post and k > PUSH_AT + PUSH_DURATION_STEPS + tc.trig_deadline_ms:
                phase = "settle"; sk = 0
        elif phase == "shift":
            if (not swing_started and sk >= tc.shift_min_ms
                    and (sw_nf < 6.0 or sk >= tc.swing_start_ms)
                    and (abs(pr) < 0.5 or sk >= tc.swing_start_ms)):
                phase = "swing"; swing_started = True; sk = 0
                swing_x0 = f[0]; swing_y0 = _foot_xy_z(m, d, swing)[1]; swing_z0 = f[2]
        elif phase == "swing":
            if not foot_lifted and sw_nf < 3.0:
                foot_lifted = True
            if sk >= tc.swing_ms:
                phase = "descend"; sk = 0
        elif phase == "descend":
            genuine = foot_lifted and getattr(bs, f"{swing.lower()}_contact") and sw_nf > tc.plant_nf
            plant_streak = plant_streak + 1 if genuine else 0
            if plant_streak >= tc.plant_hold or sk >= tc.step_max_ms:
                planted = plant_streak >= tc.plant_hold
                sf = _foot_xy_z(m, d, swing)[1]
                stf = _foot_xy_z(m, d, "L" if swing == "R" else "R")[1]
                plant_q = np.array([d.qpos[7 + sw["hip"]], d.qpos[7 + sw["knee"]],
                                    d.qpos[7 + sw["ankle"]]])
                sep = -(sf - stf) * 1000
                td = dict(k=k, planted=bool(planted), sep_mm=float(sep),
                          side=float(bs.side_lean_deg), sole=float(np.degrees(F._sole_pitch(m, d, swing))),
                          nf=float(sw_nf), vfwd=float(bs.com_vfwd), vlat=float(bs.com_vel[0]),
                          torso_pitch=float(bs.fwd_lean_deg), clr=float(peak_clear))
                steps.append(td)
                phase = "transfer"; sk = 0
        elif phase == "transfer":
            done = (st_nf > 6 and sw_nf > 6 and abs(bs.side_lean_deg) < 8
                    and abs(float(bs.com_vel[0])) < 0.12)
            if sk >= tc.transfer_ms or (done and sk > 50):
                phase = "settle"; sk = 0
        elif phase == "settle":
            if sk >= tc.settle_ms:
                break

        if vw is not None:
            if not vw.is_running():
                break
            vw.sync(); time.sleep(0.004)

    bs = sample_balance(m, d)
    ds = bs.l_contact and bs.r_contact
    n_planted = sum(1 for s in steps if s["planted"])
    success = (result["triggered"] and not result["fell"] and ds
               and bs.up_tilt_deg < 8 and abs(bs.side_lean_deg) < 8
               and bs.com_speed_horiz < 0.10 and bs.chest_z > NOMINAL_CHEST_Z - 0.10
               and n_planted >= 1 and peak_up <= 35)
    sep = float(td.get("sep_mm", steps[0]["sep_mm"] if steps else 0.0))
    result.update(
        success=bool(success), n_planted=n_planted, end_ds=bool(ds),
        end_up=float(bs.up_tilt_deg), end_side=float(bs.side_lean_deg),
        end_speed=float(bs.com_speed_horiz), peak_up=float(peak_up),
        peak_fwd=float(peak_fwd), peak_clear=float(peak_clear), step_sep_mm=sep,
        touchdown=td,
        Lwb_pitch=I["Lwb_pitch"], Lwb_mag=I["Lwb_mag"], Lswing_pitch=I["Lswing_pitch"],
        Lswing_signed=I["Lswing_signed_max"], Larm_pitch=I["Larm_pitch"],
        Larm_signed_range=[I["Larm_pitch_signed_min"], I["Larm_pitch_signed_max"]],
        arm_cancel_frac=(I["arm_cancels"] / max(I["n_steps"], 1)),
        peak_pitch_rate=I["pitch_rate"], peak_torso_pitch=I["torso_pitch"],
        tau_leg_peak=I["tau_leg_peak"], tau_arm_peak=I["tau_arm_peak"],
        tau_ankleroll_peak=I["tau_ankleroll_peak"],
        sat_frac=(I["sat_steps"] / max(I["n_steps"], 1)),
        arm_series=I["arm_series"], swing_series=I["swing_series"])
    if vw is not None:
        try:
            while vw.is_running():
                vw.sync(); time.sleep(0.004)
        except KeyboardInterrupt:
            pass
        vw.close()
    return result


def _all_joint_specs(tc):
    """joint specs implied by the knots dict, with default bounds/ref sources."""
    spec = {ci: s for ci, s, _ in JOINTS_B}
    out = []
    for ci in tc.knots:
        out.append((ci, spec.get(ci, "ss" if ci in (11, 12, 13) else "def"), None))
    return out


# ============================================================ optimisation
def _pack_layout(joint_list):
    """flat-vector layout: [swing_ms, descend_ms, x_ss_mm] + 4*len(joints)."""
    base = [("swing_ms", 110, 240, 150), ("descend_ms", 45, 130, 80),
            ("x_ss_mm", 40, 74, 60)]
    knot = []
    for ci, src, (lo, hi) in joint_list:
        for j in range(4):
            knot.append((f"j{ci}_k{j}", lo, hi, 0.0))
    return base + knot, len(base)


def _vec_to_tc(x, joint_list, base_tc: TrajCfg):
    layout, nb = _pack_layout(joint_list)
    lo = np.array([p[1] for p in layout]); hi = np.array([p[2] for p in layout])
    x = np.clip(np.asarray(x, float), lo, hi)
    tc = TrajCfg(**{k: getattr(base_tc, k) for k in base_tc.__dataclass_fields__ if k != "knots"})
    tc.swing_ms = int(x[0]); tc.descend_ms = int(x[1]); tc.x_ss_mm = float(x[2])
    knots = {}
    p = nb
    for ci, src, _ in joint_list:
        knots[ci] = np.array(x[p:p + 4], float)
        p += 4
    tc.knots = knots
    return tc


def _score(r):
    if r is None:
        return 0.0
    rec = 0.0
    if not r["fell"] and r["triggered"]:
        rec = 0.25
        if r["n_planted"]:
            rec += 0.15
            td = r["touchdown"]
            rec *= (0.65 + 0.35 * max(0.0, 1.0 - abs(td.get("sole", 90)) / 22.0))
        if r["end_ds"]:
            rec += 0.10
        rec += 0.22 * max(0.0, 1.0 - r["end_up"] / 16.0)
        rec += 0.13 * max(0.0, 1.0 - abs(r["end_side"]) / 16.0)
        rec += 0.15 * max(0.0, 1.0 - r["end_speed"] / 0.30)
    if r["success"]:
        rec = 1.0
    sep = r.get("step_sep_mm", 0.0)
    sep_f = float(np.clip((sep - 15.0) / (65.0 - 15.0), 0.0, 1.25))
    return float(rec * (0.30 + 0.70 * sep_f))


_W = {}


def _winit(model_path):
    cache = {}
    o_sl, o_ab, o_ss = F.StandingLQR, F._LQRAbout, F._SingleSupportLQR

    def SL(m, d, verbose=False):
        return cache.setdefault("sl", o_sl(m, d, verbose=False))

    def AB(m, d, pose, tag="ab", verbose=False):
        return cache.setdefault("ab", o_ab(m, d, pose, tag="ab", verbose=False))

    def SS(m, d, ab, scfg, verbose=False):
        return cache.setdefault("ss", o_ss(m, d, ab, scfg, verbose=False))

    F.StandingLQR, F._LQRAbout, F._SingleSupportLQR = SL, AB, SS
    _W["ready"] = True
    _W["model"] = model_path


def _eval(task):
    x, joint_list, base_tc, push_n, model_path = task
    if not _W.get("ready"):
        _winit(model_path)
    try:
        tc = _vec_to_tc(x, joint_list, base_tc)
        return run_traj(push_n, tc, model_path)
    except Exception as e:
        return {"fell": True, "triggered": False, "success": False, "err": repr(e),
                "step_sep_mm": 0.0, "n_planted": 0, "end_ds": False, "end_up": 99,
                "end_side": 99, "end_speed": 9, "peak_up": 99, "touchdown": {}}


def optimise(joint_list, base_tc, model_path, pool, grid, gens, popsize, seed, x0=None):
    import cma
    layout, _ = _pack_layout(joint_list)
    lo = np.array([p[1] for p in layout]); hi = np.array([p[2] for p in layout])
    x00 = np.array([p[3] for p in layout]) if x0 is None else np.asarray(x0, float)
    x0n = (np.clip(x00, lo, hi) - lo) / (hi - lo)
    es = cma.CMAEvolutionStrategy(list(x0n), 0.30,
                                  {"bounds": [0, 1], "popsize": popsize, "seed": seed,
                                   "maxiter": gens, "verbose": -9,
                                   "tolfun": 1e-4, "tolfunhist": 1e-4,
                                   "tolflatfitness": gens, "tolstagnation": gens,
                                   "tolx": 1e-5})
    best = {"fit": 1e9, "x": None}
    t0 = time.time()
    g = 0
    while not es.stop():
        g += 1
        sols = es.ask()
        tasks = []
        for sn in sols:
            xr = lo + np.asarray(sn) * (hi - lo)
            for pn in grid:
                tasks.append((xr, joint_list, base_tc, pn, model_path))
        res = pool.map(_eval, tasks)
        ng = len(grid)
        fits = []
        for i, sn in enumerate(sols):
            rs = res[i * ng:(i + 1) * ng]
            sc = np.array([_score(r) for r in rs])
            fit = -(sc.mean()) + 0.12 * sc.std()
            fits.append(fit)
            if fit < best["fit"]:
                xr = lo + np.asarray(sn) * (hi - lo)
                best = {"fit": float(fit), "x": xr.tolist(),
                        "grid_succ": float(np.mean([r["success"] for r in rs])),
                        "grid_sep": [round(r["step_sep_mm"], 1) for r in rs],
                        "grid_seps_ok": [round(r["step_sep_mm"], 1) for r in rs if r["success"]]}
        es.tell(sols, fits)
        if g % 5 == 0 or g == 1:
            print(f"    gen {g:3d}  fit {best['fit']:+.3f}  gridSucc {best.get('grid_succ',0):.2f}  "
                  f"sepsOK {best.get('grid_seps_ok',[])}  ({time.time()-t0:.0f}s)")
    return best


# ============================================================ analysis
FINE = tuple(float(x) for x in range(122, 147, 2))


def analyse(name, x, joint_list, base_tc, model_path, pool):
    tasks = [(x, joint_list, base_tc, pn, model_path) for pn in FINE]
    res = pool.map(_eval, tasks)
    ok = [r["success"] for r in res]
    seps_ok = [r["step_sep_mm"] for r, s in zip(res, ok) if s]
    seps_all = [r["step_sep_mm"] for r in res]
    # failure classes
    modes = {}
    for r in res:
        if r["success"]:
            c = "success"
        elif not r["triggered"]:
            c = "no_trigger"
        elif r["fell"]:
            c = "fell_lateral" if abs(r["end_side"]) > 22 else "fell_sagittal"
        elif r["n_planted"] == 0:
            c = "no_plant"
        else:
            c = "planted_unstable"
        modes[c] = modes.get(c, 0) + 1
    tds = [r["touchdown"] for r in res if r["success"] and r["touchdown"]]
    rr = [r for r in res]
    out = dict(
        name=name, model=os.path.basename(model_path),
        n=len(FINE), success_rate=round(float(np.mean(ok)), 3),
        band=_band(FINE, ok),
        best_sep=round(float(np.max(seps_all)), 1),
        best_sep_recovered=round(float(np.max(seps_ok)), 1) if seps_ok else None,
        median_sep_ok=round(float(np.median(seps_ok)), 1) if seps_ok else None,
        mean_sep_ok=round(float(np.mean(seps_ok)), 1) if seps_ok else None,
        sep_ok_list=[round(s, 1) for s in seps_ok],
        failure_modes=modes,
        peak_torso_pitch_mean=round(float(np.mean([r["peak_torso_pitch"] for r in rr])), 1),
        peak_torso_pitch_max=round(float(np.max([r["peak_torso_pitch"] for r in rr])), 1),
        peak_pitch_rate_mean=round(float(np.mean([r["peak_pitch_rate"] for r in rr])), 2),
        Lwb_pitch_mean=round(float(np.mean([r["Lwb_pitch"] for r in rr])), 3),
        Lswing_pitch_mean=round(float(np.mean([r["Lswing_pitch"] for r in rr])), 3),
        Larm_pitch_mean=round(float(np.mean([r["Larm_pitch"] for r in rr])), 3),
        arm_cancel_frac_mean=round(float(np.mean([r["arm_cancel_frac"] for r in rr])), 2),
        tau_leg_peak_mean=round(float(np.mean([r["tau_leg_peak"] for r in rr])), 2),
        tau_arm_peak_mean=round(float(np.mean([r["tau_arm_peak"] for r in rr])), 2),
        sat_frac_mean=round(float(np.mean([r["sat_frac"] for r in rr])), 3),
        touchdown_sole_mean=round(float(np.mean([t["sole"] for t in tds])), 1) if tds else None,
        per_push=[dict(push=pn, success=bool(r["success"]),
                       sep=round(r["step_sep_mm"], 1), peak_up=round(r["peak_up"], 1),
                       torso_pitch=round(r["peak_torso_pitch"], 1),
                       Lsw=round(r["Lswing_signed"], 3),
                       Larm_rng=[round(v, 3) for v in r["Larm_signed_range"]],
                       tau_leg=round(r["tau_leg_peak"], 2), tau_arm=round(r["tau_arm_peak"], 2))
                  for pn, r in zip(FINE, res)],
        x=list(map(float, x)),
    )
    return out


def _band(grid, ok):
    best = (None, None, 0)
    i = 0
    while i < len(grid):
        if ok[i]:
            j = i
            while j < len(grid) and ok[j]:
                j += 1
            if j - i > best[2]:
                best = (grid[i], grid[j - 1], j - i)
            i = j
        else:
            i += 1
    return dict(lo=best[0], hi=best[1], width_N=(best[1] - best[0]) if best[0] else 0)


# ============================================================ heavier hands
def make_hand_model(mult, out_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse("robot/robot.xml")
    root = tree.getroot()
    n = 0
    for body in root.iter("body"):
        if body.get("name") in ("L_hand", "R_hand"):
            for inert in body.iter("inertial"):
                inert.set("mass", f"{float(inert.get('mass')) * mult:.6f}")
                dia = inert.get("diaginertia")
                if dia:
                    inert.set("diaginertia", " ".join(f"{float(v) * mult:.8f}" for v in dia.split()))
                n += 1
    root.set("model", f"Full_hands{mult}x")
    tree.write(out_path)
    return n


# ============================================================ driver
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C"], default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--push", type=float, default=132.0)
    a = ap.parse_args(argv)
    os.makedirs("results", exist_ok=True)

    if a.replay:
        data = json.load(open(a.replay))
        jl = JOINTS_B if "stageB" in a.replay or "hands" in a.replay else JOINTS_A
        tc = _vec_to_tc(data["x"], jl, TrajCfg())
        print(f"replay {a.replay} @ push {a.push}")
        r = run_traj(a.push, tc, data.get("model_path", "robot/robot.xml"),
                     trace=True, viewer=True)
        print(json.dumps({k: r[k] for k in ("success", "step_sep_mm", "peak_up",
              "peak_torso_pitch", "end_up", "end_side", "Lswing_pitch", "Larm_pitch",
              "arm_cancel_frac", "tau_leg_peak", "tau_arm_peak", "touchdown")}, indent=2, default=float))
        return

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    G_ = (128.0, 134.0) if a.quick else (126.0, 130.0, 134.0, 138.0)
    gens = 6 if a.quick else 38
    pop = 8 if a.quick else 16
    pool = ctx.Pool(a.workers, initializer=_winit, initargs=("robot/robot.xml",))
    OUT = {}
    try:
        stages = (["A", "B", "C"] if a.all else [a.stage])
        base = TrajCfg()

        if "A" in stages:
            print("=== STAGE A: swing-leg trajectory only ===")
            bA = optimise(JOINTS_A, base, "robot/robot.xml", pool, G_, gens, pop, seed=1)
            anA = analyse("stageA", bA["x"], JOINTS_A, base, "robot/robot.xml", pool)
            OUT["stageA"] = dict(best=bA, analysis=anA)
            json.dump(OUT, open("results/wb_stages.json", "w"), indent=2, default=float)
            print(f"  -> success {anA['success_rate']:.2f}  band {anA['band']['width_N']}N  "
                  f"bestSepRec {anA['best_sep_recovered']}  medSepOK {anA['median_sep_ok']}")

        if "B" in stages:
            print("\n=== STAGE B: whole-body (swing leg + arms + stance) ===")
            x0B = None
            if "stageA" in OUT:
                # warm-start the shared params from A
                x0B = list(OUT["stageA"]["best"]["x"][:3]) + \
                      list(OUT["stageA"]["best"]["x"][3:]) + [0.0] * (4 * (len(JOINTS_B) - len(JOINTS_A)))
            bB = optimise(JOINTS_B, base, "robot/robot.xml", pool, G_,
                          gens + 10, pop + 4, seed=2, x0=x0B)
            anB = analyse("stageB", bB["x"], JOINTS_B, base, "robot/robot.xml", pool)
            OUT["stageB"] = dict(best=bB, analysis=anB)
            json.dump(OUT, open("results/wb_stages.json", "w"), indent=2, default=float)
            print(f"  -> success {anB['success_rate']:.2f}  band {anB['band']['width_N']}N  "
                  f"bestSepRec {anB['best_sep_recovered']}  medSepOK {anB['median_sep_ok']}  "
                  f"armCancel {anB['analysis']['arm_cancel_frac_mean'] if 'analysis' in anB else anB['arm_cancel_frac_mean']}")

        if "C" in stages:
            print("\n=== STAGE C: heavier hands ===")
            OUT["stageC"] = {}
            x0C = OUT["stageB"]["best"]["x"] if "stageB" in OUT else None
            for mult in ([2.0, 3.0] if a.quick else [1.0, 2.0, 3.0, 5.0]):
                mp_path = f"robot/_exp_hands_{mult:g}x.xml"
                nch = make_hand_model(mult, mp_path)
                _BUILD.pop(mp_path, None)
                print(f"  -- hands {mult}x  ({nch} inertials scaled) --")
                pool2 = ctx.Pool(a.workers, initializer=_winit, initargs=(mp_path,))
                try:
                    bC = optimise(JOINTS_B, base, mp_path, pool2, G_,
                                  (gens if mult == 1.0 else 25), pop + 4, seed=int(mult * 10),
                                  x0=x0C)
                    anC = analyse(f"hands{mult}x", bC["x"], JOINTS_B, base, mp_path, pool2)
                finally:
                    pool2.close(); pool2.join()
                OUT["stageC"][f"{mult}x"] = dict(best=bC, analysis=anC, model_path=mp_path)
                json.dump(OUT, open("results/wb_stages.json", "w"), indent=2, default=float)
                print(f"     -> success {anC['success_rate']:.2f}  bestSepRec {anC['best_sep_recovered']}  "
                      f"medSepOK {anC['median_sep_ok']}  torsoPitch {anC['peak_torso_pitch_mean']}  "
                      f"armL {anC['Larm_pitch_mean']}  sat {anC['sat_frac_mean']}")

        json.dump(OUT, open("results/wb_stages.json", "w"), indent=2, default=float)
        _report(OUT)
    finally:
        pool.close(); pool.join()


def _report(o):
    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    rows = []
    rows.append(("1 analytical (hand-tuned)", "27", "0 N band", "6% (1/16)", "55", "-", "needle point"))
    rows.append(("2 CMA-ES fixed controller", "32", "31 / 6 N", "50% (8/16)", "~48*", "-", "lateral topple @ edge"))
    for key, lab in (("stageA", "3 swing-leg trajectory"),
                     ("stageB", "4 whole-body + normal hands")):
        if key in o:
            an = o[key]["analysis"]
            rows.append((lab, str(an["best_sep_recovered"]), str(an["median_sep_ok"]),
                         f"{an['success_rate']*100:.0f}% ({int(an['success_rate']*an['n'])}/{an['n']})",
                         f"{an['peak_torso_pitch_mean']}", f"{an['sat_frac_mean']*100:.0f}%",
                         max(an["failure_modes"], key=an["failure_modes"].get)))
    if "stageC" in o:
        for mk, mv in o["stageC"].items():
            an = mv["analysis"]
            rows.append((f"5 whole-body, hands {mk}", str(an["best_sep_recovered"]),
                         str(an["median_sep_ok"]),
                         f"{an['success_rate']*100:.0f}% ({int(an['success_rate']*an['n'])}/{an['n']})",
                         f"{an['peak_torso_pitch_mean']}", f"{an['sat_frac_mean']*100:.0f}%",
                         max(an["failure_modes"], key=an["failure_modes"].get)))
    print(f"\n{'controller':<30} {'bestSep':>7} {'robust/band':>12} {'success':>13} "
          f"{'pitch':>6} {'satur':>6}  failure")
    for r in rows:
        print(f"{r[0]:<30} {r[1]:>7} {r[2]:>12} {r[3]:>13} {r[4]:>6} {r[5]:>6}  {r[6]}")
    if "stageB" in o:
        an = o["stageB"]["analysis"]
        print(f"\narm angular-momentum: |Larm_pitch| mean {an['Larm_pitch_mean']}  vs  "
              f"|Lswing_pitch| mean {an['Lswing_pitch_mean']}  "
              f"(arm opposes swing {an['arm_cancel_frac_mean']*100:.0f}% of swing steps)")


if __name__ == "__main__":
    main(sys.argv[1:])
