"""Isolate the source of the torso pitch/roll during the 77 mm SS-LQR swing.

Question from the user: is the whole-body disturbance (a) the SS-LQR actively
commanding it, (b) the SS-LQR reacting to swing-leg reaction forces, (c) the
stance leg pushing the pelvis, (d) the swing trajectory's own angular momentum,
or (e) the SS-LQR leaned reference itself?  And how long is single support?

Runs several controlled variants from the SAME post-impulse state and logs, each
tick: torso attitude + rates, total angular momentum (subtree_angmom of the
floating base), the swing-leg subtree angular momentum, the pelvis free-joint
twist, and the LQR feedback effort on every stance / roll joint.

    python ss_swing_diag.py                 # all variants, summary table
    python ss_swing_diag.py --variant swing_slow --full   # per-tick trace
"""
from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from biped_env import DEFAULT_POSE
from recovery_metrics import CHEST_BODY, _foot_normal_force, _foot_xy_z, sample_balance
from standing_balance_lqr import StandingLQR
from push_step_recovery_test import (
    StepConfig, _LQRAbout, _SingleSupportLQR, _pos_error, _smooth,
)
from step_primitive import AnkleSolver, _sole_pitch, LEG, AR, FWD_HIP_SIGN, KNEE_FLEX_SIGN

PUSH_AT = 20
PUSH_DUR = 5
AR_UNLOAD_SIGN = {"R": -1.0, "L": +1.0}
SWING = "R"
BASE_BID = CHEST_BODY          # floating-base body = root of the tree = whole robot
SW_HIP_BID = 12                # R_hip body: swing-leg subtree root
ST_HIP_BID = 7                 # L_hip body: stance-leg subtree root


def build():
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)
    stand = StandingLQR(m, d, verbose=False)
    scfg = StepConfig(swing=SWING, ankle_roll_amp_rad=0.12, ss_reach_steps=340)
    stand_ab = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="ab", verbose=False)
    ss = _SingleSupportLQR(m, d, stand_ab, scfg, verbose=False)
    return m, d, stand, ss


def swing_arc(variant, w, h0, k0, hip_amp=0.45, knee_amp=0.30):
    """(hip, knee) joint targets over swing phase progress w in [0,1]."""
    hf = FWD_HIP_SIGN[SWING]
    if variant == "hold":
        return h0, k0
    if variant in ("swing_slow", "swing_fast"):
        # demo shape: hip smooth-step forward, knee sin bump for clearance
        s = _smooth(w)
        hip = h0 + hf * hip_amp * s
        knee = k0 + KNEE_FLEX_SIGN * (0.05 + (knee_amp - 0.05) * np.sin(np.pi * w))
        return hip, knee
    if variant == "knee_lead":
        # bicycle style: bend the knee FIRST (foot up, low MoI), rotate the hip
        # while MoI is low, then extend the knee forward to place the foot.
        # hip lags the knee by ~0.25 of the phase.
        wk = _smooth(min(1.0, w / 0.55))                 # knee flexes early
        wh = _smooth(max(0.0, (w - 0.20) / 0.80))        # hip starts later
        we = _smooth(max(0.0, (w - 0.55) / 0.45))        # knee re-extends late
        hip = h0 + hf * hip_amp * wh
        knee = k0 + KNEE_FLEX_SIGN * (0.05 + (knee_amp + 0.10) * wk - (knee_amp + 0.05) * we)
        return hip, knee
    raise ValueError(variant)


def run(variant, full=False, hip_amp=0.45, knee_amp=0.30):
    bvar = variant[:-3] if variant.endswith("_rd") else variant
    m, d, stand, ss = build()
    ank = AnkleSolver(m)
    sw = LEG[SWING]
    stn = LEG["L"]
    hf = FWD_HIP_SIGN[SWING]
    ss_idx = (sw["hip"], sw["knee"], sw["ankle"])
    lean_ss = float(ss.ctrl0[AR[SWING]])
    h0, k0, a0 = (ss.qpos0[7 + sw["hip"]], ss.qpos0[7 + sw["knee"]],
                  ss.qpos0[7 + sw["ankle"]])

    d.qpos[:] = stand.qpos0
    d.qvel[:] = stand.qvel0
    mujoco.mj_forward(m, d)

    swing_ms = 75 if bvar == "swing_fast" else 150
    push_n = 135.0
    fxy = np.array([0.0, -push_n])

    roll_decouple = variant.endswith("_rd")
    ROLL_CI = (LEG["L"]["hip_roll"], LEG["R"]["hip_roll"], AR["L"], AR["R"])

    def ss_u(hip, knee, ankle):
        qref = ss.qpos0.copy(); cref = ss.ctrl0.copy()
        for ci, v in zip(ss_idx, (hip, knee, ankle)):
            qref[7 + ci] = v; cref[ci] = v
        dx = np.concatenate([_pos_error(m, qref, d.qpos), d.qvel - ss.qvel0])
        for ci in ss_idx:
            dx[6 + ci] = 0.0; dx[m.nv + 6 + ci] = 0.0
        dx[0] = dx[m.nv + 0] = 0.0
        if roll_decouple:
            # don't let ss.K regulate ROLL at all - it's entered ~7 deg off its
            # leaned reference and slams the roll joints.  Zero every roll DOF:
            # both hip_roll + ankle_roll joints, and the base fore-aft-axis tilt.
            for ci in ROLL_CI:
                dx[6 + ci] = 0.0; dx[m.nv + 6 + ci] = 0.0
            for bi in (3, 4, 5):        # base rotation dofs - find the roll one empirically
                pass
            dx[3] = dx[m.nv + 3] = 0.0  # base roll (about fore-aft world axis)
            dx[5] = dx[m.nv + 5] = 0.0  # base yaw
        u = cref - ss.K @ dx
        for ci in ss_idx:
            u[ci] = cref[ci]
        return u, (u - cref)          # also return the raw feedback effort

    phase = "stand"
    sk = 0
    an_cmd = a0
    swing_y0 = _foot_xy_z(m, d, SWING)[1]
    swing_z0 = _foot_xy_z(m, d, SWING)[2]
    t_trig = t_swing = t_liftoff = t_touchdown = None
    rows = []
    peak = dict(up=0.0, side=0.0, fwd=0.0, Lam=0.0, roll_rate=0.0)
    fb_accum = np.zeros(15)
    fb_n = 0

    for k in range(1400):
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT <= k < PUSH_AT + PUSH_DUR:
            d.xfrc_applied[CHEST_BODY, 0:2] = fxy
        bs = sample_balance(m, d)
        sw_nf = _foot_normal_force(m, d, SWING)
        st_nf = _foot_normal_force(m, d, "L")
        fb = np.zeros(15)

        if phase == "stand":
            u = stand.control(m, d)
            if k > PUSH_AT + PUSH_DUR + 40:
                if (bs.capture_fwd_rel_support_mm > 6.0 and bs.com_vfwd > 0.05):
                    t_trig = k
                    phase = "impulse"
                    sk = 0
        elif phase == "impulse":
            imp_end = 8 + 40
            a = (AR_UNLOAD_SIGN[SWING] * 0.22 * min(1.0, sk / 8) if sk < imp_end
                 else 0.0)
            u = stand.control(m, d)
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] + a
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] + a
            if sk >= 90 or sw_nf < 6.0:
                # hand to SS-LQR
                phase = "ss"
                t_swing = k
                sk = 0
        else:  # SS phase (hold / swing_slow / swing_fast / knee_lead)
            w = min(1.0, sk / swing_ms)
            hip, knee = swing_arc(bvar, w, h0, k0, hip_amp, knee_amp)
            sole_tgt = 0.10 * np.sin(np.pi * w) if bvar != "hold" else 0.0
            an_cmd = ank.solve(d.qpos, SWING, hip, knee, sole_tgt, prev=an_cmd)
            u, fb = ss_u(hip, knee, an_cmd)
            u = np.array(u, float)
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] + lean_ss
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] + lean_ss

        u = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        d.ctrl[:15] = u
        mujoco.mj_step(m, d)
        if phase != "stand":
            sk += 1

        mujoco.mj_subtreeVel(m, d)          # fills subtree_linvel + subtree_angmom
        bs = sample_balance(m, d)
        sw_nf = _foot_normal_force(m, d, SWING)
        f = _foot_xy_z(m, d, SWING)
        fwd_mm = -(f[1] - swing_y0) * 1000.0
        clr_mm = (f[2] - swing_z0) * 1000.0

        # total angular momentum about the whole-body CoM (world frame)
        Lam = d.subtree_angmom[BASE_BID].copy()
        Lam_sw = d.subtree_angmom[SW_HIP_BID].copy()
        Lam_st = d.subtree_angmom[ST_HIP_BID].copy()
        # pelvis free-joint twist
        pel_w = d.qvel[3:6].copy()          # base angular velocity (world)
        pel_v = d.qvel[0:3].copy()

        in_ss = t_swing is not None and t_touchdown is None
        if in_ss and t_liftoff is None and sw_nf < 2.0:
            t_liftoff = k
        if t_liftoff is not None and t_touchdown is None and sw_nf > 12.0 and clr_mm < 15:
            t_touchdown = k

        if phase not in ("stand", "impulse"):
            peak["up"] = max(peak["up"], bs.up_tilt_deg)
            peak["side"] = max(peak["side"], abs(bs.side_lean_deg))
            peak["fwd"] = max(peak["fwd"], abs(bs.fwd_lean_deg))
            peak["Lam"] = max(peak["Lam"], np.linalg.norm(Lam))
            peak["roll_rate"] = max(peak["roll_rate"], abs(pel_w[1]))
            fb_accum += np.abs(fb)
            fb_n += 1

        if full and k % 8 == 0 and phase not in ("stand",):
            rows.append((k, phase[:4], bs.up_tilt_deg, bs.fwd_lean_deg, bs.side_lean_deg,
                         sw_nf, fwd_mm, clr_mm,
                         Lam[0], Lam[1], Lam[2],
                         Lam_sw[0], Lam_sw[2],
                         pel_w[0], pel_w[1], pel_w[2],
                         fb[stn["hip"]], fb[stn["knee"]], fb[stn["ankle"]],
                         fb[stn["hip_roll"]], fb[AR["L"]]))

        if bs.up_tilt_deg > 45:
            break

    fb_mean = fb_accum / max(fb_n, 1)
    if full:
        print(f"\n=== {variant}  (swing_ms={swing_ms}, hip_amp={hip_amp}, knee_amp={knee_amp}) ===")
        hdr = ("k phs up fwd side swNF fFwd clr | Lam_x Lam_y Lam_z | LamSw_x LamSw_z "
               "| pelW_x pelW_y pelW_z | fb:Lhip Lknee Lank LhipR LankR")
        print(hdr)
        for r in rows:
            print(f"{r[0]:4d} {r[1]:>4} {r[2]:5.1f} {r[3]:+5.1f} {r[4]:+6.1f} {r[5]:4.0f} "
                  f"{r[6]:5.0f} {r[7]:4.0f} | {r[8]:+6.3f} {r[9]:+6.3f} {r[10]:+6.3f} | "
                  f"{r[11]:+6.3f} {r[12]:+6.3f} | {r[13]:+5.2f} {r[14]:+5.2f} {r[15]:+5.2f} | "
                  f"{r[16]:+5.2f} {r[17]:+5.2f} {r[18]:+5.2f} {r[19]:+5.2f} {r[20]:+5.2f}")

    ss_dur = (t_touchdown - t_swing) if (t_touchdown and t_swing) else None
    air = (t_touchdown - t_liftoff) if (t_touchdown and t_liftoff) else None
    print(f"\n  {variant:11s}: trig@{t_trig} ss_hand@{t_swing} liftoff@{t_liftoff} "
          f"touchdown@{t_touchdown}  ss_dur={ss_dur}ms air={air}ms")
    print(f"     peak up={peak['up']:.1f}  side={peak['side']:.1f}  fwd={peak['fwd']:.1f}  "
          f"|L|max={peak['Lam']:.3f}  rollRate_max={peak['roll_rate']:.2f} rad/s")
    print(f"     mean|fb effort|: Lhip={fb_mean[stn['hip']]:.3f} Lknee={fb_mean[stn['knee']]:.3f} "
          f"Lank={fb_mean[stn['ankle']]:.3f} LhipRoll={fb_mean[stn['hip_roll']]:.3f} "
          f"LankRoll={fb_mean[AR['L']]:.3f} Rhip={fb_mean[sw['hip_roll']]:.3f}")
    return peak, ss_dur


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None)
    p.add_argument("--full", action="store_true")
    p.add_argument("--hip", type=float, default=0.45)
    p.add_argument("--knee", type=float, default=0.30)
    a = p.parse_args(argv)
    variants = ([a.variant] if a.variant else
               ["hold", "swing_slow", "swing_fast", "knee_lead"])
    for v in variants:
        run(v, full=a.full, hip_amp=a.hip, knee_amp=a.knee)


if __name__ == "__main__":
    main(sys.argv[1:])
