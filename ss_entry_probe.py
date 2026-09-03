"""How fast can the robot enter the leaned single-support state such that the
SS-LQR then actually holds it?

The demo reaches it in 340 quasi-static steps then ss.K holds forever.  The
recovery has ~150-250 ms.  This sweeps the pre-shift duration: impulse to break
the foot loose, ramp the both-ankle-roll bias to the ss value under StandingLQR
(reference-biased, exactly the demo's sequence), hold to `preshift`, hand to
ss.K, then just HOLD the ss pose (no swing) for 500 ms and see if it stays up.

    python ss_entry_probe.py
    python ss_entry_probe.py --push 132 --full
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
    StepConfig, _LQRAbout, _SingleSupportLQR, _pos_error,
)
from step_primitive import LEG, AR

PUSH_AT, PUSH_DUR = 20, 5
AR_UNLOAD_SIGN = {"R": -1.0, "L": +1.0}
SWING = "R"


def build(amp):
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)
    stand = StandingLQR(m, d, verbose=False)
    ab = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="ab", verbose=False)
    ss = _SingleSupportLQR(m, d, ab, StepConfig(swing=SWING, ankle_roll_amp_rad=amp,
                                                ss_reach_steps=340), verbose=False)
    return m, d, stand, ss


def probe(preshift, amp=0.12, push_n=135.0, ramp=60, settle_hold=True,
          decouple_roll=False, full=False):
    m, d, stand, ss = build(amp)
    sw = LEG[SWING]
    stn = LEG["L"]
    ss_idx = (sw["hip"], sw["knee"], sw["ankle"])
    lean_ss = float(ss.ctrl0[AR[SWING]])
    ROLL_CI = (stn["hip_roll"], sw["hip_roll"], AR["L"], AR["R"])

    d.qpos[:] = stand.qpos0
    d.qvel[:] = stand.qvel0
    mujoco.mj_forward(m, d)
    fxy = np.array([0.0, -push_n])

    def ss_hold_u():
        qref = ss.qpos0.copy()
        cref = ss.ctrl0.copy()
        dx = np.concatenate([_pos_error(m, qref, d.qpos), d.qvel - ss.qvel0])
        for ci in ss_idx:
            dx[6 + ci] = 0.0
            dx[m.nv + 6 + ci] = 0.0
        if decouple_roll:
            for ci in ROLL_CI:
                dx[6 + ci] = 0.0
                dx[m.nv + 6 + ci] = 0.0
            dx[3] = dx[m.nv + 3] = 0.0
        u = cref - ss.K @ dx
        for ci in ss_idx:
            u[ci] = cref[ci]
        return u

    phase = "stand"
    sk = 0
    t_hand = None
    hand_side = hand_latvel = hand_rnf = None
    peak_up = peak_side = 0.0
    rows = []

    for k in range(1400):
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT <= k < PUSH_AT + PUSH_DUR:
            d.xfrc_applied[CHEST_BODY, 0:2] = fxy
        bs = sample_balance(m, d)
        rnf = _foot_normal_force(m, d, SWING)

        if phase == "stand":
            u = stand.control(m, d)
            trig = (bs.capture_fwd_rel_support_mm > 6.0 and bs.com_vfwd > 0.05
                    if push_n >= 20 else k >= PUSH_AT + PUSH_DUR + 40)
            if k > PUSH_AT + PUSH_DUR + 40 and trig:
                phase = "pre"
                sk = 0
        elif phase == "pre":
            imp_end = 48
            if sk < 8:
                a = AR_UNLOAD_SIGN[SWING] * 0.22 * (sk / 8)
            elif sk < imp_end:
                a = AR_UNLOAD_SIGN[SWING] * 0.22
            else:
                pf = min(1.0, (sk - imp_end) / max(ramp, 1))
                a = (1.0 - pf) * (AR_UNLOAD_SIGN[SWING] * 0.22) + pf * lean_ss
            qref = stand.qpos0.copy()
            cref = stand.ctrl0.copy()
            for ci in (AR["L"], AR["R"]):
                qref[7 + ci] += a
                cref[ci] += a
            dq = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, qref, d.qpos)
            u = cref - stand.K @ np.concatenate([dq, d.qvel - stand.qvel0])
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] + a
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] + a
            if sk >= imp_end + preshift:
                phase = "ss"
                sk = 0
                t_hand = k
                mujoco.mj_subtreeVel(m, d)
                hand_side = bs.side_lean_deg
                hand_latvel = float(d.subtree_linvel[CHEST_BODY][0])
                hand_rnf = rnf
        else:  # ss hold  -- exactly the demo's hold: ss.K commands ankle-roll too
            u = np.array(ss_hold_u(), float)

        u = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        d.ctrl[:15] = u
        mujoco.mj_step(m, d)
        if phase != "stand":
            sk += 1
        bs = sample_balance(m, d)
        if phase == "ss":
            peak_up = max(peak_up, bs.up_tilt_deg)
            peak_side = max(peak_side, abs(bs.side_lean_deg))
            if full and sk % 20 == 0:
                rows.append((k, bs.up_tilt_deg, bs.fwd_lean_deg, bs.side_lean_deg,
                             _foot_normal_force(m, d, SWING), _foot_normal_force(m, d, "L"),
                             bs.com_vfwd))
        if bs.up_tilt_deg > 40:
            break
    else:
        pass

    if t_hand is None:
        print(f"  preshift={preshift:3d} push~{push_n:.0f}: never handed off "
              f"(no trigger / fell in pre)")
        return False, 99.0
    held = peak_up <= 20.0
    ss_ms = (k - t_hand) if t_hand else 0
    if full:
        for r in rows:
            print(f"    t{r[0]:4d} up{r[1]:5.1f} fwd{r[2]:+5.1f} side{r[3]:+6.1f} "
                  f"Rnf{r[4]:4.0f} Lnf{r[5]:4.0f} vF{r[6]:+.2f}")
    print(f"  preshift={preshift:3d} ramp={ramp} amp={amp} decR={int(decouple_roll)}: "
          f"hand@side {hand_side:+.1f}  latvel {hand_latvel:+.3f}  Rnf {hand_rnf:.0f}  "
          f"-> ss {ss_ms}ms  peakUp {peak_up:.1f}  peakSide {peak_side:.1f}  "
          f"{'HELD' if held else 'FELL'}")
    return held, peak_up


def democheck():
    """replicate the demo's own entry (340-step quasi-static ramp, NO push) then
    hold under ss.K in THIS harness - sanity check that ss_hold_u is not buggy."""
    from push_step_recovery_test import AR_L_IDX, AR_R_IDX, AR_SIGN
    m, d, stand_lqr, ss = build(0.12)
    ab = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="ab2", verbose=False)
    sw = LEG[SWING]
    ss_idx = (sw["hip"], sw["knee"], sw["ankle"])
    d.qpos[:] = ab.qpos0
    d.qvel[:] = ab.qvel0
    mujoco.mj_forward(m, d)
    ar = AR_SIGN[SWING] * 0.12
    for _ in range(120):
        d.ctrl[:15] = np.clip(ab.ctrl0 - ab.K @ np.concatenate(
            [_pos_error(m, ab.qpos0, d.qpos), d.qvel - ab.qvel0]),
            m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(m, d)
    for j in range(340):
        a = ar * min(1.0, j / 60.0)
        qr = ab.qpos0.copy(); cr = ab.ctrl0.copy()
        for ci in (AR_L_IDX, AR_R_IDX):
            qr[7 + ci] += a; cr[ci] += a
        u = cr - ab.K @ np.concatenate([_pos_error(m, qr, d.qpos), d.qvel - ab.qvel0])
        u[AR_L_IDX] = cr[AR_L_IDX]; u[AR_R_IDX] = cr[AR_R_IDX]
        d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(m, d)
    b = sample_balance(m, d)
    print(f"  demo entry done: side {b.side_lean_deg:+.1f}  Rnf {_foot_normal_force(m,d,SWING):.0f}")
    peak_up = 0.0
    for j in range(700):
        qref = ss.qpos0.copy(); cref = ss.ctrl0.copy()
        dx = np.concatenate([_pos_error(m, qref, d.qpos), d.qvel - ss.qvel0])
        for ci in ss_idx:
            dx[6 + ci] = 0.0; dx[m.nv + 6 + ci] = 0.0
        u = cref - ss.K @ dx
        for ci in ss_idx:
            u[ci] = cref[ci]
        d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(m, d)
        b = sample_balance(m, d)
        peak_up = max(peak_up, b.up_tilt_deg)
    print(f"  demo-style ss.K hold 700ms: peakUp {peak_up:.1f}  endSide {b.side_lean_deg:+.1f}  "
          f"{'HELD' if peak_up < 20 else 'FELL'}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--push", type=float, default=135.0)
    p.add_argument("--full", action="store_true")
    p.add_argument("--democheck", action="store_true")
    a = p.parse_args(argv)
    if a.democheck:
        democheck()
        return
    print("== push sweep at preshift=300 (near-full quasi-static entry) ==")
    for pn in (0.1, 60, 90, 110, 125, 135):
        probe(300, push_n=pn, full=a.full)
    print("\n== push sweep at preshift=160 ==")
    for pn in (0.1, 60, 90, 110, 125, 135):
        probe(160, push_n=pn, full=a.full)
    print("\n== sweep preshift duration (amp 0.12, ramp 60), push", a.push, "==")
    for ps in (40, 80, 120, 160, 220, 300):
        probe(ps, push_n=a.push, full=a.full)
    print("\n== same, roll decoupled from ss.K ==")
    for ps in (40, 80, 120, 160, 220, 300):
        probe(ps, push_n=a.push, decouple_roll=True, full=a.full)
    print("\n== amp 0.16 (bigger lean target), ramp 60 ==")
    for ps in (80, 160, 220, 300):
        probe(ps, amp=0.16, push_n=a.push, full=a.full)


if __name__ == "__main__":
    main(sys.argv[1:])
