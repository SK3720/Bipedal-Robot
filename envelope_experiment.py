"""Does stronger leg torque + a wider symmetric stance increase the physical
recovery envelope?  Controlled A/B/C/D over model_variants.py.

For each variant it measures (reusing the established building blocks):
  1. standing-LQR forward-push ceiling  (full StandingLQR)          -> no-step limit
  2. torso-only-LQR forward-push ceiling (joint pos+vel fb removed)  -> no-step limit
  3. lateral-push tolerance (torso-only LQR)                         -> stance-width probe
  4. dynamic foot-unload under a supra-ceiling forward push + ankle-roll shift
  5. single-support swing from REST (SS-LQR about the leaned state)
  6. capture-point excursion just above the no-step ceiling

Run:   python envelope_experiment.py                    # all variants, all tests
       python envelope_experiment.py --variant strong   # one variant
       python envelope_experiment.py --test 1,2,3       # subset of tests
"""

from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from biped_env import CHEST_Z_CONTACT, DEFAULT_POSE, PUSH_DURATION_STEPS, STANDING_QUAT
from model_variants import VARIANTS, describe, load_model
from recovery_metrics import (
    FLOOR_Z, NOMINAL_CHEST_Z, _foot_normal_force, _foot_xy_z, sample_balance,
)
from standing_balance_lqr import StandingLQR, _dare
from push_step_recovery_test import (
    AR_L_IDX, AR_R_IDX, AR_SIGN, LEG_IDX, _SingleSupportLQR, _pos_error, _smooth,
)

FWD = -np.pi / 2.0
SETTLE = 400
WINDOW = 4000


# ---------------------------------------------------------------- controllers
class TorsoOnlyLQR:
    """StandingLQR with the 15 joint-position AND joint-velocity errors zeroed,
    so it only stabilises the floating base (raised the baseline no-step ceiling
    from ~125 N to ~145 N in turn 7)."""
    def __init__(self, model, data):
        self.lqr = StandingLQR(model, data, verbose=False)
        self.model = model

    @property
    def qpos0(self):
        return self.lqr.qpos0

    @property
    def qvel0(self):
        return self.lqr.qvel0

    def control(self, model, data):
        dq = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, dq, 1.0, self.lqr.qpos0, data.qpos)
        dq[6:21] = 0.0
        dv = data.qvel - self.lqr.qvel0
        dv[6:21] = 0.0
        u = self.lqr.ctrl0 - self.lqr.K @ np.concatenate([dq, dv])
        return np.clip(u, model.actuator_ctrlrange[:15, 0], model.actuator_ctrlrange[:15, 1])


def _reset_stand(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = DEFAULT_POSE
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _run_push(model, data, ctrl_fn, push_n, direction=FWD, settle_ctrl=None,
              window=WINDOW, overlay=None):
    """Reset -> settle under ctrl_fn -> impulse -> run.  Returns a metrics dict."""
    _reset_stand(model, data)
    sc = settle_ctrl or ctrl_fn
    for _ in range(SETTLE):
        data.ctrl[:15] = sc(model, data)
        mujoco.mj_step(model, data)
    com0 = sample_balance(model, data).com.copy()
    lf0 = _foot_xy_z(model, data, "L")[1]
    rf0 = _foot_xy_z(model, data, "R")[1]
    fxy = push_n * np.array([np.cos(direction), np.sin(direction)])

    peak_up = 0.0
    peak_side = 0.0
    peak_capture = -1e9
    min_swing_nf = 1e9
    min_z = 9.0
    ever_air = False
    fell = False
    for k in range(window):
        data.xfrc_applied[1, :] = 0.0
        if 5 <= k < 5 + PUSH_DURATION_STEPS:
            data.xfrc_applied[1, 0:2] = fxy
        u = ctrl_fn(model, data)
        if overlay is not None:
            u = overlay(model, data, k, u)
        data.ctrl[:15] = np.clip(u, model.actuator_ctrlrange[:15, 0],
                                 model.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(model, data)
        bs = sample_balance(model, data)
        peak_up = max(peak_up, bs.up_tilt_deg)
        peak_side = max(peak_side, abs(bs.side_lean_deg))
        peak_capture = max(peak_capture, bs.capture_fwd_rel_support_mm)
        min_z = min(min_z, bs.chest_z)
        lnf = _foot_normal_force(model, data, "L")
        rnf = _foot_normal_force(model, data, "R")
        min_swing_nf = min(min_swing_nf, min(lnf, rnf))
        if not bs.l_contact and not bs.r_contact:
            ever_air = True
        if bs.up_tilt_deg > 50.0 or bs.chest_z < NOMINAL_CHEST_Z - 0.20:
            fell = True
            break
    bs = sample_balance(model, data)
    com = bs.com
    lf = _foot_xy_z(model, data, "L")[1]
    rf = _foot_xy_z(model, data, "R")[1]
    recovered = (not fell and bs.up_tilt_deg < 12 and abs(bs.side_lean_deg) < 12
                 and bs.l_contact and bs.r_contact and bs.com_speed_horiz < 0.15
                 and bs.chest_z > NOMINAL_CHEST_Z - 0.06)
    return dict(
        fell=fell, recovered=recovered, peak_up=peak_up, peak_side=peak_side,
        peak_capture_mm=peak_capture, min_swing_nf=min_swing_nf, min_z=min_z,
        ever_air=ever_air,
        com_drift_fwd_mm=-(com[1] - com0[1]) * 1000.0,
        lfoot_fwd_mm=-(lf - lf0) * 1000.0, rfoot_fwd_mm=-(rf - rf0) * 1000.0,
    )


def _ceiling(model, data, ctrl_fn, lo=80, hi=280, direction=FWD, tol=4):
    """Largest push (N) that RECOVERS in place, by bisection."""
    if not _run_push(model, data, ctrl_fn, lo, direction)["recovered"]:
        return lo - tol  # even lo fails
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _run_push(model, data, ctrl_fn, mid, direction)["recovered"]:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------- tests
def test_1_standing_lqr(model, data):
    lqr = StandingLQR(model, data, verbose=False)
    c = lqr.control
    cap = _ceiling(model, data, c)
    r = _run_push(model, data, c, cap + 20)
    return dict(ceiling_N=cap, at_ceiling_plus20_peak_up=r["peak_up"],
               at_ceiling_plus20_capture_mm=r["peak_capture_mm"])


def test_2_torso_only(model, data):
    to = TorsoOnlyLQR(model, data)
    cap = _ceiling(model, data, to.control)
    r = _run_push(model, data, to.control, cap + 20)
    return dict(ceiling_N=cap, capture_at_ceil_plus20_mm=r["peak_capture_mm"],
               peak_up_at_ceil_plus20=r["peak_up"])


def test_3_lateral(model, data):
    to = TorsoOnlyLQR(model, data)
    cap = _ceiling(model, data, to.control, lo=40, hi=320, direction=0.0)  # +X
    return dict(lateral_ceiling_N=cap)


def test_4_dynamic_unload(model, data, push_n=160):
    """Forward push (fixed 160 N, ~above every variant's torso-only ceiling) + a
    fast ankle-roll shift toward the L foot.  Can we drive the R (swing) foot to
    ~0 N while staying up, and for how long?"""
    to = TorsoOnlyLQR(model, data)
    ar_full = AR_SIGN["R"] * 0.14

    _reset_stand(model, data)
    for _ in range(SETTLE):
        data.ctrl[:15] = to.control(model, data)
        mujoco.mj_step(model, data)
    fxy = push_n * np.array([np.cos(FWD), np.sin(FWD)])
    win_lo = win_hi = None
    min_rnf = 1e9
    peak_up = 0.0
    peak_side = 0.0
    fell = False
    k = 0
    for k in range(1200):
        data.xfrc_applied[1, :] = 0.0
        if 5 <= k < 5 + PUSH_DURATION_STEPS:
            data.xfrc_applied[1, 0:2] = fxy
        u = to.control(model, data)
        if k >= 25:                      # start the shift, ramp over 60 steps
            a = ar_full * min(1.0, (k - 25) / 60.0)
            u = u.copy(); u[AR_L_IDX] += a; u[AR_R_IDX] += a
        data.ctrl[:15] = np.clip(u, model.actuator_ctrlrange[:15, 0], model.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(model, data)
        rnf = _foot_normal_force(model, data, "R")
        min_rnf = min(min_rnf, rnf)
        bs = sample_balance(model, data)
        peak_up = max(peak_up, bs.up_tilt_deg)
        peak_side = max(peak_side, abs(bs.side_lean_deg))
        if rnf < 2.0 and win_lo is None:
            win_lo = k
        if win_lo is not None and win_hi is None and rnf > 5.0 and k > win_lo + 5:
            win_hi = k
        if bs.up_tilt_deg > 45:
            fell = True
            break
    unload_ms = ((win_hi or k) - win_lo) if win_lo is not None else 0
    return dict(push_N=round(push_n), min_swing_nf=round(min_rnf, 1),
               sustained_unload_ms=unload_ms, fell=fell,
               peak_up_deg=round(peak_up, 1), peak_side_deg=round(peak_side, 1))


def test_5_ss_swing(model, data):
    """SS-LQR about the leaned single-support state: hold + swing R leg from rest."""
    class _C:
        swing = "R"
        ankle_roll_amp_rad = 0.12
        ankle_roll_ramp = 55
        ss_reach_steps = 340
    stand = StandingLQR(model, data, verbose=False)
    # build the leaned SS state + K
    try:
        ss = _SingleSupportLQR(model, data, stand, _C, verbose=False)
    except Exception as e:
        return dict(error=str(e)[:60])
    idx = LEG_IDX["R"]
    sw3 = (idx["hip"], idx["knee"], idx["ankle"])
    ss_tilt = sample_balance(model, data).up_tilt_deg
    ss_rnf = _foot_normal_force(model, data, "R")

    data.qpos[:] = ss.qpos0
    data.qvel[:] = ss.qvel0
    mujoco.mj_forward(model, data)
    y0 = _foot_xy_z(model, data, "R")[1]
    peak_fwd = 0.0
    peak_clr = 0.0
    fell = False
    for j in range(140 + 400):
        w = min(1.0, j / 140.0)
        s = _smooth(w)
        bump = np.sin(np.pi * w)
        hip = 0.45 * s
        knee = -(0.06 + (0.25 - 0.06) * bump)
        ank = -0.20 * bump
        qref = ss.qpos0.copy()
        cref = ss.ctrl0.copy()
        for ci, v in zip(sw3, (hip, knee, ank)):
            qref[7 + ci] = v
            cref[ci] = v
        dx = np.concatenate([_pos_error(model, qref, data.qpos), data.qvel - ss.qvel0])
        for ci in sw3:
            dx[6 + ci] = 0.0
            dx[model.nv + 6 + ci] = 0.0
        u = cref - ss.K @ dx
        for ci in sw3:
            u[ci] = cref[ci]
        data.ctrl[:15] = np.clip(u, model.actuator_ctrlrange[:15, 0], model.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(model, data)
        f = _foot_xy_z(model, data, "R")
        peak_fwd = max(peak_fwd, -(f[1] - y0) * 1000.0)
        peak_clr = max(peak_clr, (f[2] - FLOOR_Z) * 1000.0)
        b = sample_balance(model, data)
        if b.up_tilt_deg > 30:
            fell = True
            break
    b = sample_balance(model, data)
    stance_nf = _foot_normal_force(model, data, "L")
    return dict(ss_tilt_deg=ss_tilt, ss_swing_foot_nf=ss_rnf,
               swing_fwd_mm=peak_fwd, swing_clr_mm=peak_clr,
               end_tilt_deg=b.up_tilt_deg, end_side_deg=b.side_lean_deg,
               held=(not fell and b.up_tilt_deg < 18 and stance_nf > 3))


TESTS = {
    1: ("standing-LQR fwd-push ceiling", test_1_standing_lqr),
    2: ("torso-only-LQR fwd-push ceiling", test_2_torso_only),
    3: ("lateral-push tolerance", test_3_lateral),
    4: ("dynamic foot unload under push", test_4_dynamic_unload),
    5: ("single-support swing from rest", test_5_ss_swing),
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None, choices=VARIANTS)
    p.add_argument("--test", default=None, help="comma list of test numbers 1..5")
    a = p.parse_args(argv)
    variants = [a.variant] if a.variant else list(VARIANTS)
    tests = [int(x) for x in a.test.split(",")] if a.test else list(TESTS)

    print("=" * 78)
    for v in variants:
        print(describe(v))
    print("=" * 78)

    results = {}
    for v in variants:
        results[v] = {}
        for t in tests:
            name, fn = TESTS[t]
            model = load_model(v)          # fresh model+data per test -> deterministic
            data = mujoco.MjData(model)
            try:
                r = fn(model, data)
            except Exception as e:
                r = {"error": repr(e)[:80]}
            results[v][t] = r
            print(f"\n[{v}]  test {t}: {name}")
            for k, val in r.items():
                vv = round(val, 2) if isinstance(val, float) else val
                print(f"    {k:32s} {vv}")

    # compact comparison for the key numbers
    print("\n" + "=" * 78)
    print("SUMMARY  (N = newtons of forward-push impulse recovered IN PLACE)")
    hdr = f"{'variant':<11}"
    for t in tests:
        hdr += f" {('t%d' % t):>10}"
    print(hdr)
    for v in variants:
        row = f"{v:<11}"
        for t in tests:
            r = results[v][t]
            key = {1: "ceiling_N", 2: "ceiling_N", 3: "lateral_ceiling_N",
                   4: "sustained_unload_ms", 5: "swing_fwd_mm"}[t]
            row += f" {str(r.get(key, r.get('error', '?')))[:10]:>10}"
        print(row)


if __name__ == "__main__":
    main(sys.argv[1:])
