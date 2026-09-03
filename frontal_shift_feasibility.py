"""Can the weight-shift set up a RECOVERABLE single-support state?

frontal_capture_probe.py established the real numbers at swing-foot unload:
  x_com +65 mm, v_lat +0.262 m/s  ->  capture point xi = +109 mm
  stance-L sole outer edge = +100 mm    ->  xi is 9 mm PAST the edge already.

But the sensitivity is steep:  v_lat 0.20 -> xi +98 (just inside);
v_lat 0.15 -> xi +90 (10 mm margin).  So the whole problem may reduce to:

  get v_lat at unload down from 0.26 to <= ~0.18 m/s
  while still unloading the swing foot (Rnf -> < ~5 N).

This sweeps the weight-shift PROFILE (shape + duration + lean target) and, for
each, reports the frontal state at the instant the swing foot first goes light:
v_lat, x_com, capture point, margin to the stance-foot outer edge, swing-foot
load, and the stance ankle-roll torque headroom.  No swing, no new controller -
just: does a recoverable hand-off state exist?

    python frontal_shift_feasibility.py
    python frontal_shift_feasibility.py --trace
Does not modify any golden script.
"""
from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS
from recovery_metrics import CHEST_BODY, _foot_normal_force, _foot_xy_z, sample_balance
from standing_balance_lqr import StandingLQR
from push_step_recovery_test import StepConfig, _LQRAbout, _SingleSupportLQR, _pos_error
from step_primitive import LEG, AR

PUSH_AT = 20
AR_UNLOAD_SIGN = {"R": -1.0, "L": +1.0}
G, FLOOR_Z = 9.81, 1.0
SWING, STANCE = "R", "L"


def _stance_outer_edge_x(m, d, side):
    gid = m.geom(f"{side}_foot_collision").id
    mid = m.geom_dataid[gid]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    V = m.mesh_vert[va:va + vn]
    W = (d.geom_xmat[gid].reshape(3, 3) @ V.T).T + d.geom_xpos[gid]
    sole = W[W[:, 2] < W[:, 2].min() + 0.004]
    return float(sole[:, 0].max()), float(sole[:, 0].min())


def _smootherstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _profile(shape, sk, imp_end, dur, lean_ss, overshoot):
    """ankle-roll bias a(t) after the impulse, for the weight shift."""
    if sk < imp_end:
        return None  # impulse handled by caller
    u = (sk - imp_end) / max(dur, 1)
    if shape == "linear":                       # current behaviour
        p = min(1.0, u)
        return p * lean_ss
    if shape == "smoother":                     # S-curve, zero end-velocity
        return _smootherstep(min(1.0, u)) * lean_ss
    if shape == "overshoot":                    # go past, then settle back
        s = _smootherstep(min(1.0, u))
        if u < 0.6:
            return _smootherstep(min(1.0, u / 0.6)) * lean_ss * (1.0 + overshoot)
        b = _smootherstep((u - 0.6) / 0.4) if u < 1.0 else 1.0
        return lean_ss * ((1.0 + overshoot) * (1.0 - b) + b)
    raise ValueError(shape)


def trial(shape="smoother", dur=260, lean_scale=1.0, overshoot=0.25,
          push_n=132.0, trace=False):
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)
    stand = StandingLQR(m, d, verbose=False)
    ab = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="ab", verbose=False)
    ss = _SingleSupportLQR(m, d, ab, StepConfig(swing=SWING, ankle_roll_amp_rad=0.12,
                                                ss_reach_steps=340), verbose=False)
    lean_ss = float(ss.ctrl0[AR[SWING]]) * lean_scale
    d.qpos[:] = stand.qpos0
    d.qvel[:] = stand.qvel0
    mujoco.mj_forward(m, d)
    fxy = np.array([0.0, -push_n])

    phase = "stand"
    sk = 0
    imp_end = 48
    unload = None
    minv_after = None
    light_streak = 0
    rows = []

    for k in range(1000):
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT <= k < PUSH_AT + PUSH_DURATION_STEPS:
            d.xfrc_applied[CHEST_BODY, 0:2] = fxy
        bs = sample_balance(m, d)
        rnf = _foot_normal_force(m, d, SWING)

        if phase == "stand":
            u = stand.control(m, d)
            if k > PUSH_AT + PUSH_DURATION_STEPS + 40 and \
               bs.capture_fwd_rel_support_mm > 6.0 and bs.com_vfwd > 0.05:
                phase = "shift"
                sk = 0
        else:
            if sk < 8:
                a = AR_UNLOAD_SIGN[SWING] * 0.22 * (sk / 8)
            elif sk < imp_end:
                a = AR_UNLOAD_SIGN[SWING] * 0.22
            else:
                a = _profile(shape, sk, imp_end, dur, lean_ss, overshoot)
            qref = stand.qpos0.copy(); cref = stand.ctrl0.copy()
            for ci in (AR["L"], AR["R"]):
                qref[7 + ci] += a; cref[ci] += a
            dq = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, qref, d.qpos)
            u = cref - stand.K @ np.concatenate([dq, d.qvel - stand.qvel0])
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] + a
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] + a

        u = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        d.ctrl[:15] = u
        mujoco.mj_step(m, d)
        if phase != "stand":
            sk += 1
        bs = sample_balance(m, d)
        rnf = _foot_normal_force(m, d, SWING)

        x_com = float(bs.com[0])
        v_lat = float(bs.com_vel[0])
        h = max(float(bs.com[2]) - FLOOR_Z, 0.05)
        omega = np.sqrt(G / h)
        xi = x_com + v_lat / omega
        oe, ie = _stance_outer_edge_x(m, d, STANCE)
        tau_arl = float(d.actuator_force[AR[STANCE]])

        if phase == "shift":
            rows.append((k, sk, x_com, v_lat, xi, oe, rnf, tau_arl,
                         bs.side_lean_deg, bs.up_tilt_deg))
            light_streak = light_streak + 1 if rnf < 5.0 else 0
            if unload is None and light_streak >= 15 and sk > imp_end + 40:
                unload = (k, sk, x_com, v_lat, xi, oe, tau_arl, bs.side_lean_deg)
            if unload is not None:
                mv = abs(v_lat)
                if minv_after is None or mv < minv_after[0]:
                    minv_after = (mv, k, x_com, v_lat, xi, oe)
        if trace and phase == "shift" and sk % 10 == 0:
            print(f"    sk{sk:4d} xCoM{x_com*1000:+6.1f} vLat{v_lat:+.3f} "
                  f"xi{xi*1000:+6.1f}[oe{oe*1000:+.0f}] rnf{rnf:5.1f} "
                  f"tauAR{tau_arl:+.2f} side{bs.side_lean_deg:+5.1f}")
        if bs.up_tilt_deg > 45:
            break

    if unload is None:
        # foot never unloaded - report the lightest it got
        light = min(rows, key=lambda r: r[6]) if rows else None
        return dict(ok=False, reason="foot never < 5N",
                    min_rnf=(light[6] if light else None))
    k0, sk0, xc, vl, xi0, oe0, tau0, side0 = unload
    margin0 = (oe0 - xi0) * 1000.0
    # also the best (lowest-velocity) moment after unload, if the shift oscillates
    best = minv_after
    best_margin = (best[5] - best[4]) * 1000.0 if best else None
    return dict(ok=True, k=k0, sk_after_imp=sk0 - 48,
                x_com_mm=xc * 1000, v_lat=vl, xi_mm=xi0 * 1000,
                edge_mm=oe0 * 1000, margin_mm=margin0, tau_arl=tau0, side=side0,
                best_v=best[0] if best else None,
                best_xi_mm=best[4] * 1000 if best else None,
                best_margin_mm=best_margin)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--trace", action="store_true")
    p.add_argument("--push", type=float, default=132.0)
    a = p.parse_args(argv)

    print("legend: at the instant the swing foot first goes light (<5 N) -")
    print("  v_lat = lateral CoM speed | xi = capture point | margin = edge - xi")
    print("  (margin > 0  =>  capture point still inside the stance foot = recoverable)\n")
    print(f"  {'profile':<34} {'t*':>4} {'xCoM':>6} {'v_lat':>6} {'xi':>6} "
          f"{'edge':>5} {'margin':>7} {'tauAR':>6} {'side':>5}   best_v / best_margin")
    combos = []
    for shape in ("linear", "smoother", "overshoot"):
        for dur in (180, 260, 360, 480):
            for ls in (0.75, 0.9, 1.0):
                combos.append((shape, dur, ls))
    for shape, dur, ls in combos:
        r = trial(shape=shape, dur=dur, lean_scale=ls, push_n=a.push, trace=False)
        tag = f"{shape:<10} dur{dur:<4} lean*{ls:<4}"
        if not r["ok"]:
            print(f"  {tag:<34}  --  {r['reason']}  (min Rnf "
                  f"{r['min_rnf'] if r['min_rnf'] is not None else '?'})")
            continue
        bv = f"{r['best_v']:.3f}" if r["best_v"] is not None else "  -  "
        bm = f"{r['best_margin_mm']:+.0f}" if r["best_margin_mm"] is not None else " - "
        flag = "  <== RECOVERABLE" if r["margin_mm"] > 0 else ""
        print(f"  {tag:<34} {r['sk_after_imp']:4d} {r['x_com_mm']:+6.1f} "
              f"{r['v_lat']:+.3f} {r['xi_mm']:+6.1f} {r['edge_mm']:+5.0f} "
              f"{r['margin_mm']:+7.1f} {r['tau_arl']:+6.2f} {r['side']:+5.1f}   "
              f"{bv} / {bm}{flag}")

    if a.trace:
        print("\n--- trace: smoother dur360 lean*0.9 ---")
        trial(shape="smoother", dur=360, lean_scale=0.9, push_n=a.push, trace=True)


if __name__ == "__main__":
    main(sys.argv[1:])
