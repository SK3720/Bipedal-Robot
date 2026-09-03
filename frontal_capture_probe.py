"""Frontal-plane feasibility analysis for the forward recovery step.

The established blocker: after the ankle-roll impulse unloads the swing foot, the
lateral CoM velocity (~0.10-0.19 m/s toward the stance foot) is not arrested
before the CoM leaves the stance-foot support, and the robot rolls over.

This probe answers ONE physical question before any more controller work:

  Is there a real time window in which the swing foot can be planted BEFORE the
  lateral CoM state becomes unrecoverable, given a 66 mm-wide sole and
  +-2.3 N.m ankle-roll?

Method - measure, don't tune:
  Phase 1  impulse unloads R (reused, unchanged).
  Phase 2  the ROLL joints are frozen at neutral (L/R ankle_roll, L/R hip_roll)
           -> CoP is pinned near the stance-foot centre, no roll control at all.
           The sagittal plane + a feed-forward forward swing still run so the
           geometry is realistic.  We log the frontal LIPM state every tick:
             x_com, v_com (lateral, world X), h, omega, capture point
             xi = x_com + v_com/omega, stance-foot lateral edges (live, from
             contacts), CoP (from contact forces), ankle-roll / hip-roll actuator
             torque, swing-foot x/z/vel, contact + load, and when xi crosses the
             stance-foot outer edge.

  Also a control-authority sub-probe: from the leaned single-support state, ramp
  the L ankle-roll to its limit and measure how far the CoP actually moves
  (the real CoP travel available to arrest the fall) and the torque at the stop.

    python frontal_capture_probe.py                 # full analysis
    python frontal_capture_probe.py --trace         # per-tick LIPM trace
Does not modify robot.xml / biped_env / sagittal_recovery / any golden script.
"""
from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS
from recovery_metrics import CHEST_BODY, _foot_normal_force, _foot_xy_z, sample_balance
from standing_balance_lqr import StandingLQR
from push_step_recovery_test import (
    StepConfig, _LQRAbout, _SingleSupportLQR, _pos_error, _smooth,
)
from step_primitive import AnkleSolver, _sole_pitch, LEG, AR, FWD_HIP_SIGN, KNEE_FLEX_SIGN

PUSH_AT = 20
AR_UNLOAD_SIGN = {"R": -1.0, "L": +1.0}
G = 9.81
FLOOR_Z = 1.0
SWING = "R"
STANCE = "L"
ROLL_CI = (LEG["L"]["hip_roll"], LEG["R"]["hip_roll"], AR["L"], AR["R"])


def _ground_cop_x(m, d):
    """lateral (world X) centre of pressure from all foot-floor contacts."""
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


def _stance_foot_edges_x(m, d, side):
    """live lateral extent of the stance sole (world X), from the collision mesh."""
    gid = m.geom(f"{side}_foot_collision").id
    mid = m.geom_dataid[gid]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    V = m.mesh_vert[va:va + vn]
    W = (d.geom_xmat[gid].reshape(3, 3) @ V.T).T + d.geom_xpos[gid]
    sole = W[W[:, 2] < W[:, 2].min() + 0.004]
    return float(sole[:, 0].min()), float(sole[:, 0].max())


def _swing_arc(w, h0, k0, hip_amp=0.48, knee_amp=0.30):
    hf = FWD_HIP_SIGN[SWING]
    s = _smooth(w)
    hip = h0 + hf * hip_amp * s
    knee = k0 + KNEE_FLEX_SIGN * (0.05 + (knee_amp - 0.05) * np.sin(np.pi * w))
    sole = 0.09 * np.sin(np.pi * w)
    return hip, knee, sole


def authority_subprobe():
    """From the leaned single-support state: ramp L ankle-roll to the limit,
    measure the CoP travel and the actuator torque at the stop."""
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)
    ab = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="ab", verbose=False)
    ss = _SingleSupportLQR(m, d, ab, StepConfig(swing=SWING, ankle_roll_amp_rad=0.12,
                                                ss_reach_steps=340), verbose=False)
    sw = LEG[SWING]
    ss_idx = (sw["hip"], sw["knee"], sw["ankle"])
    d.qpos[:] = ss.qpos0
    d.qvel[:] = ss.qvel0
    mujoco.mj_forward(m, d)
    lo, hi = m.jnt_range[m.actuator(AR[STANCE]).trnid[0]]
    arl_lo, arl_hi = m.actuator_ctrlrange[AR[STANCE]]
    print("\n=== control-authority sub-probe (leaned single support, stance = L) ===")
    print(f"  L ankle_roll ctrl range [{arl_lo:+.2f}, {arl_hi:+.2f}] rad, "
          f"actuator forcerange +-{m.actuator_forcerange[AR[STANCE], 1]:.2f} N.m")
    base_cop, base_load = _ground_cop_x(m, d)
    le, re = _stance_foot_edges_x(m, d, STANCE)
    print(f"  stance-L sole edges X = [{le*1000:+.0f}, {re*1000:+.0f}] mm  "
          f"(width {(re-le)*1000:.0f} mm), CoM x = {sample_balance(m,d).com[0]*1000:+.0f} mm")
    for tgt in (0.0, -0.10, -0.20, -0.30, arl_lo):
        d.qpos[:] = ss.qpos0
        d.qvel[:] = ss.qvel0
        mujoco.mj_forward(m, d)
        for _ in range(150):
            qref = ss.qpos0.copy(); cref = ss.ctrl0.copy()
            dx = np.concatenate([_pos_error(m, qref, d.qpos), d.qvel - ss.qvel0])
            for ci in ss_idx:
                dx[6 + ci] = 0.0; dx[m.nv + 6 + ci] = 0.0
            u = cref - ss.K @ dx
            for ci in ss_idx:
                u[ci] = cref[ci]
            u[AR[STANCE]] = tgt
            d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0],
                                  m.actuator_ctrlrange[:15, 1])
            mujoco.mj_step(m, d)
        cop, load = _ground_cop_x(m, d)
        b = sample_balance(m, d)
        tau = d.actuator_force[AR[STANCE]]
        print(f"  ankle_roll cmd {tgt:+.2f}: CoP_x {cop*1000:+6.1f} mm  "
              f"(move {(cop-base_cop)*1000:+5.1f})  tau {tau:+.2f} N.m  "
              f"load {load:5.1f} N  side {b.side_lean_deg:+.1f}  up {b.up_tilt_deg:.1f}")


def run(push_n=132.0, freeze_roll=True, trace=False, preshift_ms=230, swing_ms=140):
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)
    stand = StandingLQR(m, d, verbose=False)
    ab = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="ab", verbose=False)
    ss = _SingleSupportLQR(m, d, ab, StepConfig(swing=SWING, ankle_roll_amp_rad=0.12,
                                                ss_reach_steps=340), verbose=False)
    ank = AnkleSolver(m)
    sw = LEG[SWING]
    ss_idx = (sw["hip"], sw["knee"], sw["ankle"])
    lean_ss = float(ss.ctrl0[AR[SWING]])
    h0, k0, a0 = (ss.qpos0[7 + sw["hip"]], ss.qpos0[7 + sw["knee"]],
                  ss.qpos0[7 + sw["ankle"]])

    d.qpos[:] = stand.qpos0
    d.qvel[:] = stand.qvel0
    mujoco.mj_forward(m, d)
    fxy = np.array([0.0, -push_n])

    phase = "stand"
    sk = 0
    an_cmd = a0
    t_unload = t_swing0 = t_plant = None
    t_xi_exit = None
    swing_x0 = swing_z0 = None
    rows = []
    log = []

    for k in range(1600):
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT <= k < PUSH_AT + PUSH_DURATION_STEPS:
            d.xfrc_applied[CHEST_BODY, 0:2] = fxy
        bs = sample_balance(m, d)
        rnf = _foot_normal_force(m, d, SWING)
        pr = bs.pitch_rate

        if phase == "stand":
            u = stand.control(m, d)
            if k > PUSH_AT + PUSH_DURATION_STEPS + 40 and \
               bs.capture_fwd_rel_support_mm > 6.0 and bs.com_vfwd > 0.05:
                phase = "pre"
                sk = 0
        elif phase == "pre":
            imp_end = 48
            if sk < 8:
                a = AR_UNLOAD_SIGN[SWING] * 0.22 * (sk / 8)
            elif sk < imp_end:
                a = AR_UNLOAD_SIGN[SWING] * 0.22
            else:
                pf = min(1.0, (sk - imp_end) / max(preshift_ms, 1))
                a = (1 - pf) * AR_UNLOAD_SIGN[SWING] * 0.22 + pf * lean_ss
            qref = stand.qpos0.copy(); cref = stand.ctrl0.copy()
            for ci in (AR["L"], AR["R"]):
                qref[7 + ci] += a; cref[ci] += a
            dq = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, qref, d.qpos)
            u = cref - stand.K @ np.concatenate([dq, d.qvel - stand.qvel0])
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] + a
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] + a
            if sk >= imp_end + preshift_ms:
                phase = "swing"
                sk = 0
                t_swing0 = k
                swing_x0 = _foot_xy_z(m, d, SWING)[0]
                swing_z0 = _foot_xy_z(m, d, SWING)[2]
        else:  # swing: sagittal via ss.K, forward arc feed-forward; roll FROZEN
            w = min(1.0, sk / swing_ms)
            hip, knee, sole_tgt = _swing_arc(w, h0, k0)
            an_cmd = ank.solve(d.qpos, SWING, hip, knee, sole_tgt, prev=an_cmd)
            qref = ss.qpos0.copy(); cref = ss.ctrl0.copy()
            for ci, v in zip(ss_idx, (hip, knee, an_cmd)):
                qref[7 + ci] = v; cref[ci] = v
            dx = np.concatenate([_pos_error(m, qref, d.qpos), d.qvel - ss.qvel0])
            for ci in ss_idx:
                dx[6 + ci] = 0.0; dx[m.nv + 6 + ci] = 0.0
            if freeze_roll:
                for ci in ROLL_CI:
                    dx[6 + ci] = 0.0; dx[m.nv + 6 + ci] = 0.0
                dx[3] = dx[m.nv + 3] = 0.0
            u = cref - ss.K @ dx
            for ci in ss_idx:
                u[ci] = cref[ci]
            if freeze_roll:
                u[AR["L"]] = DEFAULT_POSE[AR["L"]] + lean_ss
                u[AR["R"]] = DEFAULT_POSE[AR["R"]] + lean_ss
                u[LEG["L"]["hip_roll"]] = DEFAULT_POSE[LEG["L"]["hip_roll"]]
                u[LEG["R"]["hip_roll"]] = DEFAULT_POSE[LEG["R"]["hip_roll"]]

        u = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        d.ctrl[:15] = u
        mujoco.mj_step(m, d)
        if phase != "stand":
            sk += 1
        bs = sample_balance(m, d)
        rnf = _foot_normal_force(m, d, SWING)
        lnf = _foot_normal_force(m, d, STANCE)

        # ---- frontal LIPM state ----
        x_com = float(bs.com[0])
        v_lat = float(bs.com_vel[0])
        h = max(float(bs.com[2]) - FLOOR_Z, 0.05)
        omega = np.sqrt(G / h)
        xi = x_com + v_lat / omega
        le, re = _stance_foot_edges_x(m, d, STANCE)
        cop_x, cop_load = _ground_cop_x(m, d)
        tau_arl = float(d.actuator_force[AR[STANCE]])
        tau_hrl = float(d.actuator_force[LEG[STANCE]["hip_roll"]])
        f = _foot_xy_z(m, d, SWING)
        fz = f[2] - (swing_z0 if swing_z0 is not None else f[2])
        fx = f[0] - (swing_x0 if swing_x0 is not None else f[0])

        if phase in ("pre", "swing") and t_unload is None and rnf < 3.0:
            t_unload = k
        if phase == "swing" and t_plant is None and rnf > 12.0 and fz < 0.02 \
           and t_unload is not None and k - t_unload > 20:
            t_plant = k
        if phase == "swing" and t_unload is not None and t_xi_exit is None and xi > re:
            t_xi_exit = k

        if phase in ("pre", "swing"):
            log.append(dict(k=k, phase=phase, x_com=x_com, v_lat=v_lat, xi=xi,
                            le=le, re=re, cop_x=cop_x, cop_load=cop_load,
                            tau_arl=tau_arl, tau_hrl=tau_hrl,
                            side=bs.side_lean_deg, up=bs.up_tilt_deg,
                            roll_rate=float((d.xmat[CHEST_BODY].reshape(3, 3) @ d.qvel[3:6])[1]),
                            rnf=rnf, lnf=lnf, fx=fx, fz=fz,
                            fvz=float(d.cvel[m.body(f"{SWING}_foot").id][5]),
                            sole=np.degrees(_sole_pitch(m, d, SWING))))
        if trace and phase in ("pre", "swing") and k % 8 == 0:
            print(f"  t{k:4d} {phase:5s} xCoM{x_com*1000:+6.1f} vLat{v_lat:+.3f} "
                  f"xi{xi*1000:+6.1f} [edge {re*1000:+.0f}] CoP{cop_x*1000:+6.1f}"
                  f"({cop_load:4.0f}N) tauAR{tau_arl:+.2f} tauHR{tau_hrl:+.2f} "
                  f"side{bs.side_lean_deg:+5.1f} rnf{rnf:4.0f} fz{fz*1000:+4.0f} fx{fx*1000:+4.0f}")

        if bs.up_tilt_deg > 50:
            break

    # ---------------- analysis ----------------
    print(f"\n=== frontal feasibility  (push {push_n:.0f} N, roll {'FROZEN' if freeze_roll else 'ss.K'}) ===")
    if not log:
        print("  never triggered")
        return
    unl = next((r for r in log if r["k"] == t_unload), log[0])
    print(f"  swing-foot unload @ t{t_unload}:  x_com {unl['x_com']*1000:+.1f} mm  "
          f"v_lat {unl['v_lat']:+.3f} m/s  ->  capture point xi {unl['xi']*1000:+.1f} mm")
    print(f"  stance-L outer edge at unload:    {unl['re']*1000:+.1f} mm   "
          f"(xi margin {(unl['re']-unl['xi'])*1000:+.1f} mm)")
    omega0 = np.sqrt(G / 0.27)
    print(f"  omega ~ {omega0:.2f} rad/s  (tau {1000/omega0:.0f} ms)")

    if t_xi_exit is not None:
        dt = t_xi_exit - t_unload
        print(f"  >> capture point CROSSED the stance-foot outer edge @ t{t_xi_exit} "
              f"= {dt} ms after unload")
    else:
        print(f"  >> capture point stayed INSIDE the stance foot for the whole logged window")

    # CoP behaviour: was it driven to the edge (authority used up) or slack?
    swing_log = [r for r in log if r["phase"] == "swing" and not np.isnan(r["cop_x"])]
    if swing_log:
        cop_max = max(r["cop_x"] for r in swing_log)
        tau_peak = max(abs(r["tau_arl"]) for r in swing_log)
        print(f"  during swing: CoP reached max {cop_max*1000:+.1f} mm "
              f"(outer edge ~{swing_log[0]['re']*1000:+.0f}); peak |ankle-roll torque| "
              f"{tau_peak:.2f} / 2.30 N.m")
    if t_plant is not None:
        pl = next(r for r in log if r["k"] == t_plant)
        print(f"  swing foot PLANTED @ t{t_plant} ({t_plant - t_swing0} ms into swing): "
              f"fwd {pl['fx']*1000:+.0f} mm lat, load {pl['rnf']:.0f} N, "
              f"side {pl['side']:+.1f}, sole {pl['sole']:+.1f} deg")
    else:
        print(f"  swing foot NEVER planted (no sustained load within window)")

    # the decisive numbers
    print("\n  --- decisive window ---")
    swing_dur_needed = 150   # measured elsewhere: ~105-150 ms swing + ~50 descend
    if t_xi_exit is not None:
        margin = (t_xi_exit - t_unload) - swing_dur_needed
        verdict = ("WINDOW EXISTS" if margin > 0 else "NO WINDOW")
        print(f"  time from unload to capture-point exit : {t_xi_exit - t_unload} ms")
        print(f"  time needed to swing + plant           : ~{swing_dur_needed} ms")
        print(f"  slack                                  : {margin:+d} ms  -> {verdict}")
    else:
        print(f"  capture point never exited in this run -> lateral plane is NOT the "
              f"limiter here (check whether the run actually fell, and why)")
    b = sample_balance(m, d)
    print(f"  final: up {b.up_tilt_deg:.1f}  side {b.side_lean_deg:+.1f}  "
          f"{'FELL' if b.up_tilt_deg > 50 else 'alive'}")
    return log


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--trace", action="store_true")
    p.add_argument("--push", type=float, default=132.0)
    a = p.parse_args(argv)
    authority_subprobe()
    run(push_n=a.push, freeze_roll=True, trace=a.trace)
    print("\n" + "=" * 70)
    run(push_n=a.push, freeze_roll=False, trace=a.trace)


if __name__ == "__main__":
    main(sys.argv[1:])
