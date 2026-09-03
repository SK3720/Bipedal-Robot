"""One-step push recovery attempt: SS-LQR unload/swing/plant -> post-plant LQR.

STATUS (2026-08-31): the recovery step does NOT work - but the failure is now
PHYSICAL, not a controller artifact.  Two controller bugs found and fixed:

  1. Sim.linearize_hold ran mjd_transitionFD + mj_step on the LIVE MjData, which
     teleported the running robot to the linearisation point (chest quat forced
     upright, velocities zeroed) every time an LQR was built mid-episode.  This
     faked all the earlier "stable double support" results.  Fixed: linearise
     on a scratch MjData (Sim._d_lin).

  2. The post-plant LQR was linearised at a physically-inconsistent pose (a
     staggered stance with a forcibly-uprighted torso), which is not near any
     equilibrium, so the DARE returned a garbage UNSTABLE gain (rho>1) that
     commanded every leg joint 10-30 rad past its limit -> both feet ripped off
     the ground -> the visible 'pretzel' contortion.  Fixed: _safe_lqr rejects
     any gain whose closed loop is not stable (rho>=1 or |K| absurd) and falls
     back to the feet-together standing LQR; ALL post-plant / SS-LQR feedback
     corrections are additionally hard-clamped (+/-0.7 / +/-1.2 rad).

Honest behaviour now (2026-09-01):
  * weight shift + swing work.
  * the terminal-descent phase in RecoveryILQR.seed_ctrl (retract swing hip,
    extend knee, plantarflex ankle, bend stance knee) makes the swing foot
    GENUINELY PLANT: @ ~588 ms, ~106 mm ahead, BOTH feet loaded (Lnf 29 Rnf 23),
    torso ~8 deg, residual forward CoM v only 0.13 m/s.  Physically continuous.
  * sagittally it recovers: forward lean +8 -> 0 -> -3 deg, forward CoM velocity
    arrested and reversed.
  * FAILS laterally: the SS-LQR leaned single-support state has ~zero frontal
    margin, so over the ~290 ms swing the CoM drifts +X (toward the old stance
    foot) and by plant it is rolling at a rate that carries side-lean +7 -> +35
    deg in 180 ms.  The fresh (lead) foot is on the far side of that rotation and
    lifts back off within ~20 ms; ankle-roll + hip-roll catch authority is
    SATURATED and cannot arrest it.  A physically continuous lateral topple.
  * conclusion: the first recovery step needs a lateral component - either the
    swing foot lands wider toward the fall, or a small lateral second step.
    Post-plant control alone cannot fix a rotation the CoM is already committed
    to at touchdown.

Run:
    python postplant_recovery.py --push 146 --slow
    python postplant_recovery.py --headless --push 146 --trace
"""
from __future__ import annotations

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from biped_env import STANDING_QUAT as STANDING_QUAT_LOCAL
from ilqr_recovery import RecoveryILQR, StepPlan, Weights
from recovery_metrics import (
    NOMINAL_CHEST_Z, _foot_normal_force, _foot_xy_z, sample_balance,
)
from standing_balance_lqr import _dare


def _safe_lqr(sim, q_ref, u_ref, verbose=True):
    """LQR about (q_ref, v=0) on scratch data.  RETURNS None if the result is
    not a valid stabiliser (closed-loop spectral radius >= 1, or gains absurdly
    large) - which happens when q_ref is not near a real equilibrium, e.g. a
    staggered plant pose with a forcibly-uprighted torso.  A garbage gain here
    once drove every leg joint 10-30 rad past its limit -> both feet ripped off
    the ground -> the 'pretzel'."""
    s = sim
    A, B = s.linearize_hold(np.concatenate([q_ref, np.zeros(s.nv)]),
                            np.clip(u_ref, s.ulo, s.uhi), 1)
    qp = np.ones(s.nv) * 2.0
    qp[0] = 30.0; qp[1] = 35.0; qp[2] = 40.0
    qp[3:6] = 350.0; qp[6:21] = 1.2
    qv = np.ones(s.nv) * 1.0
    qv[0] = 25.0; qv[1] = 35.0; qv[2] = 8.0
    qv[3:6] = 22.0; qv[6:21] = 0.5
    Q = np.diag(np.concatenate([qp, qv]))
    R = np.diag(np.ones(15) * 4.0)
    K, _, _ = _dare(A, B, Q, R)
    rho = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
    kmax = float(np.max(np.abs(K)))
    ok = rho < 0.9999 and kmax < 150.0
    if verbose:
        print(f"[post-plant LQR] rho={rho:.4f}  max|K|={kmax:.0f}  "
              f"{'accepted' if ok else 'REJECTED (not a valid stabiliser)'}")
    return K if ok else None


def run(push_n, swing="R", show=False, slow=False, verbose=True,
        ankle_roll_amp=0.12, unload_frac=0.26, swing_frac=0.40, trace=False):
    model = mujoco.MjModel.from_xml_path("robot/robot.xml")
    plan = StepPlan(swing=swing, push_n=push_n, N=130, H=10,
                    ankle_roll_amp=ankle_roll_amp, unload_frac=unload_frac,
                    swing_frac=swing_frac)
    prob = RecoveryILQR(model, plan, Weights(), verbose=verbose)
    s = prob.sim
    m, d = s.m, s.d
    viewer = None
    if show:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.cam.lookat[:] = [0.0, -0.2, 1.0]
        viewer.cam.distance = 2.3
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -8
    from ilqr_recovery import LEG_CTRL, FWD_HIP_SIGN, KNEE_FLEX_SIGN
    sw = LEG_CTRL[swing]
    st = LEG_CTRL["L" if swing == "R" else "R"]
    hf = FWD_HIP_SIGN[swing]

    prob._build_ss_lqr()                       # standing + SS(leaned) LQR
    s.set_x(prob.x0)
    swing_y0 = _foot_xy_z(m, d, swing)[1]
    H = prob.H

    samples = []
    step_i = 0
    phase = "maneuver"
    plant_step = None
    K_pp = None                  # post-plant feedback gain (valid stabiliser or standing-K)
    qj_freeze = None
    q_lin_ref = None
    plant_load_streak = 0
    TRANSFER_MS = 130           # ms: scripted weight-shift onto the lead foot
    peak_tilt = 0.0
    swing_fwd = swing_clear = 0.0
    post_plant_peak = 0.0
    residual_v = 0.0

    def sync():
        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            time.sleep(0.02 if slow else 0.002)

    def done(fell):
        r = _summary(m, d, samples, push_n, plant_step, residual_v, swing_fwd,
                     swing_clear, peak_tilt, post_plant_peak, fell=fell, verbose=verbose)
        if viewer is not None:
            print("\n  close viewer to exit")
            try:
                while viewer.is_running():
                    viewer.sync(); time.sleep(0.02 if slow else 0.002)
            except KeyboardInterrupt:
                pass
            viewer.close()
        return r

    settle_total = 4000
    n_outer = prob.N + settle_total // prob.H
    T_PLANT = int((prob.k_swing_end + 6) * prob.H)     # ms - end of the scripted plant push
    T_SWING = int(prob.k_swing_end * prob.H)
    for k in range(n_outer):
        for _ in range(prob.H):
            b = sample_balance(m, d)
            nf = _foot_normal_force(m, d, swing)
            if phase == "maneuver":
                kf = min(step_i / H, prob.N - 1e-3)
                u = prob.seed_ctrl(kf, d.qpos.copy(), d.qvel.copy())
            else:  # post-plant: (a) scripted WEIGHT SHIFT onto the lead foot
                   #                (reverse the ankle-roll lean, press the lead
                   #                 foot down, unload the trail knee), then
                   #                (b) balance feedback on the trail leg + ankles
                   #                    with the lead hip/knee frozen (planted).
                pk = step_i - plant_step
                if K_pp is None:
                    # An LQR linearised at the staggered plant pose is never a
                    # valid stabiliser here (rho>=1) - see _safe_lqr - and calling
                    # it stalls the viewer ~0.35 s right at touchdown, which reads
                    # as "the sim froze".  Use the feet-together standing-K
                    # (built at startup) directly.
                    q_lin_ref = np.concatenate([d.qpos[:7], qj_freeze]).copy()
                    q_lin_ref[3:7] = STANDING_QUAT_LOCAL
                    K_pp = prob._stand_lqr.K
                    lean0 = float(qj_freeze[9])             # ankle-roll lean at plant
                    if verbose:
                        print("  [post-plant] weight-shift onto lead foot + trail-leg feedback")

                # ramp every feedback correction in over ~60 ms so the phase
                # switch is not a step change in the position targets (that step
                # slammed the ankle-roll actuators to their torque limit -> a
                # visible jolt at touchdown).
                ramp = min(1.0, pk / 60.0)

                sm = 0.5 * (1 - np.cos(np.pi * min(1.0, pk / TRANSFER_MS)))
                hf_st = FWD_HIP_SIGN["L" if swing == "R" else "R"]
                u_ref = qj_freeze.copy()
                # CENTRE the pelvis between the two feet: release the ankle-roll
                # lean fully (lean0 -> 0) so BOTH feet share the load - a genuine
                # double-support catch rather than a single-support topple.  The
                # retained lean was what kept unloading the fresh plant.
                u_ref[9] = lean0 * (1 - sm); u_ref[14] = lean0 * (1 - sm)

                # STRONG explicit lateral catch (the standing-K roll gains are
                # weak for a staggered stance): drive both ankle-rolls + both
                # hip-rolls against measured side-lean + roll-rate.
                rollrate = (d.xmat[1].reshape(3, 3) @ d.qvel[3:6])[1]
                lat = 3.0 * np.radians(b.side_lean_deg) + 0.55 * rollrate
                lat = ramp * float(np.clip(lat, -0.55, 0.55))
                u_ref[9] += lat; u_ref[14] += lat
                u_ref[st["hip_roll"]] += 0.9 * lat; u_ref[sw["hip_roll"]] += 0.9 * lat
                # sagittal catch on the ankles vs fwd-lean + pitch-rate
                pitchrate = -(d.xmat[1].reshape(3, 3) @ d.qvel[3:6])[0]
                sag = 2.2 * np.radians(b.fwd_lean_deg) + 0.35 * pitchrate
                sag = ramp * float(np.clip(sag, -0.40, 0.40))
                u_ref[st["ankle"]] += hf_st * sag
                u_ref[sw["ankle"]] += FWD_HIP_SIGN[swing] * sag

                dq = np.zeros(s.nv)
                mujoco.mj_differentiatePos(m, dq, 1.0, q_lin_ref, d.qpos)
                fb = np.clip(K_pp @ np.concatenate([dq, d.qvel]), -0.25, 0.25)
                u = u_ref.copy()
                # gentle standing-K feedback on the trail leg only (lead hip/knee
                # frozen = planted); the explicit catch above does the lateral.
                for ci in (st["hip"], st["knee"], st["ankle"]):
                    u[ci] = u_ref[ci] - min(1.0, pk / 100.0) * fb[ci]
                u = np.clip(u, s.ulo, s.uhi)
            d.ctrl[:15] = np.clip(u, s.ulo, s.uhi)
            mujoco.mj_step(m, d)
            step_i += 1
            b = sample_balance(m, d)
            samples.append((step_i, b, phase))
            peak_tilt = max(peak_tilt, b.up_tilt_deg)
            f = _foot_xy_z(m, d, swing)
            swing_fwd = max(swing_fwd, -(f[1] - swing_y0) * 1000.0)
            swing_clear = max(swing_clear, (f[2] - 1.0) * 1000.0)
            if phase == "maneuver" and step_i > T_SWING:
                sc = getattr(b, f"{swing.lower()}_contact")
                # GENUINE plant: swing foot in real contact and bearing load for a
                # sustained window (a graze does not count).
                plant_load_streak = plant_load_streak + 1 if (sc and nf > 10.0) else 0
                force = step_i > T_PLANT + 400
                if plant_load_streak >= 18 or force:
                    plant_step = step_i
                    residual_v = b.com_vfwd
                    qj_freeze = np.clip(d.qpos[7:22].copy(), s.ulo, s.uhi)
                    phase = "postplant"
                    if verbose:
                        sfy = _foot_xy_z(m, d, swing)[1]
                        stfy = _foot_xy_z(m, d, "L" if swing == "R" else "R")[1]
                        genuine = plant_load_streak >= 18
                        print(f"  >> PLANT @ {plant_step} ms  ({'genuine load' if genuine else 'FORCED - foot never loaded'})  "
                              f"foot sep fwd {-(sfy-stfy)*1000:.0f} mm  residual CoM v_fwd {residual_v:.2f} m/s "
                              f"side_lean {b.side_lean_deg:+.1f}  Lnf {b.l_nf:.0f} Rnf {b.r_nf:.0f}")
            if phase == "postplant":
                post_plant_peak = max(post_plant_peak, b.up_tilt_deg)
                _ppk = step_i - plant_step
                if trace and (_ppk % 10 == 0 if _ppk < 200 else _ppk % 30 == 0):
                    mujoco.mj_subtreeVel(m, d)
                    com = d.subtree_com[1]; comv = d.subtree_linvel[1]
                    lf = _foot_xy_z(m, d, "L"); rf = _foot_xy_z(m, d, swing)
                    print(f"   pp+{step_i-plant_step:4d}  fwdLean {b.fwd_lean_deg:+6.1f}  "
                          f"sideL {b.side_lean_deg:+6.1f}  CoM(x{com[0]*1000:+.0f} y{-com[1]*1000:+.0f}) "
                          f"v(x{comv[0]*1000:+.0f} y{-comv[1]*1000:+.0f})  "
                          f"Lnf {b.l_nf:3.0f} Rnf {b.r_nf:3.0f}  Rft_z {(rf[2]-1)*1000:+.0f}")
            if b.up_tilt_deg > 55 or b.chest_z < NOMINAL_CHEST_Z - 0.24:
                if verbose:
                    print(f"  FELL @ {step_i} ms ({'tilt' if b.up_tilt_deg>55 else 'chest'})")
                return done(fell=True)
            try:
                sync()
            except KeyboardInterrupt:
                return done(fell=False)
        if phase == "maneuver" and step_i > T_PLANT + 400 and plant_step is None:
            if verbose:
                print("  swing foot never planted")
            return done(fell=False)
    return done(fell=False)


def _summary(m, d, samples, push_n, plant_step, residual_v, swing_fwd, swing_clear,
             peak_tilt, post_plant_peak, fell, verbose):
    b = sample_balance(m, d)
    tail = [sm[1] for sm in samples[-500:]]
    settle_up = float(np.mean([t.up_tilt_deg for t in tail])) if tail else 99
    settle_v = float(np.mean([t.com_speed_horiz for t in tail])) if tail else 99
    ds = b.l_contact and b.r_contact
    # longest run of genuine stable double support after plant
    stable_ms = run_ms = 0
    for step, sm, _tag in samples:
        if (plant_step is not None and step > plant_step
                and sm.up_tilt_deg < 10 and abs(sm.side_lean_deg) < 10
                and sm.l_contact and sm.r_contact and sm.com_speed_horiz < 0.12):
            run_ms += 1
            stable_ms = max(stable_ms, run_ms)
        else:
            run_ms = 0
    recovered = (not fell and settle_up < 12 and abs(b.side_lean_deg) < 14 and ds
                 and settle_v < 0.13 and b.chest_z > NOMINAL_CHEST_Z - 0.12)
    out = dict(push_n=push_n, planted=plant_step is not None, plant_step=plant_step,
               residual_com_vfwd=residual_v, swing_fwd_mm=swing_fwd,
               swing_clear_mm=swing_clear, peak_tilt_deg=peak_tilt,
               post_plant_peak_tilt_deg=post_plant_peak, end_up_tilt_deg=b.up_tilt_deg,
               end_side_deg=b.side_lean_deg, end_chest_z=b.chest_z,
               end_double_support=ds, end_com_speed=b.com_speed_horiz,
               settle_up=settle_up, settle_v=settle_v, fell=fell, recovered=recovered,
               stable_ds_ms=stable_ms)
    if verbose:
        print(f"\n  push {push_n:.0f} N   planted={out['planted']}@{plant_step}   "
              f"swing {swing_fwd:.0f} mm fwd / {swing_clear:.0f} mm peak clearance")
        print(f"  residual CoM v_fwd at plant {residual_v:.2f} m/s   "
              f"post-plant peak tilt {post_plant_peak:.0f} deg")
        print(f"  genuine stable double support after plant: {stable_ms} ms")
        print(f"  end: up_tilt {b.up_tilt_deg:.1f}  side {b.side_lean_deg:+.1f}  "
              f"chestZ {b.chest_z:.3f}  DS {ds}  CoM speed {b.com_speed_horiz:.3f}")
        print(f"  >>> {'RECOVERED' if recovered else 'FAILED - ' + ('fell' if fell else 'not settled')}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--push", type=float, default=146.0)
    p.add_argument("--swing", choices=["L", "R"], default="R")
    p.add_argument("--sweep", default=None)
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--ar", type=float, default=0.12, help="ankle-roll weight-shift amp")
    p.add_argument("--unload", type=float, default=0.26)
    p.add_argument("--swingf", type=float, default=0.40)
    p.add_argument("--trace", action="store_true")
    a = p.parse_args(argv)

    if a.sweep:
        rows = []
        for pn in (float(x) for x in a.sweep.split(",")):
            rows.append(run(pn, a.swing, verbose=True, ankle_roll_amp=a.ar,
                            unload_frac=a.unload, swing_frac=a.swingf))
        print("\n" + "=" * 64)
        for r in rows:
            print(f"  {r['push_n']:5.0f} N  plant@{str(r['plant_step']):>5}  "
                  f"resid_v {r['residual_com_vfwd']:.2f}  post_tilt {r['post_plant_peak_tilt_deg']:3.0f}  "
                  f"end_side {r['end_side_deg']:+4.0f}  "
                  f"{'RECOVERED' if r['recovered'] else ('FELL' if r['fell'] else 'no-settle')}")
        return

    run(a.push, a.swing, show=not a.headless, slow=a.slow, verbose=True,
        ankle_roll_amp=a.ar, unload_frac=a.unload, swing_frac=a.swingf, trace=a.trace)


if __name__ == "__main__":
    main(sys.argv[1:])
