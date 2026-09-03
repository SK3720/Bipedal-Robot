"""Clean commanded forward-step primitive (no push).

Architecture (chosen after a spike showed the feet-together StandingLQR CANNOT
hold a real, foot-fully-airborne swing - it only tolerated the 14 mm shuffle in
sagittal_recovery):

  stand (_LQRAbout, feet-together)              - balance in double support
  ss    (_SingleSupportLQR, leaned one-foot)    - balance while the foot is airborne
        both built ONCE, offline (no live-state linearisation -> no corruption bug)

  STAND  -> SHIFT (ramp the ankle-roll lean to the ss value under stand.K)
         -> SWING (ss.K; swing leg = decoupled feed-forward: demo hip/knee arc +
                   an ankle servo that keeps the SOLE FLAT)
         -> DESCEND (ss.K; foot comes down, forward velocity eased to ~0, sole flat)
         -> [contact: foot was airborne, now sustained load]
         -> TRANSFER (blend ss.K -> stand.K over ~150 ms, release the lean to 0,
                      let the CoM/load move onto the new foot - no hand-off jerk)
         -> SETTLE (stand.K)

Reuses single_support_step_demo's proven SS hold; adds the descent + plant +
smooth transition it left unsolved.

    python step_primitive.py                 # single step, headless, trace
    python step_primitive.py --slow
    python step_primitive.py --sweep
Does not modify robot.xml / biped_env / push_step_recovery_test / the demo.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from biped_env import DEFAULT_POSE
from recovery_metrics import (
    CHEST_BODY, NOMINAL_CHEST_Z, _foot_normal_force, _foot_xy_z, sample_balance,
)
from push_step_recovery_test import (
    AR_L_IDX, AR_R_IDX, AR_SIGN, LEG_IDX, StepConfig,
    _LQRAbout, _SingleSupportLQR, _pos_error, _smooth,
)

LEG = {"L": dict(hip_roll=5, hip=6, knee=7, ankle=8),
       "R": dict(hip_roll=10, hip=11, knee=12, ankle=13)}
AR = {"L": 9, "R": 14}
FWD_HIP_SIGN = {"L": -1.0, "R": +1.0}
KNEE_FLEX_SIGN = -1.0

_HEEL_TOE = {
    "L": (np.array([-0.0303, 0.0235, 0.0161]), np.array([-0.0302, -0.0642, 0.0161])),
    "R": (np.array([-0.0302, 0.0242, -0.0161]), np.array([-0.0303, -0.0635, -0.0161])),
}


def _sole_pitch(model, data, side):
    bid = model.body(f"{side}_foot").id
    R = data.xmat[bid].reshape(3, 3)
    p = data.xpos[bid]
    heel_l, toe_l = _HEEL_TOE[side]
    heel = R @ heel_l + p
    toe = R @ toe_l + p
    ht = toe - heel
    return float(np.arctan2(ht[2], np.linalg.norm(ht[:2]) + 1e-12))


class AnkleSolver:
    """finds the ankle_pitch joint value that gives a target world sole pitch,
    given the CURRENT torso attitude + hip/knee.  Runs on a scratch MjData
    seeded from the live qpos (read-only) - never steps or linearises live data."""

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)

    def solve(self, live_qpos, side, hip, knee, target_pitch, prev=None):
        idx = LEG[side]
        jh, jk, ja = idx["hip"], idx["knee"], idx["ankle"]
        d = self.d
        d.qpos[:] = live_qpos
        d.qpos[7 + jh] = hip
        d.qpos[7 + jk] = knee
        lo, hi = self.m.jnt_range[self.m.actuator(ja).trnid[0]]

        def pitch(av):
            d.qpos[7 + ja] = av
            mujoco.mj_kinematics(self.m, d)
            return _sole_pitch(self.m, d, side)

        a = float(np.clip(prev if prev is not None else live_qpos[7 + ja],
                          lo + 0.03, hi - 0.03))
        p = pitch(a)
        for _ in range(10):
            if abs(p - target_pitch) < 3e-3:
                break
            a2 = float(np.clip(a - 0.15, lo + 0.03, hi - 0.03))
            p2 = pitch(a2)
            slope = (p2 - p) / (a2 - a + 1e-9)
            a = float(np.clip(a - (p - target_pitch) / (slope + 1e-9),
                              lo + 0.03, hi - 0.03))
            p = pitch(a)
        return a


@dataclass
class StepCfg:
    swing: str = "R"
    shift_ms: int = 320           # reach the ss leaned state (demo uses 340)
    shift_ramp_ms: int = 60      # ankle-roll ramp within the shift
    swing_ms: int = 150          # hip forward + knee lift arc
    descend_ms: int = 110        # foot down, forward vel -> 0, sole flat
    transfer_ms: int = 200       # load the new foot, push CoM onto it, release lean
    transfer_press: float = 0.28  # trailing-ankle plantarflex push-off during transfer
    settle_ms: int = 2200

    swing_hip_rad: float = 0.45   # peak forward hip flexion (demo uses 0.45)
    swing_knee_rad: float = 0.30  # peak knee flexion (mid-swing clearance bump)
    knee_land_frac: float = 0.10  # knee flex held at touchdown (small = foot low)
    hip_retract_frac: float = 0.14  # pull hip back slightly in descend (decelerate foot)
    sole_toe_up_mid: float = 0.10  # rad toe-up mid-swing (toe-stub margin) -> 0 at TD

    ankle_roll_amp: float = 0.12   # ss lean magnitude (matches demo / StepConfig)
    ss_reach_steps: int = 340

    plant_nf: float = 12.0
    plant_hold: int = 14
    stance_cap: float = 0.35
    swing_hiproll_cap: float = 0.06


def build(cfg: StepCfg):
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)
    scfg = StepConfig(swing=cfg.swing, ankle_roll_amp_rad=cfg.ankle_roll_amp,
                      ss_reach_steps=cfg.ss_reach_steps)
    stand = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="stand", verbose=False)
    ss = _SingleSupportLQR(m, d, stand, scfg, verbose=False)
    return m, d, stand, ss


def _swing_leg_targets(cfg, phase, w, base):
    """(hip, knee) joint arc + sole-pitch target.  base = (h0, k0) from ss.qpos0.

      swing:   hip flexes forward, knee flexes to LIFT the foot (ends still bent
               and up - the descent lowers it, so touchdown is controlled).
      descend: hip held forward (foot stays out front), knee straightens toward
               knee_land_frac -> foot comes straight DOWN, sole ramped to flat.
    """
    hf = FWD_HIP_SIGN[cfg.swing]
    h0, k0 = base
    if phase == "swing":
        # demo shape: hip drives forward, knee sin-BUMP for mid-swing clearance
        # then nearly straight at the end (max forward reach).
        s = _smooth(w)
        hip = h0 + hf * cfg.swing_hip_rad * s
        knee = k0 + KNEE_FLEX_SIGN * (0.05 + (cfg.swing_knee_rad - 0.05) * np.sin(np.pi * w))
        sole_tgt = cfg.sole_toe_up_mid * np.sin(np.pi * w)
    else:  # descend: foot is out front + a bit up; lower it to the ground.
        # w can exceed 1 (contact-seeking) - keep driving the foot down + slightly
        # retract the hip to bleed off forward velocity; sole -> flat.
        s = _smooth(min(1.0, w))
        over = max(0.0, w - 1.0)
        hip = h0 + hf * cfg.swing_hip_rad * (1.0 - cfg.hip_retract_frac * s)
        knee = k0 + KNEE_FLEX_SIGN * (cfg.knee_land_frac + 0.20 * (1.0 - s) - 0.25 * over)
        sole_tgt = cfg.sole_toe_up_mid * 0.4 * (1.0 - s)
    return hip, knee, sole_tgt


def run(cfg: StepCfg, show=False, slow=False, trace=True, verbose=True,
        push_n=0.0, push_at_phase="swing"):
    m, d, stand, ss = build(cfg)
    anksolve = AnkleSolver(m)
    _pushed = [False]
    swing = cfg.swing
    sw = LEG[swing]
    stn = LEG["L" if swing == "R" else "R"]
    ar_sign = AR_SIGN[swing]
    lean_ss = float(ss.ctrl0[AR[swing]])   # ankle-roll value at the ss leaned state
    h0 = ss.qpos0[7 + sw["hip"]]
    k0 = ss.qpos0[7 + sw["knee"]]
    a0 = ss.qpos0[7 + sw["ankle"]]
    ja_lo, ja_hi = m.jnt_range[m.actuator(sw["ankle"]).trnid[0]]

    d.qpos[:] = stand.qpos0
    d.qvel[:] = stand.qvel0
    mujoco.mj_forward(m, d)

    viewer = None
    if show:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.cam.lookat[:] = [0.0, -0.25, 1.05]
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -7

    ss_idx = (sw["hip"], sw["knee"], sw["ankle"])
    phase = "stand"
    t0 = 50
    sk = 0
    foot_lifted = False
    plant_streak = 0
    contact_ms = None
    an_cmd = a0
    plant_q = np.array([h0, k0, a0])
    settle_qref = stand.qpos0.copy()
    settle_cref = stand.ctrl0.copy()
    rows = []
    swing_y0 = _foot_xy_z(m, d, swing)[1]
    swing_z0 = _foot_xy_z(m, d, swing)[2]
    peak_fwd = peak_clear = 0.0
    sat_any = 0

    def stand_u(qref=None, cref=None):
        qr = stand.qpos0 if qref is None else qref
        cr = stand.ctrl0 if cref is None else cref
        return cr - stand.K @ np.concatenate(
            [_pos_error(m, qr, d.qpos), d.qvel - stand.qvel0])

    def ss_u(hip, knee, ankle):
        qref = ss.qpos0.copy(); cref = ss.ctrl0.copy()
        for ci, v in zip(ss_idx, (hip, knee, ankle)):
            qref[7 + ci] = v; cref[ci] = v
        dx = np.concatenate([_pos_error(m, qref, d.qpos), d.qvel - ss.qvel0])
        for ci in ss_idx:
            dx[6 + ci] = 0.0; dx[m.nv + 6 + ci] = 0.0
        u = cref - ss.K @ dx
        for ci in ss_idx:
            u[ci] = cref[ci]
        return u

    k = 0
    T_END = t0 + cfg.shift_ms + cfg.swing_ms + cfg.descend_ms + 600 + cfg.transfer_ms + cfg.settle_ms
    push_k0 = [None]
    while k < T_END:
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        if push_n and phase == push_at_phase:
            if push_k0[0] is None:
                push_k0[0] = k
            if k - push_k0[0] < 5:
                d.xfrc_applied[CHEST_BODY, 0:2] = [0.0, -push_n]
        bs = sample_balance(m, d)
        sw_nf = _foot_normal_force(m, d, swing)
        st_nf = _foot_normal_force(m, d, "L" if swing == "R" else "R")
        sp = _sole_pitch(m, d, swing)

        # ------ lean schedule ------
        if phase == "stand":
            lean = 0.0
        elif phase == "shift":
            lean = lean_ss * min(1.0, sk / cfg.shift_ramp_ms)
        elif phase in ("swing", "descend"):
            lean = lean_ss
        elif phase == "transfer":
            lean = lean_ss * max(0.0, 1.0 - sk / cfg.transfer_ms)
        else:
            lean = 0.0

        # ------ base control ------
        if phase == "stand":
            u = stand_u()
        elif phase == "shift":
            # reference includes the developing lean so the LQR does NOT fight it
            qref = stand.qpos0.copy(); cref = stand.ctrl0.copy()
            for ci in (AR["L"], AR["R"]):
                qref[7 + ci] += lean; cref[ci] += lean
            u = stand_u(qref, cref)
        elif phase in ("swing", "descend"):
            seg = "swing" if phase == "swing" else "descend"
            dur = cfg.swing_ms if seg == "swing" else cfg.descend_ms
            w = (min(1.0, sk / dur) if seg == "swing"
                 else sk / dur)   # descend: allow w>1 (contact-seeking)
            hip, knee, sole_tgt = _swing_leg_targets(cfg, seg, w, (h0, k0))
            # ankle solved for the target sole pitch given the LIVE torso attitude
            an_cmd = anksolve.solve(d.qpos, swing, hip, knee, sole_tgt, prev=an_cmd)
            u = ss_u(hip, knee, an_cmd)
        elif phase == "transfer":
            # WEIGHT TRANSFER onto the new (swing) foot:
            #  - the new leg actively presses down (small knee flex + ankle
            #    plantarflex) so it takes load;
            #  - the OLD stance leg stays on ss.K but does NOT fight the CoM
            #    drifting forward, and gets a gentle plantarflex push-off that
            #    lifts the CoM forward + up onto the new foot;
            #  - the ankle-roll lean releases toward 0.
            b = _smooth(min(1.0, sk / cfg.transfer_ms))
            press = b * cfg.transfer_press
            hip_new = plant_q[0]
            knee_new = plant_q[1] + KNEE_FLEX_SIGN * 0.10 * b
            an_new = plant_q[2] + FWD_HIP_SIGN[swing] * 0.10 * b   # plantarflex a touch
            u = ss_u(hip_new, knee_new, an_new)
            u[stn["ankle"]] += FWD_HIP_SIGN["L" if swing == "R" else "R"] * press
        else:  # settle: stand.K about the achieved staggered stance, torso upright
            u = stand_u(settle_qref, settle_cref)
        # ------ ankle-roll: scheduled lean (stand/shift/swing/descend/transfer)
        #        or the settle roll damper ------
        u = np.array(u, float)
        if phase == "settle":
            rr = (d.xmat[CHEST_BODY].reshape(3, 3) @ d.qvel[3:6])[1]
            roll_corr = float(np.clip(1.2 * np.radians(bs.side_lean_deg) + 0.20 * rr,
                                      -0.20, 0.20))
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] - roll_corr
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] - roll_corr
        else:
            u[AR["L"]] = DEFAULT_POSE[AR["L"]] + lean   # lean already signed
            u[AR["R"]] = DEFAULT_POSE[AR["R"]] + lean

        # ------ safety clamps ------
        for ci in (stn["hip"], stn["knee"], stn["ankle"], stn["hip_roll"]):
            u[ci] = float(np.clip(u[ci], DEFAULT_POSE[ci] - cfg.stance_cap,
                                  DEFAULT_POSE[ci] + cfg.stance_cap))
        u[sw["hip_roll"]] = float(np.clip(u[sw["hip_roll"]],
                                          DEFAULT_POSE[sw["hip_roll"]] - cfg.swing_hiproll_cap,
                                          DEFAULT_POSE[sw["hip_roll"]] + cfg.swing_hiproll_cap))
        u = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        s_ = (np.abs(u - m.actuator_ctrlrange[:15, 1]) < 1e-3) | \
             (np.abs(u - m.actuator_ctrlrange[:15, 0]) < 1e-3)
        d.ctrl[:15] = u
        mujoco.mj_step(m, d)
        k += 1
        if phase != "stand":
            sk += 1
        bs = sample_balance(m, d)
        sw_nf = _foot_normal_force(m, d, swing)
        f = _foot_xy_z(m, d, swing)
        fwd_mm = -(f[1] - swing_y0) * 1000.0
        clr_mm = (f[2] - swing_z0) * 1000.0
        peak_fwd = max(peak_fwd, fwd_mm)
        if phase in ("swing", "descend"):
            peak_clear = max(peak_clear, clr_mm)
        sat_any |= int(s_[[sw["hip"], sw["knee"], sw["ankle"], stn["ankle"]]].any())

        # ------ transitions ------
        if phase == "stand" and k >= t0:
            phase = "shift"; sk = 0
            if verbose:
                print(f"  >> SHIFT @ {k}")
        elif phase == "shift" and sk >= cfg.shift_ms:
            phase = "swing"; sk = 0
            if verbose:
                print(f"  >> SWING @ {k}  (swing_nf {sw_nf:.1f}, side {bs.side_lean_deg:+.1f})")
        elif phase == "swing" and sk >= cfg.swing_ms:
            phase = "descend"; sk = 0
            if verbose:
                print(f"  >> DESCEND @ {k}  fwd {fwd_mm:.0f}mm  clr {clr_mm:.0f}mm  sole {np.degrees(sp):+.1f}")
        elif phase == "descend":
            if not foot_lifted and sw_nf < 2.0:
                foot_lifted = True
            genuine = (foot_lifted and getattr(bs, f"{swing.lower()}_contact")
                       and sw_nf > cfg.plant_nf)
            plant_streak = plant_streak + 1 if genuine else 0
            if plant_streak >= cfg.plant_hold:
                contact_ms = k
                plant_q = np.array([d.qpos[7 + sw["hip"]], d.qpos[7 + sw["knee"]],
                                    d.qpos[7 + sw["ankle"]]])
                phase = "transfer"; sk = 0
                if verbose:
                    print(f"  >> CONTACT @ {k}  fwd {fwd_mm:.0f}mm  sole {np.degrees(sp):+.1f}  nf {sw_nf:.0f}")
            elif sk >= cfg.descend_ms + 500:
                plant_q = np.array([d.qpos[7 + sw["hip"]], d.qpos[7 + sw["knee"]],
                                    d.qpos[7 + sw["ankle"]]])
                phase = "transfer"; sk = 0
                if verbose:
                    print(f"  >> (no clean contact) transfer @ {k}  nf {sw_nf:.0f}")
        elif phase == "transfer" and sk >= cfg.transfer_ms:
            settle_qref = d.qpos.copy()
            from biped_env import STANDING_QUAT
            settle_qref[3:7] = STANDING_QUAT
            settle_cref = np.clip(d.ctrl[:15].copy(),
                                  m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
            settle_cref[AR["L"]] = DEFAULT_POSE[AR["L"]]
            settle_cref[AR["R"]] = DEFAULT_POSE[AR["R"]]
            phase = "settle"; sk = 0
            sf = _foot_xy_z(m, d, swing)[1]; stf = _foot_xy_z(m, d, "L" if swing == "R" else "R")[1]
            if verbose:
                print(f"  >> SETTLE @ {k}  foot fwd {fwd_mm:.0f}mm  sep {-(sf-stf)*1000:.0f}mm  "
                      f"nf {sw_nf:.0f}  sole {np.degrees(sp):+.1f}")
        elif phase == "settle" and sk >= cfg.settle_ms:
            break

        if bs.up_tilt_deg > 45:
            if verbose:
                print(f"  !! FELL @ {k}  up {bs.up_tilt_deg:.0f}  fwd {bs.fwd_lean_deg:+.0f}  side {bs.side_lean_deg:+.0f}")
            break

        if trace and k % 15 == 0:
            rows.append((k, phase, bs.up_tilt_deg, bs.fwd_lean_deg, bs.side_lean_deg,
                         fwd_mm, clr_mm, np.degrees(sp), sw_nf, st_nf, bs.com_vfwd, int(s_.any())))
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(0.02 if slow else 0.002)

    bs = sample_balance(m, d)
    sf = _foot_xy_z(m, d, swing)[1]; stf = _foot_xy_z(m, d, "L" if swing == "R" else "R")[1]
    final_sep = -(sf - stf) * 1000.0
    sp = _sole_pitch(m, d, swing)
    fell = bs.up_tilt_deg > 45
    ds = bs.l_contact and bs.r_contact
    ok = (not fell and contact_ms is not None and ds
          and bs.up_tilt_deg < 9 and abs(bs.side_lean_deg) < 9
          and bs.com_speed_horiz < 0.06 and _foot_normal_force(m, d, swing) > 15
          and abs(np.degrees(sp)) < 12)

    if trace and verbose:
        print(f"\n  {'t':>5} {'phase':>8} {'up':>5} {'fwd°':>6} {'side°':>6} "
              f"{'fFwd':>5} {'clr':>4} {'sole°':>6} {'swNF':>5} {'stNF':>5} {'vF':>5} sat")
        for r in rows:
            print(f"  {r[0]:5d} {r[1]:>8} {r[2]:5.1f} {r[3]:+6.1f} {r[4]:+6.1f} "
                  f"{r[5]:5.0f} {r[6]:4.0f} {r[7]:+6.1f} {r[8]:5.0f} {r[9]:5.0f} {r[10]:+5.2f} {r[11]}")
    if verbose:
        print(f"\n  peak fwd {peak_fwd:.0f} mm  peak clearance {peak_clear:.0f} mm  contact @ {contact_ms}")
        print(f"  end: up {bs.up_tilt_deg:.1f}  side {bs.side_lean_deg:+.1f}  fwd_lean {bs.fwd_lean_deg:+.1f}  "
              f"sole {np.degrees(sp):+.1f}°  sep {final_sep:+.0f} mm  swing_nf {_foot_normal_force(m,d,swing):.0f}  "
              f"speed {bs.com_speed_horiz:.3f}")
        print(f"  >>> {'CLEAN STEP' if ok else ('FELL' if fell else 'not clean')}")

    if viewer is not None:
        try:
            while viewer.is_running():
                viewer.sync(); time.sleep(0.02 if slow else 0.002)
        except KeyboardInterrupt:
            pass
        viewer.close()
    return dict(ok=ok, fell=fell, contact_ms=contact_ms, peak_fwd=peak_fwd,
               peak_clear=peak_clear, final_sep=final_sep, end_up=bs.up_tilt_deg,
               end_side=bs.side_lean_deg, end_sole_deg=np.degrees(sp),
               end_speed=bs.com_speed_horiz)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--slow", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--swing", choices=["L", "R"], default="R")
    p.add_argument("--hip", type=float, default=0.42)
    p.add_argument("--knee", type=float, default=0.30)
    p.add_argument("--sweep", action="store_true")
    a = p.parse_args(argv)

    if a.sweep:
        for hip in (0.30, 0.42, 0.52):
            for knee in (0.25, 0.35, 0.45):
                r = run(StepCfg(swing=a.swing, swing_hip_rad=hip, swing_knee_rad=knee),
                        verbose=False, trace=False)
                print(f"  hip {hip:.2f} knee {knee:.2f}: "
                      f"{'CLEAN' if r['ok'] else ('FELL' if r['fell'] else 'meh ')}  "
                      f"peakFwd {r['peak_fwd']:3.0f}  peakClr {r['peak_clear']:3.0f}  "
                      f"contact@{str(r['contact_ms']):>5}  sep {r['final_sep']:+4.0f}  "
                      f"endUp {r['end_up']:4.1f}  sole {r['end_sole_deg']:+5.1f}")
        return

    run(StepCfg(swing=a.swing, swing_hip_rad=a.hip, swing_knee_rad=a.knee),
        show=a.show or a.slow, slow=a.slow)


if __name__ == "__main__":
    main(sys.argv[1:])
