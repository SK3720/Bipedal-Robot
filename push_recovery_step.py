"""Forward push -> recovery STEP -> stable double support -> LQR recovery.

STATUS (5th pass): from REST the single-support hold + forward leg swing works
(single_support_step_demo.py).  Under a live forward push it FAILS, and a
step-by-step trace (push_recovery_trace.py) shows exactly why:

  * The standing LQR holds the push ~0.5 s (capture pinned ~+18 mm) then loses;
    the trigger fires late (capture already +47 mm, CoM racing).
  * The ankle-roll weight shift - the whole basis of reaching single support -
    OSCILLATES the load L<->R under the forward-pitch dynamics of a push.  It
    only settles one foot free from a symmetric standing pose.
  * Switching to K_ss (a fixed-point single-support regulator) when the real
    state is off-nominal (leaning, moving, weight on the wrong foot) makes K_ss
    slam the nominal-stance leg to load an airborne foot -> that leg flails up
    and out (the "curling" seen in the viewer).  The scripted swing does nothing
    because its foot is still loaded.
  * Also tried, all failing: no pre-shift + fast direct swing under K_stand
    (topples sideways in ~0.4 s); swing L vs R; trigger 12-47 mm; widening the
    hips to 90-130 mm in-memory (breaks the SS-LQR tuning, worse).

Established across passes 2-5: the hip can swing an UNLOADED leg fine (so torque
is not the isolated limit); single-support lateral margin is ~zero (40 mm,
asymmetric stance); fixed-point LQRs are wrong the instant the state has
velocity, which it always does mid-recovery.

Conclusion: this needs a controller that reacts online to the measured state -
RL (warm-started from the standing LQR; env push machinery already exists, obs
needs torso velocity + contacts) or a capture-point/DCM walking controller with
online foot placement and a transient-tolerant single-support controller.
Scripted trajectories gated by fixed-point LQRs are not enough.

Phase machine:  BALANCE -> PRE-SHIFT -> SWING -> PLANT -> SETTLE
  (see the transition + control blocks below for the details of each)

Does not modify robot.xml / biped_env / the golden experiments.
Run:  python push_recovery_step.py --push 150 --slow      # watch it
      python push_recovery_step.py --headless --sweep 120,135,150,170
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import mujoco
import mujoco.viewer
import numpy as np

from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS, STANDING_QUAT
from recovery_metrics import (
    CHEST_BODY, NOMINAL_CHEST_Z, _foot_normal_force, _foot_xy_z,
    classify_run, sample_balance,
)
from push_step_recovery_test import (
    AR_L_IDX, AR_R_IDX, AR_SIGN, FWD_HIP_SIGN, KNEE_FLEX_SIGN, LEG_IDX,
    _LQRAbout, _SingleSupportLQR, _pos_error, _smooth,
)

FORWARD_DIR_RAD = -np.pi / 2.0
PUSH_AT_STEP = 20
WINDOW_STEPS = 6000
SAMPLE_EVERY = 5
NORMAL_SLEEP_S = 0.003
SLOW_SLEEP_S = 0.02


@dataclass
class Cfg:
    swing: str = "R"
    ankle_roll_amp_rad: float = 0.12
    ankle_roll_ramp: int = 55
    ss_reach_steps: int = 340

    # pre-shift: start the ankle-roll weight shift once the LQR is CLEARLY losing
    # (well past what it recovers in place), so we don't disturb recoverable pushes.
    preshift_capture_mm: float = 30.0
    preshift_com_vfwd: float = 0.14
    preshift_hold: int = 8
    preshift_min_step: int = 30

    # swing trigger: the push is genuinely winning
    trig_capture_mm: float = 40.0
    trig_com_vfwd: float = 0.18
    trig_hold: int = 4

    # unload (fallback if the swing foot has not gone light by the time we trigger)
    unload_nf: float = 4.0
    unload_hold: int = 8
    unload_cap: int = 90

    # swing
    swing_hip_fwd_rad: float = 0.45
    swing_knee_peak_rad: float = 0.25
    swing_ankle_dorsi_rad: float = 0.20
    swing_steps: int = 130
    swing_hip_decouple: bool = True
    swing_free_base: bool = True

    # plant
    plant_knee_rad: float = 0.14
    plant_ankle_plantar_rad: float = 0.18
    stance_knee_bend_rad: float = 0.22
    ankle_roll_plant_frac: float = 0.35    # keep this fraction of the lean during plant
    plant_steps: int = 150
    plant_nf: float = 12.0
    plant_hold: int = 15
    plant_cap: int = 320

    # settle
    settle_ankle_roll_relax_steps: int = 250
    settle_min_steps: int = 2000


@dataclass
class Result:
    push_n: float
    triggered: bool = False
    unloaded: bool = False
    planted: bool = False
    trig_step: int | None = None
    swing_step: int | None = None
    plant_step: int | None = None
    swing_foot_fwd_mm: float = 0.0
    swing_foot_clr_mm: float = 0.0
    foot_sep_fwd_at_plant_mm: float = 0.0
    min_chest_z: float = 1.3
    ever_airborne: bool = False
    peak_up_tilt_after_plant: float = 0.0
    end_up_tilt: float = 0.0
    end_side_lean: float = 0.0
    end_ds: bool = False
    end_com_speed: float = 0.0
    recovered: bool = False
    outcome: str = ""
    samples: list = field(default_factory=list)


def _swing_targets(cfg: Cfg, sk: int):
    hf, ks = FWD_HIP_SIGN[cfg.swing], KNEE_FLEX_SIGN
    w = min(1.0, sk / cfg.swing_steps)
    bump = np.sin(np.pi * w)
    return (hf * cfg.swing_hip_fwd_rad * _smooth(w),
            ks * (0.06 + (cfg.swing_knee_peak_rad - 0.06) * bump),
            -hf * cfg.swing_ankle_dorsi_rad * bump)


def _plant_targets(cfg: Cfg, pk: int, hip_at_swing_end: float):
    hf, ks = FWD_HIP_SIGN[cfg.swing], KNEE_FLEX_SIGN
    w = _smooth(min(1.0, pk / cfg.plant_steps))
    hip = hip_at_swing_end                                  # hold the forward hip
    knee = ks * (cfg.swing_knee_peak_rad + (cfg.plant_knee_rad - cfg.swing_knee_peak_rad) * w)
    ankle = hf * cfg.plant_ankle_plantar_rad * w
    return hip, knee, ankle, w


def run(model, data, cfg: Cfg, push_n: float, viewer=None, slow=False, verbose=True) -> Result:
    stand = _LQRAbout(model, data, DEFAULT_POSE.copy(), tag="stand", verbose=verbose)

    class _C:  # adapt Cfg -> the attrs _SingleSupportLQR reads
        swing = cfg.swing
        ankle_roll_amp_rad = cfg.ankle_roll_amp_rad
        ankle_roll_ramp = cfg.ankle_roll_ramp
        ss_reach_steps = cfg.ss_reach_steps
    ss = _SingleSupportLQR(model, data, stand, _C, verbose=verbose)

    idx = LEG_IDX[cfg.swing]
    stance = "R" if cfg.swing == "L" else "L"
    sidx = LEG_IDX[stance]
    sw3 = (idx["hip"], idx["knee"], idx["ankle"])
    ar_full = AR_SIGN[cfg.swing] * cfg.ankle_roll_amp_rad
    nv = model.nv

    data.qpos[:] = stand.qpos0
    data.qvel[:] = stand.qvel0
    mujoco.mj_forward(model, data)
    fxy = push_n * np.array([np.cos(FORWARD_DIR_RAD), np.sin(FORWARD_DIR_RAD)])
    res = Result(push_n=push_n)

    phase = "balance"
    man_k = pre_streak = trig_streak = calm_streak = plant_streak = 0
    preshift_k0 = swing_k0 = plant_k0 = settle_k0 = None
    swing_foot_y0 = None
    swing_end_hip = 0.0
    settle_qref = stand.qpos0.copy()
    settle_cref = stand.ctrl0.copy()
    samples = []

    for k in range(WINDOW_STEPS):
        data.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT_STEP <= k < PUSH_AT_STEP + PUSH_DURATION_STEPS:
            data.xfrc_applied[CHEST_BODY, 0:2] = fxy
        bs = sample_balance(model, data)
        post_push = k > PUSH_AT_STEP + PUSH_DURATION_STEPS
        sw_nf = _foot_normal_force(model, data, cfg.swing)

        # ---------------- control ----------------
        if phase == "balance":
            u = stand.ctrl0 - stand.K @ np.concatenate(
                [_pos_error(model, stand.qpos0, data.qpos), data.qvel - stand.qvel0])

        elif phase == "preshift":
            a = ar_full * min(1.0, (k - preshift_k0) / cfg.ankle_roll_ramp)
            qref = stand.qpos0.copy(); cref = stand.ctrl0.copy()
            for ci in (AR_L_IDX, AR_R_IDX):
                qref[7 + ci] += a; cref[ci] += a
            u = cref - stand.K @ np.concatenate(
                [_pos_error(model, qref, data.qpos), data.qvel - stand.qvel0])
            u[AR_L_IDX] = cref[AR_L_IDX]; u[AR_R_IDX] = cref[AR_R_IDX]

        elif phase == "swing":
            sk = k - swing_k0
            hip, knee, ankle = _swing_targets(cfg, sk)
            swing_end_hip = hip
            qref = ss.qpos0.copy(); cref = ss.ctrl0.copy()
            qref[3:7] = STANDING_QUAT
            for ci, v in zip(sw3, (hip, knee, ankle)):
                qref[7 + ci] = v; cref[ci] = v
            dx = np.concatenate([_pos_error(model, qref, data.qpos), data.qvel - ss.qvel0])
            if cfg.swing_free_base:
                dx[0] = dx[1] = 0.0; dx[nv + 0] = dx[nv + 1] = 0.0
            if cfg.swing_hip_decouple:
                dx[6 + idx["hip"]] = 0.0; dx[nv + 6 + idx["hip"]] = 0.0
            u = cref - ss.K @ dx
            if cfg.swing_hip_decouple:
                u[idx["hip"]] = cref[idx["hip"]]
            u[AR_L_IDX] = cref[AR_L_IDX]; u[AR_R_IDX] = cref[AR_R_IDX]

        elif phase == "plant":
            # STAY on K_ss and KEEP the lean (ankle roll ~full). Only reach the
            # swing foot down: extend its knee, plantarflex its ankle, bend the
            # stance knee to lower the pelvis. Let the forward CoM momentum roll
            # the weight onto the new front foot. Hand-off happens after it loads.
            pk = k - plant_k0
            hip, knee, ankle, w = _plant_targets(cfg, pk, swing_end_hip)
            qref = ss.qpos0.copy(); cref = ss.ctrl0.copy()
            qref[3:7] = STANDING_QUAT
            for ci, v in zip(sw3, (hip, knee, ankle)):
                qref[7 + ci] = v; cref[ci] = v
            sb = KNEE_FLEX_SIGN * cfg.stance_knee_bend_rad * w
            qref[7 + sidx["knee"]] = ss.qpos0[7 + sidx["knee"]] + sb
            cref[sidx["knee"]] = ss.ctrl0[sidx["knee"]] + sb
            dx = np.concatenate([_pos_error(model, qref, data.qpos), data.qvel - ss.qvel0])
            dx[0] = dx[1] = 0.0; dx[nv + 0] = dx[nv + 1] = 0.0
            u = cref - ss.K @ dx
            for ci in sw3:
                u[ci] = cref[ci]
            u[AR_L_IDX] = cref[AR_L_IDX]; u[AR_R_IDX] = cref[AR_R_IDX]

        else:  # settle
            sk = k - settle_k0
            a = ar_full * max(0.0, 1.0 - sk / cfg.settle_ankle_roll_relax_steps) * cfg.ankle_roll_plant_frac
            cref = settle_cref.copy(); qref = settle_qref.copy()
            for ci in (AR_L_IDX, AR_R_IDX):
                qref[7 + ci] = a; cref[ci] = a
            dx = np.concatenate([_pos_error(model, qref, data.qpos), data.qvel])
            u = cref - stand.K @ dx
            u[AR_L_IDX] = cref[AR_L_IDX]; u[AR_R_IDX] = cref[AR_R_IDX]

        data.ctrl[:15] = np.clip(u, model.actuator_ctrlrange[:15, 0],
                                 model.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(model, data)
        man_k += 1

        # ---------------- bookkeeping ----------------
        res.min_chest_z = min(res.min_chest_z, bs.chest_z)
        if post_push and not bs.l_contact and not bs.r_contact:
            res.ever_airborne = True
        if swing_foot_y0 is not None:
            f = _foot_xy_z(model, data, cfg.swing)
            res.swing_foot_fwd_mm = max(res.swing_foot_fwd_mm, -(f[1] - swing_foot_y0) * 1000.0)
            res.swing_foot_clr_mm = max(res.swing_foot_clr_mm, (f[2] - 1.0) * 1000.0)
        if res.planted:
            res.peak_up_tilt_after_plant = max(res.peak_up_tilt_after_plant, bs.up_tilt_deg)

        # ---------------- transitions ----------------
        if phase == "balance" and post_push and k >= cfg.preshift_min_step:
            losing = (bs.capture_fwd_rel_support_mm > cfg.preshift_capture_mm
                      and bs.com_vfwd > cfg.preshift_com_vfwd)
            pre_streak = pre_streak + 1 if losing else 0
            if pre_streak >= cfg.preshift_hold:
                phase, preshift_k0 = "preshift", k
                if verbose:
                    print(f"  >> PRE-SHIFT k={k} capture={bs.capture_fwd_rel_support_mm:.0f}mm")

        elif phase == "preshift":
            div = bs.capture_fwd_rel_support_mm > cfg.trig_capture_mm and bs.com_vfwd > cfg.trig_com_vfwd
            trig_streak = trig_streak + 1 if div else 0
            calm_streak = calm_streak + 1 if bs.capture_fwd_rel_support_mm < cfg.preshift_capture_mm else 0
            ramped = (k - preshift_k0) >= cfg.ankle_roll_ramp
            if calm_streak >= 60:                       # LQR won - abandon the step
                phase = "balance"; pre_streak = trig_streak = 0
                if verbose:
                    print(f"  >> LQR held; back to BALANCE k={k}")
            elif trig_streak >= cfg.trig_hold and (sw_nf < cfg.unload_nf or ramped):
                phase = "swing"; swing_k0 = k
                res.triggered = res.unloaded = True
                res.trig_step = res.swing_step = k
                swing_foot_y0 = _foot_xy_z(model, data, cfg.swing)[1]
                if verbose:
                    print(f"  >> SWING k={k} capture={bs.capture_fwd_rel_support_mm:.0f}mm "
                          f"vfwd={bs.com_vfwd:.2f} swing_nf={sw_nf:.1f}")
            elif (k - preshift_k0) >= cfg.unload_cap and sw_nf < cfg.unload_nf * 2:
                phase = "swing"; swing_k0 = k
                res.triggered = res.unloaded = True
                res.trig_step = res.swing_step = k
                swing_foot_y0 = _foot_xy_z(model, data, cfg.swing)[1]
                if verbose:
                    print(f"  >> SWING (cap) k={k} swing_nf={sw_nf:.1f}")

        elif phase == "swing":
            if (k - swing_k0) >= cfg.swing_steps:
                phase = "plant"; plant_k0 = k
                if verbose:
                    print(f"  >> PLANT-PHASE k={k} swing_fwd={res.swing_foot_fwd_mm:.0f}mm clr={res.swing_foot_clr_mm:.0f}mm")

        elif phase == "plant":
            ok = getattr(bs, f"{cfg.swing.lower()}_contact") and sw_nf > cfg.plant_nf
            plant_streak = plant_streak + 1 if ok else 0
            done = plant_streak >= cfg.plant_hold or (k - plant_k0) >= cfg.plant_cap
            if done:
                if plant_streak >= cfg.plant_hold:
                    res.planted = True; res.plant_step = k
                    sf = _foot_xy_z(model, data, cfg.swing)[1]
                    tf = _foot_xy_z(model, data, stance)[1]
                    res.foot_sep_fwd_at_plant_mm = -(sf - tf) * 1000.0
                    if verbose:
                        print(f"  >> PLANTED k={k} foot_sep_fwd={res.foot_sep_fwd_at_plant_mm:.0f}mm")
                phase = "settle"; settle_k0 = k
                settle_qref = data.qpos.copy(); settle_qref[3:7] = STANDING_QUAT
                settle_cref = np.clip(data.ctrl[:15].copy(),
                                      model.actuator_ctrlrange[:15, 0],
                                      model.actuator_ctrlrange[:15, 1])

        if k % SAMPLE_EVERY == 0:
            samples.append((k, bs))
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

    res.samples = samples
    v = classify_run(samples)
    end = samples[-1][1]
    res.end_up_tilt = end.up_tilt_deg
    res.end_side_lean = end.side_lean_deg
    res.end_ds = end.l_contact and end.r_contact
    res.end_com_speed = end.com_speed_horiz
    res.recovered = (not v.fell and v.recovered and res.end_ds
                     and res.min_chest_z > NOMINAL_CHEST_Z - 0.10)
    if res.recovered and res.triggered:
        res.outcome = "RECOVERED via step -> double support"
    elif res.recovered:
        res.outcome = "recovered in place (no step needed)"
    elif v.fell:
        res.outcome = f"FELL ({v.fell_reason})"
    elif not res.triggered:
        res.outcome = "no step triggered"
    elif not res.planted:
        res.outcome = "swing foot never planted"
    else:
        res.outcome = f"planted but not settled ({v.label})"
    return res


def print_result(r: Result, timeline=False):
    print(f"\n[push {r.push_n:.0f} N]  trig={r.triggered}@{r.trig_step}  "
          f"unload={r.unloaded}  plant={r.planted}@{r.plant_step}")
    print(f"  swing foot: fwd {r.swing_foot_fwd_mm:.0f} mm, clearance {r.swing_foot_clr_mm:.0f} mm, "
          f"sep-fwd @ plant {r.foot_sep_fwd_at_plant_mm:.0f} mm")
    print(f"  min chest z {r.min_chest_z:.3f}  airborne {r.ever_airborne}  "
          f"peak tilt after plant {r.peak_up_tilt_after_plant:.1f} deg")
    print(f"  end: up_tilt {r.end_up_tilt:.1f}  side {r.end_side_lean:+.1f}  DS {r.end_ds}  "
          f"CoM speed {r.end_com_speed:.3f}")
    print(f"  => {r.outcome}")
    if timeline:
        print("   t(s) up_tilt fwd_lean side  capt-sup com-sup  CoMvx chestZ L/R Lnf Rnf")
        for step, s in r.samples:
            if step % 40:
                continue
            print(f"  {step/1000:5.2f} {s.up_tilt_deg:6.1f} {s.fwd_lean_deg:+7.1f} {s.side_lean_deg:+5.1f} "
                  f"{s.capture_fwd_rel_support_mm:+8.0f} {s.com_fwd_rel_support_mm:+7.0f} {s.com_vfwd:+5.2f} "
                  f"{s.chest_z:6.3f} {int(s.l_contact)}/{int(s.r_contact)} {s.l_nf:3.0f} {s.r_nf:3.0f}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--push", type=float, default=150.0)
    p.add_argument("--sweep", default=None)
    p.add_argument("--swing", choices=["L", "R"], default="R")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--timeline", action="store_true")
    a = p.parse_args(argv)
    model = mujoco.MjModel.from_xml_path("robot/robot.xml")
    data = mujoco.MjData(model)
    cfg = Cfg(swing=a.swing)

    if a.sweep:
        rs = []
        for pn in (float(x) for x in a.sweep.split(",")):
            rs.append(run(model, data, cfg, pn, verbose=True))
            print_result(rs[-1], timeline=a.timeline)
        print("\n" + "=" * 70)
        for r in rs:
            print(f"  {r.push_n:5.0f} N  sep@plant {r.foot_sep_fwd_at_plant_mm:5.0f}mm  "
                  f"endTilt {r.end_up_tilt:5.1f}  {r.outcome}")
        return

    if a.headless:
        print_result(run(model, data, cfg, a.push, verbose=True), timeline=True)
        return

    with mujoco.viewer.launch_passive(model, data) as vw:
        vw.cam.lookat[:] = [0.0, -0.1, 1.05]
        vw.cam.distance = 2.0
        vw.cam.azimuth = 90
        vw.cam.elevation = -8
        r = run(model, data, cfg, a.push, viewer=vw, slow=a.slow)
        print_result(r, timeline=True)
        print("\n  close viewer to exit")
        while vw.is_running():
            vw.sync(); time.sleep(SLOW_SLEEP_S if a.slow else NORMAL_SLEEP_S)


if __name__ == "__main__":
    main(sys.argv[1:])
