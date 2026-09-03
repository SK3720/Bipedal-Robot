"""Sagittal push -> 1-2 fast forward steps -> stable double support.

Deliberately the SIMPLEST architecture that reaches the milestone (see
memory/recovery-step-next-plan.md).  One controller runs the whole time: the
feet-together StandingLQR.  A step is a brief scripted perturbation on top of it:

  * a short both-ankle-roll IMPULSE toward the stance foot (pulse, then release -
    NOT a static lean).  Measured (_frontal_lipm_probe.py): this unloads the
    swing foot in ~60 ms, moves the CoM only ~20-40 mm (the stance foot spans
    ~80 mm), and the StandingLQR recovers the small lateral disturbance itself.
  * a FAST forward swing of the freed leg (feed-forward, decoupled from the LQR).
  * the verified terminal-descent touchdown.
  * genuine sustained-load plant detection.
  * then hand straight back to StandingLQR; take a 2nd forward step if the
    capture point is still ahead of the toes.

No single-support LQR, no controller switching, no lateral foot placement, no
live-state mutation.  Stance-leg feedback is per-joint clamped (the leg-fold fix).

Refinements (2026-09-01, same reliability, cleaner look): the swing leg's
hip_roll is now capped (`swing_hiproll_cap`) - without that the balance LQR's
response to the small lateral impulse also swung the free, unloaded leg
sideways, so the step landed almost as far sideways as forward ("diagonal").
Capping it keeps the LQR's real lateral authority but stops it steering the
swing leg; forward/lateral is now consistently ~14/8 mm instead of ~10/10 mm,
still 100% reliable (11/11 across 120-140 N).

Final-pose note (2026-09-01): after the step the feet stay staggered ~40 mm
fore-aft and the robot settles at a STABLE but ~5 deg leaned pose (soles flat,
joints near nominal - a whole-body lean, the natural equilibrium of a staggered
stance).  Verified stable indefinitely (9 s).  An ankle-bias integrator pulls it
upright but then slowly destabilises (upright + staggered feet is not a stable
equilibrium here) - not used.  A square finish needs a step-together (below).

Tried and REJECTED - a deliberate "step-together" second step (bring the
trailing foot up once the robot is fully settled, so the final stance is
feet-together): measured 80-90% reliability at every amplitude tried (full
impulse, reduced impulse, no impulse/just a loaded nudge) - the trailing leg's
catch-up necessarily uses swing=L, which has less lateral margin, and even a
gentle correction there occasionally re-triggers the same slow eventual-fall
mode an unaided push produces.  Left in the code disabled (`catchup_enabled`)
for future work; not worth the reliability cost for a cosmetic finish today.
The single real step already lands close to level (7-46 mm) without it.

    python sagittal_recovery.py --slow
    python sagittal_recovery.py --headless --trace
    python sagittal_recovery.py --headless --sweep 135,140,145,150,160
Does not modify robot.xml / biped_env / any golden script.
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
    CHEST_BODY,
    NOMINAL_CHEST_Z,
    _foot_normal_force,
    _foot_xy_z,
    sample_balance,
)
from standing_balance_lqr import StandingLQR

FORWARD_DIR_RAD = -np.pi / 2.0
PUSH_AT = 20

# ctrl indices
LEG = {
    "L": dict(hip_roll=5, hip=6, knee=7, ankle=8),
    "R": dict(hip_roll=10, hip=11, knee=12, ankle=13),
}
AR = {"L": 9, "R": 14}
FWD_HIP_SIGN = {"L": -1.0, "R": +1.0}  # hip-pitch sign that swings the foot forward
KNEE_FLEX = -1.0
AR_UNLOAD_SIGN = {"R": -1.0, "L": +1.0}  # both-ankle-roll sign that unloads that foot


def _smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * t))


@dataclass
class Cfg:
    swing: str = "R"

    # trigger: capture point genuinely past the toes, moving forward
    trig_capt_mm: float = 6.0
    trig_vfwd: float = 0.05
    trig_hold: int = 12
    trig_min_ms: int = 40
    trig_deadline_ms: int = 1200

    # ankle-roll impulse (pulse then release)
    imp_amp: float = 0.22
    imp_ramp_ms: int = 8
    imp_hold_ms: int = 40

    swing_start_nf: float = 6.0
    swing_start_ms: int = 80
    swing_pr_gate: float = (
        0.2  # start the swing when pitch rate < this (torso recovering)
    )
    swing_latest_ms: int = 260  # ... or by this many ms into the step, whichever first

    # forward swing timed to the torso's backward recovery
    swing_ms: int = 150
    swing_hip_rad: float = 0.34
    swing_knee_peak: float = 0.06
    swing_ankle_dorsi: float = 0.08
    step_margin_mm: float = 30.0
    step_fwd_min_mm: float = 30.0
    step_fwd_max_mm: float = 90.0

    # terminal descent
    descend_ms: int = 55
    plant_hip_retract: float = 0.10
    plant_ankle_pf: float = 0.34
    stance_knee_bend: float = 0.18

    # genuine plant
    plant_nf: float = 14.0
    plant_hold: int = 12
    step_max_ms: int = 520

    # after a step
    settle_ms: int = 2400
    restep_capt_mm: float = 12.0
    restep_vfwd: float = 0.12
    restep_hold: int = 15
    restep_after_ms: int = 140

    max_steps: int = 2
    stance_joint_cap: float = 0.35  # max stance-leg feedback deviation (leg-fold fix)
    swing_hiproll_cap: float = (
        0.04  # max swing-leg hip_roll deviation (keeps the step straight)
    )

    # deliberate "step-together": once genuinely stable (not still falling),
    # bring the trailing foot up so the final stance is feet-together again.
    catchup_enabled: bool = False  # see docstring: measured net-harmful to
    # reliability (~80-90%, vs 100% without),
    # left here disabled for future work
    catchup_after_ms: int = 200  # earliest, within settle, to consider it
    catchup_stable_speed: float = 0.03
    catchup_stable_deg: float = 3.0
    catchup_hold: int = 400  # ms of continuous stability required
    catchup_min_sep_mm: float = 15.0  # only bother if still visibly staggered
    catchup_amp_scale: float = 0.45  # catch-up swing-hip amplitude scale
    catchup_imp_scale: float = 0.0  # catch-up ankle-roll impulse scale - the
    # trailing leg's catch-up uses swing=L,
    # which has less lateral margin (measured
    # separately); a full unload impulse there
    # occasionally reproduced the same slow
    # lateral-topple mode as an unaided push,
    # so the catch-up does NOT fully unload -
    # it nudges the trailing foot forward while
    # both feet stay loaded (never single support)


def _clip(m, u):
    return np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])


def _capture_target(bs, stance_foot_y, cfg):
    cp = bs.com_fwd + bs.com_vfwd * np.sqrt(max(bs.chest_z - 1.0, 0.05) / 9.81)
    stf = -stance_foot_y
    return float(
        np.clip(
            cp + cfg.step_margin_mm / 1000.0,
            stf + cfg.step_fwd_min_mm / 1000.0,
            stf + cfg.step_fwd_max_mm / 1000.0,
        )
    )


def run(
    push_n,
    cfg: Cfg,
    show=False,
    slow=False,
    trace=False,
    verbose=True,
    model_path="robot/robot.xml",
    push_dir=None,          # 2-vec world (x,y) unit dir; default = forward (-Y)
    model_fn=None,          # callback(model) to mutate params after load (validation)
    ctrl_delay_ms=0,        # apply the controller output this many ms late
    init_jitter_deg=0.0,    # +/- uniform jitter on the initial joint angles
    seed=0,
):
    m = mujoco.MjModel.from_xml_path(model_path)
    if model_fn is not None:
        model_fn(m)
    d = mujoco.MjData(m)
    stand = StandingLQR(m, d, verbose=verbose)
    rng = np.random.default_rng(seed)
    d.qpos[:] = stand.qpos0
    d.qvel[:] = stand.qvel0
    if init_jitter_deg > 0.0:
        j = np.radians(init_jitter_deg)
        d.qpos[7:22] += rng.uniform(-j, j, 15)
    mujoco.mj_forward(m, d)
    _ubuf = []  # control-latency FIFO

    viewer = None
    if show:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.cam.lookat[:] = [0.0, -0.35, 1.2]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 100
        viewer.cam.elevation = -6

    if push_dir is None:
        _pd = np.array([np.cos(FORWARD_DIR_RAD), np.sin(FORWARD_DIR_RAD)])
    else:
        _pd = np.asarray(push_dir, float)
        _pd = _pd / (np.linalg.norm(_pd) + 1e-12)
    fxy = push_n * _pd

    phase = "stand"
    swing = cfg.swing
    sk = 0  # ms into current step
    trig_streak = restep_streak = plant_streak = catchup_streak = 0
    is_catchup = False
    swing_started = False
    foot_lifted = False
    swing_t0 = 0
    foot_tgt = None
    swing_y0 = None
    swing_x0 = None
    steps = []
    com_fwd0 = sample_balance(m, d).com_fwd
    peak_lead_nf = 0.0
    lead_load_ms = 0
    tilt_spike = False
    peak_up = 0.0
    result = dict(push_n=push_n, triggered=False, fell=False, fell_t=None)

    T_END = PUSH_AT + PUSH_DURATION_STEPS + cfg.trig_deadline_ms + 5000
    k = 0
    while k < T_END:
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT <= k < PUSH_AT + PUSH_DURATION_STEPS:
            d.xfrc_applied[CHEST_BODY, 0:2] = fxy
        bs = sample_balance(m, d)
        post = k > PUSH_AT + PUSH_DURATION_STEPS
        sw = LEG[swing]
        stn = LEG["L" if swing == "R" else "R"]
        hf = FWD_HIP_SIGN[swing]
        sw_nf = _foot_normal_force(m, d, swing)

        # ---------------- control ----------------
        if phase in ("stand", "settle"):
            u = stand.control(m, d)
            for ci in (
                stn["hip"],
                stn["knee"],
                stn["ankle"],
                sw["hip"],
                sw["knee"],
                sw["ankle"],
            ):
                u[ci] = float(
                    np.clip(u[ci], stand.ctrl0[ci] - 0.7, stand.ctrl0[ci] + 0.7)
                )
            # NOTE (investigated 2026-09-01): after the step the feet are
            # staggered ~40 mm fore-aft and the robot settles at a stable but
            # ~5 deg leaned pose (soles stay flat, joints near nominal - it's a
            # whole-body lean, the natural equilibrium of the staggered stance).
            # A slow ankle-bias integrator DID pull it upright but then slowly
            # destabilised over the next few seconds (upright + staggered feet is
            # not a stable equilibrium for this controller) - reverted.  A
            # genuinely square finish needs a step-together, which is ~85%
            # reliable (see catchup_*), not worth the trade today.  Base
            # controller alone: stable indefinitely (verified to 9 s).
        else:  # step  --  IMPULSE + SHUFFLE + STANCE PUSH-OFF
            pr = -(d.xmat[CHEST_BODY].reshape(3, 3) @ d.qvel[3:6])[
                0
            ]  # + = pitching fwd
            amp_scale = cfg.catchup_amp_scale if is_catchup else 1.0
            imp_scale = cfg.catchup_imp_scale if is_catchup else 1.0
            imp_end = cfg.imp_ramp_ms + cfg.imp_hold_ms
            if sk < cfg.imp_ramp_ms:
                a = (
                    AR_UNLOAD_SIGN[swing]
                    * cfg.imp_amp
                    * imp_scale
                    * (sk / cfg.imp_ramp_ms)
                )
            elif sk < imp_end:
                a = AR_UNLOAD_SIGN[swing] * cfg.imp_amp * imp_scale
            else:
                a = 0.0

            # start the (low, dragging) forward shuffle once the foot is light or
            # a short time after the impulse.  The foot skims ~15-20 mm - it is
            # NOT a clean single-support swing (the swing hip is torque-limited
            # and can't lift+throw the leg while the body pitches).
            # unload first; then swing the leg forward only once the torso has
            # STOPPED pitching forward and is rotating back (pr < 0) - the hip
            # flexion then adds to the torso's own recovery instead of fighting
            # it, so the foot actually lands ahead in the world frame.
            foot_light = sw_nf < cfg.swing_start_nf or sk >= cfg.swing_start_ms
            if (
                not swing_started
                and foot_light
                and (pr < cfg.swing_pr_gate or sk >= cfg.swing_latest_ms)
            ):
                swing_started = True
                swing_t0 = sk
                stf_y = _foot_xy_z(m, d, "L" if swing == "R" else "R")[1]
                foot_tgt = _capture_target(bs, stf_y, cfg)
                swing_y0 = _foot_xy_z(m, d, swing)[1]
                swing_x0 = _foot_xy_z(m, d, swing)[0]
                if verbose:
                    print(
                        f"  >> SWING @ {k} ms  nf {sw_nf:.1f}  up {bs.up_tilt_deg:.1f}  pr {pr:+.2f}"
                    )

            # MINIMAL step: gentle forward swing of the freed leg, low + slow;
            # the StandingLQR keeps doing ALL of the sagittal balance (no brake,
            # no push-off, no hip extension - those were adding forward momentum).
            if not swing_started:
                hip = kn = an = 0.0
            else:
                s = sk - swing_t0
                sw_hip = cfg.swing_hip_rad * amp_scale
                if s < cfg.swing_ms:  # swing forward, low
                    w = s / cfg.swing_ms
                    hip = hf * sw_hip * _smooth(w)
                    kn = KNEE_FLEX * (0.04 + cfg.swing_knee_peak * np.sin(np.pi * w))
                    an = -hf * cfg.swing_ankle_dorsi * np.sin(np.pi * w)
                else:  # lower the foot, no push-off
                    w = min(1.0, (s - cfg.swing_ms) / cfg.descend_ms)
                    sm = _smooth(w)
                    hip = hf * (sw_hip - cfg.plant_hip_retract * sm)
                    kn = KNEE_FLEX * 0.04
                    an = -hf * cfg.swing_ankle_dorsi * (1 - sm) * 0.5  # ease to neutral

            qref = stand.qpos0.copy()
            cref = stand.ctrl0.copy()
            for ci, val in zip((sw["hip"], sw["knee"], sw["ankle"]), (hip, kn, an)):
                qref[7 + ci] = val
                cref[ci] = val
            for ci in (AR["L"], AR["R"]):
                qref[7 + ci] = a
                cref[ci] = a

            sw_joints = (sw["hip"], sw["knee"], sw["ankle"])
            dq = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dq, 1.0, qref, d.qpos)
            dx = np.concatenate([dq, d.qvel - stand.qvel0])
            for ci in sw_joints:
                dx[6 + ci] = 0.0
                dx[m.nv + 6 + ci] = 0.0
            dx[0] = dx[m.nv + 0] = 0.0  # don't chase lateral drift
            u = cref - stand.K @ dx
            for ci in sw_joints:
                u[ci] = cref[ci]
            u[AR["L"]] = cref[AR["L"]]
            u[AR["R"]] = cref[AR["R"]]
            # swing hip_roll: let the LQR keep using it for real balance, but cap
            # how far it can swing the free, unloaded leg sideways (that's what
            # made the step travel diagonally instead of straight forward).
            u[sw["hip_roll"]] = float(
                np.clip(
                    u[sw["hip_roll"]],
                    cref[sw["hip_roll"]] - cfg.swing_hiproll_cap,
                    cref[sw["hip_roll"]] + cfg.swing_hiproll_cap,
                )
            )
            # stance-leg per-joint clamp (leg-fold safety only)
            for ci in (stn["hip"], stn["knee"], stn["ankle"], stn["hip_roll"]):
                u[ci] = float(
                    np.clip(
                        u[ci],
                        cref[ci] - cfg.stance_joint_cap,
                        cref[ci] + cfg.stance_joint_cap,
                    )
                )

        u = _clip(m, u)
        if ctrl_delay_ms > 0:
            _ubuf.append(u)
            u = _ubuf.pop(0) if len(_ubuf) > ctrl_delay_ms else _ubuf[0]
        d.ctrl[:15] = u
        mujoco.mj_step(m, d)
        k += 1
        if phase != "stand":
            sk += 1
        bs = sample_balance(m, d)
        sw_nf = _foot_normal_force(m, d, swing)
        if phase in ("settle",):
            peak_lead_nf = max(peak_lead_nf, sw_nf)
            if sw_nf > cfg.plant_nf:
                lead_load_ms += 1
        if bs.up_tilt_deg > 45:
            tilt_spike = True
        peak_up = max(peak_up, bs.up_tilt_deg)

        if bs.up_tilt_deg > 55 or bs.chest_z < NOMINAL_CHEST_Z - 0.24:
            result.update(fell=True, fell_t=k)
            if verbose:
                print(
                    f"  FELL @ {k} ms  up {bs.up_tilt_deg:.0f}  fwd {bs.fwd_lean_deg:+.0f}  side {bs.side_lean_deg:+.0f}"
                )
            break

        # ---------------- transitions ----------------
        if phase == "stand":
            if post and k > PUSH_AT + PUSH_DURATION_STEPS + cfg.trig_min_ms:
                diverging = (
                    bs.capture_fwd_rel_support_mm > cfg.trig_capt_mm
                    and bs.com_vfwd > cfg.trig_vfwd
                )
                trig_streak = trig_streak + 1 if diverging else 0
                if trig_streak >= cfg.trig_hold:
                    phase = "step"
                    sk = 0
                    swing_started = False
                    foot_lifted = False
                    plant_streak = 0
                    result["triggered"] = True
                    if verbose:
                        print(
                            f"  >> STEP {len(steps)+1} @ {k} ms  swing={swing}  "
                            f"capt {bs.capture_fwd_rel_support_mm:+.0f}mm  vfwd {bs.com_vfwd:.2f}  "
                            f"lean {bs.fwd_lean_deg:+.1f}"
                        )
                elif k > PUSH_AT + PUSH_DURATION_STEPS + cfg.trig_deadline_ms:
                    phase = "settle"
                    sk = 0
                    if verbose:
                        print("  no step needed (LQR held)")

        elif phase == "step":
            if swing_started and not foot_lifted and sw_nf < 3.0:
                foot_lifted = True
            # only a plant AFTER the foot has genuinely left the ground counts
            genuine = (
                foot_lifted
                and getattr(bs, f"{swing.lower()}_contact")
                and sw_nf > cfg.plant_nf
            )
            plant_streak = plant_streak + 1 if genuine else 0
            done = plant_streak >= cfg.plant_hold
            if done or sk >= cfg.step_max_ms:
                planted = plant_streak >= cfg.plant_hold
                sfx, sf = _foot_xy_z(m, d, swing)[0], _foot_xy_z(m, d, swing)[1]
                stf = _foot_xy_z(m, d, "L" if swing == "R" else "R")[1]
                lat_mm = (sfx - swing_x0) * 1000.0 if swing_x0 is not None else 0.0
                steps.append(
                    dict(
                        k=k,
                        planted=planted,
                        sep_mm=-(sf - stf) * 1000.0,
                        lat_mm=lat_mm,
                        vfwd=bs.com_vfwd,
                        side=bs.side_lean_deg,
                    )
                )
                if verbose:
                    print(
                        f"  >> {'PLANT' if planted else 'TIMEOUT (no load)'} @ {k} ms  "
                        f"foot sep fwd {-(sf-stf)*1000:.0f} mm  nf {sw_nf:.0f}  "
                        f"residual vfwd {bs.com_vfwd:.2f}  side {bs.side_lean_deg:+.1f}"
                    )
                phase = "settle"
                sk = 0
                restep_streak = 0
                catchup_streak = 0

        elif phase == "settle":
            if sk >= cfg.settle_ms:
                break
            if len(steps) < cfg.max_steps and sk > cfg.restep_after_ms:
                need = (
                    bs.capture_fwd_rel_support_mm > cfg.restep_capt_mm
                    and bs.com_vfwd > cfg.restep_vfwd
                )
                restep_streak = restep_streak + 1 if need else 0
                if restep_streak >= cfg.restep_hold:
                    swing = "L" if swing == "R" else "R"
                    phase = "step"
                    sk = 0
                    swing_started = False
                    foot_lifted = False
                    plant_streak = 0
                    is_catchup = False
                    if verbose:
                        print(
                            f"  >> STEP {len(steps)+1} @ {k} ms  still diverging "
                            f"(capt {bs.capture_fwd_rel_support_mm:+.0f}mm) - swing {swing}"
                        )
                elif cfg.catchup_enabled and sk > cfg.catchup_after_ms:
                    # NOT still falling - a deliberate, low-risk "bring the
                    # trailing foot up" step so the robot returns to a normal
                    # feet-together stance instead of leaving them staggered.
                    # Only fires once the robot is genuinely settled (small
                    # speed/tilt held for a while), so it never competes with
                    # the emergency recovery above.
                    stable = (
                        bs.com_speed_horiz < cfg.catchup_stable_speed
                        and bs.up_tilt_deg < cfg.catchup_stable_deg
                        and abs(bs.side_lean_deg) < cfg.catchup_stable_deg
                    )
                    catchup_streak = catchup_streak + 1 if stable else 0
                    cur_sep = (
                        -(
                            _foot_xy_z(m, d, swing)[1]
                            - _foot_xy_z(m, d, "L" if swing == "R" else "R")[1]
                        )
                        * 1000.0
                    )
                    if (
                        catchup_streak >= cfg.catchup_hold
                        and abs(cur_sep) > cfg.catchup_min_sep_mm
                    ):
                        swing = "L" if swing == "R" else "R"
                        phase = "step"
                        sk = 0
                        swing_started = False
                        foot_lifted = False
                        plant_streak = 0
                        is_catchup = True
                        if verbose:
                            print(
                                f"  >> STEP {len(steps)+1} @ {k} ms  stable - bringing "
                                f"trailing foot up (sep {cur_sep:+.0f}mm) - swing {swing}"
                            )

        if trace and k % 20 == 0:
            print(
                f"   t{k:5d} [{phase:6s}] up{bs.up_tilt_deg:5.1f} fwd{bs.fwd_lean_deg:+6.1f} "
                f"side{bs.side_lean_deg:+6.1f} capt{bs.capture_fwd_rel_support_mm:+7.0f} "
                f"vF{bs.com_vfwd:+5.2f} Lnf{bs.l_nf:4.0f} Rnf{bs.r_nf:4.0f} "
                f"swZ{(_foot_xy_z(m,d,swing)[2]-1)*1000:+4.0f}"
            )

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(0.02 if slow else 0.0015)

    bs = sample_balance(m, d)
    ds = bs.l_contact and bs.r_contact
    com_moved = bs.com_fwd - com_fwd0
    n_planted = sum(1 for s in steps if s["planted"])
    ended_stable = (
        not result["fell"]
        and ds
        and bs.up_tilt_deg < 8
        and abs(bs.side_lean_deg) < 8
        and bs.com_speed_horiz < 0.09
        and bs.chest_z > NOMINAL_CHEST_Z - 0.10
    )
    physically_continuous = peak_up <= 35.0  # a stumble-and-catch, not a fall
    recovered = ended_stable and physically_continuous
    success = result["triggered"] and recovered and n_planted >= 1
    result.update(
        n_steps=len(steps),
        n_planted=n_planted,
        end_ds=ds,
        end_up=bs.up_tilt_deg,
        end_side=bs.side_lean_deg,
        end_speed=bs.com_speed_horiz,
        com_moved_mm=com_moved * 1000,
        peak_up=peak_up,
        peak_lead_nf=peak_lead_nf,
        lead_load_ms=lead_load_ms,
        tilt_spike=tilt_spike,
        steps=steps,
        recovered=recovered,
        success=success,
    )
    final_sep_mm = -(_foot_xy_z(m, d, "R")[1] - _foot_xy_z(m, d, "L")[1]) * 1000.0
    result["final_sep_mm"] = final_sep_mm
    if verbose:
        print(
            f"\n  push {push_n:.0f} N   steps={len(steps)} planted={n_planted}   "
            f"CoM moved fwd {com_moved*1000:+.0f} mm   final foot sep {final_sep_mm:+.0f} mm"
        )
        for i, s in enumerate(steps):
            print(
                f"    step {i+1}: {'PLANT' if s['planted'] else 'no-load'} @ {s['k']} ms  "
                f"sep {s['sep_mm']:+.0f} mm  vfwd {s['vfwd']:.2f}  side {s['side']:+.1f}"
            )
        print(f"  lead foot: peak {peak_lead_nf:.0f} N   loaded {lead_load_ms} ms")
        print(
            f"  end: up {bs.up_tilt_deg:.1f}  side {bs.side_lean_deg:+.1f}  DS {ds}  "
            f"speed {bs.com_speed_horiz:.3f}  spike {tilt_spike}"
        )
        print(
            f"  >>> {'SUCCESS' if success else ('FELL' if result['fell'] else 'FAILED - not a clean recovery')}"
        )
    if viewer is not None:
        try:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02 if slow else 0.0015)
        except KeyboardInterrupt:
            pass
        viewer.close()
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--push", type=float, default=135.0)
    p.add_argument("--swing", choices=["L", "R"], default="R")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--sweep", default=None)
    p.add_argument("--model", default="robot/robot.xml")
    a = p.parse_args(argv)
    cfg = Cfg(swing=a.swing)

    if a.sweep:
        rows = [
            run(float(x), cfg, verbose=True, model_path=a.model)
            for x in a.sweep.split(",")
        ]
        print("\n" + "=" * 66)
        for r in rows:
            print(
                f"  {r['push_n']:5.0f} N  trig={str(r['triggered']):>5}  "
                f"steps={r.get('n_steps',0)} planted={r.get('n_planted',0)}  "
                f"CoM{r.get('com_moved_mm',0):+.0f}mm  leadNF {r.get('peak_lead_nf',0):.0f}  "
                f"{'SUCCESS' if r['success'] else ('FELL' if r['fell'] else 'fail')}"
            )
        return
    run(
        a.push, cfg, show=not a.headless, slow=a.slow, trace=a.trace, model_path=a.model
    )


if __name__ == "__main__":
    main(sys.argv[1:])
