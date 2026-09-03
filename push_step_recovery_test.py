"""Forward push -> LQR -> recovery STEP -> double support -> LQR.  Integration test.

STATUS (2026-08-31, 3rd pass):

SOLVED - single-support balance (was declared impossible the prior pass):
  * single_support_unload_probe.py: a small both-ankle-roll bias toward the
    stance foot, held under the standing LQR, unloads the swing foot to ~0 N and
    holds a ~8 deg lean indefinitely.
  * single_support_lqr_probe.py + _SingleSupportLQR here: linearising an LQR
    about THAT leaned single-support state gives a K that holds single support
    AND lets the free leg swing forward ~80-150 mm without the both-feet-airborne
    hop that the stiff feet-together standing K causes.
  * single_support_step_demo.py: from standing (no push) the robot shifts weight,
    holds single support at ~8 deg, and swings the free leg 97 mm forward while
    staying balanced (final up_tilt 9 deg).  Watch it:  python single_support_step_demo.py --slow

NOT YET CLOSED - the full push -> step -> double-support -> recover chain:
  * From a live >=150 N forward push the CoM is already moving forward ~0.3-0.6
    m/s and the ~0.45 s pre-faceplant window is shorter than the unload+swing
    takes; triggered mid-fall, the swing still hops and the plant lands too late.
  * The best partial: at 175 N one run reached up_tilt ~3 deg / double support
    ~0.9 s after the plant, then a residual ~0.5 m/s forward CoM velocity
    re-toppled it forward.
  * Lowering the swung foot back to the ground to finish in stepped double
    support is unsolved on its own (weight-shift relax + K hand-off destabilise).

Next: a time-varying / capture-point controller for the swing (so it tolerates
the push momentum), plus active braking or a 2nd step for the residual velocity.

Does not modify robot.xml / biped_env / any golden experiment.
Run:  --selftest-step  (step from standing) ;  --push N  (full push case) ;  --sweep a,b,c
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np

from biped_env import CHEST_Z_CONTACT, DEFAULT_POSE, PUSH_DURATION_STEPS, SETTLE_STEPS, STANDING_QUAT
from recovery_metrics import (
    CHEST_BODY, NOMINAL_CHEST_Z, _foot_normal_force, _foot_xy_z,
    classify_run, sample_balance,
)
from standing_balance_lqr import StandingLQR, _dare

FORWARD_DIR_RAD = -np.pi / 2.0
PUSH_AT_STEP = 20
WINDOW_STEPS = 7000
SAMPLE_EVERY = 5
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.02

LEG_IDX = {
    "L": dict(hip_roll=5, hip=6, knee=7, ankle=8, qhip=13, qknee=14, qankle=15, qhiproll=12),
    "R": dict(hip_roll=10, hip=11, knee=12, ankle=13, qhip=18, qknee=19, qankle=20, qhiproll=17),
}
FWD_HIP_SIGN = {"L": -1.0, "R": +1.0}   # hip-pitch sign that swings the foot forward (-Y)
KNEE_FLEX_SIGN = -1.0                    # both knees: negative == flex/lift


def _smooth(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * t))


@dataclass
class StepConfig:
    swing: str = "R"
    trig_capture_mm: float = 28.0
    trig_hold: int = 20
    trig_com_vfwd: float = 0.08
    trig_min_step: int = 60          # let the LQR attempt in-place recovery first
    trig_deadline_steps: int = 800

    # --- ankle-roll weight-shift onto the stance foot (unloads the swing foot) ---
    # Verified (_dynamic_unload_probe, _ss_swing_probe): a small both-ankle-roll
    # bias toward the stance side, held under the standing LQR, unloads the swing
    # foot to ~0 N at ~11 deg lean and holds that quasi-single-support state.
    ankle_roll_amp_rad: float = 0.12   # magnitude; sign set by swing side
    ankle_roll_ramp: int = 60
    unload_steps: int = 260            # hard cap on the unload wait
    ss_reach_steps: int = 340          # steps to settle the leaned SS state (offline)

    # --- swing (hip drives the leg forward; LQR NOT decoupled - the cross terms
    #     actually help here) ---
    step_hip_fwd_rad: float = 0.45     # forward hip angle (held from swing through plant)
    swing_knee_peak_rad: float = 0.22  # peak knee flex mid-swing (sin bump) - low = less lift
    swing_ankle_dorsi_rad: float = 0.18
    swing_steps: int = 150             # slower swing = less ballistic whip
    plant_steps: int = 110             # knee straightens, foot settles ahead
    ankle_roll_relax_steps: int = 200  # over plant+settle, bring weight back to centre

    swing_unload_nf: float = 3.0      # swing foot considered free below this
    swing_unload_hold: int = 12
    plant_detect_nf: float = 10.0
    plant_detect_hold: int = 15

    # --- post-plant brake (arrest residual forward CoM velocity) ---
    lead_brake_ankle_rad: float = 0.35   # lead ankle plantarflex (push CoM back)
    lead_brake_knee_rad: float = 0.30    # lead knee bend (absorb)
    trail_settle_hip_rad: float = 0.10   # trailing hip comes slightly forward

    settle_min_steps: int = 2500


@dataclass
class StepResult:
    push_n: float
    swing: str
    triggered: bool = False
    trigger_step: int | None = None
    planted: bool = False
    plant_step: int | None = None
    maneuver_done_step: int | None = None
    swing_foot_fwd_travel_mm: float = 0.0
    swing_foot_peak_clear_mm: float = 0.0
    foot_sep_fwd_at_plant_mm: float = 0.0
    min_chest_z: float = 1.30
    ever_both_airborne: bool = False
    end_double_support: bool = False
    end_up_tilt_deg: float = 0.0
    end_com_speed: float = 0.0
    outcome: str = ""
    recovered_strict: bool = False
    samples: list = field(default_factory=list)


# --------------------------------------------------------------------------

def _settle_pose(model, data, pose):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = pose
    data.qvel[:] = 0.0
    data.ctrl[:15] = pose
    mujoco.mj_forward(model, data)
    for _ in range(SETTLE_STEPS):
        data.ctrl[:15] = pose
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


class _LQRAbout:
    """LQR linearised about an arbitrary settled pose."""
    def __init__(self, model, data, pose, tag="", verbose=True):
        self.model = model
        self.nv = model.nv
        self.ctrl0 = pose.copy()
        self.qpos0, self.qvel0 = _settle_pose(model, data, pose)
        data.qpos[:] = self.qpos0
        data.qvel[:] = self.qvel0
        data.ctrl[:15] = self.ctrl0
        mujoco.mj_forward(model, data)
        n2 = 2 * self.nv
        A = np.zeros((n2, n2)); B = np.zeros((n2, 15))
        mujoco.mjd_transitionFD(model, data, 1e-6, 1, A, B, None, None)
        q_pos = np.ones(self.nv) * 2.0
        q_pos[0:2] = 3.0; q_pos[2] = 40.0; q_pos[3:6] = 400.0; q_pos[6:21] = 1.0
        q_vel = np.ones(self.nv) * 1.0
        q_vel[0:3] = 6.0; q_vel[3:6] = 25.0; q_vel[6:21] = 0.4
        Q = np.diag(np.concatenate([q_pos, q_vel]))
        R = np.diag(np.ones(15) * 3.0)
        self.K, _, it = _dare(A, B, Q, R)
        if verbose:
            eig = np.linalg.eigvals(A - B @ self.K)
            print(f"[LQR {tag}] DARE it={it} rho={np.max(np.abs(eig)):.4f}  "
                  f"foot_sep_fwd={self._foot_sep(model, data):.0f}mm")

    def _foot_sep(self, model, data):
        return -(data.xpos[model.body('R_foot').id][1] - data.xpos[model.body('L_foot').id][1]) * 1000.0


def _lqr_gain(model, data, q0, v0, c0, ori_w=400.0):
    """DARE gain for the system linearised about (q0, v0, c0)."""
    data.qpos[:] = q0; data.qvel[:] = v0; data.ctrl[:15] = c0
    mujoco.mj_forward(model, data)
    nv = model.nv
    A = np.zeros((2 * nv, 2 * nv)); B = np.zeros((2 * nv, 15))
    mujoco.mjd_transitionFD(model, data, 1e-6, 1, A, B, None, None)
    qp = np.ones(nv) * 2.0
    qp[0:2] = 3.0; qp[2] = 40.0; qp[3:6] = ori_w; qp[6:21] = 1.0
    qv = np.ones(nv) * 1.0
    qv[0:3] = 6.0; qv[3:6] = 22.0; qv[6:21] = 0.4
    Q = np.diag(np.concatenate([qp, qv]))
    R = np.diag(np.ones(15) * 3.0)
    K, _, _ = _dare(A, B, Q, R)
    return K


# NOTE: commanding the swing hip-roll toward neutral (~-8 deg) during the reach
# gives a much more upright single-support state (~2 deg tilt vs ~8 deg) - but
# that state's K_ss then destabilises during the leg swing, worse than the leaned
# state. Left leaned. See the 3rd-pass notes at the top of the file.
class _SingleSupportLQR:
    """LQR about the LEANED single-support state reached by stand.K + an
    ankle-roll bias toward the stance foot. Verified (single_support_lqr_probe):
    this K holds single support AND lets the free leg swing forward ~80-150 mm
    without the both-feet-airborne hop that the stiff feet-together standing K
    causes.
    """
    def __init__(self, model, data, stand, cfg, verbose=True):
        ar = AR_SIGN[cfg.swing] * cfg.ankle_roll_amp_rad
        data.qpos[:] = stand.qpos0; data.qvel[:] = stand.qvel0
        mujoco.mj_forward(model, data)
        for k in range(cfg.ss_reach_steps):
            a = ar * min(1.0, k / cfg.ankle_roll_ramp)
            qr = stand.qpos0.copy(); cr = stand.ctrl0.copy()
            for ci in (AR_L_IDX, AR_R_IDX):
                qr[7 + ci] += a; cr[ci] += a
            u = cr - stand.K @ np.concatenate(
                [_pos_error(model, qr, data.qpos), data.qvel - stand.qvel0])
            u[AR_L_IDX] = cr[AR_L_IDX]; u[AR_R_IDX] = cr[AR_R_IDX]
            data.ctrl[:15] = np.clip(u, model.actuator_ctrlrange[:15, 0],
                                     model.actuator_ctrlrange[:15, 1])
            mujoco.mj_step(model, data)
        self.qpos0 = data.qpos.copy()
        self.qvel0 = np.zeros(model.nv)
        self.ctrl0 = np.clip(data.ctrl[:15].copy(), model.actuator_ctrlrange[:15, 0],
                             model.actuator_ctrlrange[:15, 1])
        self.K = _lqr_gain(model, data, self.qpos0, self.qvel0, self.ctrl0, ori_w=300.0)
        if verbose:
            b = sample_balance(model, data)
            print(f"[SS-LQR {cfg.swing}] leaned state up_tilt={b.up_tilt_deg:.1f} "
                  f"swing_nf={_foot_normal_force(model, data, cfg.swing):.1f}N")


def _pos_error(model, qpos_ref, qpos):
    dq = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, dq, 1.0, qpos_ref, qpos)
    return dq


def _seg(x, a, b):
    """progress within [a,b], smoothed, clamped."""
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return _smooth((x - a) / (b - a))


AR_SIGN = {"R": -1.0, "L": +1.0}   # both-ankle-roll bias that unloads that foot
AR_L_IDX, AR_R_IDX = 9, 14         # ankle-roll ctrl indices


def _unload_ref(cfg: StepConfig, stand, man_k: int):
    """Unload phase reference: ramp both ankle rolls toward the stance side."""
    q = stand.qpos0.copy()
    c = stand.ctrl0.copy()
    ar = AR_SIGN[cfg.swing] * cfg.ankle_roll_amp_rad * min(1.0, man_k / cfg.ankle_roll_ramp)
    for ci in (AR_L_IDX, AR_R_IDX):
        q[7 + ci] += ar
        c[ci] += ar
    return q, c


def _swing_leg_targets(cfg: StepConfig, sk: int):
    """Swing-leg (hip, knee, ankle) targets for local step `sk` of the swing."""
    hf, ks = FWD_HIP_SIGN[cfg.swing], KNEE_FLEX_SIGN
    w_sw = min(1.0, sk / cfg.swing_steps)
    w_pl = _seg(sk, cfg.swing_steps, cfg.swing_steps + cfg.plant_steps)
    bump = np.sin(np.pi * w_sw)
    hip = hf * cfg.step_hip_fwd_rad * _smooth(w_sw)
    knee = ks * (0.06 + (cfg.swing_knee_peak_rad - 0.06) * bump * (1.0 - w_pl))
    ankle = -hf * cfg.swing_ankle_dorsi_rad * bump * (1.0 - w_pl) + hf * 0.04 * w_pl
    return hip, knee, ankle


def run_push_step(model, data, cfg: StepConfig, push_n: float,
                  viewer=None, slow=False, verbose=True,
                  force_trigger_at: int | None = None) -> StepResult:
    stand = _LQRAbout(model, data, DEFAULT_POSE.copy(), tag="stand", verbose=verbose)
    ss = _SingleSupportLQR(model, data, stand, cfg, verbose=verbose)
    idx = LEG_IDX[cfg.swing]
    ss_idx = (idx["hip"], idx["knee"], idx["ankle"])

    data.qpos[:] = stand.qpos0
    data.qvel[:] = stand.qvel0
    mujoco.mj_forward(model, data)

    fxy = push_n * np.array([np.cos(FORWARD_DIR_RAD), np.sin(FORWARD_DIR_RAD)])
    res = StepResult(push_n=push_n, swing=cfg.swing)

    phase = "balance"
    trig_streak = 0
    plant_streak = 0
    unload_streak = 0
    man_k = 0
    settle_k = 0
    swing_start_k = None
    settle_qref = stand.qpos0.copy()
    settle_cref = stand.ctrl0.copy()
    swing_start_foot_y = None
    samples = []

    for k in range(WINDOW_STEPS):
        data.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT_STEP <= k < PUSH_AT_STEP + PUSH_DURATION_STEPS:
            data.xfrc_applied[CHEST_BODY, 0:2] = fxy

        bs = sample_balance(model, data)
        post_push = k > PUSH_AT_STEP + PUSH_DURATION_STEPS

        if phase == "balance":
            u = stand.ctrl0 - stand.K @ np.concatenate(
                [_pos_error(model, stand.qpos0, data.qpos), data.qvel - stand.qvel0])
        elif phase == "maneuver":
            if swing_start_k is None:
                # UNLOAD: standing LQR drives the ankle-roll weight shift.
                qref, cref = _unload_ref(cfg, stand, man_k)
                dx = np.concatenate(
                    [_pos_error(model, qref, data.qpos), data.qvel - stand.qvel0])
                u = cref - stand.K @ dx
                u[AR_L_IDX] = cref[AR_L_IDX]
                u[AR_R_IDX] = cref[AR_R_IDX]
            else:
                # SWING + PLANT: single-support LQR (knows the leaned dynamics);
                # swing leg decoupled from ss.K and driven by feed-forward.
                sk_l = man_k - swing_start_k
                hip, knee, ankle = _swing_leg_targets(cfg, sk_l)
                qref = ss.qpos0.copy()
                cref = ss.ctrl0.copy()
                for ci, val in zip(ss_idx, (hip, knee, ankle)):
                    qref[7 + ci] = val
                    cref[ci] = val
                dx = np.concatenate(
                    [_pos_error(model, qref, data.qpos), data.qvel - ss.qvel0])
                for ci in ss_idx:                       # decouple swing leg
                    dx[6 + ci] = 0.0
                    dx[model.nv + 6 + ci] = 0.0
                # a recovery step SHOULD let the CoM keep translating forward -
                # don't let ss.K (a stationary-point regulator) fight the base
                # x/y position + velocity error the push created.
                dx[0] = dx[1] = 0.0
                dx[model.nv + 0] = dx[model.nv + 1] = 0.0
                u = cref - ss.K @ dx
                for ci in ss_idx:
                    u[ci] = cref[ci]
            man_k += 1
            if swing_start_foot_y is not None:
                foot = _foot_xy_z(model, data, cfg.swing)
                res.swing_foot_fwd_travel_mm = max(
                    res.swing_foot_fwd_travel_mm, -(foot[1] - swing_start_foot_y) * 1000.0)
                res.swing_foot_peak_clear_mm = max(
                    res.swing_foot_peak_clear_mm, (foot[2] - 1.0) * 1000.0)
        else:  # settle: hold the ACTUAL planted stance, torso forced upright, stand.K
            #   (a stepped-stance LQR launches here - the real plant is far from
            #    any pre-linearised stepped pose. Freezing the achieved joint
            #    config as the reference and letting stand.K damp + right the
            #    torso is what actually settles it.)
            dx = np.concatenate([_pos_error(model, settle_qref, data.qpos), data.qvel])
            u = settle_cref - stand.K @ dx
            u[AR_L_IDX] = settle_cref[AR_L_IDX]
            u[AR_R_IDX] = settle_cref[AR_R_IDX]
            settle_k += 1

        data.ctrl[:15] = np.clip(u, model.actuator_ctrlrange[:15, 0],
                                 model.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(model, data)

        res.min_chest_z = min(res.min_chest_z, bs.chest_z)
        if post_push and not bs.l_contact and not bs.r_contact:
            res.ever_both_airborne = True

        if phase == "balance" and force_trigger_at is not None and k >= force_trigger_at:
            phase, man_k = "maneuver", 0
            res.triggered = True
            res.trigger_step = k
            swing_start_foot_y = None
            if verbose:
                print(f"  >> FORCED TRIGGER step {k} (selftest)")
        elif phase == "balance" and post_push and k >= cfg.trig_min_step:
            diverging = (bs.capture_fwd_rel_support_mm > cfg.trig_capture_mm
                         and bs.com_vfwd > cfg.trig_com_vfwd)
            trig_streak = trig_streak + 1 if diverging else 0
            if trig_streak >= cfg.trig_hold:
                phase, man_k = "maneuver", 0
                res.triggered = True
                res.trigger_step = k
                swing_start_foot_y = None
                if verbose:
                    print(f"  >> TRIGGER step {k} capture={bs.capture_fwd_rel_support_mm:.0f}mm "
                          f"com_vfwd={bs.com_vfwd:.2f} lean={bs.fwd_lean_deg:.1f} swing={cfg.swing}")
            elif k - (PUSH_AT_STEP + PUSH_DURATION_STEPS) > cfg.trig_deadline_steps:
                phase = "settle_nostep"
        elif phase == "maneuver":
            nf = _foot_normal_force(model, data, cfg.swing)
            swing_contact = getattr(bs, f"{cfg.swing.lower()}_contact")

            # state-triggered swing start: begin the moment the swing foot is
            # genuinely unloaded (or a hard fallback wait), not a fixed delay.
            if swing_start_k is None:
                unload_streak = unload_streak + 1 if nf < cfg.swing_unload_nf else 0
                if unload_streak >= cfg.swing_unload_hold or man_k >= cfg.unload_steps:
                    swing_start_k = man_k
                    swing_start_foot_y = _foot_xy_z(model, data, cfg.swing)[1]
                    if verbose:
                        print(f"  >> SWING START man_k={man_k} (swing_nf={nf:.1f})")

            sk_local = (man_k - swing_start_k) if swing_start_k is not None else -1
            swing_done = sk_local >= cfg.swing_steps + cfg.plant_steps // 2
            if swing_done:
                plant_streak = plant_streak + 1 if (swing_contact and nf > cfg.plant_detect_nf) else 0
            man_cap = (swing_start_k or cfg.unload_steps) + cfg.swing_steps + cfg.plant_steps + 200
            enter_settle = (plant_streak >= cfg.plant_detect_hold and not res.planted) or man_k >= man_cap
            if enter_settle:
                if plant_streak >= cfg.plant_detect_hold:
                    _mark_plant(res, model, data, cfg, k)
                res.maneuver_done_step = k
                # freeze the achieved joint config as the settle reference, force
                # the torso reference upright, relax the ankle-roll bias, and add
                # a BRAKE bias on the lead leg: plantarflex its ankle + bend its
                # knee so it decelerates the still-forward CoM (the plant alone
                # leaves ~0.5 m/s of residual forward CoM velocity).
                idx = LEG_IDX[cfg.swing]
                stance = "R" if cfg.swing == "L" else "L"
                sidx = LEG_IDX[stance]
                settle_qref = data.qpos.copy()
                settle_qref[3:7] = STANDING_QUAT
                settle_cref = np.clip(data.ctrl[:15].copy(),
                                      model.actuator_ctrlrange[:15, 0],
                                      model.actuator_ctrlrange[:15, 1])
                for ci in (AR_L_IDX, AR_R_IDX):
                    settle_cref[ci] = 0.0
                    settle_qref[7 + ci] = 0.0
                # lead-leg brake
                settle_cref[idx["ankle"]] = FWD_HIP_SIGN[cfg.swing] * cfg.lead_brake_ankle_rad
                settle_qref[7 + idx["ankle"]] = FWD_HIP_SIGN[cfg.swing] * cfg.lead_brake_ankle_rad
                settle_cref[idx["knee"]] = KNEE_FLEX_SIGN * cfg.lead_brake_knee_rad
                settle_qref[7 + idx["knee"]] = KNEE_FLEX_SIGN * cfg.lead_brake_knee_rad
                # trailing (stance) leg: let it come forward under the body
                settle_cref[sidx["hip"]] = FWD_HIP_SIGN[stance] * cfg.trail_settle_hip_rad
                settle_qref[7 + sidx["hip"]] = FWD_HIP_SIGN[stance] * cfg.trail_settle_hip_rad
                phase = "settle"

        if k % SAMPLE_EVERY == 0:
            samples.append((k, bs))

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

    res.samples = samples
    verdict = classify_run(samples)
    end = samples[-1][1]
    res.end_double_support = end.l_contact and end.r_contact
    res.end_up_tilt_deg = end.up_tilt_deg
    res.end_com_speed = end.com_speed_horiz
    res.recovered_strict = (
        not verdict.fell and verdict.recovered and res.end_double_support
        and res.min_chest_z > NOMINAL_CHEST_Z - 0.10
    )
    if res.recovered_strict and res.triggered:
        res.outcome = "RECOVERED via step (double support, upright, still)"
    elif res.recovered_strict:
        res.outcome = "RECOVERED in place (LQR, no step needed)"
    elif verdict.fell:
        res.outcome = f"FELL ({verdict.fell_reason})"
    elif not res.triggered:
        res.outcome = "no step triggered (LQR handled it or gave up)"
    elif not res.planted:
        res.outcome = "stepped but swing foot never planted cleanly"
    else:
        res.outcome = f"stepped+planted, not settled: {verdict.label}"
    return res


def _mark_plant(res, model, data, cfg, k):
    res.planted = True
    res.plant_step = k
    stance = "R" if cfg.swing == "L" else "L"
    sf = _foot_xy_z(model, data, cfg.swing)[1]
    stf = _foot_xy_z(model, data, stance)[1]
    res.foot_sep_fwd_at_plant_mm = -(sf - stf) * 1000.0


def print_result(r: StepResult, timeline=False):
    print(f"\n[{r.swing}-step]  push {r.push_n:.0f} N")
    print(f"  triggered={r.triggered} @ {r.trigger_step}   planted={r.planted} @ {r.plant_step}")
    print(f"  swing foot: fwd travel {r.swing_foot_fwd_travel_mm:.0f} mm, peak clearance "
          f"{r.swing_foot_peak_clear_mm:.0f} mm,  foot sep fwd @ plant {r.foot_sep_fwd_at_plant_mm:.0f} mm")
    print(f"  min chest z {r.min_chest_z:.3f}   both-feet-airborne {r.ever_both_airborne}   "
          f"end DS {r.end_double_support}  end up_tilt {r.end_up_tilt_deg:.1f} deg  end CoM speed {r.end_com_speed:.3f}")
    print(f"  => {r.outcome}")
    if timeline:
        print("    t(s)  up_tilt fwd_lean side_lean capt-supp com-supp CoMvx chestZ L/R  Lnf  Rnf")
        for step, s in r.samples:
            if step % 50 != 0:
                continue
            print(f"    {step/1000:5.2f} {s.up_tilt_deg:6.1f} {s.fwd_lean_deg:+7.1f} {s.side_lean_deg:+8.1f} "
                  f"{s.capture_fwd_rel_support_mm:+9.0f} {s.com_fwd_rel_support_mm:+8.0f} {s.com_vfwd:+5.2f} "
                  f"{s.chest_z:6.3f} {int(s.l_contact)}/{int(s.r_contact)} {s.l_nf:4.0f} {s.r_nf:4.0f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Push -> LQR -> recovery step -> double support -> LQR.")
    p.add_argument("--push", type=float, default=None)
    p.add_argument("--sweep", default=None)
    _d = StepConfig()
    p.add_argument("--swing", choices=["L", "R"], default="R")
    p.add_argument("--trig", type=float, default=_d.trig_capture_mm)
    p.add_argument("--step-hip", type=float, default=_d.step_hip_fwd_rad)
    p.add_argument("--unload", type=int, default=_d.unload_steps)
    p.add_argument("--ankle-roll", type=float, default=_d.ankle_roll_amp_rad)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--timeline", action="store_true")
    p.add_argument("--selftest-step", action="store_true",
                   help="no push: trigger the step maneuver from standing, check it reaches stable DS")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    model = mujoco.MjModel.from_xml_path("robot/robot.xml")
    data = mujoco.MjData(model)
    cfg = StepConfig(swing=args.swing, trig_capture_mm=args.trig,
                     step_hip_fwd_rad=args.step_hip, unload_steps=args.unload,
                     ankle_roll_amp_rad=args.ankle_roll)

    if args.selftest_step:
        r = run_push_step(model, data, cfg, 0.0, verbose=True, force_trigger_at=200)
        print_result(r, timeline=True)
        return

    if args.push is not None and not args.headless:
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.lookat[:] = [0.0, -0.10, 1.05]
            v.cam.distance = 2.0
            v.cam.azimuth = 90
            v.cam.elevation = -10
            r = run_push_step(model, data, cfg, args.push, viewer=v, slow=args.slow)
            print_result(r, timeline=True)
            print("\nClose viewer to exit.")
            while v.is_running():
                v.sync(); time.sleep(SLOW_SLEEP_S if args.slow else NORMAL_SLEEP_S)
        return

    sweep = [args.push] if args.push else [130, 150, 170, 190, 210]
    if args.sweep:
        sweep = [float(x) for x in args.sweep.split(",")]
    results = []
    for pn in sweep:
        r = run_push_step(model, data, cfg, pn, verbose=True)
        print_result(r, timeline=args.timeline)
        results.append(r)
    print("\n" + "=" * 82)
    print(f"SUMMARY  swing={args.swing} trig={args.trig}mm step_hip={args.step_hip} "
          f"unload={args.unload} ankle_roll={args.ankle_roll}")
    print(f"{'push':>6} {'trig':>6} {'plant':>6} {'swing_mm':>9} {'clr_mm':>7} {'sep_mm':>7} {'minZ':>6}  outcome")
    for r in results:
        print(f"{r.push_n:6.0f} {str(r.triggered):>6} {str(r.planted):>6} {r.swing_foot_fwd_travel_mm:9.0f} "
              f"{r.swing_foot_peak_clear_mm:7.0f} {r.foot_sep_fwd_at_plant_mm:7.0f} {r.min_chest_z:6.3f}  {r.outcome}")


if __name__ == "__main__":
    main(sys.argv[1:])
