"""Post-touchdown balance ARREST — the real bottleneck after the golden catch step.

Established by inspection + headless runs of the existing experiments:
  * staged_forward_catch_test.py produces a good swing step (L foot ~+60 mm
    airborne, lands ~+70 mm ahead of R at global step 958).
  * BUT every continuation (staged_forward_catch_continue, staged_forward_r_recovery,
    falling_forward_step_test) ends with the torso pitching forward to
    tilt 1.4-1.7 rad — i.e. the robot face-plants. Nothing arrests the forward
    rotation after the foot lands.
  * The base golden summary's touchdown metrics are measured at global step ~590
    (a transient L-foot unload during RAPID SHIFT), NOT the real landing. This
    script uses _is_heel_touchdown() (catch-phase, post-airborne) which is correct
    and matches the 958 that the continue/r-recovery scripts report.

This experiment keeps the golden swing + catch lerp BYTE-IDENTICAL through heel
touchdown (imported locked prefix), then runs a parameterized ARREST controller
for 4 s and measures whether the forward pitch is actually stopped and the robot
settles.

Variants (cumulative):
  hold        - freeze the touchdown pose (== current behaviour / baseline)
  hip         - L stance hip extends to drive the torso back upright
  hipankle    - hip + L knee stiffens to a strut + L ankle plantarflexes
  brace       - hipankle + trailing R leg drops as a rear brace (2nd contact)
  full        - brace + arms thrown up/back for counter-rotation

Evaluation only - does not modify robot.xml, biped_env, or the golden scripts.
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

from biped_env import BipedalWalkEnv, DEFAULT_POSE

from flat_foot_touchdown_test import _run_locked_prefix
from staged_forward_catch_continue_test import _is_heel_touchdown
from staged_forward_catch_test import (
    CATCH_MAX_STEPS,
    FLOOR_Z,
    IDX_L_ANKLE_P,
    IDX_L_HIP_PITCH,
    IDX_L_KNEE,
    IDX_R_ANKLE_P,
    IDX_R_HIP_PITCH,
    IDX_R_KNEE,
    L_ANKLE_CATCH,
    L_HIP_CATCH,
    L_KNEE_CATCH,
    NORMAL_SLEEP_S,
    Phase,
    RunState,
    SLOW_SLEEP_S,
    StepDiagnostics,
    _foot_contact,
    _foot_normal_force,
    _foot_pos,
    _lerp_ctrl,
    _print_phase,
    _reset,
    _smooth,
    _sync_viewer,
    left_leg_pose,
    rapid_shift_pose,
)

IDX_L_SHOULDER = 1
IDX_R_SHOULDER = 3
QPOS_L_ANKLE = 15
QPOS_R_HIP = 18
QPOS_R_KNEE = 19

CHEST_BODY = 1
G = 9.81

ARREST_STEPS = 4000          # 4 s settle window
ARREST_RAMP_STEPS = 50       # ramp from touchdown pose to arrest targets
SETTLE_WINDOW_STEPS = 500    # final 0.5 s used for the "settled" test
FALL_TILT_RAD = 0.9          # == biped_env MAX_TILT_RAD
NOMINAL_CHEST_Z = 1.26

VARIANTS = ("hold", "hip", "hipankle", "brace", "full")


@dataclass
class ArrestConfig:
    variant: str = "hold"
    l_hip_arrest: float = 0.15
    l_knee_arrest: float = -0.02
    l_ankle_arrest: float = -0.35
    r_hip_brace: float = 0.10
    r_knee_brace: float = -0.55
    l_shoulder_arrest: float = -1.4
    r_shoulder_arrest: float = 1.4
    ramp_steps: int = ARREST_RAMP_STEPS


@dataclass
class Sample:
    step: int
    t_rel: float
    tilt: float
    tilt_rate: float
    chest_z: float
    com_fwd: float
    com_vfwd: float
    com_vlat: float
    angmom_x: float          # whole-body angular momentum about CoM, world X (pitch)
    l_fwd: float
    r_fwd: float
    com_minus_lfoot_fwd: float
    capture_minus_lfoot_fwd: float
    l_contact: bool
    r_contact: bool
    l_nf: float
    r_nf: float


@dataclass
class ArrestResult:
    label: str
    heel_step: int
    heel_tilt: float
    heel_com_vfwd: float
    heel_angmom_x: float
    heel_com_minus_lfoot_fwd: float
    heel_capture_minus_lfoot_fwd: float
    fell: bool = False
    fell_step: int | None = None
    peak_tilt: float = 0.0
    peak_tilt_step: int | None = None
    end_tilt: float = 0.0
    end_tilt_rate: float = 0.0
    end_com_speed: float = 0.0
    end_chest_z: float = 0.0
    end_l_contact: bool = False
    end_r_contact: bool = False
    settled_mean_tilt: float = 0.0
    recovered: bool = False
    verdict: str = ""
    timeline: list[Sample] = field(default_factory=list)


def _whole_body_com(data):
    return data.subtree_com[CHEST_BODY].copy()


def _sample(env, model, data, step, t_rel, prev_tilt, dt):
    mujoco.mj_subtreeVel(model, data)
    com = _whole_body_com(data)
    com_v = data.subtree_linvel[CHEST_BODY].copy()
    angmom = data.subtree_angmom[CHEST_BODY].copy()
    l = _foot_pos(model, data, "L")
    r = _foot_pos(model, data, "R")
    tilt = env._quat_tilt_rad()
    tilt_rate = (tilt - prev_tilt) / dt if prev_tilt is not None else 0.0

    com_fwd = -float(com[1])
    com_vfwd = -float(com_v[1])
    com_vlat = float(com_v[0])
    l_fwd = -float(l[1])
    r_fwd = -float(r[1])
    h = max(float(com[2]) - FLOOR_Z, 0.05)
    capture_fwd = com_fwd + com_vfwd * np.sqrt(h / G)

    return Sample(
        step=step,
        t_rel=t_rel,
        tilt=tilt,
        tilt_rate=tilt_rate,
        chest_z=float(data.qpos[2]),
        com_fwd=com_fwd,
        com_vfwd=com_vfwd,
        com_vlat=com_vlat,
        angmom_x=float(angmom[0]),
        l_fwd=l_fwd,
        r_fwd=r_fwd,
        com_minus_lfoot_fwd=(com_fwd - l_fwd) * 1000.0,
        capture_minus_lfoot_fwd=(capture_fwd - l_fwd) * 1000.0,
        l_contact=_foot_contact(model, data, "L"),
        r_contact=_foot_contact(model, data, "R"),
        l_nf=_foot_normal_force(model, data, "L"),
        r_nf=_foot_normal_force(model, data, "R"),
    )


def _run_to_heel(env, viewer, slow):
    """Golden locked prefix + catch lerp, stop exactly at heel touchdown."""
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)
    shifted = rapid_shift_pose()
    st = RunState(
        ctrl=DEFAULT_POSE.copy(),
        phase=Phase.STAND,
        stand_r_xy=_foot_pos(model, data, "R")[:2].copy(),
        diag=StepDiagnostics(),
        knee_cmd=float(shifted[IDX_L_KNEE]),
        hip_cmd=float(shifted[IDX_L_HIP_PITCH]),
    )
    _run_locked_prefix(env, model, data, cr, st, viewer, slow, verbose=False)

    catch_start = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l = _foot_contact(model, data, "L")
    was_air = not prev_l
    heel_ctrl = st.ctrl.copy()
    heel_step = None

    while st.catch_steps < CATCH_MAX_STEPS:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        st.ctrl = _lerp_ctrl(catch_start, catch_target, alpha, cr)
        data.ctrl[:15] = st.ctrl
        mujoco.mj_step(model, data)
        st.diag.global_step += 1
        st.catch_steps += 1
        if not _sync_viewer(viewer, slow):
            break
        if _is_heel_touchdown(
            model, data,
            catch_started=True, was_airborne_in_catch=was_air, prev_l_contact=prev_l,
        ):
            heel_ctrl = st.ctrl.copy()
            heel_step = st.diag.global_step
            break
        prev_l = _foot_contact(model, data, "L")
        if not prev_l:
            was_air = True

    if heel_step is None:
        heel_ctrl = st.ctrl.copy()
        heel_step = st.diag.global_step
    return st, heel_ctrl, heel_step


def _arrest_targets(cfg: ArrestConfig, heel_ctrl: np.ndarray) -> np.ndarray:
    tgt = heel_ctrl.copy()
    v = cfg.variant
    if v in ("hip", "hipankle", "brace", "full"):
        tgt[IDX_L_HIP_PITCH] = cfg.l_hip_arrest
    if v in ("hipankle", "brace", "full"):
        tgt[IDX_L_KNEE] = cfg.l_knee_arrest
        tgt[IDX_L_ANKLE_P] = cfg.l_ankle_arrest
    if v in ("brace", "full"):
        tgt[IDX_R_HIP_PITCH] = cfg.r_hip_brace
        tgt[IDX_R_KNEE] = cfg.r_knee_brace
    if v == "full":
        tgt[IDX_L_SHOULDER] = cfg.l_shoulder_arrest
        tgt[IDX_R_SHOULDER] = cfg.r_shoulder_arrest
    return tgt


def run_arrest(env: BipedalWalkEnv, cfg: ArrestConfig,
               viewer: mujoco.viewer.Handle | None = None,
               slow: bool = False, verbose: bool = True) -> ArrestResult:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    dt = model.opt.timestep

    st, heel_ctrl, heel_step = _run_to_heel(env, viewer, slow)
    if verbose:
        _print_phase(f"ARREST [{cfg.variant}]  (heel touchdown at step {heel_step})")

    heel_sample = _sample(env, model, data, heel_step, 0.0, None, dt)
    res = ArrestResult(
        label=cfg.variant,
        heel_step=heel_step,
        heel_tilt=heel_sample.tilt,
        heel_com_vfwd=heel_sample.com_vfwd,
        heel_angmom_x=heel_sample.angmom_x,
        heel_com_minus_lfoot_fwd=heel_sample.com_minus_lfoot_fwd,
        heel_capture_minus_lfoot_fwd=heel_sample.capture_minus_lfoot_fwd,
    )

    arrest_start = heel_ctrl.copy()
    arrest_tgt = _arrest_targets(cfg, heel_ctrl)

    prev_tilt = heel_sample.tilt
    for k in range(ARREST_STEPS):
        alpha = _smooth(min(k + 1, cfg.ramp_steps) / cfg.ramp_steps)
        ctrl = _lerp_ctrl(arrest_start, arrest_tgt, alpha, cr)
        data.ctrl[:15] = ctrl
        mujoco.mj_step(model, data)
        st.diag.global_step += 1

        s = _sample(env, model, data, st.diag.global_step, (k + 1) * dt, prev_tilt, dt)
        prev_tilt = s.tilt
        if k % 20 == 0 or k < 20:
            res.timeline.append(s)

        if s.tilt > res.peak_tilt:
            res.peak_tilt = s.tilt
            res.peak_tilt_step = s.step
        if not res.fell and (s.tilt > FALL_TILT_RAD or s.chest_z < NOMINAL_CHEST_Z - 0.15):
            res.fell = True
            res.fell_step = s.step

        if not _sync_viewer(viewer, slow):
            break

    # settled test on final window
    tail = [s for s in res.timeline if s.step >= st.diag.global_step - SETTLE_WINDOW_STEPS]
    if not tail:
        tail = res.timeline[-5:]
    end = res.timeline[-1]
    res.end_tilt = end.tilt
    res.end_tilt_rate = float(np.mean([abs(s.tilt_rate) for s in tail]))
    res.end_com_speed = float(np.hypot(end.com_vfwd, end.com_vlat))
    res.end_chest_z = end.chest_z
    res.end_l_contact = end.l_contact
    res.end_r_contact = end.r_contact
    res.settled_mean_tilt = float(np.mean([s.tilt for s in tail]))

    res.recovered = (
        not res.fell
        and res.settled_mean_tilt < 0.35
        and res.end_tilt < 0.40
        and res.end_tilt_rate < 0.6
        and res.end_com_speed < 0.20
        and res.end_l_contact
        and res.end_chest_z > NOMINAL_CHEST_Z - 0.12
    )
    if res.fell:
        res.verdict = f"FELL at step {res.fell_step} (tilt>{FALL_TILT_RAD})"
    elif res.recovered:
        res.verdict = "RECOVERED - forward pitch arrested, settled upright"
    else:
        res.verdict = "NO FALL but not settled (drifting / leaning / bouncing)"
    return res


def print_result(res: ArrestResult) -> None:
    print("\n" + "=" * 74)
    print(f"ARREST VARIANT: {res.label}")
    print("=" * 74)
    print(f"heel touchdown step         = {res.heel_step}")
    print(f"tilt at touchdown           = {res.heel_tilt:.3f} rad")
    print(f"CoM fwd vel at touchdown    = {res.heel_com_vfwd:+.3f} m/s")
    print(f"ang.mom (pitch) at touchdown= {res.heel_angmom_x:+.4f} kg m^2/s")
    print(f"CoM - Lfoot (fwd) at TD     = {res.heel_com_minus_lfoot_fwd:+.1f} mm")
    print(f"capture pt - Lfoot at TD    = {res.heel_capture_minus_lfoot_fwd:+.1f} mm  (>0 => still toppling fwd)")
    print("-" * 74)
    print(f"peak tilt (post-TD)         = {res.peak_tilt:.3f} rad @ step {res.peak_tilt_step}")
    print(f"end tilt                    = {res.end_tilt:.3f} rad")
    print(f"end |tilt rate| (0.5s mean) = {res.end_tilt_rate:.3f} rad/s")
    print(f"end CoM speed               = {res.end_com_speed:.3f} m/s")
    print(f"end chest z                 = {res.end_chest_z:.3f} m  (nominal {NOMINAL_CHEST_Z})")
    print(f"end contacts L / R          = {res.end_l_contact} / {res.end_r_contact}")
    print(f"settled-window mean tilt    = {res.settled_mean_tilt:.3f} rad")
    print(f"\nVERDICT: {res.verdict}")


def print_timeline(res: ArrestResult) -> None:
    print("\n  t(s)  tilt  d_tilt  chestZ  CoM-Lf  capt-Lf  CoMvx  Lc Rc  L_nf  R_nf")
    for s in res.timeline:
        if s.step % 100 != 0 and s.t_rel > 0.05:
            continue
        print(
            f"  {s.t_rel:4.2f} {s.tilt:5.2f} {s.tilt_rate:+6.2f} {s.chest_z:6.3f} "
            f"{s.com_minus_lfoot_fwd:+7.1f} {s.capture_minus_lfoot_fwd:+7.1f} "
            f"{s.com_vfwd:+5.2f} {int(s.l_contact)}  {int(s.r_contact)}  "
            f"{s.l_nf:5.1f} {s.r_nf:5.1f}"
        )


def print_comparison(results: list[ArrestResult]) -> None:
    print("\n" + "=" * 90)
    print("ARREST VARIANT COMPARISON  (golden swing locked; heel TD identical across all)")
    print("=" * 90)
    h = results[0]
    print(f"Disturbance at touchdown: tilt {h.heel_tilt:.3f} rad, "
          f"CoM v_fwd {h.heel_com_vfwd:+.3f} m/s, "
          f"pitch ang.mom {h.heel_angmom_x:+.4f}, "
          f"capture pt {h.heel_capture_minus_lfoot_fwd:+.0f} mm past L foot")
    print("-" * 90)
    print(f"{'variant':<10} {'fell':<6} {'peak tilt':<10} {'end tilt':<9} "
          f"{'end rate':<9} {'end vCoM':<9} {'settled':<8} verdict")
    for r in results:
        print(f"{r.label:<10} {str(r.fell):<6} {r.peak_tilt:<10.3f} {r.end_tilt:<9.3f} "
              f"{r.end_tilt_rate:<9.3f} {r.end_com_speed:<9.3f} "
              f"{str(r.recovered):<8} {r.verdict}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-touchdown balance arrest experiment.")
    p.add_argument("--variant", choices=VARIANTS, default=None,
                   help="run a single variant (with viewer unless --headless)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--timeline", action="store_true", help="print per-variant timeline")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()
    print("Post-touchdown ARREST experiment")
    print("Golden swing + catch locked (staged_forward_catch_test) through heel touchdown.\n")

    if args.headless:
        variants = [args.variant] if args.variant else list(VARIANTS)
        results = []
        for v in variants:
            res = run_arrest(env, ArrestConfig(variant=v), viewer=None,
                             slow=False, verbose=True)
            print_result(res)
            if args.timeline:
                print_timeline(res)
            results.append(res)
        if len(results) > 1:
            print_comparison(results)
        return

    variant = args.variant or "brace"
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.7
        v.cam.azimuth = 88
        v.cam.elevation = -12
        _reset(env.model, env.data)
        v.sync()
        res = run_arrest(env, ArrestConfig(variant=variant), viewer=v,
                         slow=args.slow, verbose=True)
        print_result(res)
        if args.timeline:
            print_timeline(res)
        print("\nClose viewer window to exit.")
        while v.is_running():
            v.sync()
            time.sleep(SLOW_SLEEP_S if args.slow else NORMAL_SLEEP_S)


if __name__ == "__main__":
    main(sys.argv[1:])
