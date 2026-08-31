"""Staged forward catch — continuation past left heel touchdown.

Runs the LOCKED trajectory from staged_forward_catch_test.py through the
successful catch-phase heel strike, then continues simulating to observe
weight transfer onto the planted left foot.

Does NOT modify the pre-touchdown motion.

Evaluation only — does not modify robot.xml, biped_env, or other scripts.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np
from ctypes import wintypes

from biped_env import BipedalWalkEnv, DEFAULT_POSE

import staged_forward_catch_test as base

from staged_forward_catch_test import (
    CATCH_MAX_STEPS,
    IDX_L_ANKLE_P,
    IDX_L_HIP_PITCH,
    IDX_L_KNEE,
    L_ANKLE_CATCH,
    L_HIP_CATCH,
    L_KNEE_CATCH,
    Phase,
    RunState,
    StepDiagnostics,
    _foot_contact,
    _foot_normal_force,
    _foot_pos,
    _forward_mm,
    _forward_vel,
    _is_unloaded,
    _lerp_ctrl,
    _print_phase,
    _reset,
    _sim_step,
    _smooth,
    _sync_viewer,
    forward_lean_pose,
    left_leg_pose,
    rapid_shift_pose,
)

POST_HEEL_MAX_STEPS = 600
PLANT_FORCE_N = 15.0
PLANT_SUSTAIN_STEPS = 40
RECOVERY_ANKLE_DELTA = 0.03
RECOVERY_MAX_STEPS = 120

VIEWER_WIDTH = base.VIEWER_WIDTH
VIEWER_HEIGHT = base.VIEWER_HEIGHT
VIEWER_TITLE_PREFIX = base.VIEWER_TITLE_PREFIX
VIEWER_POSITION_TIMEOUT_S = base.VIEWER_POSITION_TIMEOUT_S


class PostPhase(str):
    POST_HEEL = "POST HEEL - NATURAL SETTLE"
    OBSERVE = "OBSERVE WEIGHT TRANSFER"
    MINI_RECOVERY = "MINI RECOVERY (planted only)"


@dataclass
class PostSample:
    step: int
    phase: str
    l_contact: bool
    l_normal_n: float
    r_normal_n: float
    r_load_frac: float
    l_foot_xyz: np.ndarray
    r_foot_xyz: np.ndarray
    l_rel_r_fwd_mm: float
    torso_tilt_rad: float
    torso_angvel_y: float
    forward_vel_m_s: float


@dataclass
class ContinueDiagnostics:
    base: StepDiagnostics
    heel_touchdown_step: int | None = None
    heel_touchdown_l_rel_r_mm: float | None = None
    heel_touchdown_ctrl: np.ndarray | None = None
    heel_touchdown_l_nf: float | None = None
    peak_l_normal_post_heel: float = 0.0
    peak_r_normal_post_heel: float = 0.0
    min_r_load_frac_post_heel: float = 1.0
    max_r_load_frac_post_heel: float = 0.0
    l_foot_lost_contact_post_heel: bool = False
    l_foot_lost_contact_step: int | None = None
    sustained_plant: bool = False
    plant_confirmed_step: int | None = None
    mini_recovery_applied: bool = False
    final_torso_tilt_rad: float = 0.0
    final_forward_vel_m_s: float = 0.0
    final_l_rel_r_mm: float = 0.0
    passed_over_l_foot: bool | None = None
    post_samples: list[PostSample] = field(default_factory=list)


def _r_load_fraction(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    total = l_nf + r_nf
    return (r_nf / total) if total > 1e-6 else float("nan")


def _torso_angvel_y(data: mujoco.MjData) -> float:
    return float(data.qvel[4])


def _is_heel_touchdown(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    catch_started: bool,
    was_airborne_in_catch: bool,
    prev_l_contact: bool,
) -> bool:
    """Heel strike = first foot contact during CATCH after being airborne there."""
    if not catch_started or not was_airborne_in_catch:
        return False
    l_contact = _foot_contact(model, data, "L")
    return l_contact and not prev_l_contact


def _record_post(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    post: ContinueDiagnostics,
    phase: str,
) -> None:
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    r_frac = _r_load_fraction(model, data)
    l_contact = _foot_contact(model, data, "L")

    post.peak_l_normal_post_heel = max(post.peak_l_normal_post_heel, l_nf)
    post.peak_r_normal_post_heel = max(post.peak_r_normal_post_heel, r_nf)
    if not np.isnan(r_frac):
        post.min_r_load_frac_post_heel = min(post.min_r_load_frac_post_heel, r_frac)
        post.max_r_load_frac_post_heel = max(post.max_r_load_frac_post_heel, r_frac)

    if post.heel_touchdown_step is not None and not l_contact:
        if not post.l_foot_lost_contact_post_heel:
            post.l_foot_lost_contact_post_heel = True
            post.l_foot_lost_contact_step = post.base.global_step

    if post.base.global_step % 25 == 0:
        post.post_samples.append(
            PostSample(
                step=post.base.global_step,
                phase=phase,
                l_contact=l_contact,
                l_normal_n=l_nf,
                r_normal_n=r_nf,
                r_load_frac=r_frac,
                l_foot_xyz=l_pos.copy(),
                r_foot_xyz=r_pos.copy(),
                l_rel_r_fwd_mm=_forward_mm(l_pos[1], r_pos[1]),
                torso_tilt_rad=env._quat_tilt_rad(),
                torso_angvel_y=_torso_angvel_y(data),
                forward_vel_m_s=_forward_vel(data),
            )
        )


def run_staged_forward_catch_continue(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> ContinueDiagnostics:
    """Locked pre-heel motion from staged_forward_catch_test, then continue."""
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    shifted = rapid_shift_pose()
    lean = forward_lean_pose()
    ctrl = DEFAULT_POSE.copy()
    stand_r_xy = _foot_pos(model, data, "R")[:2].copy()
    base_diag = StepDiagnostics()
    st = RunState(
        ctrl=ctrl,
        phase=Phase.STAND,
        stand_r_xy=stand_r_xy,
        diag=base_diag,
        knee_cmd=float(shifted[IDX_L_KNEE]),
        hip_cmd=float(shifted[IDX_L_HIP_PITCH]),
    )
    post = ContinueDiagnostics(base=base_diag)

    # --- LOCKED PREFIX (identical to staged_forward_catch_test.py) ---
    _print_phase(Phase.STAND.value)
    for s in range(base.STAND_STEPS):
        st.ctrl = _lerp_ctrl(st.ctrl, DEFAULT_POSE, _smooth((s + 1) / base.STAND_STEPS), cr)
        _sim_step(env, model, data, st, viewer, slow)
    st.stand_r_xy = _foot_pos(model, data, "R")[:2].copy()

    _print_phase(Phase.FORWARD_FALL.value)
    for s in range(base.FALL_RAMP_STEPS):
        alpha = (s + 1) / base.FALL_RAMP_STEPS
        st.ctrl = _lerp_ctrl(st.ctrl, lean, alpha, cr)
        _sim_step(env, model, data, st, viewer, slow)
    for _ in range(base.FALL_MOMENTUM_STEPS):
        st.ctrl = lean.copy()
        _sim_step(env, model, data, st, viewer, slow)

    _print_phase(Phase.RAPID_SHIFT.value)
    shift_start = st.ctrl.copy()
    for s in range(base.RAPID_SHIFT_STEPS):
        alpha = (s + 1) / base.RAPID_SHIFT_STEPS
        st.ctrl = _lerp_ctrl(shift_start, shifted, alpha, cr)
        st.knee_cmd = float(st.ctrl[IDX_L_KNEE])
        st.hip_cmd = float(st.ctrl[IDX_L_HIP_PITCH])
        _sim_step(env, model, data, st, viewer, slow)

    post_shift = 0
    while post_shift < base.POST_SHIFT_MAX_STEPS and st.phase == Phase.RAPID_SHIFT:
        st.ctrl = shifted.copy()
        st.knee_cmd = float(st.ctrl[IDX_L_KNEE])
        st.hip_cmd = float(st.ctrl[IDX_L_HIP_PITCH])
        _sim_step(env, model, data, st, viewer, slow)
        post_shift += 1

    _print_phase(Phase.KNEE_LIFT.value)
    st.phase = Phase.KNEE_LIFT
    while st.knee_lift_steps < base.KNEE_LIFT_MAX_STEPS and st.phase == Phase.KNEE_LIFT:
        if _is_unloaded(model, data):
            st.clearance_knee = st.knee_cmd
            st.diag.clearance_knee_cmd = st.knee_cmd
            st.phase = Phase.HIP_SWING
            st.diag.hip_swing_start_step = st.diag.global_step + 1
            break
        st.knee_cmd = max(st.knee_cmd - base.KNEE_LIFT_STEP, base.KNEE_LIFT_TARGET)
        st.ctrl = left_leg_pose(st.knee_cmd, st.hip_cmd)
        _sim_step(env, model, data, st, viewer, slow)
        st.knee_lift_steps += 1

    if st.phase == Phase.KNEE_LIFT:
        st.clearance_knee = st.knee_cmd
        st.diag.clearance_knee_cmd = st.knee_cmd
        st.phase = Phase.HIP_SWING
        st.diag.hip_swing_start_step = st.diag.global_step + 1

    _print_phase(Phase.HIP_SWING.value)
    while st.hip_swing_steps < base.HIP_SWING_MAX_STEPS and st.phase == Phase.HIP_SWING:
        if st.airborne_ref_y is None and not _foot_contact(model, data, "L"):
            st.airborne_ref_y = float(_foot_pos(model, data, "L")[1])

        knee_hold = st.clearance_knee if st.clearance_knee is not None else st.knee_cmd
        st.hip_cmd = max(st.hip_cmd - base.HIP_RAMP_PER_STEP, base.HIP_SWING_TARGET)
        st.ctrl = left_leg_pose(knee_hold, st.hip_cmd)
        _sim_step(env, model, data, st, viewer, slow)
        st.hip_swing_steps += 1

        fwd_air = 0.0
        if st.airborne_ref_y is not None:
            fwd_air = _forward_mm(_foot_pos(model, data, "L")[1], st.airborne_ref_y)

        hip_at_target = st.hip_cmd <= base.HIP_SWING_TARGET + 1e-6
        if fwd_air >= base.MIN_FWD_AIRBORNE_MM and not _foot_contact(model, data, "L"):
            st.phase = Phase.CATCH
            st.diag.catch_start_step = st.diag.global_step + 1
            break
        if hip_at_target and st.hip_swing_steps > 80 and fwd_air >= 15.0:
            st.phase = Phase.CATCH
            st.diag.catch_start_step = st.diag.global_step + 1
            break

    if st.phase == Phase.HIP_SWING:
        st.phase = Phase.CATCH
        st.diag.catch_start_step = st.diag.global_step + 1

    # --- LOCKED CATCH until heel touchdown (same ramp; no early STOP) ---
    _print_phase(Phase.CATCH.value)
    catch_start_ctrl = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l_contact = _foot_contact(model, data, "L")
    was_airborne_in_catch = not prev_l_contact
    heel_ctrl: np.ndarray | None = None

    while st.catch_steps < CATCH_MAX_STEPS and post.heel_touchdown_step is None:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        st.ctrl = _lerp_ctrl(catch_start_ctrl, catch_target, alpha, cr)
        _sim_step(env, model, data, st, viewer, slow)
        st.catch_steps += 1

        if _is_heel_touchdown(
            model, data,
            catch_started=True,
            was_airborne_in_catch=was_airborne_in_catch,
            prev_l_contact=prev_l_contact,
        ):
            l_pos = _foot_pos(model, data, "L")
            r_pos = _foot_pos(model, data, "R")
            post.heel_touchdown_step = st.diag.global_step
            post.heel_touchdown_l_rel_r_mm = _forward_mm(l_pos[1], r_pos[1])
            post.heel_touchdown_ctrl = st.ctrl.copy()
            post.heel_touchdown_l_nf = _foot_normal_force(model, data, "L")
            heel_ctrl = st.ctrl.copy()
            print(
                f"\n  >> HEEL TOUCHDOWN at step {post.heel_touchdown_step} "
                f"L-R fwd={post.heel_touchdown_l_rel_r_mm:.1f} mm "
                f"L_nf={post.heel_touchdown_l_nf:.1f} N"
            )
            break

        prev_l_contact = _foot_contact(model, data, "L")
        if not prev_l_contact:
            was_airborne_in_catch = True

    if heel_ctrl is None:
        heel_ctrl = st.ctrl.copy()
        post.heel_touchdown_step = st.diag.global_step
        l_pos = _foot_pos(model, data, "L")
        r_pos = _foot_pos(model, data, "R")
        post.heel_touchdown_l_rel_r_mm = _forward_mm(l_pos[1], r_pos[1])
        post.heel_touchdown_ctrl = heel_ctrl.copy()
        post.heel_touchdown_l_nf = _foot_normal_force(model, data, "L")
        print(
            f"\n  >> No distinct heel transition detected; continuing from step "
            f"{post.heel_touchdown_step} (L contact={_foot_contact(model, data, 'L')})"
        )

    # --- POST HEEL: hold touchdown configuration, let dynamics continue ---
    _print_phase(PostPhase.POST_HEEL)
    st.phase = Phase.STOP
    plant_streak = 0
    recovery_started = False
    recovery_steps = 0
    heel_l_y = float(_foot_pos(model, data, "L")[1])

    for post_step in range(POST_HEEL_MAX_STEPS):
        phase_name = PostPhase.POST_HEEL
        if post_step > POST_HEEL_MAX_STEPS // 3:
            phase_name = PostPhase.OBSERVE

        l_nf = _foot_normal_force(model, data, "L")
        l_contact = _foot_contact(model, data, "L")

        if l_contact and l_nf >= PLANT_FORCE_N:
            plant_streak += 1
            if plant_streak >= PLANT_SUSTAIN_STEPS and not post.sustained_plant:
                post.sustained_plant = True
                post.plant_confirmed_step = st.diag.global_step
                print(f"\n  >> L FOOT PLANT confirmed at step {post.plant_confirmed_step} (L_nf={l_nf:.1f} N)")
        else:
            plant_streak = 0

        if post.sustained_plant and not recovery_started and not post.mini_recovery_applied:
            phase_name = PostPhase.MINI_RECOVERY
            recovery_started = True
            post.mini_recovery_applied = True
            heel_ctrl = heel_ctrl.copy()
            heel_ctrl[IDX_L_ANKLE_P] = float(np.clip(
                heel_ctrl[IDX_L_ANKLE_P] + RECOVERY_ANKLE_DELTA,
                cr[IDX_L_ANKLE_P, 0], cr[IDX_L_ANKLE_P, 1],
            ))
            print(f"\n  >> MINI RECOVERY: L ankle +{RECOVERY_ANKLE_DELTA:.2f} rad (conservative)")

        if recovery_started and recovery_steps < RECOVERY_MAX_STEPS:
            recovery_steps += 1

        st.ctrl = heel_ctrl.copy()
        data.ctrl[:15] = st.ctrl
        mujoco.mj_step(model, data)
        st.diag.global_step += 1
        _record_post(env, model, data, post, phase_name)
        if not _sync_viewer(viewer, slow):
            break

    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    post.final_torso_tilt_rad = env._quat_tilt_rad()
    post.final_forward_vel_m_s = _forward_vel(data)
    post.final_l_rel_r_mm = _forward_mm(l_pos[1], r_pos[1])

    chest_y = float(data.qpos[1])
    post.passed_over_l_foot = chest_y < heel_l_y - 0.02

    return post


def print_summary(post: ContinueDiagnostics) -> None:
    d = post.base
    print("\n" + "=" * 60)
    print("STAGED FORWARD CATCH CONTINUE - DIAGNOSTICS")
    print("=" * 60)
    print("--- LOCKED PREFIX (same as staged_forward_catch_test) ---")
    print(f"HIP_SWING_START_STEP = {d.hip_swing_start_step}")
    print(f"CATCH_START_STEP = {d.catch_start_step}")
    print(f"PEAK_FWD_AIRBORNE_MM = {d.peak_fwd_airborne_mm:.1f}")
    print(f"HEEL_TOUCHDOWN_STEP = {post.heel_touchdown_step}")
    print(f"HEEL_L_REL_R_MM = {post.heel_touchdown_l_rel_r_mm}")
    print(f"HEEL_L_NORMAL_N = {post.heel_touchdown_l_nf}")

    print("\n--- POST HEEL ---")
    print(f"PEAK_L_NORMAL_N = {post.peak_l_normal_post_heel:.1f}")
    print(f"PEAK_R_NORMAL_N = {post.peak_r_normal_post_heel:.1f}")
    print(f"R_LOAD_FRAC_RANGE = {post.min_r_load_frac_post_heel:.3f} .. {post.max_r_load_frac_post_heel:.3f}")
    print(f"SUSTAINED_L_PLANT = {post.sustained_plant}")
    print(f"PLANT_CONFIRMED_STEP = {post.plant_confirmed_step}")
    print(f"L_FOOT_LOST_CONTACT_POST_HEEL = {post.l_foot_lost_contact_post_heel}")
    print(f"L_FOOT_LOST_CONTACT_STEP = {post.l_foot_lost_contact_step}")
    print(f"MINI_RECOVERY_APPLIED = {post.mini_recovery_applied}")
    print(f"FINAL_TORSO_TILT_RAD = {post.final_torso_tilt_rad:.3f}")
    print(f"FINAL_FORWARD_VEL_M_S = {post.final_forward_vel_m_s:.3f}")
    print(f"FINAL_L_REL_R_MM = {post.final_l_rel_r_mm:.1f}")
    print(f"TORSO_PASSED_OVER_L_FOOT = {post.passed_over_l_foot}")

    if post.post_samples:
        print("\n  Post-heel timeline (every 25 steps):")
        print(
            "  step  phase                      L_ct L_nf  R_nf  R_frac  "
            "L-R_fwd  tilt   fwd_vel  angvel_y"
        )
        for s in post.post_samples:
            print(
                f"  {s.step:5d}  {s.phase:<26s} "
                f"{'Y' if s.l_contact else 'N':4s} {s.l_normal_n:5.1f} {s.r_normal_n:5.1f} "
                f"{s.r_load_frac:6.3f} {s.l_rel_r_fwd_mm:7.1f} "
                f"{s.torso_tilt_rad:5.2f} {s.forward_vel_m_s:7.3f} {s.torso_angvel_y:8.3f}"
            )

    print("\nContinuation experiment - locked prefix, no auto-tuning.")


def _find_mujoco_viewer_hwnd():
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    end = time.time() + VIEWER_POSITION_TIMEOUT_S
    while time.time() < end:
        found: list[int] = []

        def cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                n = user32.GetWindowTextLengthW(hwnd)
                if n > 0:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, buf, n + 1)
                    if buf.value.startswith(VIEWER_TITLE_PREFIX):
                        found.append(hwnd)
            return True

        fn = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(cb)
        user32.EnumWindows(fn, 0)
        if found:
            return found[0]
        time.sleep(0.05)
    return None


def _configure_viewer_window() -> None:
    if sys.platform != "win32":
        return
    hwnd = _find_mujoco_viewer_hwnd()
    if hwnd is None:
        return
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.wintypes.DWORD), ("rcMonitor", RECT),
                    ("rcWork", RECT), ("dwFlags", ctypes.wintypes.DWORD)]

    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 1), ctypes.byref(info))
    wa = info.rcWork
    x = wa.left + (wa.right - wa.left - VIEWER_WIDTH) // 2
    y = wa.top + (wa.bottom - wa.top - VIEWER_HEIGHT) // 2
    user32.SetWindowPos(hwnd, 0, x, y, VIEWER_WIDTH, VIEWER_HEIGHT, 0x0004)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continue staged forward catch past heel touchdown.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Staged forward catch CONTINUE: locked prefix -> past heel touchdown")
    print("Viewer: robot only.\n")

    if args.headless:
        post = run_staged_forward_catch_continue(env, viewer=None, slow=False)
        print_summary(post)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        post = run_staged_forward_catch_continue(env, viewer=v, slow=args.slow)
    print_summary(post)


if __name__ == "__main__":
    main(sys.argv[1:])
