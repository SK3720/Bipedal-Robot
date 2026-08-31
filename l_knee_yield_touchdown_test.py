"""L knee yield at touchdown — flat ankle + conservative knee flexion on contact.

LOCKED: staged_forward_catch_test prefix + flat-at-last-moment ankle (flat_foot_touchdown_test).
NEW: at first L contact, smooth L knee flexion yield while L hip / R leg unchanged.

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

from flat_foot_touchdown_test import (
    ANKLE_FLAT_RAMP_STEP,
    CLEARANCE_TRIGGER_MM,
    L_ANKLE_FLAT_TARGET,
    SOLE_DESCENT_VEL_M_S,
    VIEWER_HEIGHT,
    VIEWER_POSITION_TIMEOUT_S,
    VIEWER_TITLE_PREFIX,
    VIEWER_WIDTH,
    _foot_pitch_rad,
    _l_foot_vert_vel_m_s,
    _run_locked_prefix,
    _sole_clearance_mm,
)
from staged_forward_catch_test import (
    CATCH_MAX_STEPS,
    IDX_L_ANKLE_P,
    IDX_L_HIP_PITCH,
    IDX_L_KNEE,
    L_ANKLE_CATCH,
    L_HIP_CATCH,
    L_KNEE_CATCH,
    Phase,
    QPOS_L_KNEE,
    RunState,
    StepDiagnostics,
    _foot_contact,
    _foot_normal_force,
    _foot_pos,
    _forward_vel,
    _lerp_ctrl,
    _print_phase,
    _reset,
    _sim_step,
    _smooth,
    _sync_viewer,
    left_leg_pose,
    rapid_shift_pose,
)
from staged_forward_catch_continue_test import _is_heel_touchdown, _r_load_fraction

QPOS_L_ANKLE = 15

# Conservative single-shot knee yield (negative = flex).
KNEE_YIELD_DELTA_RAD = -0.15
KNEE_YIELD_STEPS = 20
POST_YIELD_OBSERVE_STEPS = 100
L_PLANT_DRIFT_THRESH_MM = 25.0


@dataclass
class TrajCheck:
    max_hip_cmd_diff: float = 0.0
    max_knee_cmd_diff: float = 0.0
    max_ankle_cmd_diff: float = 0.0
    max_hip_qpos_diff: float = 0.0
    max_knee_qpos_diff: float = 0.0
    steps_compared: int = 0
    ankle_trigger_step: int | None = None
    clearance_at_ankle_trigger_mm: float | None = None


@dataclass
class YieldDiagnostics:
    base: StepDiagnostics
    traj_check: TrajCheck = field(default_factory=TrajCheck)
    heel_touchdown_step: int | None = None
    knee_yield_trigger_step: int | None = None
    knee_cmd_before_td: float | None = None
    knee_qpos_before_td: float | None = None
    knee_cmd_after_yield: float | None = None
    knee_qpos_after_yield: float | None = None
    ankle_cmd_at_td: float | None = None
    ankle_qpos_at_td: float | None = None
    foot_pitch_at_td_rad: float | None = None
    torso_tilt_before_td_rad: float | None = None
    torso_tilt_after_yield_rad: float | None = None
    torso_angvel_before_td_rad_s: float | None = None
    torso_angvel_after_yield_rad_s: float | None = None
    torso_fwd_vel_before_td_m_s: float | None = None
    torso_fwd_vel_after_yield_m_s: float | None = None
    fwd_vel_kick_at_td_m_s: float | None = None
    peak_l_normal_n: float = 0.0
    peak_r_normal_n: float = 0.0
    r_load_fraction_at_td: float | None = None
    r_load_fraction_after_yield: float | None = None
    l_foot_max_drift_mm: float = 0.0
    l_foot_remains_planted: bool = False
    r_foot_in_contact_at_end: bool = False
    r_foot_in_contact_at_td: bool = False


def _torso_angvel_pitch_rad_s(data: mujoco.MjData) -> float:
    return float(data.qvel[4])


def _record_flat_catch_reference(
    env: BipedalWalkEnv,
) -> dict[int, tuple[float, float, float]]:
    """Flat-ankle catch reference: hip_cmd, knee_cmd, ankle_cmd per step."""
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
    _run_locked_prefix(env, model, data, cr, st, viewer=None, slow=False, verbose=False)

    catch_start_ctrl = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l_contact = _foot_contact(model, data, "L")
    was_airborne_in_catch = not prev_l_contact
    ankle_cmd = float(catch_start_ctrl[IDX_L_ANKLE_P])
    ankle_flat_active = False
    ref: dict[int, tuple[float, float, float]] = {}

    while st.catch_steps < CATCH_MAX_STEPS:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        full_lerp = _lerp_ctrl(catch_start_ctrl, catch_target, alpha, cr)
        knee_val = float(full_lerp[IDX_L_KNEE])
        hip_val = float(full_lerp[IDX_L_HIP_PITCH])

        if not _foot_contact(model, data, "L"):
            if not ankle_flat_active:
                clearance = _sole_clearance_mm(model, data)
                descending = _l_foot_vert_vel_m_s(model, data) < -SOLE_DESCENT_VEL_M_S
                if clearance < CLEARANCE_TRIGGER_MM and descending:
                    ankle_flat_active = True
            if ankle_flat_active:
                ankle_cmd = min(ankle_cmd + ANKLE_FLAT_RAMP_STEP, L_ANKLE_FLAT_TARGET)

        st.ctrl = left_leg_pose(knee_val, hip_val, ankle_cmd)
        step = st.diag.global_step + 1
        ref[step] = (hip_val, knee_val, ankle_cmd)
        _sim_step(env, model, data, st, viewer=None, slow=False)
        st.catch_steps += 1

        if _is_heel_touchdown(
            model,
            data,
            catch_started=True,
            was_airborne_in_catch=was_airborne_in_catch,
            prev_l_contact=prev_l_contact,
        ):
            break
        prev_l_contact = _foot_contact(model, data, "L")
        if not prev_l_contact:
            was_airborne_in_catch = True

    return ref


def run_knee_yield_experiment(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    ref_cmds: dict[int, tuple[float, float, float]] | None = None,
    *,
    verbose: bool = True,
) -> YieldDiagnostics:
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
    diag = YieldDiagnostics(base=st.diag)

    _run_locked_prefix(env, model, data, cr, st, viewer, slow, verbose=verbose)

    if verbose:
        _print_phase("FLAT ANKLE CATCH + KNEE YIELD AT TOUCHDOWN")

    catch_start_ctrl = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l_contact = _foot_contact(model, data, "L")
    was_airborne_in_catch = not prev_l_contact

    ankle_cmd = float(catch_start_ctrl[IDX_L_ANKLE_P])
    ankle_flat_active = False
    knee_yield_active = False
    pre_td_tilt: float | None = None
    pre_td_vel: float | None = None
    pre_td_angvel: float | None = None
    heel_l_xy: np.ndarray | None = None
    stance_ctrl: np.ndarray | None = None
    hip_hold = 0.0
    ankle_hold = 0.0
    knee_start = 0.0
    yield_steps = 0

    while st.catch_steps < CATCH_MAX_STEPS:
        if pre_td_vel is None and not _foot_contact(model, data, "L"):
            pre_td_vel = _forward_vel(data)
            pre_td_tilt = env._quat_tilt_rad()
            pre_td_angvel = _torso_angvel_pitch_rad_s(data)

        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        full_lerp = _lerp_ctrl(catch_start_ctrl, catch_target, alpha, cr)
        knee_val = float(full_lerp[IDX_L_KNEE])
        hip_val = float(full_lerp[IDX_L_HIP_PITCH])

        if ref_cmds is not None and not knee_yield_active:
            step = st.diag.global_step + 1
            if step in ref_cmds:
                ref_hip, ref_knee, ref_ankle = ref_cmds[step]
                tc = diag.traj_check
                tc.max_hip_cmd_diff = max(tc.max_hip_cmd_diff, abs(hip_val - ref_hip))
                tc.max_knee_cmd_diff = max(tc.max_knee_cmd_diff, abs(knee_val - ref_knee))
                tc.max_ankle_cmd_diff = max(tc.max_ankle_cmd_diff, abs(ankle_cmd - ref_ankle))
                tc.steps_compared += 1

        if knee_yield_active:
            t = _smooth(yield_steps / KNEE_YIELD_STEPS)
            knee_cmd = knee_start + t * KNEE_YIELD_DELTA_RAD
            st.ctrl = left_leg_pose(knee_cmd, hip_hold, ankle_hold)
            for idx in range(15):
                if idx not in (IDX_L_KNEE, IDX_L_HIP_PITCH, IDX_L_ANKLE_P):
                    st.ctrl[idx] = stance_ctrl[idx]  # type: ignore[index]
        else:
            if not _foot_contact(model, data, "L"):
                if not ankle_flat_active:
                    clearance = _sole_clearance_mm(model, data)
                    descending = _l_foot_vert_vel_m_s(model, data) < -SOLE_DESCENT_VEL_M_S
                    if clearance < CLEARANCE_TRIGGER_MM and descending:
                        ankle_flat_active = True
                        diag.traj_check.ankle_trigger_step = st.diag.global_step
                        diag.traj_check.clearance_at_ankle_trigger_mm = clearance
                if ankle_flat_active:
                    ankle_cmd = min(ankle_cmd + ANKLE_FLAT_RAMP_STEP, L_ANKLE_FLAT_TARGET)
            st.ctrl = left_leg_pose(knee_val, hip_val, ankle_cmd)

        _sim_step(env, model, data, st, viewer, slow)
        st.catch_steps += 1

        l_nf = _foot_normal_force(model, data, "L")
        r_nf = _foot_normal_force(model, data, "R")
        diag.peak_l_normal_n = max(diag.peak_l_normal_n, l_nf)
        diag.peak_r_normal_n = max(diag.peak_r_normal_n, r_nf)

        if (
            not knee_yield_active
            and _is_heel_touchdown(
                model,
                data,
                catch_started=True,
                was_airborne_in_catch=was_airborne_in_catch,
                prev_l_contact=prev_l_contact,
            )
        ):
            diag.heel_touchdown_step = st.diag.global_step
            diag.knee_yield_trigger_step = st.diag.global_step + 1
            diag.knee_cmd_before_td = float(data.ctrl[IDX_L_KNEE])
            diag.knee_qpos_before_td = float(data.qpos[QPOS_L_KNEE])
            diag.ankle_cmd_at_td = float(data.ctrl[IDX_L_ANKLE_P])
            diag.ankle_qpos_at_td = float(data.qpos[QPOS_L_ANKLE])
            diag.foot_pitch_at_td_rad = _foot_pitch_rad(model, data)
            diag.torso_tilt_before_td_rad = pre_td_tilt
            diag.torso_angvel_before_td_rad_s = pre_td_angvel
            diag.torso_fwd_vel_before_td_m_s = pre_td_vel
            diag.r_load_fraction_at_td = _r_load_fraction(model, data)
            diag.r_foot_in_contact_at_td = _foot_contact(model, data, "R")
            heel_l_xy = _foot_pos(model, data, "L")[:2].copy()
            stance_ctrl = st.ctrl.copy()
            hip_hold = float(stance_ctrl[IDX_L_HIP_PITCH])
            ankle_hold = float(stance_ctrl[IDX_L_ANKLE_P])
            knee_start = float(stance_ctrl[IDX_L_KNEE])
            knee_yield_active = True
            yield_steps = 0
            if verbose:
                print(
                    f"\n  >> TOUCHDOWN step {diag.heel_touchdown_step} "
                    f"knee_cmd={diag.knee_cmd_before_td:.3f} "
                    f"ankle_cmd={diag.ankle_cmd_at_td:.3f} -> knee yield next step"
                )

        if knee_yield_active:
            yield_steps += 1
            l_pos = _foot_pos(model, data, "L")
            if heel_l_xy is not None:
                diag.l_foot_max_drift_mm = max(
                    diag.l_foot_max_drift_mm,
                    float(np.linalg.norm(l_pos[:2] - heel_l_xy) * 1000.0),
                )
            if yield_steps >= KNEE_YIELD_STEPS:
                diag.knee_cmd_after_yield = float(data.ctrl[IDX_L_KNEE])
                diag.knee_qpos_after_yield = float(data.qpos[QPOS_L_KNEE])
                diag.torso_tilt_after_yield_rad = env._quat_tilt_rad()
                diag.torso_angvel_after_yield_rad_s = _torso_angvel_pitch_rad_s(data)
                diag.torso_fwd_vel_after_yield_m_s = _forward_vel(data)
                diag.r_load_fraction_after_yield = _r_load_fraction(model, data)
                if diag.torso_fwd_vel_before_td_m_s is not None:
                    diag.fwd_vel_kick_at_td_m_s = (
                        diag.torso_fwd_vel_after_yield_m_s - diag.torso_fwd_vel_before_td_m_s
                    )
                if verbose:
                    print(
                        f"\n  >> KNEE YIELD COMPLETE step {st.diag.global_step} "
                        f"knee_cmd={diag.knee_cmd_after_yield:.3f} "
                        f"knee_qpos={diag.knee_qpos_after_yield:.3f}"
                    )
                break

        prev_l_contact = _foot_contact(model, data, "L")
        if not prev_l_contact:
            was_airborne_in_catch = True

    if stance_ctrl is None:
        stance_ctrl = st.ctrl.copy()
        if heel_l_xy is None:
            heel_l_xy = _foot_pos(model, data, "L")[:2].copy()

    hold_ctrl = stance_ctrl.copy()
    if knee_yield_active and diag.knee_cmd_after_yield is not None:
        hold_ctrl[IDX_L_KNEE] = diag.knee_cmd_after_yield

    if verbose:
        _print_phase("POST YIELD OBSERVE")
    for _ in range(POST_YIELD_OBSERVE_STEPS):
        l_nf = _foot_normal_force(model, data, "L")
        r_nf = _foot_normal_force(model, data, "R")
        diag.peak_l_normal_n = max(diag.peak_l_normal_n, l_nf)
        diag.peak_r_normal_n = max(diag.peak_r_normal_n, r_nf)
        l_pos = _foot_pos(model, data, "L")
        if heel_l_xy is not None:
            diag.l_foot_max_drift_mm = max(
                diag.l_foot_max_drift_mm,
                float(np.linalg.norm(l_pos[:2] - heel_l_xy) * 1000.0),
            )
        st.ctrl = hold_ctrl.copy()
        data.ctrl[:15] = st.ctrl
        mujoco.mj_step(model, data)
        st.diag.global_step += 1
        _sync_viewer(viewer, slow)

    diag.l_foot_remains_planted = diag.l_foot_max_drift_mm < L_PLANT_DRIFT_THRESH_MM
    diag.r_foot_in_contact_at_end = _foot_contact(model, data, "R")

    if verbose:
        _print_phase("STOP")
    return diag


def print_summary(d: YieldDiagnostics) -> None:
    print("\n" + "=" * 60)
    print("L KNEE YIELD AT TOUCHDOWN (flat ankle + conservative flex)")
    print("=" * 60)
    print(f"KNEE_YIELD_DELTA = {KNEE_YIELD_DELTA_RAD:.2f} rad over {KNEE_YIELD_STEPS} steps")
    print(f"Ankle trigger: clearance < {CLEARANCE_TRIGGER_MM:.0f} mm, descending")

    print("\n--- TOUCHDOWN ---")
    print(f"HEEL_TOUCHDOWN_STEP = {d.heel_touchdown_step}")
    print(f"KNEE_YIELD_TRIGGER_STEP = {d.knee_yield_trigger_step}")
    print(f"L_KNEE_CMD_BEFORE_TD = {d.knee_cmd_before_td}")
    print(f"L_KNEE_QPOS_BEFORE_TD = {d.knee_qpos_before_td}")
    print(f"L_KNEE_CMD_AFTER_YIELD = {d.knee_cmd_after_yield}")
    print(f"L_KNEE_QPOS_AFTER_YIELD = {d.knee_qpos_after_yield}")
    print(f"L_ANKLE_CMD_AT_TD = {d.ankle_cmd_at_td}")
    print(f"L_ANKLE_QPOS_AT_TD = {d.ankle_qpos_at_td}")
    print(f"L_FOOT_PITCH_AT_TD_RAD = {d.foot_pitch_at_td_rad}")

    print("\n--- FORCES / LOAD ---")
    print(f"PEAK_L_NORMAL_N = {d.peak_l_normal_n:.1f}")
    print(f"PEAK_R_NORMAL_N = {d.peak_r_normal_n:.1f}")
    print(f"R_LOAD_FRACTION_AT_TD = {d.r_load_fraction_at_td}")
    print(f"R_LOAD_FRACTION_AFTER_YIELD = {d.r_load_fraction_after_yield}")

    print("\n--- TORSO ---")
    print(f"TORSO_TILT_BEFORE_TD_RAD = {d.torso_tilt_before_td_rad}")
    print(f"TORSO_TILT_AFTER_YIELD_RAD = {d.torso_tilt_after_yield_rad}")
    print(f"TORSO_ANGVEL_BEFORE_TD_RAD_S = {d.torso_angvel_before_td_rad_s}")
    print(f"TORSO_ANGVEL_AFTER_YIELD_RAD_S = {d.torso_angvel_after_yield_rad_s}")
    print(f"TORSO_FWD_VEL_BEFORE_TD_M_S = {d.torso_fwd_vel_before_td_m_s}")
    print(f"TORSO_FWD_VEL_AFTER_YIELD_M_S = {d.torso_fwd_vel_after_yield_m_s}")
    print(f"FWD_VEL_CHANGE_TD_TO_POST_YIELD_M_S = {d.fwd_vel_kick_at_td_m_s}")

    print("\n--- FOOT STATE ---")
    print(f"L_FOOT_MAX_DRIFT_MM = {d.l_foot_max_drift_mm:.1f}")
    print(f"L_FOOT_REMAINS_PLANTED = {d.l_foot_remains_planted}")
    print(f"R_FOOT_IN_CONTACT_AT_TD = {d.r_foot_in_contact_at_td}")
    print(f"R_FOOT_IN_CONTACT_AT_END = {d.r_foot_in_contact_at_end}")

    tc = d.traj_check
    print("\n--- TRAJECTORY MATCH vs FLAT-AT-LAST-MOMENT (before knee yield) ---")
    print(f"ANKLE_TRIGGER_STEP = {tc.ankle_trigger_step}")
    print(f"CLEARANCE_AT_ANKLE_TRIGGER_MM = {tc.clearance_at_ankle_trigger_mm}")
    print(f"STEPS_COMPARED = {tc.steps_compared}")
    print(f"MAX_HIP_CMD_DIFF_RAD = {tc.max_hip_cmd_diff:.6f}")
    print(f"MAX_KNEE_CMD_DIFF_RAD = {tc.max_knee_cmd_diff:.6f}")
    print(f"MAX_ANKLE_CMD_DIFF_RAD = {tc.max_ankle_cmd_diff:.6f}")
    match_ok = (
        tc.max_hip_cmd_diff < 1e-4
        and tc.max_knee_cmd_diff < 1e-4
        and tc.max_ankle_cmd_diff <= ANKLE_FLAT_RAMP_STEP + 1e-6
    )
    print(f"Pre-yield trajectory match: {'YES' if match_ok else 'CHECK'}")

    knee_bent = (
        d.knee_qpos_after_yield is not None
        and d.knee_qpos_before_td is not None
        and d.knee_qpos_after_yield < d.knee_qpos_before_td - 0.03
    )
    print(f"\nKnee visibly yielded (qpos flexed > 0.03 rad): {'YES' if knee_bent else 'NO'}")
    print("\nSingle fixed experiment - no auto-tuning.")


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
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.wintypes.DWORD),
        ]

    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 1), ctypes.byref(info))
    wa = info.rcWork
    x = wa.left + (wa.right - wa.left - VIEWER_WIDTH) // 2
    y = wa.top + (wa.bottom - wa.top - VIEWER_HEIGHT) // 2
    user32.SetWindowPos(hwnd, 0, x, y, VIEWER_WIDTH, VIEWER_HEIGHT, 0x0004)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L knee yield at touchdown experiment.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("L knee yield at touchdown: locked swing + flat ankle + knee flex on contact")
    print(f"Knee yield: {KNEE_YIELD_DELTA_RAD:.2f} rad over {KNEE_YIELD_STEPS} steps")
    print("Viewer: robot only.\n")

    ref_cmds = _record_flat_catch_reference(env)

    if args.headless:
        diag = run_knee_yield_experiment(
            env, viewer=None, slow=False, ref_cmds=ref_cmds, verbose=False
        )
        print_summary(diag)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        diag = run_knee_yield_experiment(
            env, viewer=v, slow=args.slow, ref_cmds=ref_cmds, verbose=True
        )
    print_summary(diag)


if __name__ == "__main__":
    main(sys.argv[1:])
