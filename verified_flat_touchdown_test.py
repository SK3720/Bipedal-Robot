"""Corrected flat touchdown experiment using verified sole geometry.

FINDINGS from ankle_axis_diagnostic + touchdown_timeline_diagnostic:
- L ankle PITCH (not roll) controls sagittal sole orientation.
- POSITIVE ankle pitch STEEPENS the sole; flatter target is NEGATIVE (~-0.10 to -0.20).
- Legacy foot_pitch metric (~72 deg at TD) is misleading; mesh sole angle ~2 deg.
- Baseline catch lerp moves ankle slightly positive (+0.08) — wrong direction for flattening.

LOCKED: staged_forward_catch_test through swing + catch lerp until prep trigger.
PREP: ramp L ankle PITCH toward FLAT_TARGET + knee yield; L hip frozen; R unchanged.
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
    SOLE_DESCENT_VEL_M_S,
    VIEWER_HEIGHT,
    VIEWER_POSITION_TIMEOUT_S,
    VIEWER_TITLE_PREFIX,
    VIEWER_WIDTH,
    _l_foot_vert_vel_m_s,
    _run_locked_prefix,
    _sole_clearance_mm,
)
from foot_geometry import sole_metrics
from staged_forward_catch_test import (
    CATCH_MAX_STEPS,
    IDX_L_ANKLE_P,
    IDX_L_HIP_PITCH,
    IDX_L_KNEE,
    L_ANKLE_CATCH,
    L_HIP_CATCH,
    L_KNEE_CATCH,
    NORMAL_SLEEP_S,
    Phase,
    QPOS_L_KNEE,
    RunState,
    SLOW_SLEEP_S,
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
from staged_forward_catch_continue_test import _is_heel_touchdown

QPOS_L_ANKLE = 15

# Verified at catch pose: flattest sole near ankle pitch -0.10 to -0.20.
FLAT_ANKLE_PITCH_TARGET = -0.15
ANKLE_PITCH_RAMP_STEP = 0.012
KNEE_YIELD_DELTA_RAD = -0.15
KNEE_YIELD_STEPS = 20
PREP_CLEARANCE_MM = 28.0
POST_HOLD_STEPS = 120
HEEL_TOE_FLAT_TOL_DEG = 1.5

_TRACE_ENABLED = False
_TRACE_LOG: list[dict] = []


def _trace_record(
    *,
    step: int,
    phase: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer: mujoco.viewer.Handle | None,
    loop_running: bool,
    extra: str = "",
) -> None:
    if not _TRACE_ENABLED:
        return
    viewer_running = None
    if viewer is not None:
        viewer_running = viewer.is_running()
    entry = {
        "step": step,
        "phase": phase,
        "l_clr_mm": _sole_clearance_mm(model, data),
        "l_contact": _foot_contact(model, data, "L"),
        "ankle_q": float(data.qpos[QPOS_L_ANKLE]),
        "knee_q": float(data.qpos[QPOS_L_KNEE]),
        "loop_running": loop_running,
        "viewer_running": viewer_running,
        "extra": extra,
    }
    _TRACE_LOG.append(entry)
    near_td = 930 <= step <= 1000
    if near_td or step % 100 == 0 or extra:
        vr = "n/a" if viewer_running is None else str(viewer_running)
        print(
            f"  [trace] step={step:4d} phase={phase:<22} "
            f"clr={entry['l_clr_mm']:6.1f}mm contact={entry['l_contact']} "
            f"ankle_q={entry['ankle_q']:.3f} knee_q={entry['knee_q']:.3f} "
            f"loop={loop_running} viewer={vr} {extra}"
        )


def _hold_viewer(viewer: mujoco.viewer.Handle, slow: bool) -> None:
    if not viewer.is_running():
        return
    print("\nTouchdown complete — close viewer window to exit.")
    while viewer.is_running():
        viewer.sync()
        time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)


@dataclass
class TrajCheck:
    max_hip_cmd_diff: float = 0.0
    max_knee_cmd_diff: float = 0.0
    max_ankle_cmd_diff: float = 0.0
    max_fwd_mm_diff: float = 0.0
    steps_compared: int = 0


@dataclass
class TouchdownDiagnostics:
    label: str
    base: StepDiagnostics
    traj: TrajCheck = field(default_factory=TrajCheck)
    prep_trigger_step: int | None = None
    touchdown_step: int | None = None
    sole_horiz_at_trigger_deg: float | None = None
    sole_horiz_before_contact_deg: float | None = None
    sole_horiz_at_td_deg: float | None = None
    heel_toe_pitch_at_td_deg: float | None = None
    ankle_cmd_at_td: float | None = None
    ankle_qpos_at_td: float | None = None
    knee_qpos_at_td: float | None = None
    l_normal_at_td: float | None = None
    peak_l_normal: float = 0.0
    torso_fwd_vel_at_td: float | None = None
    torso_angvel_at_td: float | None = None
    l_foot_drift_mm: float = 0.0
    approach_rows: list[dict] = field(default_factory=list)
    final_global_step: int = 0
    termination_reason: str = "unknown"
    viewer_running_at_end: bool | None = None


def _record_baseline_catch_ref(env: BipedalWalkEnv) -> dict[int, tuple[float, float, float, float]]:
    """step -> (hip_cmd, knee_cmd, ankle_cmd, fwd_air_mm)."""
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
    _run_locked_prefix(env, model, data, cr, st, None, False, verbose=False)
    catch_start = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev = _foot_contact(model, data, "L")
    was_air = not prev
    ref: dict[int, tuple[float, float, float, float]] = {}
    while st.catch_steps < CATCH_MAX_STEPS:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        st.ctrl = _lerp_ctrl(catch_start, catch_target, alpha, cr)
        step = st.diag.global_step + 1
        fwd = 0.0
        if st.airborne_ref_y is not None:
            fwd = float(-(_foot_pos(model, data, "L")[1] - st.airborne_ref_y) * 1000.0)
        ref[step] = (
            float(st.ctrl[IDX_L_HIP_PITCH]),
            float(st.ctrl[IDX_L_KNEE]),
            float(st.ctrl[IDX_L_ANKLE_P]),
            fwd,
        )
        _sim_step(env, model, data, st, None, False)
        st.catch_steps += 1
        if _is_heel_touchdown(
            model, data,
            catch_started=True, was_airborne_in_catch=was_air, prev_l_contact=prev,
        ):
            break
        prev = _foot_contact(model, data, "L")
        if not prev:
            was_air = True
    return ref


def run_baseline_touchdown(env: BipedalWalkEnv) -> TouchdownDiagnostics:
    from touchdown_timeline_diagnostic import run_baseline_timeline

    rec = run_baseline_timeline()
    d = TouchdownDiagnostics(label="BASELINE", base=StepDiagnostics())
    d.touchdown_step = rec.first_contact_step
    if rec.samples:
        last_air = next((s for s in reversed(rec.samples) if not s["contact"]), None)
        at = next((s for s in rec.samples if s["step"] == rec.first_contact_step), None)
        if last_air:
            d.sole_horiz_before_contact_deg = last_air["sole_horiz_deg"]
            d.heel_toe_pitch_at_td_deg = last_air["heel_toe_pitch_deg"]
            d.ankle_qpos_at_td = last_air["ankle_qpos"]
            d.knee_qpos_at_td = last_air["knee_qpos"]
        if at:
            d.sole_horiz_at_td_deg = at["sole_horiz_deg"]
            d.heel_toe_pitch_at_td_deg = at["heel_toe_pitch_deg"]
            d.ankle_cmd_at_td = at["ankle_cmd"]
            d.ankle_qpos_at_td = at["ankle_qpos"]
            d.knee_qpos_at_td = at["knee_qpos"]
            d.l_normal_at_td = at["l_nf"]
            d.torso_fwd_vel_at_td = at["torso_fwd_vel"]
            d.peak_l_normal = max(s["l_nf"] for s in rec.samples)
    return d


def run_corrected_experiment(
    env: BipedalWalkEnv,
    ref: dict[int, tuple[float, float, float, float]] | None,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    verbose: bool = True,
) -> TouchdownDiagnostics:
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
    d = TouchdownDiagnostics(label="CORRECTED_PREP", base=st.diag)

    _run_locked_prefix(env, model, data, cr, st, viewer, slow, verbose=verbose)
    if verbose:
        _print_phase("CORRECTED ANKLE PITCH + KNEE YIELD PREP")

    catch_start = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    st.phase = Phase.CATCH
    prev_l = _foot_contact(model, data, "L")
    was_air = not prev_l

    ankle_cmd = float(catch_start[IDX_L_ANKLE_P])
    prep = False
    hip_hold = knee_start = 0.0
    prep_steps = 0
    stance: np.ndarray | None = None
    heel_xy: np.ndarray | None = None
    peak_clear = 0.0
    catch_loop_running = True

    while catch_loop_running and st.catch_steps < CATCH_MAX_STEPS:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        full = _lerp_ctrl(catch_start, catch_target, alpha, cr)
        knee_val = float(full[IDX_L_KNEE])
        hip_val = float(full[IDX_L_HIP_PITCH])

        if ref is not None and not prep:
            step = st.diag.global_step + 1
            if step in ref:
                rh, rk, ra, rf = ref[step]
                tc = d.traj
                tc.max_hip_cmd_diff = max(tc.max_hip_cmd_diff, abs(hip_val - rh))
                tc.max_knee_cmd_diff = max(tc.max_knee_cmd_diff, abs(knee_val - rk))
                tc.max_ankle_cmd_diff = max(tc.max_ankle_cmd_diff, abs(ankle_cmd - ra))
                if st.airborne_ref_y is not None:
                    fwd = float(-(_foot_pos(model, data, "L")[1] - st.airborne_ref_y) * 1000.0)
                    tc.max_fwd_mm_diff = max(tc.max_fwd_mm_diff, abs(fwd - rf))
                tc.steps_compared += 1

        if not prep:
            clr = _sole_clearance_mm(model, data)
            if not _foot_contact(model, data, "L"):
                peak_clear = max(peak_clear, clr)
                descending = _l_foot_vert_vel_m_s(model, data) < -SOLE_DESCENT_VEL_M_S
                trigger = (
                    clr < PREP_CLEARANCE_MM
                    and descending
                    and peak_clear >= 8.0
                    and clr <= peak_clear - 2.0
                )
                if trigger:
                    prep = True
                    d.prep_trigger_step = st.diag.global_step + 1
                    m = sole_metrics(model, data)
                    d.sole_horiz_at_trigger_deg = float(np.degrees(m["sole_angle_from_horizontal_rad"]))
                    hip_hold = hip_val
                    knee_start = knee_val
                    stance = st.ctrl.copy()
                    if verbose:
                        print(
                            f"\n  >> PREP step {d.prep_trigger_step} "
                            f"sole={d.sole_horiz_at_trigger_deg:.1f}° "
                            f"ankle_q={data.qpos[QPOS_L_ANKLE]:.3f}"
                        )
            st.ctrl = left_leg_pose(knee_val, hip_val, ankle_cmd)
        else:
            prep_steps += 1
            t = _smooth(min(prep_steps, KNEE_YIELD_STEPS) / KNEE_YIELD_STEPS)
            knee_cmd = knee_start + t * KNEE_YIELD_DELTA_RAD
            step_dir = -1.0 if FLAT_ANKLE_PITCH_TARGET < ankle_cmd else 1.0
            if abs(ankle_cmd - FLAT_ANKLE_PITCH_TARGET) > 1e-6:
                ankle_cmd += step_dir * ANKLE_PITCH_RAMP_STEP
                if step_dir < 0:
                    ankle_cmd = max(ankle_cmd, FLAT_ANKLE_PITCH_TARGET)
                else:
                    ankle_cmd = min(ankle_cmd, FLAT_ANKLE_PITCH_TARGET)
            st.ctrl = left_leg_pose(knee_cmd, hip_hold, ankle_cmd)
            if stance is not None:
                for i in range(15):
                    if i not in (IDX_L_KNEE, IDX_L_HIP_PITCH, IDX_L_ANKLE_P):
                        st.ctrl[i] = stance[i]

        _sim_step(env, model, data, st, viewer, slow)
        st.catch_steps += 1
        step_now = st.diag.global_step
        _trace_record(
            step=step_now,
            phase=st.phase.value + ("+PREP" if prep else ""),
            model=model,
            data=data,
            viewer=viewer,
            loop_running=catch_loop_running and st.catch_steps < CATCH_MAX_STEPS,
        )

        if prep:
            m = sole_metrics(model, data)
            d.approach_rows.append(
                {
                    "step": step_now,
                    "contact": _foot_contact(model, data, "L"),
                    "sole_horiz_deg": float(np.degrees(m["sole_angle_from_horizontal_rad"])),
                    "heel_toe_deg": float(np.degrees(m["heel_toe_pitch_rad"])),
                    "ankle_q": float(data.qpos[QPOS_L_ANKLE]),
                    "knee_q": float(data.qpos[QPOS_L_KNEE]),
                    "heel_clr": m["heel_clearance_mm"],
                    "toe_clr": m["toe_clearance_mm"],
                }
            )

        ln = _foot_normal_force(model, data, "L")
        d.peak_l_normal = max(d.peak_l_normal, ln)

        if _is_heel_touchdown(
            model, data,
            catch_started=True, was_airborne_in_catch=was_air, prev_l_contact=prev_l,
        ):
            d.touchdown_step = step_now
            m = sole_metrics(model, data)
            d.sole_horiz_at_td_deg = float(np.degrees(m["sole_angle_from_horizontal_rad"]))
            d.heel_toe_pitch_at_td_deg = float(np.degrees(m["heel_toe_pitch_rad"]))
            d.ankle_cmd_at_td = float(data.ctrl[IDX_L_ANKLE_P])
            d.ankle_qpos_at_td = float(data.qpos[QPOS_L_ANKLE])
            d.knee_qpos_at_td = float(data.qpos[QPOS_L_KNEE])
            d.l_normal_at_td = ln
            d.torso_fwd_vel_at_td = _forward_vel(data)
            d.torso_angvel_at_td = float(data.qvel[4])
            heel_xy = _foot_pos(model, data, "L")[:2].copy()
            if d.approach_rows:
                last_air = next((r for r in reversed(d.approach_rows) if not r["contact"]), None)
                if last_air:
                    d.sole_horiz_before_contact_deg = last_air["sole_horiz_deg"]
            hold = st.ctrl.copy()
            if verbose:
                print(
                    f"\n  >> TOUCHDOWN step {d.touchdown_step} "
                    f"sole={d.sole_horiz_at_td_deg:.1f}° "
                    f"heel-toe={d.heel_toe_pitch_at_td_deg:.1f}° "
                    f"ankle_q={d.ankle_qpos_at_td:.3f}"
                )
            d.termination_reason = f"touchdown at step {d.touchdown_step}"
            for hold_i in range(POST_HOLD_STEPS):
                lpos = _foot_pos(model, data, "L")
                if heel_xy is not None:
                    d.l_foot_drift_mm = max(
                        d.l_foot_drift_mm,
                        float(np.linalg.norm(lpos[:2] - heel_xy) * 1000.0),
                    )
                st.ctrl = hold.copy()
                data.ctrl[:15] = st.ctrl
                mujoco.mj_step(model, data)
                d.peak_l_normal = max(d.peak_l_normal, _foot_normal_force(model, data, "L"))
                _sync_viewer(viewer, slow)
                _trace_record(
                    step=st.diag.global_step,
                    phase="POST_HOLD",
                    model=model,
                    data=data,
                    viewer=viewer,
                    loop_running=hold_i + 1 < POST_HOLD_STEPS,
                    extra=f"hold {hold_i + 1}/{POST_HOLD_STEPS}",
                )
            catch_loop_running = False
            break

        prev_l = _foot_contact(model, data, "L")
        if not prev_l:
            was_air = True

    if d.termination_reason == "unknown":
        if st.catch_steps >= CATCH_MAX_STEPS:
            d.termination_reason = f"catch_steps reached CATCH_MAX_STEPS ({CATCH_MAX_STEPS})"
        else:
            d.termination_reason = "catch loop exited without touchdown"
    d.final_global_step = st.diag.global_step
    d.viewer_running_at_end = viewer.is_running() if viewer is not None else None
    if _TRACE_ENABLED:
        print(
            f"\n  [trace] TERMINATION: {d.termination_reason} "
            f"final_step={d.final_global_step} "
            f"touchdown_step={d.touchdown_step} "
            f"viewer_running={d.viewer_running_at_end}"
        )
    if verbose:
        _print_phase("STOP")
    return d


def print_comparison(base: TouchdownDiagnostics, corr: TouchdownDiagnostics) -> None:
    print("\n" + "=" * 72)
    print("CORRECTED FLAT TOUCHDOWN (negative ankle pitch target)")
    print("=" * 72)
    print(f"FLAT_ANKLE_PITCH_TARGET = {FLAT_ANKLE_PITCH_TARGET:.2f} rad (verified flatter than +0.38)")
    print(f"Prior experiments used +0.38 — that STEEPENS the sole (see ankle_axis_diagnostic.py)")

    tc = corr.traj
    print("\n--- TRAJECTORY LOCK (before prep) ---")
    print(f"steps_compared = {tc.steps_compared}")
    print(f"max_hip_cmd_diff = {tc.max_hip_cmd_diff:.6f}")
    print(f"max_knee_cmd_diff = {tc.max_knee_cmd_diff:.6f}")
    print(f"max_ankle_cmd_diff = {tc.max_ankle_cmd_diff:.6f}")
    print(f"max_fwd_mm_diff = {tc.max_fwd_mm_diff:.2f}")

    rows = [
        ("Touchdown step", base.touchdown_step, corr.touchdown_step),
        ("Prep trigger step", None, corr.prep_trigger_step),
        ("Sole horiz before contact (deg)", base.sole_horiz_before_contact_deg, corr.sole_horiz_before_contact_deg),
        ("Sole horiz at TD (deg)", base.sole_horiz_at_td_deg, corr.sole_horiz_at_td_deg),
        ("Heel-toe pitch at TD (deg)", base.heel_toe_pitch_at_td_deg, corr.heel_toe_pitch_at_td_deg),
        ("Ankle qpos at TD", base.ankle_qpos_at_td, corr.ankle_qpos_at_td),
        ("Knee qpos at TD", base.knee_qpos_at_td, corr.knee_qpos_at_td),
        ("L normal at TD (N)", base.l_normal_at_td, corr.l_normal_at_td),
        ("Peak L normal (N)", base.peak_l_normal, corr.peak_l_normal),
        ("Torso fwd vel at TD", base.torso_fwd_vel_at_td, corr.torso_fwd_vel_at_td),
    ]
    print(f"\n{'Metric':<32} {'BASELINE':<16} {'CORRECTED':<16}")
    print("-" * 64)
    for name, a, b in rows:
        print(f"{name:<32} {str(a):<16} {str(b):<16}")

    if corr.approach_rows:
        print("\n--- CORRECTED APPROACH (last 12 samples) ---")
        for r in corr.approach_rows[-12:]:
            print(
                f"  {r['step']:4d} sole={r['sole_horiz_deg']:5.1f}° "
                f"ht={r['heel_toe_deg']:5.1f}° ankle_q={r['ankle_q']:.3f} "
                f"knee_q={r['knee_q']:.3f} heel={r['heel_clr']:.1f} toe={r['toe_clr']:.1f}"
            )

    ht_ok = (
        corr.heel_toe_pitch_at_td_deg is not None
        and base.heel_toe_pitch_at_td_deg is not None
        and abs(corr.heel_toe_pitch_at_td_deg) < abs(base.heel_toe_pitch_at_td_deg)
    )
    sole_ok = (
        corr.sole_horiz_at_td_deg is not None
        and base.sole_horiz_at_td_deg is not None
        and corr.sole_horiz_at_td_deg <= base.sole_horiz_at_td_deg + 0.5
    )
    print(f"\nHeel-toe leveling improved: {'YES' if ht_ok else 'NO'}")
    print(f"Sole remained flat at TD: {'YES' if sole_ok else 'CHECK'}")
    print("\nRun: python verified_flat_touchdown_test.py --slow  (visual confirmation required)")


def _configure_viewer_window() -> None:
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.wintypes.DWORD), ("rcMonitor", RECT),
                    ("rcWork", RECT), ("dwFlags", ctypes.wintypes.DWORD)]

    hwnd = None
    end = time.time() + VIEWER_POSITION_TIMEOUT_S
    while time.time() < end:
        found: list[int] = []

        def cb(h, _):
            if user32.IsWindowVisible(h):
                n = user32.GetWindowTextLengthW(h)
                if n > 0:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(h, buf, n + 1)
                    if buf.value.startswith(VIEWER_TITLE_PREFIX):
                        found.append(h)
            return True

        fn = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(cb)
        user32.EnumWindows(fn, 0)
        if found:
            hwnd = found[0]
            break
        time.sleep(0.05)
    if hwnd is None:
        return
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 1), ctypes.byref(info))
    wa = info.rcWork
    x = wa.left + (wa.right - wa.left - VIEWER_WIDTH) // 2
    y = wa.top + (wa.bottom - wa.top - VIEWER_HEIGHT) // 2
    user32.SetWindowPos(hwnd, 0, x, y, VIEWER_WIDTH, VIEWER_HEIGHT, 0x0004)


def _run_step_count_compare(env: BipedalWalkEnv, ref: dict) -> None:
    """Run corrected experiment twice (fast vs slow-sleep) and compare step counts."""
    fast = run_corrected_experiment(env, ref, None, False, verbose=False)
    slow = run_corrected_experiment(env, ref, None, True, verbose=False)
    print("\n" + "=" * 72)
    print("STEP COUNT COMPARE (headless fast vs headless slow-sleep)")
    print("=" * 72)
    print(f"fast final_step={fast.final_global_step} td={fast.touchdown_step} reason={fast.termination_reason}")
    print(f"slow final_step={slow.final_global_step} td={slow.touchdown_step} reason={slow.termination_reason}")
    match = (
        fast.final_global_step == slow.final_global_step
        and fast.touchdown_step == slow.touchdown_step
    )
    print(f"trajectory match: {'YES' if match else 'NO'}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verified flat touchdown experiment.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--trace",
        action="store_true",
        help="Log step/phase/clearance/contact/joint state during corrected experiment.",
    )
    p.add_argument(
        "--trace-compare",
        action="store_true",
        help="Headless: run fast vs slow-sleep and compare final step counts.",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    global _TRACE_ENABLED, _TRACE_LOG
    args = parse_args(argv)
    _TRACE_ENABLED = args.trace
    _TRACE_LOG = []
    env = BipedalWalkEnv()
    print("Verified flat touchdown — negative ankle pitch + knee yield")
    print("See ankle_axis_diagnostic.py for joint axis verification.\n")

    base = run_baseline_touchdown(env)
    ref = _record_baseline_catch_ref(env)

    if args.trace_compare:
        _run_step_count_compare(env, ref)
        return

    if args.headless:
        corr = run_corrected_experiment(env, ref, None, False, verbose=args.trace)
        print_comparison(base, corr)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        corr = run_corrected_experiment(env, ref, v, args.slow, verbose=True)
        _hold_viewer(v, args.slow)
    print_comparison(base, corr)


if __name__ == "__main__":
    main(sys.argv[1:])
