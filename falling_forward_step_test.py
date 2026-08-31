"""Falling forward step — controlled dynamic fall and catch with the swing foot.

Sequence:
  STAND -> forward lean / momentum -> rapid R weight shift ->
  concurrent L knee lift + hip swing -> L foot catch ahead -> brief stop.

The robot is ALLOWED to fall. Success = fall forward and plant L foot ahead
to catch momentum — not static single-leg balance.

Evaluation only — does not modify robot.xml, biped_env, or other scripts.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np
from ctypes import wintypes

from biped_env import (
    BipedalWalkEnv,
    CHEST_Z_CONTACT,
    DEFAULT_POSE,
    STANDING_QUAT,
)

FLOOR_Z = 1.0

IDX_L_HIP_ROLL = 5
IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_R_HIP_ROLL = 10
IDX_R_HIP_PITCH = 11
IDX_R_KNEE = 12
IDX_R_ANKLE_P = 13
IDX_R_ANKLE_ROLL = 14

STAND_STEPS = 500
FALL_RAMP_STEPS = 90
FALL_MOMENTUM_STEPS = 200
RAPID_SHIFT_STEPS = 40
POST_SHIFT_WAIT_STEPS = 25
SWING_STEPS = 420
CATCH_STEPS = 300
STOP_STEPS = 200

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    FORWARD_FALL = "INITIATE FORWARD FALL"
    RAPID_SHIFT = "RAPID RIGHTWARD WEIGHT TRANSFER"
    SWING = "LEFT LEG SWING"
    CATCH = "LEFT FOOT CATCH"
    STOP = "STOP"


@dataclass
class StepMetrics:
    global_step: int = 0
    peak_com_forward_vel_m_s: float = 0.0
    peak_torso_forward_vel_m_s: float = 0.0
    contact_loss_step: int | None = None
    contact_loss_phase: str | None = None
    max_l_clearance_mm: float = 0.0
    peak_l_forward_mm: float = 0.0
    peak_l_forward_airborne_mm: float = 0.0
    r_max_drift_mm: float = 0.0
    left_foot_airborne: bool = False
    left_touchdown_occurred: bool = False
    touchdown_l_rel_r_forward_mm: float | None = None
    touchdown_tilt_rad: float | None = None
    touchdown_forward_vel_m_s: float | None = None
    touchdown_step: int | None = None
    swing_forward_mm_while_airborne: float = 0.0
    swing_backward_mm_while_airborne: float = 0.0
    fall_forward_sign: float = 1.0
    swing_same_direction_as_fall: bool | None = None
    peak_r_load_during_shift: float = 0.0
    final_torso_tilt_rad: float = 0.0
    final_l_rel_r_forward_mm: float = 0.0


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _forward_mm(y: float, y_ref: float) -> float:
    return float(-(y - y_ref) * 1000.0)


def _forward_vel_m_s(data: mujoco.MjData) -> float:
    """Positive when chest/COM moves along anatomical forward (world -Y)."""
    return float(-data.qvel[1])


def _foot_pos(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    return data.xpos[model.body(f"{side}_foot").id].copy()


def _foot_contact(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> bool:
    for ci in range(data.ncon):
        for gid in (data.contact[ci].geom1, data.contact[ci].geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if f"{side}_foot_collision" in name:
                return True
    return False


def _foot_normal_force(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
    total = 0.0
    for ci in range(data.ncon):
        g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        if f"{side}_foot_collision" not in n1 and f"{side}_foot_collision" not in n2:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, wrench)
        total += max(0.0, float(wrench[0]))
    return total


def _r_load_fraction(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    l_nf = _foot_normal_force(model, data, "L")
    r_nf = _foot_normal_force(model, data, "R")
    total = l_nf + r_nf
    return (r_nf / total) if total > 1e-6 else float("nan")


def _reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = DEFAULT_POSE
    data.qvel[:] = 0.0
    data.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(model, data)


def _lerp_ctrl(ctrl: np.ndarray, target: np.ndarray, alpha: float, cr: np.ndarray) -> np.ndarray:
    return np.clip((1.0 - alpha) * ctrl + alpha * target, cr[:, 0], cr[:, 1])


def _apply_joints(base: np.ndarray, values: dict[int, float]) -> np.ndarray:
    pose = base.copy()
    for idx, val in values.items():
        pose[idx] = val
    return pose


def forward_lean_pose() -> np.ndarray:
    """Sagittal lean — bilateral hip/ankle pitch, slight knee flex. No lateral shift."""
    return _apply_joints(
        DEFAULT_POSE,
        {
            IDX_L_HIP_PITCH: -0.14,
            IDX_R_HIP_PITCH: 0.14,
            IDX_L_KNEE: -0.08,
            IDX_R_KNEE: -0.08,
            IDX_L_ANKLE_P: -0.12,
            IDX_R_ANKLE_P: -0.12,
        },
    )


def weight_shift_on_lean() -> np.ndarray:
    """Verified rapid lateral unload on top of forward lean."""
    return _apply_joints(
        forward_lean_pose(),
        {
            IDX_R_HIP_ROLL: -0.04,
            IDX_R_ANKLE_ROLL: 0.03,
            IDX_L_HIP_ROLL: -0.03,
        },
    )


def swing_pose(t: float) -> np.ndarray:
    """Concurrent L knee lift + hip forward swing on weight-shifted lean base."""
    alpha = _smooth(t)
    knee = (1.0 - alpha) * (-0.10) + alpha * (-0.55)
    hip = (1.0 - alpha) * (-0.12) + alpha * (-0.42)
    ankle = (1.0 - alpha) * 0.0 + alpha * 0.22
    return _apply_joints(
        weight_shift_on_lean(),
        {
            IDX_L_KNEE: knee,
            IDX_L_HIP_PITCH: hip,
            IDX_L_ANKLE_P: ankle,
        },
    )


def catch_pose(t: float) -> np.ndarray:
    """Lower L foot while keeping it ahead — extend knee, moderate hip forward."""
    alpha = _smooth(t)
    knee = (1.0 - alpha) * (-0.55) + alpha * (-0.14)
    hip = (1.0 - alpha) * (-0.42) + alpha * (-0.30)
    ankle = (1.0 - alpha) * 0.22 + alpha * 0.08
    return _apply_joints(
        weight_shift_on_lean(),
        {
            IDX_L_KNEE: knee,
            IDX_L_HIP_PITCH: hip,
            IDX_L_ANKLE_P: ankle,
        },
    )


def catch_hold_pose() -> np.ndarray:
    return catch_pose(1.0)


@dataclass
class RunContext:
    ctrl: np.ndarray
    stand_l_y: float
    stand_r_xy: np.ndarray
    metrics: StepMetrics
    airborne_seen: bool = False
    airborne_start_l_y: float | None = None
    swing_started: bool = False
    catch_started: bool = False


def _step_sim(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctx: RunContext,
    phase: Phase,
    *,
    during_shift: bool = False,
) -> None:
    ctx.metrics.global_step += 1
    step = ctx.metrics.global_step

    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    l_contact = _foot_contact(model, data, "L")
    clearance_mm = (l_pos[2] - FLOOR_Z) * 1000.0
    l_fwd = _forward_mm(l_pos[1], ctx.stand_l_y)
    r_drift = float(np.linalg.norm(r_pos[:2] - ctx.stand_r_xy) * 1000.0)
    fwd_vel = _forward_vel_m_s(data)
    com_fwd_vel = fwd_vel

    ctx.metrics.peak_com_forward_vel_m_s = max(ctx.metrics.peak_com_forward_vel_m_s, com_fwd_vel)
    ctx.metrics.peak_torso_forward_vel_m_s = max(ctx.metrics.peak_torso_forward_vel_m_s, fwd_vel)
    ctx.metrics.max_l_clearance_mm = max(ctx.metrics.max_l_clearance_mm, clearance_mm)
    ctx.metrics.peak_l_forward_mm = max(ctx.metrics.peak_l_forward_mm, l_fwd)
    ctx.metrics.r_max_drift_mm = max(ctx.metrics.r_max_drift_mm, r_drift)
    ctx.metrics.final_l_rel_r_forward_mm = _forward_mm(l_pos[1], r_pos[1])

    if during_shift:
        r_load = _r_load_fraction(model, data)
        if not np.isnan(r_load):
            ctx.metrics.peak_r_load_during_shift = max(ctx.metrics.peak_r_load_during_shift, r_load)

    if not l_contact and not ctx.metrics.left_foot_airborne:
        ctx.metrics.left_foot_airborne = True
        ctx.metrics.contact_loss_step = step
        ctx.metrics.contact_loss_phase = phase.value
        ctx.airborne_seen = True
        ctx.airborne_start_l_y = float(l_pos[1])
        if abs(fwd_vel) > 0.01:
            ctx.metrics.fall_forward_sign = 1.0 if fwd_vel > 0 else -1.0

    if ctx.airborne_seen and l_contact and not ctx.metrics.left_touchdown_occurred:
        ctx.metrics.left_touchdown_occurred = True
        ctx.metrics.touchdown_step = step
        ctx.metrics.touchdown_l_rel_r_forward_mm = _forward_mm(l_pos[1], r_pos[1])
        ctx.metrics.touchdown_tilt_rad = env._quat_tilt_rad()
        ctx.metrics.touchdown_forward_vel_m_s = fwd_vel

    if ctx.airborne_seen and not l_contact and ctx.airborne_start_l_y is not None:
        delta_fwd = _forward_mm(l_pos[1], ctx.airborne_start_l_y)
        ctx.metrics.peak_l_forward_airborne_mm = max(
            ctx.metrics.peak_l_forward_airborne_mm,
            abs(delta_fwd),
        )
        if delta_fwd > 0:
            ctx.metrics.swing_forward_mm_while_airborne = max(
                ctx.metrics.swing_forward_mm_while_airborne, delta_fwd,
            )
        elif delta_fwd < 0:
            ctx.metrics.swing_backward_mm_while_airborne = max(
                ctx.metrics.swing_backward_mm_while_airborne, -delta_fwd,
            )


def _sync_viewer(viewer: mujoco.viewer.Handle | None, slow: bool) -> bool:
    if viewer is None:
        return True
    if not viewer.is_running():
        return False
    viewer.sync()
    time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)
    return True


def run_falling_forward_step(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> StepMetrics:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    ctrl = DEFAULT_POSE.copy()
    stand_l_y = float(_foot_pos(model, data, "L")[1])
    stand_r_xy = _foot_pos(model, data, "R")[:2].copy()
    ctx = RunContext(ctrl=ctrl, stand_l_y=stand_l_y, stand_r_xy=stand_r_xy, metrics=StepMetrics())
    lean = forward_lean_pose()
    shifted = weight_shift_on_lean()

    print(f"\n--- {Phase.STAND.value} ---")
    for s in range(STAND_STEPS):
        ctx.ctrl = _lerp_ctrl(ctx.ctrl, DEFAULT_POSE, _smooth((s + 1) / STAND_STEPS), cr)
        data.ctrl[:15] = ctx.ctrl
        mujoco.mj_step(model, data)
        _step_sim(env, model, data, ctx, Phase.STAND)
        if not _sync_viewer(viewer, slow):
            return ctx.metrics
    ctx.stand_l_y = float(_foot_pos(model, data, "L")[1])
    ctx.stand_r_xy = _foot_pos(model, data, "R")[:2].copy()
    print(
        f"  end: fwd_vel={_forward_vel_m_s(data):.3f} m/s "
        f"tilt={env._quat_tilt_rad():.3f}"
    )

    print(f"\n--- {Phase.FORWARD_FALL.value} ---")
    for s in range(FALL_RAMP_STEPS):
        alpha = (s + 1) / FALL_RAMP_STEPS
        ctx.ctrl = _lerp_ctrl(ctx.ctrl, lean, alpha, cr)
        data.ctrl[:15] = ctx.ctrl
        mujoco.mj_step(model, data)
        _step_sim(env, model, data, ctx, Phase.FORWARD_FALL)
        if not _sync_viewer(viewer, slow):
            return ctx.metrics
    for s in range(FALL_MOMENTUM_STEPS):
        ctx.ctrl = lean.copy()
        data.ctrl[:15] = ctx.ctrl
        mujoco.mj_step(model, data)
        _step_sim(env, model, data, ctx, Phase.FORWARD_FALL)
        if not _sync_viewer(viewer, slow):
            return ctx.metrics
    print(
        f"  end: fwd_vel={_forward_vel_m_s(data):.3f} m/s "
        f"peak_fwd_vel={ctx.metrics.peak_torso_forward_vel_m_s:.3f} m/s "
        f"tilt={env._quat_tilt_rad():.3f}"
    )

    print(f"\n--- {Phase.RAPID_SHIFT.value} ---")
    shift_base = ctx.ctrl.copy()
    for s in range(RAPID_SHIFT_STEPS):
        alpha = (s + 1) / RAPID_SHIFT_STEPS
        ctx.ctrl = _lerp_ctrl(shift_base, shifted, alpha, cr)
        data.ctrl[:15] = ctx.ctrl
        mujoco.mj_step(model, data)
        _step_sim(env, model, data, ctx, Phase.RAPID_SHIFT, during_shift=True)
        if ctx.metrics.left_foot_airborne and not ctx.swing_started:
            ctx.swing_started = True
            break
        if not _sync_viewer(viewer, slow):
            return ctx.metrics

    post_wait = 0
    while not ctx.swing_started and post_wait < POST_SHIFT_WAIT_STEPS:
        ctx.ctrl = shifted.copy()
        data.ctrl[:15] = ctx.ctrl
        mujoco.mj_step(model, data)
        _step_sim(env, model, data, ctx, Phase.RAPID_SHIFT, during_shift=True)
        post_wait += 1
        if ctx.metrics.left_foot_airborne:
            ctx.swing_started = True
            break
        if not _sync_viewer(viewer, slow):
            return ctx.metrics

    if not ctx.swing_started:
        ctx.swing_started = True

    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    print(
        f"  end: R_load_peak={ctx.metrics.peak_r_load_during_shift:.3f} "
        f"L_airborne={ctx.metrics.left_foot_airborne} "
        f"fwd_vel={_forward_vel_m_s(data):.3f} m/s "
        f"L-R_fwd={_forward_mm(l_pos[1], r_pos[1]):.1f}mm"
    )

    print(f"\n--- {Phase.SWING.value} ---")
    swing_base = ctx.ctrl.copy()
    for s in range(SWING_STEPS):
        t = (s + 1) / SWING_STEPS
        target = swing_pose(t)
        ctx.ctrl = _lerp_ctrl(swing_base if s == 0 else ctx.ctrl, target, 1.0, cr)
        data.ctrl[:15] = ctx.ctrl
        mujoco.mj_step(model, data)
        _step_sim(env, model, data, ctx, Phase.SWING)
        if not _sync_viewer(viewer, slow):
            return ctx.metrics
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    print(
        f"  end: clearance={(l_pos[2]-FLOOR_Z)*1000:.0f}mm "
        f"L_fwd={_forward_mm(l_pos[1], ctx.stand_l_y):.1f}mm "
        f"L-R_fwd={_forward_mm(l_pos[1], r_pos[1]):.1f}mm "
        f"airborne_fwd={ctx.metrics.swing_forward_mm_while_airborne:.1f}mm"
    )

    print(f"\n--- {Phase.CATCH.value} ---")
    catch_base = ctx.ctrl.copy()
    for s in range(CATCH_STEPS):
        t = (s + 1) / CATCH_STEPS
        target = catch_pose(t)
        ctx.ctrl = _lerp_ctrl(catch_base if s == 0 else ctx.ctrl, target, 1.0, cr)
        data.ctrl[:15] = ctx.ctrl
        mujoco.mj_step(model, data)
        _step_sim(env, model, data, ctx, Phase.CATCH)
        if not _sync_viewer(viewer, slow):
            return ctx.metrics
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    print(
        f"  end: L_contact={_foot_contact(model, data, 'L')} "
        f"L-R_fwd={_forward_mm(l_pos[1], r_pos[1]):.1f}mm "
        f"touchdown={ctx.metrics.left_touchdown_occurred}"
    )

    print(f"\n--- {Phase.STOP.value} ---")
    hold = catch_hold_pose()
    for s in range(STOP_STEPS):
        ctx.ctrl = _lerp_ctrl(ctx.ctrl, hold, _smooth((s + 1) / min(STOP_STEPS, 80)), cr)
        data.ctrl[:15] = ctx.ctrl
        mujoco.mj_step(model, data)
        _step_sim(env, model, data, ctx, Phase.STOP)
        if not _sync_viewer(viewer, slow):
            return ctx.metrics

    ctx.metrics.final_torso_tilt_rad = env._quat_tilt_rad()
    fwd = ctx.metrics.swing_forward_mm_while_airborne
    bwd = ctx.metrics.swing_backward_mm_while_airborne
    if ctx.metrics.left_foot_airborne:
        ctx.metrics.swing_same_direction_as_fall = (
            ctx.metrics.swing_forward_mm_while_airborne
            >= ctx.metrics.swing_backward_mm_while_airborne
        )

    return ctx.metrics


def _verdict_line(ok: bool, text: str) -> str:
    return f"  [{'YES' if ok else 'NO '}] {text}"


def print_summary(m: StepMetrics) -> None:
    print("\n" + "=" * 60)
    print("FALLING FORWARD STEP - CONTROLLED DYNAMIC CATCH")
    print("=" * 60)
    print(f"PEAK_COM_FORWARD_VEL_M_S = {m.peak_com_forward_vel_m_s:.4f}")
    print(f"PEAK_TORSO_FORWARD_VEL_M_S = {m.peak_torso_forward_vel_m_s:.4f}")
    print(f"L_CONTACT_LOSS_STEP = {m.contact_loss_step}")
    print(f"L_CONTACT_LOSS_PHASE = {m.contact_loss_phase}")
    print(f"MAX_L_CLEARANCE_MM = {m.max_l_clearance_mm:.1f}")
    print(f"PEAK_L_FORWARD_MM (from stand) = {m.peak_l_forward_mm:.1f}")
    print(f"PEAK_L_FORWARD_WHILE_AIRBORNE_MM = {m.peak_l_forward_airborne_mm:.1f}")
    print(f"SWING_FORWARD_MM_WHILE_AIRBORNE = {m.swing_forward_mm_while_airborne:.1f}")
    print(f"SWING_BACKWARD_MM_WHILE_AIRBORNE = {m.swing_backward_mm_while_airborne:.1f}")
    print(f"R_MAX_DRIFT_MM = {m.r_max_drift_mm:.1f}")
    print(f"PEAK_R_LOAD_DURING_SHIFT = {m.peak_r_load_during_shift:.3f}")
    print(f"L_TOUCHDOWN_OCCURRED = {m.left_touchdown_occurred}")
    if m.touchdown_l_rel_r_forward_mm is not None:
        print(f"L_TOUCHDOWN_REL_R_FORWARD_MM = {m.touchdown_l_rel_r_forward_mm:.1f}")
        print(f"TORSO_TILT_AT_TOUCHDOWN_RAD = {m.touchdown_tilt_rad:.3f}")
        print(f"FORWARD_VEL_AT_TOUCHDOWN_M_S = {m.touchdown_forward_vel_m_s:.4f}")
    print(f"FINAL_L_REL_R_FORWARD_MM = {m.final_l_rel_r_forward_mm:.1f}")
    print(f"FINAL_TORSO_TILT_RAD = {m.final_torso_tilt_rad:.3f}")

    if m.swing_same_direction_as_fall is None:
        swing_dir = "unknown (no airborne swing)"
    elif m.swing_same_direction_as_fall:
        swing_dir = "FORWARD (net displacement along world -Y)"
    else:
        swing_dir = "BACKWARD (net opposite fall axis)"
    print(f"\nL FOOT SWING DIRECTION: {swing_dir}")

    has_fwd_vel = m.peak_torso_forward_vel_m_s > 0.05
    airborne = m.left_foot_airborne
    swing_ok = m.swing_same_direction_as_fall is True
    ahead = (
        m.touchdown_l_rel_r_forward_mm is not None
        and m.touchdown_l_rel_r_forward_mm > 0.0
    )
    momentum_td = (
        m.touchdown_forward_vel_m_s is not None
        and m.touchdown_forward_vel_m_s > 0.02
    )

    print("\n--- SUCCESS CRITERIA (fall-forward catch) ---")
    print(_verdict_line(has_fwd_vel, f"measurable forward velocity ({m.peak_torso_forward_vel_m_s:.3f} m/s)"))
    print(_verdict_line(airborne, "L foot became airborne"))
    print(_verdict_line(swing_ok, "L leg swung in same direction as forward fall"))
    print(_verdict_line(ahead, "L foot landed ahead of R foot"))
    print(_verdict_line(momentum_td, "touchdown while still moving forward"))

    experiment_ok = has_fwd_vel and airborne and ahead
    print()
    if experiment_ok and swing_ok and momentum_td:
        print("VERDICT: SUCCESS - falling dynamics produced a forward catch step.")
    elif experiment_ok:
        print("VERDICT: PARTIAL - fall and catch occurred; see criteria above.")
    else:
        print("VERDICT: INCOMPLETE - controlled fall did not produce a clear catch step.")

    print("\nTrajectory: lean ramp 90 + momentum 200 -> rapid shift 40 -> swing/catch")
    print("Forward = world -Y. Diagnostic only - no parameter sweep.")


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
    p = argparse.ArgumentParser(description="Falling forward step — dynamic catch test.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Falling forward step: lean -> momentum -> rapid shift -> swing -> catch")
    print("Robot may fall. Viewer: robot only.\n")

    if args.headless:
        metrics = run_falling_forward_step(env, viewer=None, slow=False)
        print_summary(metrics)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        _reset(env.model, env.data)
        v.sync()
        metrics = run_falling_forward_step(env, viewer=v, slow=args.slow)
    print_summary(metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
