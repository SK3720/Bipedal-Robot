"""Staged forward catch — knee clearance then hip-only swing while falling.

STAND -> FORWARD FALL -> RAPID RIGHT SHIFT ->
state-dependent KNEE LIFT (clearance only) ->
state-dependent HIP SWING (knee held) ->
state-dependent CATCH -> STOP.

Based on airborne diagnostic: knee flexion cancels hip forward sweep.
Knee = clearance. Hip = forward swing. No stabilization.

Evaluation only — does not modify robot.xml, biped_env, or other scripts.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass, field
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
QPOS_L_HIP = 13
QPOS_L_KNEE = 14

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
FALL_RAMP_STEPS = 80
FALL_MOMENTUM_STEPS = 180
RAPID_SHIFT_STEPS = 40
POST_SHIFT_MAX_STEPS = 30

KNEE_LIFT_MAX_STEPS = 180
KNEE_LIFT_TARGET = -0.40
KNEE_LIFT_STEP = 0.006
L_NORMAL_LOSS_N = 2.0

HIP_SWING_TARGET = -0.50
HIP_RAMP_PER_STEP = 0.008
HIP_SWING_MAX_STEPS = 320
MIN_FWD_AIRBORNE_MM = 35.0

CATCH_MAX_STEPS = 260
STOP_STEPS = 80

L_KNEE_CATCH = -0.14
L_HIP_CATCH = -0.32
L_ANKLE_CATCH = 0.08

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018


class Phase(str, Enum):
    STAND = "STAND"
    FORWARD_FALL = "FORWARD FALL"
    RAPID_SHIFT = "RAPID RIGHT SHIFT"
    KNEE_LIFT = "LEFT KNEE LIFT"
    HIP_SWING = "LEFT HIP SWING"
    CATCH = "CATCH / TOUCHDOWN"
    STOP = "STOP"


@dataclass
class TrajSample:
    step: int
    phase: str
    l_foot_xyz: np.ndarray
    fwd_airborne_mm: float
    hip_cmd: float
    hip_qpos: float
    knee_cmd: float
    knee_qpos: float
    airborne: bool


@dataclass
class StepDiagnostics:
    global_step: int = 0
    contact_loss_step: int | None = None
    contact_loss_knee_cmd: float | None = None
    contact_loss_knee_qpos: float | None = None
    contact_loss_clearance_mm: float | None = None
    hip_swing_start_step: int | None = None
    catch_start_step: int | None = None
    clearance_knee_cmd: float | None = None
    peak_clearance_mm: float = 0.0
    peak_fwd_airborne_mm: float = 0.0
    peak_l_foot_fwd_vel_m_s: float = 0.0
    peak_torso_fwd_vel_m_s: float = 0.0
    peak_torso_tilt_rad: float = 0.0
    r_max_drift_mm: float = 0.0
    hip_cmd_at_peak_fwd: float | None = None
    hip_qpos_at_peak_fwd: float | None = None
    first_touchdown_step: int | None = None
    touchdown_l_rel_r_mm: float | None = None
    touchdown_torso_fwd_vel_m_s: float | None = None
    touchdown_torso_tilt_rad: float | None = None
    touchdown_while_moving_forward: bool | None = None
    swing_traj: list[TrajSample] = field(default_factory=list)


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _forward_mm(y: float, y_ref: float) -> float:
    return float(-(y - y_ref) * 1000.0)


def _forward_vel(data: mujoco.MjData) -> float:
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
    return _apply_joints(
        DEFAULT_POSE,
        {
            IDX_L_HIP_PITCH: -0.14,
            IDX_R_HIP_PITCH: 0.14,
            IDX_L_KNEE: -0.06,
            IDX_R_KNEE: -0.06,
            IDX_L_ANKLE_P: -0.12,
            IDX_R_ANKLE_P: -0.12,
        },
    )


def rapid_shift_pose() -> np.ndarray:
    return _apply_joints(
        forward_lean_pose(),
        {
            IDX_R_HIP_ROLL: -0.04,
            IDX_R_ANKLE_ROLL: 0.03,
            IDX_L_HIP_ROLL: -0.03,
        },
    )


def left_leg_pose(knee: float, hip: float, ankle: float = 0.0) -> np.ndarray:
    return _apply_joints(
        rapid_shift_pose(),
        {IDX_L_KNEE: knee, IDX_L_HIP_PITCH: hip, IDX_L_ANKLE_P: ankle},
    )


def _print_phase(name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"PHASE: {name}")
    print(f"{'=' * 60}")


def _sync_viewer(viewer: mujoco.viewer.Handle | None, slow: bool) -> bool:
    if viewer is None:
        return True
    if not viewer.is_running():
        return False
    viewer.sync()
    time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)
    return True


@dataclass
class RunState:
    ctrl: np.ndarray
    phase: Phase
    stand_r_xy: np.ndarray
    diag: StepDiagnostics
    knee_cmd: float
    hip_cmd: float
    clearance_knee: float | None = None
    airborne_ref_y: float | None = None
    airborne_seen: bool = False
    hip_swing_steps: int = 0
    knee_lift_steps: int = 0
    catch_steps: int = 0
    stop_steps: int = 0


def _sim_step(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    st: RunState,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
) -> None:
    data.ctrl[:15] = st.ctrl
    mujoco.mj_step(model, data)
    st.diag.global_step += 1

    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    l_contact = _foot_contact(model, data, "L")
    l_nf = _foot_normal_force(model, data, "L")
    clearance = (l_pos[2] - FLOOR_Z) * 1000.0
    tilt = env._quat_tilt_rad()
    torso_fwd = _forward_vel(data)
    r_drift = float(np.linalg.norm(r_pos[:2] - st.stand_r_xy) * 1000.0)

    st.diag.peak_clearance_mm = max(st.diag.peak_clearance_mm, clearance)
    st.diag.peak_torso_fwd_vel_m_s = max(st.diag.peak_torso_fwd_vel_m_s, torso_fwd)
    st.diag.peak_torso_tilt_rad = max(st.diag.peak_torso_tilt_rad, tilt)
    st.diag.r_max_drift_mm = max(st.diag.r_max_drift_mm, r_drift)

    if st.airborne_ref_y is not None and not l_contact:
        fwd_air = _forward_mm(l_pos[1], st.airborne_ref_y)
        if fwd_air > st.diag.peak_fwd_airborne_mm:
            st.diag.peak_fwd_airborne_mm = fwd_air
            st.diag.hip_cmd_at_peak_fwd = float(data.ctrl[IDX_L_HIP_PITCH])
            st.diag.hip_qpos_at_peak_fwd = float(data.qpos[QPOS_L_HIP])
        l_body = model.body("L_foot").id
        foot_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, l_body, foot_vel, 0)
        foot_fwd_vel = float(-foot_vel[1])
        st.diag.peak_l_foot_fwd_vel_m_s = max(st.diag.peak_l_foot_fwd_vel_m_s, foot_fwd_vel)

    if st.phase in (Phase.HIP_SWING, Phase.CATCH):
        if st.diag.global_step % 20 == 0:
            fwd_a = 0.0
            if st.airborne_ref_y is not None:
                fwd_a = _forward_mm(l_pos[1], st.airborne_ref_y)
            st.diag.swing_traj.append(
                TrajSample(
                    step=st.diag.global_step,
                    phase=st.phase.value,
                    l_foot_xyz=l_pos.copy(),
                    fwd_airborne_mm=fwd_a,
                    hip_cmd=float(data.ctrl[IDX_L_HIP_PITCH]),
                    hip_qpos=float(data.qpos[QPOS_L_HIP]),
                    knee_cmd=float(data.ctrl[IDX_L_KNEE]),
                    knee_qpos=float(data.qpos[QPOS_L_KNEE]),
                    airborne=not l_contact,
                )
            )

    if not l_contact and st.diag.contact_loss_step is None:
        st.diag.contact_loss_step = st.diag.global_step
        st.diag.contact_loss_knee_cmd = float(data.ctrl[IDX_L_KNEE])
        st.diag.contact_loss_knee_qpos = float(data.qpos[QPOS_L_KNEE])
        st.diag.contact_loss_clearance_mm = clearance
        st.airborne_ref_y = float(l_pos[1])
        st.airborne_seen = True

    if st.airborne_seen and l_contact and st.diag.first_touchdown_step is None:
        st.diag.first_touchdown_step = st.diag.global_step
        st.diag.touchdown_l_rel_r_mm = _forward_mm(l_pos[1], r_pos[1])
        st.diag.touchdown_torso_fwd_vel_m_s = torso_fwd
        st.diag.touchdown_torso_tilt_rad = tilt
        st.diag.touchdown_while_moving_forward = torso_fwd > 0.02

    _sync_viewer(viewer, slow)


def _is_unloaded(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return (not _foot_contact(model, data, "L")) or (
        _foot_normal_force(model, data, "L") < L_NORMAL_LOSS_N
    )


def run_staged_forward_catch(
    env: BipedalWalkEnv,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
) -> StepDiagnostics:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    shifted = rapid_shift_pose()
    lean = forward_lean_pose()
    ctrl = DEFAULT_POSE.copy()
    stand_r_xy = _foot_pos(model, data, "R")[:2].copy()
    diag = StepDiagnostics()
    st = RunState(
        ctrl=ctrl,
        phase=Phase.STAND,
        stand_r_xy=stand_r_xy,
        diag=diag,
        knee_cmd=float(shifted[IDX_L_KNEE]),
        hip_cmd=float(shifted[IDX_L_HIP_PITCH]),
    )

    _print_phase(Phase.STAND.value)
    for s in range(STAND_STEPS):
        st.ctrl = _lerp_ctrl(st.ctrl, DEFAULT_POSE, _smooth((s + 1) / STAND_STEPS), cr)
        _sim_step(env, model, data, st, viewer, slow)
    st.stand_r_xy = _foot_pos(model, data, "R")[:2].copy()

    _print_phase(Phase.FORWARD_FALL.value)
    for s in range(FALL_RAMP_STEPS):
        alpha = (s + 1) / FALL_RAMP_STEPS
        st.ctrl = _lerp_ctrl(st.ctrl, lean, alpha, cr)
        _sim_step(env, model, data, st, viewer, slow)
    for _ in range(FALL_MOMENTUM_STEPS):
        st.ctrl = lean.copy()
        _sim_step(env, model, data, st, viewer, slow)

    _print_phase(Phase.RAPID_SHIFT.value)
    shift_start = st.ctrl.copy()
    for s in range(RAPID_SHIFT_STEPS):
        alpha = (s + 1) / RAPID_SHIFT_STEPS
        st.ctrl = _lerp_ctrl(shift_start, shifted, alpha, cr)
        st.knee_cmd = float(st.ctrl[IDX_L_KNEE])
        st.hip_cmd = float(st.ctrl[IDX_L_HIP_PITCH])
        _sim_step(env, model, data, st, viewer, slow)

    post = 0
    while post < POST_SHIFT_MAX_STEPS and st.phase == Phase.RAPID_SHIFT:
        st.ctrl = shifted.copy()
        st.knee_cmd = float(st.ctrl[IDX_L_KNEE])
        st.hip_cmd = float(st.ctrl[IDX_L_HIP_PITCH])
        _sim_step(env, model, data, st, viewer, slow)
        post += 1

    _print_phase(Phase.KNEE_LIFT.value)
    st.phase = Phase.KNEE_LIFT
    while st.knee_lift_steps < KNEE_LIFT_MAX_STEPS and st.phase == Phase.KNEE_LIFT:
        if _is_unloaded(model, data):
            st.clearance_knee = st.knee_cmd
            st.diag.clearance_knee_cmd = st.knee_cmd
            st.phase = Phase.HIP_SWING
            st.diag.hip_swing_start_step = st.diag.global_step + 1
            break
        st.knee_cmd = max(st.knee_cmd - KNEE_LIFT_STEP, KNEE_LIFT_TARGET)
        st.ctrl = left_leg_pose(st.knee_cmd, st.hip_cmd)
        _sim_step(env, model, data, st, viewer, slow)
        st.knee_lift_steps += 1

    if st.phase == Phase.KNEE_LIFT:
        st.clearance_knee = st.knee_cmd
        st.diag.clearance_knee_cmd = st.knee_cmd
        st.phase = Phase.HIP_SWING
        st.diag.hip_swing_start_step = st.diag.global_step + 1

    _print_phase(Phase.HIP_SWING.value)
    while st.hip_swing_steps < HIP_SWING_MAX_STEPS and st.phase == Phase.HIP_SWING:
        if st.airborne_ref_y is None and not _foot_contact(model, data, "L"):
            st.airborne_ref_y = float(_foot_pos(model, data, "L")[1])

        knee_hold = st.clearance_knee if st.clearance_knee is not None else st.knee_cmd
        st.hip_cmd = max(st.hip_cmd - HIP_RAMP_PER_STEP, HIP_SWING_TARGET)
        st.ctrl = left_leg_pose(knee_hold, st.hip_cmd)
        _sim_step(env, model, data, st, viewer, slow)
        st.hip_swing_steps += 1

        fwd_air = 0.0
        if st.airborne_ref_y is not None:
            fwd_air = _forward_mm(_foot_pos(model, data, "L")[1], st.airborne_ref_y)

        hip_at_target = st.hip_cmd <= HIP_SWING_TARGET + 1e-6
        if fwd_air >= MIN_FWD_AIRBORNE_MM and not _foot_contact(model, data, "L"):
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

    _print_phase(Phase.CATCH.value)
    catch_start_ctrl = st.ctrl.copy()
    catch_target = left_leg_pose(L_KNEE_CATCH, L_HIP_CATCH, L_ANKLE_CATCH)
    while st.catch_steps < CATCH_MAX_STEPS and st.phase == Phase.CATCH:
        alpha = _smooth((st.catch_steps + 1) / min(CATCH_MAX_STEPS, 120))
        st.ctrl = _lerp_ctrl(catch_start_ctrl, catch_target, alpha, cr)
        _sim_step(env, model, data, st, viewer, slow)
        st.catch_steps += 1
        if st.diag.first_touchdown_step is not None and st.catch_steps > 40:
            break

    _print_phase(Phase.STOP.value)
    st.phase = Phase.STOP
    hold = catch_target.copy()
    for _ in range(STOP_STEPS):
        st.ctrl = hold.copy()
        _sim_step(env, model, data, st, viewer, slow)

    return diag


def print_summary(d: StepDiagnostics) -> None:
    print("\n" + "=" * 60)
    print("STAGED FORWARD CATCH - DIAGNOSTICS")
    print("=" * 60)
    print(f"L_CONTACT_LOSS_STEP = {d.contact_loss_step}")
    print(f"L_KNEE_CMD_AT_CONTACT_LOSS = {d.contact_loss_knee_cmd}")
    print(f"L_KNEE_QPOS_AT_CONTACT_LOSS = {d.contact_loss_knee_qpos}")
    print(f"L_CLEARANCE_AT_CONTACT_LOSS_MM = {d.contact_loss_clearance_mm}")
    print(f"CLEARANCE_KNEE_CMD_HELD = {d.clearance_knee_cmd}")
    print(f"HIP_SWING_START_STEP = {d.hip_swing_start_step}")
    print(f"CATCH_START_STEP = {d.catch_start_step}")
    print(f"PEAK_L_CLEARANCE_MM = {d.peak_clearance_mm:.1f}")
    print(f"PEAK_FWD_DISPLACEMENT_WHILE_AIRBORNE_MM = {d.peak_fwd_airborne_mm:.1f}")
    print(f"HIP_CMD_AT_PEAK_FWD = {d.hip_cmd_at_peak_fwd}")
    print(f"HIP_QPOS_AT_PEAK_FWD = {d.hip_qpos_at_peak_fwd}")
    print(f"PEAK_L_FOOT_FORWARD_VEL_M_S = {d.peak_l_foot_fwd_vel_m_s:.4f}")
    print(f"PEAK_TORSO_FORWARD_VEL_M_S = {d.peak_torso_fwd_vel_m_s:.4f}")
    print(f"PEAK_TORSO_TILT_RAD = {d.peak_torso_tilt_rad:.3f}")
    print(f"R_MAX_DRIFT_MM = {d.r_max_drift_mm:.1f}")
    print(f"FIRST_L_TOUCHDOWN_STEP = {d.first_touchdown_step}")
    print(f"L_REL_R_FORWARD_AT_TOUCHDOWN_MM = {d.touchdown_l_rel_r_mm}")
    print(f"TORSO_FORWARD_VEL_AT_TOUCHDOWN_M_S = {d.touchdown_torso_fwd_vel_m_s}")
    print(f"TORSO_TILT_AT_TOUCHDOWN_RAD = {d.touchdown_torso_tilt_rad}")
    print(f"TOUCHDOWN_WHILE_MOVING_FORWARD = {d.touchdown_while_moving_forward}")

    if d.swing_traj:
        print("\n  Foot trajectory (HIP SWING + CATCH, every 20 steps):")
        print("  step   phase                    fwd_air  hip_c  hip_q  knee_c  knee_q  L_foot_Y")
        for s in d.swing_traj:
            print(
                f"  {s.step:5d}  {s.phase:<24s} {s.fwd_airborne_mm:7.1f} "
                f"{s.hip_cmd:5.2f} {s.hip_qpos:5.2f} {s.knee_cmd:5.2f} {s.knee_qpos:5.2f} "
                f"{s.l_foot_xyz[1]:.4f}"
            )

    print("\nPhases: STAND -> FORWARD FALL -> RAPID RIGHT SHIFT ->")
    print("        KNEE LIFT (clearance) -> HIP SWING (knee held) -> CATCH -> STOP")
    if d.peak_fwd_airborne_mm >= MIN_FWD_AIRBORNE_MM:
        print(f"Forward swing reproduced HIP-ONLY scale ({d.peak_fwd_airborne_mm:.0f} mm airborne).")
    else:
        print(
            f"WARNING: peak airborne forward = {d.peak_fwd_airborne_mm:.1f} mm "
            f"(target >= {MIN_FWD_AIRBORNE_MM:.0f} mm). See trajectory above."
        )
    print("Single fixed experiment - no auto-tuning.")


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
    p = argparse.ArgumentParser(description="Staged forward catch stepping experiment.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()

    print("Staged forward catch: knee clearance -> hip-only swing -> touchdown")
    print(f"Knee lift to ~{KNEE_LIFT_TARGET:.2f} rad, hip swing to {HIP_SWING_TARGET:.2f} rad")
    print("State-dependent transitions. Viewer: robot only.\n")

    if args.headless:
        diag = run_staged_forward_catch(env, viewer=None, slow=False)
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
        diag = run_staged_forward_catch(env, viewer=v, slow=args.slow)
    print_summary(diag)


if __name__ == "__main__":
    main(sys.argv[1:])
