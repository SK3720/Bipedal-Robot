"""Sagittal step-catching feasibility test on the ORIGINAL robot (no geometry changes).

Push-disturbance followed by scripted hip/knee/ankle sagittal step trajectories.
Evaluation only — does not modify biped_env, robot.xml, or PPO artifacts.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np
from ctypes import wintypes

from biped_env import (
    BipedalWalkEnv,
    CHEST_Z_CONTACT,
    DEFAULT_POSE,
    PUSH_DURATION_STEPS,
    SETTLE_STEPS,
    STANDING_QUAT,
)

FLOOR_Z = 1.0
FOOT_CONTACT_Z = 1.042
MIN_CLEARANCE_M = 0.025
MIN_DISPLACEMENT_M = 0.05
MAX_UPRIGHT_TILT = 0.45
POST_STEP_HOLD_STEPS = 500

STAND_MS = 200
PUSH_MS = PUSH_DURATION_STEPS
REACT_MS = 30
WEIGHT_SHIFT_MS = 250
UNLOAD_MS = 50
PLACE_MS = 120

PUSH_MAGNITUDES_N = [20, 40, 60, 80, 100]

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.015

# Actuator indices (ctrl[:15])
IDX_L_HIP_ROLL, IDX_L_HIP_PITCH, IDX_L_KNEE, IDX_L_ANKLE_P = 5, 6, 7, 8
IDX_R_HIP_ROLL, IDX_R_HIP_PITCH, IDX_R_KNEE, IDX_R_ANKLE_P = 10, 11, 12, 13

# Verified: +ctrl[L_HIP_PITCH] moves L foot -X; -ctrl moves +X (forward).
# Verified: +ctrl[R_HIP_PITCH] moves R foot -X; -ctrl moves +X (forward).
IDX_L_ANKLE_R = 9
IDX_R_ANKLE_R = 14


class Phase(Enum):
    RESET_SETTLE = "RESET / SETTLE"
    WAIT_STANDING = "WAIT / STANDING"
    DISTURBANCE = "DISTURBANCE"
    WEIGHT_SHIFT = "WEIGHT SHIFT"
    SWING_LIFT = "SWING FOOT LIFT"
    SWING_MOVE = "SWING FORWARD/BACK"
    FOOT_PLACE = "FOOT PLACEMENT"
    STABILIZATION = "DOUBLE-SUPPORT STABILIZATION"
    HOLD = "HOLD"


@dataclass(frozen=True)
class TrajectoryParams:
    name: str
    swing_leg: str
    hip_pitch: float
    knee: float
    ankle: float
    ws_stance_roll: float
    swing_ms: int
    stance_ankle_roll: float = 0.0
    stabilize_ms: int = 200
    place_hip_scale: float = 0.75
    place_knee_scale: float = 0.70


@dataclass
class SagittalCase:
    direction: str
    push_force_n: float
    swing_leg: str
    traj: TrajectoryParams
    push_axis_sign: float
    step_axis_sign: float


@dataclass
class TrialMetrics:
    case_name: str
    direction: str
    push_force_n: float
    swing_leg: str
    success: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    push_world_axis: str = "+X"
    step_world_axis: str = "+X"
    peak_clearance_m: float = 0.0
    peak_displacement_m: float = 0.0
    max_torso_tilt_rad: float = 0.0
    max_tilt_during_push: float = 0.0
    max_tilt_pre_swing: float = 0.0
    tilt_at_lift: float = 0.0
    tilt_at_landing: float | None = None
    max_tilt_after_landing: float = 0.0
    final_tilt_rad: float = 0.0
    max_torso_angvel: float = 0.0
    stance_ok_swing: bool = True
    lost_contact: bool = False
    regained_contact: bool = False
    landing_step: int | None = None
    total_steps: int = 0
    phase_summaries: list[dict] = field(default_factory=list)


def _smooth(alpha: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(alpha, 0.0, 1.0)))


def _ms_to_steps(ms: int) -> int:
    return max(1, int(ms))


def _foot_state(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[np.ndarray, int]:
    body_id = model.body(f"{side}_foot").id
    pos = data.xpos[body_id].copy()
    contacts = 0
    for ci in range(data.ncon):
        contact = data.contact[ci]
        for geom_id in (contact.geom1, contact.geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if f"{side}_foot_collision" in name:
                contacts += 1
                break
    return pos, contacts


def _leg_indices(swing_leg: str) -> tuple[int, int, int, int]:
    if swing_leg == "L":
        return IDX_L_HIP_ROLL, IDX_L_HIP_PITCH, IDX_L_KNEE, IDX_L_ANKLE_P
    return IDX_R_HIP_ROLL, IDX_R_HIP_PITCH, IDX_R_KNEE, IDX_R_ANKLE_P


def _stance_leg(swing_leg: str) -> str:
    return "R" if swing_leg == "L" else "L"


def _make_poses(model: mujoco.MjModel, case: SagittalCase) -> dict[str, np.ndarray]:
    cr = model.actuator_ctrlrange[:15]
    swing = case.swing_leg
    stance = _stance_leg(swing)
    _, hip_i, knee_i, ankle_i = _leg_indices(swing)
    stance_roll_i = IDX_R_HIP_ROLL if stance == "R" else IDX_L_HIP_ROLL
    stance_ankle_i = IDX_R_ANKLE_R if stance == "R" else IDX_L_ANKLE_R

    stand = DEFAULT_POSE.copy()
    ws = stand.copy()
    ws[stance_roll_i] = case.traj.ws_stance_roll
    ws[stance_ankle_i] = case.traj.stance_ankle_roll

    unload = ws.copy()
    unload[knee_i] = case.traj.knee * 0.45
    unload[hip_i] = case.traj.hip_pitch * 0.25

    apex = ws.copy()
    apex[hip_i] = np.clip(case.traj.hip_pitch, cr[hip_i, 0], cr[hip_i, 1])
    apex[knee_i] = np.clip(case.traj.knee, cr[knee_i, 0], cr[knee_i, 1])
    apex[ankle_i] = np.clip(case.traj.ankle, cr[ankle_i, 0], cr[ankle_i, 1])

    place = ws.copy()
    place[hip_i] = np.clip(case.traj.hip_pitch * case.traj.place_hip_scale, cr[hip_i, 0], cr[hip_i, 1])
    place[knee_i] = np.clip(case.traj.knee * case.traj.place_knee_scale, cr[knee_i, 0], cr[knee_i, 1])
    place[ankle_i] = np.clip(case.traj.ankle * 0.85, cr[ankle_i, 0], cr[ankle_i, 1])

    return {"stand": stand, "ws": ws, "unload": unload, "apex": apex, "place": place}


def _reset_sim(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = DEFAULT_POSE
    data.qvel[:] = 0.0
    data.ctrl[:15] = DEFAULT_POSE
    for _ in range(SETTLE_STEPS):
        data.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_step(model, data)


def verify_world_frame(env: BipedalWalkEnv) -> dict[str, str]:
    """Confirm push and step directions from physics, not naming convention."""
    model, data = env.model, env.data
    chest = env.chest_body_id
    _reset_sim(model, data)
    x0 = float(data.qpos[0])

    for fx, label in ((80.0, "+X_force"), (-80.0, "-X_force")):
        _reset_sim(model, data)
        x0 = float(data.qpos[0])
        for _ in range(PUSH_DURATION_STEPS):
            data.xfrc_applied[chest, :3] = [fx, 0.0, 0.0]
            data.ctrl[:15] = DEFAULT_POSE
            mujoco.mj_step(model, data)
        data.xfrc_applied[chest, :3] = 0.0
        for _ in range(100):
            data.ctrl[:15] = DEFAULT_POSE
            mujoco.mj_step(model, data)
        dx = (data.qpos[0] - x0) * 1000.0
        print(f"  Frame check {label}: chest drifts {dx:+.1f} mm in world X after 100 ms")

    _reset_sim(model, data)
    lf0 = float(data.xpos[model.body("L_foot").id, 0])
    c = DEFAULT_POSE.copy()
    c[IDX_L_HIP_PITCH] = -0.18
    for _ in range(250):
        data.ctrl[:15] = c
        mujoco.mj_step(model, data)
    lf1 = float(data.xpos[model.body("L_foot").id, 0])
    print(f"  L swing (-hip_pitch ctrl): foot moves {(lf1-lf0)*1000:+.1f} mm in world X")

    return {
        "forward_push": "+X",
        "forward_step": "+X",
        "backward_push": "-X",
        "backward_step": "-X",
    }


def _run_trial(
    env: BipedalWalkEnv,
    case: SagittalCase,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    global_step: list[int] | None = None,
    slow_window: tuple[int, int] | None = None,
) -> TrialMetrics:
    model = env.model
    data = env.data
    chest = env.chest_body_id
    poses = _make_poses(model, case)
    swing = case.swing_leg
    stance = _stance_leg(swing)
    metrics = TrialMetrics(
        case_name=f"{case.direction}_{case.push_force_n:.0f}N_{swing}_{case.traj.name}",
        direction=case.direction,
        push_force_n=case.push_force_n,
        swing_leg=swing,
        push_world_axis="+X" if case.push_axis_sign > 0 else "-X",
        step_world_axis="+X" if case.step_axis_sign > 0 else "-X",
    )

    _reset_sim(model, data)
    swing_init, swing_init_c = _foot_state(model, data, swing)

    ctrl = DEFAULT_POSE.copy()
    step_counter = global_step if global_step is not None else [0]
    slow_lo, slow_hi = slow_window or (0, 10**9)
    cr = model.actuator_ctrlrange[:15]

    push_force = case.push_axis_sign * case.push_force_n
    segments: list[tuple[Phase, np.ndarray, int, float | None]] = [
        (Phase.WAIT_STANDING, poses["stand"], _ms_to_steps(STAND_MS), None),
        (Phase.DISTURBANCE, poses["stand"], _ms_to_steps(PUSH_MS), push_force),
        (Phase.WEIGHT_SHIFT, poses["ws"], _ms_to_steps(REACT_MS + WEIGHT_SHIFT_MS), None),
        (Phase.SWING_LIFT, poses["unload"], _ms_to_steps(UNLOAD_MS), None),
        (Phase.SWING_MOVE, poses["apex"], _ms_to_steps(case.traj.swing_ms), None),
        (Phase.FOOT_PLACE, poses["place"], _ms_to_steps(PLACE_MS), None),
        (Phase.STABILIZATION, poses["place"], _ms_to_steps(case.traj.stabilize_ms), None),
        (Phase.HOLD, poses["place"], POST_STEP_HOLD_STEPS, None),
    ]

    lift_started = False
    landing_seen = False
    after_landing = False
    max_tilt_push = 0.0

    for phase, target, n_steps, phase_push in segments:
        for s in range(n_steps):
            alpha = _smooth((s + 1) / n_steps)
            ctrl = (1.0 - alpha) * ctrl + alpha * target
            ctrl = np.clip(ctrl, cr[:, 0], cr[:, 1])
            data.ctrl[:15] = ctrl
            if phase_push is not None:
                data.xfrc_applied[chest, :3] = [phase_push, 0.0, 0.0]
            else:
                data.xfrc_applied[chest, :3] = 0.0
            mujoco.mj_step(model, data)

            swing_pos, swing_c = _foot_state(model, data, swing)
            _, stance_c = _foot_state(model, data, stance)
            tilt = env._quat_tilt_rad()
            angvel = float(np.linalg.norm(data.qvel[3:6]))
            clearance = swing_pos[2] - FLOOR_Z
            disp = (swing_pos[0] - swing_init[0]) * case.step_axis_sign

            if phase == Phase.DISTURBANCE:
                max_tilt_push = max(max_tilt_push, tilt)
            if phase in (Phase.WAIT_STANDING, Phase.DISTURBANCE, Phase.WEIGHT_SHIFT):
                metrics.max_tilt_pre_swing = max(metrics.max_tilt_pre_swing, tilt)
            if phase in (Phase.SWING_LIFT, Phase.SWING_MOVE) and not lift_started:
                if swing_c == 0 and clearance > 0.01:
                    lift_started = True
                    metrics.tilt_at_lift = tilt
            if swing_c == 0 and swing_pos[2] > FOOT_CONTACT_Z:
                metrics.lost_contact = True
            if clearance >= MIN_CLEARANCE_M:
                metrics.peak_clearance_m = max(metrics.peak_clearance_m, clearance)
            if disp >= MIN_DISPLACEMENT_M:
                metrics.peak_displacement_m = max(metrics.peak_displacement_m, disp)
            if stance_c < 1 and phase in (Phase.SWING_LIFT, Phase.SWING_MOVE):
                metrics.stance_ok_swing = False

            metrics.max_torso_angvel = max(metrics.max_torso_angvel, angvel)
            if not after_landing:
                metrics.max_torso_tilt_rad = max(metrics.max_torso_tilt_rad, tilt)
            else:
                metrics.max_tilt_after_landing = max(metrics.max_tilt_after_landing, tilt)

            if (
                not landing_seen
                and metrics.lost_contact
                and swing_c > 0
                and phase in (Phase.FOOT_PLACE, Phase.STABILIZATION, Phase.HOLD)
            ):
                landing_seen = True
                after_landing = True
                metrics.regained_contact = True
                metrics.landing_step = step_counter[0]
                metrics.tilt_at_landing = tilt

            if viewer is not None and viewer.is_running():
                viewer.sync()
                if slow and slow_lo <= step_counter[0] <= slow_hi:
                    time.sleep(SLOW_SLEEP_S)
                else:
                    time.sleep(NORMAL_SLEEP_S)
            step_counter[0] += 1

    metrics.max_tilt_during_push = max_tilt_push
    metrics.total_steps = step_counter[0]
    final_swing_c = _foot_state(model, data, swing)[1]
    if metrics.lost_contact and final_swing_c > 0:
        metrics.regained_contact = True
    metrics.final_tilt_rad = env._quat_tilt_rad()

    reasons: list[str] = []
    if not metrics.lost_contact:
        reasons.append("swing foot never lost contact")
    if metrics.peak_clearance_m < MIN_CLEARANCE_M:
        reasons.append(f"peak clearance {metrics.peak_clearance_m:.3f} m < {MIN_CLEARANCE_M:.3f} m")
    if metrics.peak_displacement_m < MIN_DISPLACEMENT_M:
        reasons.append(f"peak displacement {metrics.peak_displacement_m:.3f} m < {MIN_DISPLACEMENT_M:.3f} m")
    if not metrics.regained_contact:
        reasons.append("swing foot did not regain contact")
    if metrics.final_tilt_rad >= MAX_UPRIGHT_TILT:
        reasons.append(f"final tilt {metrics.final_tilt_rad:.3f} rad >= {MAX_UPRIGHT_TILT:.3f} rad")
    if not metrics.stance_ok_swing:
        reasons.append("stance foot lost contact during swing")
    if metrics.max_tilt_pre_swing >= MAX_UPRIGHT_TILT:
        reasons.append(
            f"pre-swing instability (tilt {metrics.max_tilt_pre_swing:.3f} rad >= {MAX_UPRIGHT_TILT:.3f} rad)"
        )

    metrics.success = not reasons
    metrics.failure_reasons = reasons
    return metrics


def _narrow_grid(direction: str) -> list[TrajectoryParams]:
    """Physics-informed subset — not full Cartesian product."""
    if direction == "forward":
        return [
            TrajectoryParams("L_base", "L", -0.18, -0.28, 0.25, -0.183, 150),
            TrajectoryParams("L_fast", "L", -0.22, -0.35, 0.30, -0.183, 100),
            TrajectoryParams("L_slow", "L", -0.15, -0.22, 0.20, -0.183, 200),
            TrajectoryParams("L_shallow", "L", -0.12, -0.18, 0.18, -0.15, 150),
            TrajectoryParams("L_deep", "L", -0.24, -0.38, 0.35, -0.183, 120),
            TrajectoryParams("L_light_ws", "L", -0.18, -0.28, 0.25, -0.12, 150),
            TrajectoryParams("L_fast_light_ws", "L", -0.22, -0.35, 0.30, -0.12, 100),
            TrajectoryParams("R_base", "R", -0.18, -0.28, 0.25, 0.15, 150),
            TrajectoryParams("R_fast", "R", -0.22, -0.35, 0.30, 0.18, 100),
        ]
    return [
        TrajectoryParams("L_back_base", "L", 0.18, -0.25, -0.20, -0.183, 150),
        TrajectoryParams("L_back_fast", "L", 0.22, -0.32, -0.25, -0.183, 100),
        TrajectoryParams("R_back_base", "R", 0.18, -0.25, -0.20, 0.15, 150),
        TrajectoryParams("R_back_fast", "R", 0.22, -0.32, -0.25, 0.18, 100),
        TrajectoryParams("R_back_light_ws", "R", 0.18, -0.25, -0.20, 0.12, 150),
        TrajectoryParams("L_back_deep", "L", 0.24, -0.35, -0.28, -0.183, 120),
    ]


def _build_cases(direction: str, push_forces: list[float], trajs: list[TrajectoryParams]) -> list[SagittalCase]:
    if direction == "forward":
        push_sign, step_sign = 1.0, 1.0
    else:
        push_sign, step_sign = -1.0, -1.0
    return [
        SagittalCase(direction, push_n, traj.swing_leg, traj, push_sign, step_sign)
        for push_n, traj in product(push_forces, trajs)
    ]


def _print_trial(m: TrialMetrics) -> None:
    print(f"\n--- {m.case_name} ---")
    print(f"  push: {m.push_force_n:.0f} N {m.push_world_axis} | step: {m.step_world_axis} | swing: {m.swing_leg}")
    print(f"  clearance: {m.peak_clearance_m:.3f} m | displacement: {m.peak_displacement_m:.3f} m")
    print(f"  tilt push: {m.max_tilt_during_push:.3f} | at lift: {m.tilt_at_lift:.3f} | "
          f"at land: {m.tilt_at_landing} | max after: {m.max_tilt_after_landing:.3f} | final: {m.final_tilt_rad:.3f}")
    print(f"  max angvel: {m.max_torso_angvel:.3f} | stance_ok: {m.stance_ok_swing} | "
          f"lost/regain: {m.lost_contact}/{m.regained_contact} | steps: {m.total_steps}")
    print(f"  RESULT: {'SUCCESS' if m.success else 'FAILED'}")
    for r in m.failure_reasons:
        print(f"    - {r}")


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

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                    ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]

    user32 = ctypes.windll.user32
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 1), ctypes.byref(info))
    wa = info.rcWork
    x = wa.left + (wa.right - wa.left - VIEWER_WIDTH) // 2
    y = wa.top + (wa.bottom - wa.top - VIEWER_HEIGHT) // 2
    user32.SetWindowPos(hwnd, 0, x, y, VIEWER_WIDTH, VIEWER_HEIGHT, 0x0004)


def run_search(direction: str, headless: bool = True, push_forces: list[float] | None = None) -> list[TrialMetrics]:
    env = BipedalWalkEnv()
    if headless:
        print("=== World-frame verification (original robot) ===")
        verify_world_frame(env)
        print()

    forces = push_forces or PUSH_MAGNITUDES_N
    trajs = _narrow_grid(direction)
    cases = _build_cases(direction, forces, trajs)
    print(f"=== Sagittal step-catching: {direction} ({len(cases)} trials) ===")

    results: list[TrialMetrics] = []
    for case in cases:
        m = _run_trial(env, case)
        results.append(m)
        if headless and (m.success or case.push_force_n == forces[len(forces) // 2]):
            pass  # summary only at end

    successes = [m for m in results if m.success]
    print(f"\n=== {direction} summary: {len(successes)} / {len(results)} strict successes ===")
    if successes:
        best = min(successes, key=lambda m: m.final_tilt_rad)
        _print_trial(best)
        print("\nSTOP: scripted step demonstrated. Do not proceed to PPO yet.")
        return results

    # Report best near-miss per push magnitude
    print("\nBest near-miss per push magnitude:")
    for fn in forces:
        subset = [m for m in results if m.push_force_n == fn]
        if not subset:
            continue
        best = min(subset, key=lambda m: (0 if m.regained_contact else 1, m.final_tilt_rad, -m.peak_clearance_m))
        print(
            f"  {fn:.0f}N: {best.case_name} | clear={best.peak_clearance_m*1000:.0f}mm "
            f"disp={best.peak_displacement_m*1000:.0f}mm final_tilt={best.final_tilt_rad:.3f} "
            f"fail={'; '.join(best.failure_reasons[:2])}"
        )
    _print_limiting_mechanism(results)
    return results


def _print_limiting_mechanism(results: list[TrialMetrics]) -> None:
    print("\nLimiting mechanism:")
    if not any(m.lost_contact for m in results):
        print("  A. Cannot unload swing foot — trajectory/control coordination")
        return
    if not any(m.peak_clearance_m >= MIN_CLEARANCE_M for m in results):
        print("  D. Insufficient foot clearance")
        return
    if not any(m.peak_displacement_m >= MIN_DISPLACEMENT_M for m in results):
        print("  A. Trajectory/control — foot clears but does not reach target displacement")
        return
    if not any(m.regained_contact for m in results):
        print("  E. Landing instability — no reliable re-contact")
        return
    upright = [m for m in results if m.regained_contact and m.final_tilt_rad < MAX_UPRIGHT_TILT]
    if not upright:
        near = [m for m in results if m.regained_contact and m.peak_clearance_m >= MIN_CLEARANCE_M
                and m.peak_displacement_m >= MIN_DISPLACEMENT_M and m.final_tilt_rad < MAX_UPRIGHT_TILT]
        if near:
            print("  A/E. Trajectory/control — step kinematics succeed but stance foot lifts during swing")
        else:
            print("  E/F. Landing or post-landing stabilization — re-contact without upright hold")
        return
    print("  Unknown — inspect individual trials")


def run_viewer(direction: str, slow: bool, push_n: float = 60.0, traj_name: str | None = None) -> TrialMetrics:
    env = BipedalWalkEnv()
    print("=== World-frame verification ===")
    verify_world_frame(env)
    trajs = _narrow_grid(direction)
    traj = next((t for t in trajs if t.name == traj_name), trajs[0])
    if direction == "forward":
        case = SagittalCase(direction, push_n, traj.swing_leg, traj, 1.0, 1.0)
    else:
        case = SagittalCase(direction, push_n, traj.swing_leg, traj, -1.0, -1.0)

    slow_start = SETTLE_STEPS + _ms_to_steps(STAND_MS) - 20
    slow_end = slow_start + _ms_to_steps(PUSH_MS + REACT_MS + WEIGHT_SHIFT_MS + UNLOAD_MS + traj.swing_ms + PLACE_MS + 400)

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.05, 0.0, 1.05]
        v.cam.distance = 1.35
        v.cam.azimuth = 140
        v.cam.elevation = -15
        _configure_viewer_window()
        metrics = _run_trial(
            env, case, viewer=v, slow=slow,
            global_step=[0], slow_window=(slow_start, slow_end),
        )
    _print_trial(metrics)
    return metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sagittal push-and-catch step test (original robot)")
    p.add_argument("--case", choices=["forward", "backward"], help="Step-catching direction")
    p.add_argument("--headless", action="store_true", help="Run bounded search without viewer")
    p.add_argument("--slow", action="store_true", help="Slow-motion in viewer mode")
    p.add_argument("--push-n", type=float, default=60.0, help="Push magnitude for viewer mode (N)")
    p.add_argument("--traj", type=str, default=None, help="Trajectory template name for viewer")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless:
        run_search("forward", headless=True)
        run_search("backward", headless=True)
    elif args.case:
        run_viewer(args.case, slow=args.slow, push_n=args.push_n, traj_name=args.traj)
    else:
        print("Specify --headless or --case forward|backward")
        sys.exit(1)


if __name__ == "__main__":
    main()
