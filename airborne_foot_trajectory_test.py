"""Airborne L-foot trajectory diagnostic — hip vs knee vs concurrent coupling.

Common prefix: STAND -> FORWARD FALL -> RAPID RIGHT SHIFT -> L foot airborne.
Then three isolated tests while airborne:
  A) L hip pitch only (~-0.45 rad)
  B) L knee only (~-0.62 rad)
  C) L hip + L knee concurrent

Records foot XYZ trajectory over time and compares max forward (-Y) displacement
while the L foot is actually airborne.

Evaluation only — does not modify robot.xml, biped_env, or other scripts.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Literal

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
UNLOAD_HOLD_STEPS = 120
TEST_RAMP_STEPS = 50
TEST_HOLD_STEPS = 200
TRAJ_PRINT_EVERY = 25

L_HIP_TEST = -0.45
L_KNEE_TEST = -0.62

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.018

TestId = Literal["A", "B", "C"]


class PrefixPhase(str, Enum):
    STAND = "STAND"
    FORWARD_FALL = "FORWARD FALL"
    RAPID_SHIFT = "RAPID RIGHT SHIFT"
    UNLOAD_HOLD = "UNLOAD HOLD (await airborne)"


@dataclass
class FootSample:
    step: int
    test_step: int
    l_foot_xyz: np.ndarray
    r_foot_xyz: np.ndarray
    fwd_mm: float
    lat_mm: float
    l_rel_r_fwd_mm: float
    clearance_mm: float
    hip_cmd: float
    hip_qpos: float
    knee_cmd: float
    knee_qpos: float
    torso_tilt_rad: float
    forward_vel_m_s: float
    airborne: bool


@dataclass
class TestResult:
    test_id: TestId
    label: str
    prefix_airborne: bool
    airborne_start_step: int | None
    test_start_fwd_mm: float
    test_start_lat_mm: float
    test_start_l_rel_r_mm: float
    max_fwd_while_airborne_mm: float = 0.0
    max_lat_while_airborne_mm: float = 0.0
    max_clearance_mm: float = 0.0
    peak_forward_vel_m_s: float = 0.0
    hip_cmd_end: float = 0.0
    hip_qpos_end: float = 0.0
    knee_cmd_end: float = 0.0
    knee_qpos_end: float = 0.0
    foot_end_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    samples: list[FootSample] = field(default_factory=list)


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _forward_mm(y: float, y_ref: float) -> float:
    return float(-(y - y_ref) * 1000.0)


def _lateral_mm(x: float, x_ref: float) -> float:
    return float((x - x_ref) * 1000.0)


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


def _test_pose(hip: float, knee: float) -> np.ndarray:
    return _apply_joints(
        rapid_shift_pose(),
        {IDX_L_HIP_PITCH: hip, IDX_L_KNEE: knee},
    )


def _sample_foot(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    step: int,
    test_step: int,
    ref_l: np.ndarray,
    ref_r: np.ndarray,
) -> FootSample:
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    return FootSample(
        step=step,
        test_step=test_step,
        l_foot_xyz=l_pos,
        r_foot_xyz=r_pos,
        fwd_mm=_forward_mm(l_pos[1], ref_l[1]),
        lat_mm=_lateral_mm(l_pos[0], ref_l[0]),
        l_rel_r_fwd_mm=_forward_mm(l_pos[1], r_pos[1]),
        clearance_mm=(l_pos[2] - FLOOR_Z) * 1000.0,
        hip_cmd=float(data.ctrl[IDX_L_HIP_PITCH]),
        hip_qpos=float(data.qpos[QPOS_L_HIP]),
        knee_cmd=float(data.ctrl[IDX_L_KNEE]),
        knee_qpos=float(data.qpos[QPOS_L_KNEE]),
        torso_tilt_rad=env._quat_tilt_rad(),
        forward_vel_m_s=_forward_vel(data),
        airborne=not _foot_contact(model, data, "L"),
    )


def run_prefix(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cr: np.ndarray,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
    *,
    verbose: bool,
) -> tuple[np.ndarray, bool, int | None]:
    """Return (ctrl at test start, foot was airborne before test, global step at airborne)."""
    ctrl = DEFAULT_POSE.copy()
    lean = forward_lean_pose()
    shifted = rapid_shift_pose()
    global_step = 0
    airborne_seen = False
    airborne_step: int | None = None

    def step_once(phase: PrefixPhase, target: np.ndarray) -> None:
        nonlocal ctrl, global_step, airborne_seen, airborne_step
        data.ctrl[:15] = ctrl
        mujoco.mj_step(model, data)
        global_step += 1
        if not _foot_contact(model, data, "L") and not airborne_seen:
            airborne_seen = True
            airborne_step = global_step
        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

    if verbose:
        print(f"\n--- {PrefixPhase.STAND.value} ---")
    for s in range(STAND_STEPS):
        ctrl = _lerp_ctrl(ctrl, DEFAULT_POSE, _smooth((s + 1) / STAND_STEPS), cr)
        step_once(PrefixPhase.STAND, ctrl)

    if verbose:
        print(f"--- {PrefixPhase.FORWARD_FALL.value} ---")
    for s in range(FALL_RAMP_STEPS):
        alpha = (s + 1) / FALL_RAMP_STEPS
        ctrl = _lerp_ctrl(ctrl, lean, alpha, cr)
        step_once(PrefixPhase.FORWARD_FALL, ctrl)
    for _ in range(FALL_MOMENTUM_STEPS):
        ctrl = lean.copy()
        step_once(PrefixPhase.FORWARD_FALL, ctrl)

    if verbose:
        print(f"--- {PrefixPhase.RAPID_SHIFT.value} ---")
    shift_start = ctrl.copy()
    for s in range(RAPID_SHIFT_STEPS):
        alpha = (s + 1) / RAPID_SHIFT_STEPS
        ctrl = _lerp_ctrl(shift_start, shifted, alpha, cr)
        step_once(PrefixPhase.RAPID_SHIFT, ctrl)

    if verbose:
        print(f"--- {PrefixPhase.UNLOAD_HOLD.value} ---")
    for _ in range(UNLOAD_HOLD_STEPS):
        ctrl = shifted.copy()
        step_once(PrefixPhase.UNLOAD_HOLD, ctrl)

    return ctrl, airborne_seen, airborne_step


def run_isolated_test(
    env: BipedalWalkEnv,
    test_id: TestId,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    verbose: bool = True,
) -> TestResult:
    labels = {
        "A": "TEST A - HIP ONLY",
        "B": "TEST B - KNEE ONLY",
        "C": "TEST C - HIP + KNEE CONCURRENT",
    }
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    prefix_ctrl, prefix_airborne, airborne_step = run_prefix(
        env, model, data, cr, viewer, slow, verbose=verbose,
    )

    l_ref = _foot_pos(model, data, "L")
    r_ref = _foot_pos(model, data, "R")
    hip0 = float(prefix_ctrl[IDX_L_HIP_PITCH])
    knee0 = float(prefix_ctrl[IDX_L_KNEE])

    result = TestResult(
        test_id=test_id,
        label=labels[test_id],
        prefix_airborne=prefix_airborne,
        airborne_start_step=airborne_step,
        test_start_fwd_mm=0.0,
        test_start_lat_mm=0.0,
        test_start_l_rel_r_mm=_forward_mm(l_ref[1], r_ref[1]),
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(labels[test_id])
        print(f"{'=' * 60}")
        print(f"  prefix airborne = {prefix_airborne} (step {airborne_step})")
        print(f"  test start: hip cmd={hip0:.3f} knee cmd={knee0:.3f}")
        print(f"  L foot XYZ = {l_ref}")
        print(f"  R foot XYZ = {r_ref}")
        print(f"  L-R forward = {result.test_start_l_rel_r_mm:.1f} mm")

    test_start_l = l_ref.copy()
    global_step = STAND_STEPS + FALL_RAMP_STEPS + FALL_MOMENTUM_STEPS + RAPID_SHIFT_STEPS + UNLOAD_HOLD_STEPS
    test_global = global_step

    def run_test_steps(n_steps: int, hip: float, knee: float) -> None:
        nonlocal test_global, prefix_ctrl
        ctrl = _test_pose(hip, knee)
        data.ctrl[:15] = ctrl
        mujoco.mj_step(model, data)
        test_global += 1
        sample = _sample_foot(
            env, model, data,
            step=test_global,
            test_step=n_steps,
            ref_l=test_start_l,
            ref_r=r_ref,
        )
        result.samples.append(sample)
        if sample.airborne:
            result.max_fwd_while_airborne_mm = max(
                result.max_fwd_while_airborne_mm, sample.fwd_mm,
            )
            result.max_lat_while_airborne_mm = max(
                result.max_lat_while_airborne_mm, abs(sample.lat_mm),
            )
            result.max_clearance_mm = max(result.max_clearance_mm, sample.clearance_mm)
        result.peak_forward_vel_m_s = max(result.peak_forward_vel_m_s, sample.forward_vel_m_s)
        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

    total_test = TEST_RAMP_STEPS + TEST_HOLD_STEPS
    for s in range(total_test):
        if s < TEST_RAMP_STEPS:
            t = (s + 1) / TEST_RAMP_STEPS
            alpha = _smooth(t)
            if test_id == "A":
                hip = (1.0 - alpha) * hip0 + alpha * L_HIP_TEST
                knee = knee0
            elif test_id == "B":
                hip = hip0
                knee = (1.0 - alpha) * knee0 + alpha * L_KNEE_TEST
            else:
                hip = (1.0 - alpha) * hip0 + alpha * L_HIP_TEST
                knee = (1.0 - alpha) * knee0 + alpha * L_KNEE_TEST
        else:
            if test_id == "A":
                hip, knee = L_HIP_TEST, knee0
            elif test_id == "B":
                hip, knee = hip0, L_KNEE_TEST
            else:
                hip, knee = L_HIP_TEST, L_KNEE_TEST

        run_test_steps(s + 1, hip, knee)

    last = result.samples[-1]
    result.hip_cmd_end = last.hip_cmd
    result.hip_qpos_end = last.hip_qpos
    result.knee_cmd_end = last.knee_cmd
    result.knee_qpos_end = last.knee_qpos
    result.foot_end_xyz = last.l_foot_xyz.copy()
    return result


def _print_trajectory(result: TestResult) -> None:
    print(f"\n  Foot trajectory ({result.label}):")
    print(
        "  step  airborne  fwd_mm  lat_mm  L-R_fwd  clr_mm  "
        "hip_c  hip_q  knee_c  knee_q  L_foot_Y"
    )
    for samp in result.samples:
        if samp.test_step == 1 or samp.test_step % TRAJ_PRINT_EVERY == 0 or samp.test_step == len(result.samples):
            print(
                f"  {samp.test_step:4d}  "
                f"{'Y' if samp.airborne else 'N':7s}  "
                f"{samp.fwd_mm:6.1f}  {samp.lat_mm:6.1f}  "
                f"{samp.l_rel_r_fwd_mm:7.1f}  {samp.clearance_mm:6.1f}  "
                f"{samp.hip_cmd:5.2f} {samp.hip_qpos:5.2f} "
                f"{samp.knee_cmd:5.2f} {samp.knee_qpos:5.2f}  "
                f"{samp.l_foot_xyz[1]:.4f}"
            )


def _print_test_summary(result: TestResult) -> None:
    print(f"\n  --- {result.label} summary ---")
    print(f"  L hip cmd/actual end: {result.hip_cmd_end:.3f} / {result.hip_qpos_end:.3f} rad")
    print(f"  L knee cmd/actual end: {result.knee_cmd_end:.3f} / {result.knee_qpos_end:.3f} rad")
    print(f"  L foot end XYZ: {result.foot_end_xyz}")
    print(f"  MAX forward displacement WHILE AIRBORNE: {result.max_fwd_while_airborne_mm:.1f} mm")
    print(f"  MAX lateral displacement while airborne: {result.max_lat_while_airborne_mm:.1f} mm")
    print(f"  MAX clearance while airborne: {result.max_clearance_mm:.1f} mm")
    print(f"  Peak forward velocity: {result.peak_forward_vel_m_s:.3f} m/s")
    airborne_samples = [s for s in result.samples if s.airborne]
    if airborne_samples:
        dy = [_forward_mm(s.l_foot_xyz[1], airborne_samples[0].l_foot_xyz[1]) for s in airborne_samples]
        print(f"  Airborne foot dY range (world Y, mm): {min(dy):.1f} to {max(dy):.1f}")
        print(f"  (negative dY in world coords = forward along -Y)")
    else:
        print("  WARNING: foot was never airborne during isolated test phase")


def print_comparison(results: list[TestResult]) -> None:
    print("\n" + "=" * 60)
    print("AIRBORNE FOOT TRAJECTORY COMPARISON")
    print("=" * 60)
    print(f"{'Test':<32} {'max_fwd_air_mm':>14} {'max_clr_mm':>10} {'hip_q_end':>10} {'knee_q_end':>10}")
    for r in results:
        print(
            f"{r.label:<32} {r.max_fwd_while_airborne_mm:14.1f} "
            f"{r.max_clearance_mm:10.1f} {r.hip_qpos_end:10.3f} {r.knee_qpos_end:10.3f}"
        )

    a = next((r for r in results if r.test_id == "A"), None)
    b = next((r for r in results if r.test_id == "B"), None)
    c = next((r for r in results if r.test_id == "C"), None)

    print("\n--- INTERPRETATION ---")
    if a and a.max_fwd_while_airborne_mm > 15.0:
        print("HIP ONLY produces meaningful forward foot motion while airborne.")
        if c and c.max_fwd_while_airborne_mm < a.max_fwd_while_airborne_mm * 0.6:
            print("HIP+KNEE concurrent REDUCES forward motion vs HIP ONLY.")
            print("=> Kinematic coupling / knee flexion is likely interfering with forward swing.")
        elif c and c.max_fwd_while_airborne_mm >= a.max_fwd_while_airborne_mm * 0.8:
            print("HIP+KNEE preserves most of the HIP ONLY forward motion.")
            print("=> Coupling is not the primary blocker; timing or whole-body dynamics may dominate.")
    elif a and a.max_fwd_while_airborne_mm <= 15.0:
        print("HIP ONLY produces little forward foot motion despite hip tracking.")
        print("=> Investigate joint axis/sign convention and foot Jacobian before retuning trajectory.")
        if b and b.max_fwd_while_airborne_mm > a.max_fwd_while_airborne_mm + 10.0:
            print("KNEE ONLY moves the foot more forward than HIP ONLY — hip pitch axis may not")
            print("couple strongly to world -Y at this configuration.")

    if b:
        print(f"\nKNEE ONLY max forward while airborne: {b.max_fwd_while_airborne_mm:.1f} mm")
    if a and c:
        print(f"HIP ONLY:  {a.max_fwd_while_airborne_mm:.1f} mm forward (airborne)")
        print(f"HIP+KNEE:  {c.max_fwd_while_airborne_mm:.1f} mm forward (airborne)")

    print("\nForward = world -Y. fwd_mm is displacement along -Y from test-start L foot.")
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
    p = argparse.ArgumentParser(description="Airborne L-foot hip/knee trajectory diagnostic.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--test",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Run one isolated test or all three (default: all).",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()
    tests: list[TestId] = ["A", "B", "C"] if args.test == "all" else [args.test]  # type: ignore[list-item]

    print("Airborne foot trajectory diagnostic")
    print("Prefix: STAND -> FORWARD FALL -> RAPID RIGHT SHIFT -> unload hold")
    print(f"Isolated tests: hip={L_HIP_TEST:.2f} rad, knee={L_KNEE_TEST:.2f} rad")
    print("Viewer: robot only.\n")

    results: list[TestResult] = []

    if args.headless:
        for tid in tests:
            r = run_isolated_test(env, tid, viewer=None, slow=False, verbose=True)
            _print_trajectory(r)
            _print_test_summary(r)
            results.append(r)
        if len(results) > 1:
            print_comparison(results)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.02]
        v.cam.distance = 1.55
        v.cam.azimuth = 88
        v.cam.elevation = -18
        _configure_viewer_window()
        for tid in tests:
            _reset(env.model, env.data)
            v.sync()
            r = run_isolated_test(env, tid, viewer=v, slow=args.slow, verbose=True)
            _print_trajectory(r)
            _print_test_summary(r)
            results.append(r)
    if len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main(sys.argv[1:])
