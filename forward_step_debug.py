"""Debug anatomical forward vs world axes; minimal scripted forward step (no push).

The prior BEST_FORWARD_CATCH trajectory was mislabeled: it moved the swing
foot primarily along world +X (lateral / hip-roll direction), not along the
robot's anatomical forward axis.

Evaluation only — does not modify robot.xml, biped_env, or PPO artifacts.
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
    SETTLE_STEPS,
    STANDING_QUAT,
)

FLOOR_Z = 1.0
FOOT_CONTACT_Z = 1.042

# Actuator ctrl indices
IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_R_HIP_ROLL = 10

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.015

ARROW_LEN_M = 0.12
TRAIL_DECIMATE = 5


class Phase(Enum):
    SETTLE = "SETTLE"
    STAND = "STAND"
    UNLOAD = "UNLOAD SWING FOOT"
    LIFT = "LIFT"
    SWING_FORWARD = "SWING FORWARD"
    PLACE = "PLACE"
    STABILIZE = "STABILIZE"


@dataclass
class FootTrace:
    swing: list[np.ndarray] = field(default_factory=list)
    stance: list[np.ndarray] = field(default_factory=list)
    swing_init: np.ndarray = field(default_factory=lambda: np.zeros(3))
    stance_init: np.ndarray = field(default_factory=lambda: np.zeros(3))
    swing_final: np.ndarray = field(default_factory=lambda: np.zeros(3))
    stance_final: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class StepReport:
    anatomical_forward: np.ndarray
    lateral_axis: np.ndarray
    delta_swing_world: np.ndarray
    delta_swing_forward_mm: float
    delta_swing_lateral_mm: float
    delta_swing_vertical_mm: float
    swing_ahead_of_stance_mm: float
    peak_clearance_mm: float
    final_tilt_rad: float
    lost_contact: bool
    regained_contact: bool


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _foot_pos(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    return data.xpos[model.body(f"{side}_foot").id].copy()


def _foot_contacts(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> int:
    n = 0
    for ci in range(data.ncon):
        for gid in (data.contact[ci].geom1, data.contact[ci].geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if f"{side}_foot_collision" in name:
                n += 1
                break
    return n


def compute_axes(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Anatomical forward from cross(world_up, lateral); lateral = L foot -> R foot."""
    l_pos = _foot_pos(model, data, "L")
    r_pos = _foot_pos(model, data, "R")
    lateral = r_pos - l_pos
    lateral[2] = 0.0
    lat_norm = np.linalg.norm(lateral)
    if lat_norm < 1e-6:
        lateral = np.array([1.0, 0.0, 0.0])
    else:
        lateral /= lat_norm
    up = np.array([0.0, 0.0, 1.0])
    forward = np.cross(up, lateral)
    fn = np.linalg.norm(forward)
    if fn < 1e-6:
        forward = np.array([0.0, -1.0, 0.0])
    else:
        forward /= fn
    return forward, lateral


def probe_interactive_forward(env: BipedalWalkEnv) -> dict[str, float]:
    """Confirm key-2 behavior: negative L hip pitch ctrl moves foot along anatomical forward."""
    model, data = env.model, env.data
    _reset(model, data)
    fwd, lat = compute_axes(model, data)
    l0 = _foot_pos(model, data, "L")
    ctrl = DEFAULT_POSE.copy()
    ctrl[IDX_L_HIP_PITCH] = -0.15  # key 2 direction (forward per interactive test)
    for _ in range(300):
        data.ctrl[:15] = ctrl
        mujoco.mj_step(model, data)
    l1 = _foot_pos(model, data, "L")
    d = l1 - l0
    return {
        "d_forward_mm": float(np.dot(d, fwd) * 1000),
        "d_lateral_mm": float(np.dot(d, lat) * 1000),
        "d_world_x_mm": float(d[0] * 1000),
        "d_world_y_mm": float(d[1] * 1000),
    }


def _reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = DEFAULT_POSE
    data.qvel[:] = 0.0
    data.ctrl[:15] = DEFAULT_POSE
    for _ in range(SETTLE_STEPS):
        data.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_step(model, data)


def _lerp_ctrl(ctrl: np.ndarray, target: np.ndarray, alpha: float, cr: np.ndarray) -> np.ndarray:
    out = (1.0 - alpha) * ctrl + alpha * target
    return np.clip(out, cr[:, 0], cr[:, 1])


@dataclass(frozen=True)
class SimpleForwardStep:
    """Minimal joint targets — sagittal only, no hip-roll weight shift."""

    name: str
    unload_knee: float = -0.10
    swing_hip_pitch: float = -0.25
    swing_knee: float = -0.30
    swing_ankle: float = 0.10
    place_hip_scale: float = 0.75
    unload_steps: int = 100
    lift_steps: int = 50
    swing_steps: int = 180
    place_steps: int = 120
    stabilize_steps: int = 400


DEFAULT_SIMPLE = SimpleForwardStep("minimal_L_forward")

# Gentler profile: smaller knee flex, slower swing (for viewer inspection)
GENTLE_SIMPLE = SimpleForwardStep(
    "gentle_L_forward",
    unload_knee=-0.06,
    swing_hip_pitch=-0.22,
    swing_knee=-0.18,
    swing_ankle=0.06,
    swing_steps=220,
    place_steps=150,
    stabilize_steps=500,
)


def run_simple_forward_step(
    env: BipedalWalkEnv,
    profile: SimpleForwardStep = DEFAULT_SIMPLE,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    draw_debug: bool = True,
) -> tuple[FootTrace, StepReport]:
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset(model, data)

    fwd, lat = compute_axes(model, data)
    trace = FootTrace()
    trace.swing_init = _foot_pos(model, data, "L")
    trace.stance_init = _foot_pos(model, data, "R")

    stand = DEFAULT_POSE.copy()
    unload = stand.copy()
    unload[IDX_L_KNEE] = profile.unload_knee

    apex = stand.copy()
    apex[IDX_L_HIP_PITCH] = profile.swing_hip_pitch
    apex[IDX_L_KNEE] = profile.swing_knee
    apex[IDX_L_ANKLE_P] = profile.swing_ankle

    place = stand.copy()
    place[IDX_L_HIP_PITCH] = profile.swing_hip_pitch * profile.place_hip_scale
    place[IDX_L_KNEE] = profile.swing_knee * 0.65
    place[IDX_L_ANKLE_P] = profile.swing_ankle * 0.8

    segments = [
        (Phase.STAND, stand, 100),
        (Phase.UNLOAD, unload, profile.unload_steps),
        (Phase.LIFT, unload, profile.lift_steps),
        (Phase.SWING_FORWARD, apex, profile.swing_steps),
        (Phase.PLACE, place, profile.place_steps),
        (Phase.STABILIZE, place, profile.stabilize_steps),
    ]

    ctrl = stand.copy()
    step_i = 0
    lost = False
    regained = False
    peak_clear = 0.0
    slow_lo, slow_hi = SETTLE_STEPS, SETTLE_STEPS + 800

    for phase, target, n_steps in segments:
        for s in range(n_steps):
            ctrl = _lerp_ctrl(ctrl, target, _smooth((s + 1) / n_steps), cr)
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)

            swing_p = _foot_pos(model, data, "L")
            stance_p = _foot_pos(model, data, "R")
            if step_i % TRAIL_DECIMATE == 0:
                trace.swing.append(swing_p.copy())
                trace.stance.append(stance_p.copy())

            clearance = swing_p[2] - FLOOR_Z
            peak_clear = max(peak_clear, clearance)
            swing_c = _foot_contacts(model, data, "L")
            if swing_c == 0 and swing_p[2] > FOOT_CONTACT_Z:
                lost = True
            if lost and swing_c > 0:
                regained = True

            if viewer is not None and viewer.is_running():
                if draw_debug:
                    origin = 0.5 * (trace.stance_init + trace.swing_init)
                    origin[2] = FLOOR_Z + 0.02
                    arrow_fwd = origin + fwd * ARROW_LEN_M
                    arrow_lat = origin + lat * (ARROW_LEN_M * 0.5)
                    _overlay_debug(viewer, trace, origin, arrow_fwd, arrow_lat)
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow and slow_lo <= step_i <= slow_hi else NORMAL_SLEEP_S)
            step_i += 1

    trace.swing_final = _foot_pos(model, data, "L")
    trace.stance_final = _foot_pos(model, data, "R")
    delta = trace.swing_final - trace.swing_init
    ahead = float(np.dot(trace.swing_final - trace.stance_final, fwd) * 1000)

    report = StepReport(
        anatomical_forward=fwd,
        lateral_axis=lat,
        delta_swing_world=delta,
        delta_swing_forward_mm=float(np.dot(delta, fwd) * 1000),
        delta_swing_lateral_mm=float(np.dot(delta, lat) * 1000),
        delta_swing_vertical_mm=float(delta[2] * 1000),
        swing_ahead_of_stance_mm=ahead,
        peak_clearance_mm=peak_clear * 1000,
        final_tilt_rad=env._quat_tilt_rad(),
        lost_contact=lost,
        regained_contact=regained,
    )
    return trace, report


def _overlay_debug(
    viewer: mujoco.viewer.Handle,
    trace: FootTrace,
    origin: np.ndarray,
    arrow_fwd_end: np.ndarray,
    arrow_lat_end: np.ndarray,
) -> None:
    """Draw debug overlays into viewer.user_scn (MuJoCo 3.x API)."""
    scene = viewer.user_scn
    scene.ngeom = 0
    _add_sphere(scene, trace.swing_init, 0.012, np.array([0.2, 0.9, 0.2, 0.9]))
    _add_sphere(scene, trace.stance_init, 0.012, np.array([0.9, 0.2, 0.2, 0.9]))
    if len(trace.swing) > 0:
        _add_sphere(scene, trace.swing[-1], 0.010, np.array([0.1, 0.6, 1.0, 0.9]))
    for i in range(1, len(trace.swing)):
        _add_line(scene, trace.swing[i - 1], trace.swing[i], np.array([0.2, 0.7, 1.0, 0.8]))
    _add_arrow(scene, origin, arrow_fwd_end, np.array([1.0, 0.85, 0.0, 1.0]))
    _add_arrow(scene, origin, arrow_lat_end, np.array([1.0, 0.3, 0.3, 0.7]))
    label_pt = origin + (arrow_fwd_end - origin) / max(np.linalg.norm(arrow_fwd_end - origin), 1e-6) * (ARROW_LEN_M + 0.03)
    _add_sphere(scene, label_pt, 0.008, np.array([1.0, 0.85, 0.0, 1.0]))


def _add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
        pos.astype(np.float64), np.eye(3).flatten(), rgba.astype(np.float32),
    )
    scene.ngeom += 1


def _add_line(scene: mujoco.MjvScene, p0: np.ndarray, p1: np.ndarray, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004,
        p0.astype(np.float64), p1.astype(np.float64),
    )
    g.rgba[:] = rgba
    scene.ngeom += 1


def _add_arrow(scene: mujoco.MjvScene, base: np.ndarray, tip: np.ndarray, rgba: np.ndarray) -> None:
    _add_line(scene, base, tip, rgba)
    dir_vec = tip - base
    n = np.linalg.norm(dir_vec)
    if n < 1e-6:
        return
    dir_vec /= n
    head_base = tip - dir_vec * 0.02
    for offset in (lat_perp(dir_vec, np.array([0, 0, 1.0])), lat_perp(dir_vec, np.array([0, 1, 0]))):
        if offset is None:
            continue
        _add_line(scene, tip, head_base + offset * 0.012, rgba)


def lat_perp(v: np.ndarray, hint: np.ndarray) -> np.ndarray | None:
    p = np.cross(v, hint)
    n = np.linalg.norm(p)
    if n < 1e-6:
        return None
    return p / n


def audit_best_forward_catch() -> None:
    """Print why BEST_FORWARD_CATCH is lateral, not forward."""
    from forward_catch_refine import BEST_FORWARD_CATCH, _case
    from sagittal_step_test import _run_trial

    env = BipedalWalkEnv()
    _reset(env.model, env.data)
    fwd, lat = compute_axes(env.model, env.data)
    probe = probe_interactive_forward(env)

    print("=" * 60)
    print("AUDIT: BEST_FORWARD_CATCH vs anatomical forward")
    print("=" * 60)
    print(f"Anatomical forward (horizontal): {fwd}")
    print(f"Lateral axis (L->R feet):        {lat}")
    print(f"Interactive probe (L hip -0.15): forward {probe['d_forward_mm']:+.1f} mm, "
          f"lateral {probe['d_lateral_mm']:+.1f} mm, "
          f"world dX {probe['d_world_x_mm']:+.1f} mm, dY {probe['d_world_y_mm']:+.1f} mm")
    print()
    print("BEST_FORWARD_CATCH joint targets:")
    print(f"  R hip_roll (weight shift): {BEST_FORWARD_CATCH.ws_stance_roll:.3f}  <- LATERAL")
    print(f"  R ankle_roll:              {BEST_FORWARD_CATCH.stance_ankle_roll:+.3f}  <- LATERAL")
    print(f"  L hip_pitch:               {BEST_FORWARD_CATCH.hip_pitch:.3f}")
    print(f"  L knee / ankle:            {BEST_FORWARD_CATCH.knee:.3f} / {BEST_FORWARD_CATCH.ankle:.3f}")
    print()
    print("Prior code measured displacement along world +X, which is the LATERAL axis.")
    print("Running BEST_FORWARD_CATCH once to quantify...")
    _reset(env.model, env.data)
    l0 = _foot_pos(env.model, env.data, "L")
    m = _run_trial(env, _case(BEST_FORWARD_CATCH, 40.0))
    lf = _foot_pos(env.model, env.data, "L")
    d = lf - l0
    print(f"  L foot delta along forward:  {np.dot(d,fwd)*1000:+.1f} mm")
    print(f"  L foot delta along lateral:  {np.dot(d,lat)*1000:+.1f} mm")
    print(f"  Reported 'forward' (+X):     {m.peak_displacement_m*1000:.1f} mm  (WRONG AXIS)")
    print(f"  Stance ok / success flags:   {m.stance_ok_swing} / {m.success}")
    print()
    print("CONCLUSION: BEST_FORWARD_CATCH is a lateral shuffle/reposition, not a")
    print("genuine forward step. Do not treat prior strict success as valid.")


def print_report(profile: SimpleForwardStep, report: StepReport) -> None:
    print(f"\n=== Simple forward step: {profile.name} ===")
    print(f"Anatomical forward axis: {report.anatomical_forward}")
    print(f"Swing foot delta - forward: {report.delta_swing_forward_mm:+.1f} mm | "
          f"lateral: {report.delta_swing_lateral_mm:+.1f} mm | "
          f"vertical: {report.delta_swing_vertical_mm:+.1f} mm")
    print(f"Swing foot ahead of stance (forward): {report.swing_ahead_of_stance_mm:+.1f} mm")
    print(f"Peak clearance: {report.peak_clearance_mm:.1f} mm | "
          f"lost/regain contact: {report.lost_contact}/{report.regained_contact}")
    print(f"Final tilt: {report.final_tilt_rad:.3f} rad")
    forward_ok = report.delta_swing_forward_mm > 30 and report.swing_ahead_of_stance_mm > 20
    print(f"Visually-forward criterion (heuristic): {'LIKELY YES' if forward_ok else 'NOT YET — inspect viewer'}")
    print("  (Use viewer to confirm swing foot lands in front of stance foot along yellow arrow)")


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
    p = argparse.ArgumentParser(description="Forward step axis debug and minimal scripted step")
    p.add_argument("--audit", action="store_true", help="Print BEST_FORWARD_CATCH axis audit (headless)")
    p.add_argument("--profile", choices=["default", "gentle"], default="default")
    p.add_argument("--headless", action="store_true", help="Run minimal step without viewer")
    p.add_argument("--slow", action="store_true", help="Slow motion in viewer")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.audit:
        audit_best_forward_catch()
        return

    env = BipedalWalkEnv()
    print("=== Interactive joint mapping check ===")
    _reset(env.model, env.data)
    probe = probe_interactive_forward(env)
    fwd, _ = compute_axes(env.model, env.data)
    print(f"  Anatomical forward: {fwd}")
    print(f"  L hip pitch -0.15 -> forward {probe['d_forward_mm']:+.1f} mm, lateral {probe['d_lateral_mm']:+.1f} mm")
    print("  (Yellow arrow in viewer = anatomical forward; red = lateral)")
    print()

    profile = GENTLE_SIMPLE if args.profile == "gentle" else DEFAULT_SIMPLE

    if args.headless:
        _, report = run_simple_forward_step(env, profile, viewer=None, draw_debug=False)
        print_report(profile, report)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.06, 1.05]
        v.cam.distance = 1.45
        v.cam.azimuth = 90
        v.cam.elevation = -15
        _configure_viewer_window()
        _, report = run_simple_forward_step(
            env, profile, viewer=v, slow=args.slow, draw_debug=True,
        )
    print_report(profile, report)


if __name__ == "__main__":
    main(sys.argv[1:])
