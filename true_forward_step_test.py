"""True forward step feasibility on original robot geometry (WORLD -Y = forward).

Evaluation only. Does not modify robot.xml, biped_env, rewards, obs, or PPO artifacts.
Prior +X \"forward\" results are lateral and must not be reused.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path
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

# --- Constants ---
FLOOR_Z = 1.0
FOOT_CONTACT_Z = 1.042
MIN_CLEARANCE_M = 0.025
MIN_FORWARD_M = 0.050
MAX_LATERAL_RATIO = 0.65
MAX_UPRIGHT_TILT = 0.45
POST_HOLD_STEPS = 500
MAX_PRE_SWING_TILT = 0.45

# Actuator ctrl indices
IDX = {
    "L_HIP_ROLL": 5,
    "L_HIP_PITCH": 6,
    "L_KNEE": 7,
    "L_ANKLE_P": 8,
    "R_HIP_ROLL": 10,
    "R_HIP_PITCH": 11,
    "R_KNEE": 12,
    "R_ANKLE_P": 13,
}

# Verified under position servos (see module doc / kinematic audit):
# L forward: decrease hip pitch ctrl (interactive key 2)
# R forward: increase hip pitch ctrl (interactive key 7) — grid also tests nearby values

PUSH_OPTIONS_N = [0.0, 20.0, 30.0, 40.0]
PUSH_DURATION = PUSH_DURATION_STEPS
REACT_MS = 30

VIEWER_WIDTH = 1280
VIEWER_HEIGHT = 720
VIEWER_TITLE_PREFIX = "MuJoCo"
VIEWER_POSITION_TIMEOUT_S = 2.0
NORMAL_SLEEP_S = 0.002
SLOW_SLEEP_S = 0.015
ARROW_LEN_M = 0.12
TRAIL_DECIMATE = 4

RESULTS_PATH = Path("true_forward_step_best.json")

WORLD_FORWARD = np.array([0.0, -1.0, 0.0])
WORLD_LATERAL = np.array([-1.0, 0.0, 0.0])


class Phase(Enum):
    RESET_SETTLE = "RESET / SETTLE"
    WEIGHT_SHIFT = "WEIGHT SHIFT"
    UNLOAD = "UNLOAD"
    LIFT = "LIFT"
    FORWARD_SWING = "FORWARD SWING"
    PLACE = "PLACE"
    STABILIZE = "STABILIZE"
    HOLD = "HOLD"
    DISTURBANCE = "DISTURBANCE"


@dataclass(frozen=True)
class TrajectorySpec:
    name: str
    swing_leg: str
    hip_pitch: float
    knee: float
    ankle: float
    ws_stance_pitch: float
    swing_ms: int = 140
    unload_ms: int = 60
    lift_ms: int = 50
    place_ms: int = 120
    weight_shift_ms: int = 200
    stabilize_ms: int = 300
    place_hip_scale: float = 0.72
    place_knee_scale: float = 0.65


@dataclass
class TrialMetrics:
    name: str
    swing_leg: str
    push_n: float
    success: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    peak_forward_m: float = 0.0
    peak_lateral_m: float = 0.0
    lateral_forward_ratio: float = float("inf")
    peak_clearance_m: float = 0.0
    lost_contact: bool = False
    regained_contact: bool = False
    stance_ok_swing: bool = True
    max_torso_tilt_rad: float = 0.0
    final_torso_tilt_rad: float = 0.0
    max_torso_angvel: float = 0.0
    max_pre_swing_tilt_rad: float = 0.0
    landing_step: int | None = None
    total_steps: int = 0
    spec: TrajectorySpec | None = None


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _ms(ms: int) -> int:
    return max(1, int(ms))


def _forward_m(pos_y: float, y0: float) -> float:
    return -(pos_y - y0)


def _lateral_m(pos_x: float, x0: float) -> float:
    return pos_x - x0


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


def _swing_indices(swing: str) -> tuple[int, int, int]:
    if swing == "L":
        return IDX["L_HIP_PITCH"], IDX["L_KNEE"], IDX["L_ANKLE_P"]
    return IDX["R_HIP_PITCH"], IDX["R_KNEE"], IDX["R_ANKLE_P"]


def _stance_leg(swing: str) -> str:
    return "R" if swing == "L" else "L"


def _stance_pitch_idx(stance: str) -> int:
    return IDX["R_HIP_PITCH"] if stance == "R" else IDX["L_HIP_PITCH"]


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


def _make_poses(model: mujoco.MjModel, spec: TrajectorySpec) -> dict[str, np.ndarray]:
    cr = model.actuator_ctrlrange[:15]
    hip_i, knee_i, ankle_i = _swing_indices(spec.swing_leg)
    stance = _stance_leg(spec.swing_leg)
    st_pitch_i = _stance_pitch_idx(stance)

    stand = DEFAULT_POSE.copy()
    ws = stand.copy()
    ws[st_pitch_i] = np.clip(spec.ws_stance_pitch, cr[st_pitch_i, 0], cr[st_pitch_i, 1])

    unload = ws.copy()
    unload[knee_i] = spec.knee * 0.40
    unload[hip_i] = spec.hip_pitch * 0.20

    apex = ws.copy()
    apex[hip_i] = np.clip(spec.hip_pitch, cr[hip_i, 0], cr[hip_i, 1])
    apex[knee_i] = np.clip(spec.knee, cr[knee_i, 0], cr[knee_i, 1])
    apex[ankle_i] = np.clip(spec.ankle, cr[ankle_i, 0], cr[ankle_i, 1])

    place = ws.copy()
    place[hip_i] = np.clip(spec.hip_pitch * spec.place_hip_scale, cr[hip_i, 0], cr[hip_i, 1])
    place[knee_i] = np.clip(spec.knee * spec.place_knee_scale, cr[knee_i, 0], cr[knee_i, 1])
    place[ankle_i] = np.clip(spec.ankle * 0.85, cr[ankle_i, 0], cr[ankle_i, 1])

    return {"stand": stand, "ws": ws, "unload": unload, "apex": apex, "place": place}


def _run_trial(
    env: BipedalWalkEnv,
    spec: TrajectorySpec,
    push_n: float,
    viewer: mujoco.viewer.Handle | None = None,
    slow: bool = False,
    trace: list[np.ndarray] | None = None,
    phase_label: list[str] | None = None,
) -> TrialMetrics:
    model, data = env.model, env.data
    chest = env.chest_body_id
    swing = spec.swing_leg
    stance = _stance_leg(swing)
    poses = _make_poses(model, spec)
    metrics = TrialMetrics(f"{spec.name}_push{push_n:.0f}N", swing, push_n, spec=spec)

    _reset(model, data)
    swing_init = _foot_pos(model, data, swing)
    x0, y0 = float(swing_init[0]), float(swing_init[1])

    segments: list[tuple[Phase, np.ndarray, int, np.ndarray | None]] = []
    if push_n > 0:
        segments.append((Phase.DISTURBANCE, poses["stand"], PUSH_DURATION, np.array([0.0, -push_n, 0.0])))
        segments.append((Phase.WEIGHT_SHIFT, poses["ws"], _ms(REACT_MS + spec.weight_shift_ms), None))
    else:
        segments.append((Phase.WEIGHT_SHIFT, poses["ws"], _ms(spec.weight_shift_ms), None))

    segments.extend([
        (Phase.UNLOAD, poses["unload"], _ms(spec.unload_ms), None),
        (Phase.LIFT, poses["unload"], _ms(spec.lift_ms), None),
        (Phase.FORWARD_SWING, poses["apex"], _ms(spec.swing_ms), None),
        (Phase.PLACE, poses["place"], _ms(spec.place_ms), None),
        (Phase.STABILIZE, poses["place"], _ms(spec.stabilize_ms), None),
        (Phase.HOLD, poses["place"], POST_HOLD_STEPS, None),
    ])

    cr = model.actuator_ctrlrange[:15]
    ctrl = DEFAULT_POSE.copy()
    step_i = 0
    landing_seen = False
    after_landing = False
    slow_lo, slow_hi = SETTLE_STEPS, SETTLE_STEPS + 900

    for phase, target, n_steps, frc in segments:
        if phase_label is not None:
            phase_label[0] = phase.value
        for s in range(n_steps):
            alpha = _smooth((s + 1) / n_steps)
            ctrl = np.clip((1.0 - alpha) * ctrl + alpha * target, cr[:, 0], cr[:, 1])
            data.ctrl[:15] = ctrl
            if frc is not None:
                data.xfrc_applied[chest, :3] = frc
            else:
                data.xfrc_applied[chest, :3] = 0.0
            mujoco.mj_step(model, data)

            sp = _foot_pos(model, data, swing)
            sc = _foot_contacts(model, data, swing)
            stc = _foot_contacts(model, data, stance)
            tilt = env._quat_tilt_rad()
            angvel = float(np.linalg.norm(data.qvel[3:6]))
            clearance = sp[2] - FLOOR_Z
            fwd = _forward_m(sp[1], y0)
            lat = _lateral_m(sp[0], x0)

            if phase in (Phase.RESET_SETTLE, Phase.WEIGHT_SHIFT, Phase.DISTURBANCE):
                metrics.max_pre_swing_tilt_rad = max(metrics.max_pre_swing_tilt_rad, tilt)

            metrics.peak_forward_m = max(metrics.peak_forward_m, fwd)
            metrics.peak_lateral_m = max(metrics.peak_lateral_m, abs(lat))
            metrics.peak_clearance_m = max(metrics.peak_clearance_m, clearance)
            metrics.max_torso_angvel = max(metrics.max_torso_angvel, angvel)
            if not after_landing:
                metrics.max_torso_tilt_rad = max(metrics.max_torso_tilt_rad, tilt)

            if sc == 0 and sp[2] > FOOT_CONTACT_Z:
                metrics.lost_contact = True
            if stc < 1 and phase in (Phase.LIFT, Phase.FORWARD_SWING):
                metrics.stance_ok_swing = False

            if (
                not landing_seen
                and metrics.lost_contact
                and sc > 0
                and phase in (Phase.PLACE, Phase.STABILIZE, Phase.HOLD)
            ):
                landing_seen = True
                after_landing = True
                metrics.regained_contact = True
                metrics.landing_step = step_i

            if trace is not None and step_i % TRAIL_DECIMATE == 0:
                trace.append(sp.copy())

            if viewer is not None and viewer.is_running():
                _draw_overlay(viewer, swing_init, _foot_pos(model, data, stance), trace or [], phase.value)
                viewer.sync()
                time.sleep(SLOW_SLEEP_S if slow and slow_lo <= step_i <= slow_hi else NORMAL_SLEEP_S)
            step_i += 1

    if metrics.lost_contact and _foot_contacts(model, data, swing) > 0:
        metrics.regained_contact = True
    metrics.final_torso_tilt_rad = env._quat_tilt_rad()
    metrics.total_steps = step_i
    if metrics.peak_forward_m > 1e-6:
        metrics.lateral_forward_ratio = metrics.peak_lateral_m / metrics.peak_forward_m

    reasons: list[str] = []
    if not metrics.lost_contact:
        reasons.append("swing foot never lost contact")
    if metrics.peak_clearance_m < MIN_CLEARANCE_M:
        reasons.append(f"clearance {metrics.peak_clearance_m:.3f} m < {MIN_CLEARANCE_M:.3f} m")
    if metrics.peak_forward_m < MIN_FORWARD_M:
        reasons.append(f"forward {metrics.peak_forward_m:.3f} m < {MIN_FORWARD_M:.3f} m (WORLD -Y)")
    if metrics.peak_forward_m > 1e-6 and metrics.lateral_forward_ratio > MAX_LATERAL_RATIO:
        reasons.append(f"lateral/forward ratio {metrics.lateral_forward_ratio:.2f} > {MAX_LATERAL_RATIO:.2f}")
    if not metrics.regained_contact:
        reasons.append("swing foot did not recontact")
    if not metrics.stance_ok_swing:
        reasons.append("stance foot lost contact during swing")
    if metrics.max_pre_swing_tilt_rad >= MAX_PRE_SWING_TILT:
        reasons.append(f"pre-swing tilt {metrics.max_pre_swing_tilt_rad:.3f} rad too high")
    if metrics.final_torso_tilt_rad >= MAX_UPRIGHT_TILT:
        reasons.append(f"final tilt {metrics.final_torso_tilt_rad:.3f} rad >= {MAX_UPRIGHT_TILT:.3f} rad")

    metrics.success = not reasons
    metrics.failure_reasons = reasons
    return metrics


def _draw_overlay(
    viewer: mujoco.viewer.Handle,
    swing_init: np.ndarray,
    stance_pos: np.ndarray,
    trail: list[np.ndarray],
    phase_name: str,
) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0
    origin = 0.5 * (swing_init + stance_pos)
    origin[2] = FLOOR_Z + 0.02
    fwd_tip = origin + WORLD_FORWARD * ARROW_LEN_M
    lat_tip = origin + WORLD_LATERAL * (ARROW_LEN_M * 0.5)
    _sphere(scn, swing_init, 0.012, [0.2, 0.9, 0.2, 0.95])
    _sphere(scn, stance_pos, 0.012, [0.9, 0.2, 0.2, 0.95])
    for i in range(1, len(trail)):
        _capsule(scn, trail[i - 1], trail[i], 0.004, [0.2, 0.7, 1.0, 0.85])
    if trail:
        _sphere(scn, trail[-1], 0.010, [0.1, 0.6, 1.0, 0.95])
    _capsule(scn, origin, fwd_tip, 0.006, [1.0, 0.85, 0.0, 1.0])
    _capsule(scn, origin, lat_tip, 0.004, [1.0, 0.35, 0.35, 0.8])


def _sphere(scn: mujoco.MjvScene, pos: np.ndarray, r: float, rgba: list[float]) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]),
        pos.astype(np.float64), np.eye(3).flatten(), np.array(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _capsule(scn: mujoco.MjvScene, p0: np.ndarray, p1: np.ndarray, r: float, rgba: list[float]) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r, p0.astype(np.float64), p1.astype(np.float64))
    g.rgba[:] = np.array(rgba, dtype=np.float32)
    scn.ngeom += 1


def _candidate_grid() -> list[TrajectorySpec]:
    """Bounded feasibility grid — L swing primary, R swing secondary."""
    specs: list[TrajectorySpec] = []
    l_hips = [-0.20, -0.26, -0.30]
    knees = [-0.24, -0.30, -0.34]
    ankles = [0.08, 0.12]
    ws_r = [0.08, 0.12]
    swings = [140, 170]

    for hp, kn, an, ws, sw in product(l_hips, knees, ankles, ws_r, swings):
        specs.append(
            TrajectorySpec(
                name=f"L_hp{abs(hp):.2f}_k{abs(kn):.2f}_a{an:.2f}_ws{ws:.2f}_t{sw}",
                swing_leg="L",
                hip_pitch=hp,
                knee=kn,
                ankle=an,
                ws_stance_pitch=ws,
                swing_ms=sw,
            )
        )

    r_hips = [0.24, 0.28]
    ws_l = [-0.08, -0.12]
    for hp, kn, an, ws, sw in product(r_hips, [-0.28, -0.32], [0.10], ws_l, [150]):
        specs.append(
            TrajectorySpec(
                name=f"R_hp{hp:.2f}_k{abs(kn):.2f}_a{an:.2f}_ws{ws:.2f}_t{sw}",
                swing_leg="R",
                hip_pitch=hp,
                knee=kn,
                ankle=an,
                ws_stance_pitch=ws,
                swing_ms=sw,
            )
        )
    return specs


def _rank(m: TrialMetrics) -> tuple:
    return (
        0 if m.success else 1,
        0 if m.stance_ok_swing else 1,
        -m.peak_forward_m,
        m.lateral_forward_ratio,
        m.final_torso_tilt_rad,
    )


def run_headless() -> list[TrialMetrics]:
    env = BipedalWalkEnv()
    specs = _candidate_grid()
    results: list[TrialMetrics] = []
    print("=== True forward step feasibility (WORLD -Y forward) ===")
    print(f"Candidates: {len(specs)} trajectories x {len(PUSH_OPTIONS_N)} push levels = {len(specs)*len(PUSH_OPTIONS_N)} trials")
    print(f"Forward metric: -(delta foot_y); lateral: |delta foot_x|; max ratio {MAX_LATERAL_RATIO}")
    print()

    for push_n in PUSH_OPTIONS_N:
        for spec in specs:
            results.append(_run_trial(env, spec, push_n))

    results.sort(key=_rank)
    successes = [m for m in results if m.success]
    print(f"Strict successes: {len(successes)} / {len(results)}")
    if successes:
        best = successes[0]
        _save_best(best)
        _print_metrics(best, header="BEST SUCCESS")
        print("\nVerdict: A) TRUE FORWARD STEP FOUND")
        return results

    # Prefer best true-forward candidate (low lateral ratio) for saved viewer params
    forwardish = [m for m in results if m.swing_leg == "L" and m.lateral_forward_ratio <= MAX_LATERAL_RATIO]
    save_candidate = min(forwardish, key=_rank) if forwardish else results[0]
    print("\nTop 8 near-misses:")
    for m in results[:8]:
        _print_metrics(m)
    _save_best(save_candidate)
    _print_verdict(save_candidate)
    return results


def _print_metrics(m: TrialMetrics, header: str | None = None) -> None:
    if header:
        print(f"\n--- {header} ---")
    print(
        f"  {m.name} | swing={m.swing_leg} push={m.push_n:.0f}N | "
        f"ok={m.success} | fwd={m.peak_forward_m*1000:.0f}mm lat={m.peak_lateral_m*1000:.0f}mm "
        f"ratio={m.lateral_forward_ratio:.2f} | clear={m.peak_clearance_m*1000:.0f}mm | "
        f"stance_ok={m.stance_ok_swing} | final_tilt={m.final_torso_tilt_rad:.3f} | "
        f"fail={'; '.join(m.failure_reasons[:2])}"
    )


def _print_verdict(best: TrialMetrics) -> None:
    print("\n=== VERDICT ===")
    if best.peak_forward_m < MIN_FORWARD_M:
        print("C) FORWARD FOOT MOTION IS INSUFFICIENT")
        print("   Swing leg cannot reach 50 mm along WORLD -Y with bounded hip/knee/ankle commands.")
    elif best.peak_clearance_m >= MIN_CLEARANCE_M and best.regained_contact and best.final_torso_tilt_rad >= MAX_UPRIGHT_TILT:
        print("B) FORWARD STEP IS KINEMATICALLY POSSIBLE BUT BALANCE FAILS")
    elif best.lateral_forward_ratio > MAX_LATERAL_RATIO:
        print("D) CONTROL/TRAJECTORY NEEDS MORE REFINEMENT (accidental lateral stepping)")
    else:
        print("D) CONTROL/TRAJECTORY NEEDS MORE REFINEMENT")
    print(f"   Best forward: {best.peak_forward_m*1000:.0f} mm, ratio: {best.lateral_forward_ratio:.2f}")


def _save_best(m: TrialMetrics) -> None:
    if m.spec is None:
        return
    payload = {
        "metrics": {
            k: v
            for k, v in asdict(m).items()
            if k not in ("spec", "failure_reasons")
        },
        "failure_reasons": m.failure_reasons,
        "spec": asdict(m.spec),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved best candidate parameters to {RESULTS_PATH}")


def _load_best() -> tuple[TrajectorySpec, float]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"No saved best candidate at {RESULTS_PATH}. Run --headless first.")
    data = json.loads(RESULTS_PATH.read_text())
    spec = TrajectorySpec(**data["spec"])
    push_n = float(data["metrics"]["push_n"])
    return spec, push_n


def run_viewer_best(slow: bool) -> None:
    spec, push_n = _load_best()
    env = BipedalWalkEnv()
    trace: list[np.ndarray] = []
    phase = ["?"]
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0.0, -0.08, 1.05]
        v.cam.distance = 1.5
        v.cam.azimuth = 90
        v.cam.elevation = -15
        _configure_viewer_window()
        m = _run_trial(env, spec, push_n, viewer=v, slow=slow, trace=trace, phase_label=phase)
    _print_metrics(m, header="VIEWER RUN")
    print(f"Phase overlays shown during run; final phase: {phase[0]}")


def _configure_viewer_window() -> None:
    if sys.platform != "win32":
        return
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
            hwnd = found[0]
            break
        time.sleep(0.05)
    else:
        return

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                    ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]

    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 1), ctypes.byref(info))
    wa = info.rcWork
    x = wa.left + (wa.right - wa.left - VIEWER_WIDTH) // 2
    y = wa.top + (wa.bottom - wa.top - VIEWER_HEIGHT) // 2
    user32.SetWindowPos(hwnd, 0, x, y, VIEWER_WIDTH, VIEWER_HEIGHT, 0x0004)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="True forward step test (WORLD -Y)")
    p.add_argument("--headless", action="store_true", help="Bounded search")
    p.add_argument("--best", action="store_true", help="Viewer for saved best candidate")
    p.add_argument("--slow", action="store_true", help="Slow motion in viewer")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.headless:
        run_headless()
    elif args.best:
        run_viewer_best(slow=args.slow)
    else:
        print("Use --headless for search or --best [--slow] for viewer")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
