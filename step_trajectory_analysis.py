"""Compare manual vs dynamic forward-step trajectories with pelvis-relative metrics.

Read-only analysis — does not modify robot.xml, trajectories, or actuators.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable

import mujoco
import numpy as np

from biped_env import BipedalWalkEnv, CHEST_Z_CONTACT, DEFAULT_POSE, STANDING_QUAT

IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
QPOS_L_HIP = 13
QPOS_L_KNEE = 14
QPOS_L_ANKLE = 15

FOOT_CONTACT_Z = 1.042
FLOOR_Z = 1.0


@dataclass
class FrameRecord:
    traj: str
    phase: str
    step: int
    pelvis_xyz: np.ndarray
    pelvis_quat: np.ndarray
    pelvis_pitch_rad: float
    torso_tilt_rad: float
    l_foot_xyz: np.ndarray
    r_foot_xyz: np.ndarray
    l_contact: int
    r_contact: int
    l_normal_n: float
    r_normal_n: float
    l_rel_pelvis: np.ndarray
    l_rel_r_forward_mm: float
    l_rel_r_lateral_mm: float
    l_world_forward_mm: float
    r_world_forward_mm: float
    l_clearance_mm: float
    l_hip_cmd: float
    l_knee_cmd: float
    l_ankle_cmd: float
    l_hip_qpos: float
    l_knee_qpos: float
    l_ankle_qpos: float


@dataclass
class TrajSummary:
    name: str
    records: list[FrameRecord] = field(default_factory=list)
    peak_l_clearance_mm: float = 0.0
    peak_l_rel_r_forward_mm: float = 0.0
    peak_l_rel_r_forward_airborne_mm: float = 0.0
    min_l_rel_r_forward_swing_mm: float = field(default_factory=lambda: float("inf"))
    r_max_drift_swing_mm: float = 0.0
    tilt_at_first_touchdown: float | None = None
    l_rel_r_at_first_touchdown_mm: float | None = None
    misleading_world_success: bool = False


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _foot_pos(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    return data.xpos[model.body(f"{side}_foot").id].copy()


def _foot_contact(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> int:
    for ci in range(data.ncon):
        for gid in (data.contact[ci].geom1, data.contact[ci].geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if f"{side}_foot_collision" in name:
                return 1
    return 0


def _foot_normal(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
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


def _pelvis_pitch_rad(model: mujoco.MjModel, data: mujoco.MjData, chest_id: int) -> float:
    """Signed sagittal pitch: + = chest leans toward world -Y (forward), - = backward."""
    rot = data.xmat[chest_id].reshape(3, 3)
    # Chest local +Y is nominal up; project onto world sagittal plane.
    up_world = rot @ np.array([0.0, 1.0, 0.0])
    return float(np.arctan2(-up_world[1], up_world[2]))


def _record_frame(
    env: BipedalWalkEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    traj: str,
    phase: str,
    step: int,
    stand_l_y: float,
    stand_r_y: float,
    stand_r_xy: np.ndarray,
) -> FrameRecord:
    chest_id = env.chest_body_id
    pelvis = data.xpos[chest_id].copy()
    quat = data.qpos[3:7].copy()
    l_foot = _foot_pos(model, data, "L")
    r_foot = _foot_pos(model, data, "R")
    rel_r_fwd = -(l_foot[1] - r_foot[1]) * 1000.0
    rel_r_lat = (l_foot[0] - r_foot[0]) * 1000.0
    return FrameRecord(
        traj=traj,
        phase=phase,
        step=step,
        pelvis_xyz=pelvis,
        pelvis_quat=quat,
        pelvis_pitch_rad=_pelvis_pitch_rad(model, data, chest_id),
        torso_tilt_rad=env._quat_tilt_rad(),
        l_foot_xyz=l_foot,
        r_foot_xyz=r_foot,
        l_contact=_foot_contact(model, data, "L"),
        r_contact=_foot_contact(model, data, "R"),
        l_normal_n=_foot_normal(model, data, "L"),
        r_normal_n=_foot_normal(model, data, "R"),
        l_rel_pelvis=l_foot - pelvis,
        l_rel_r_forward_mm=rel_r_fwd,
        l_rel_r_lateral_mm=rel_r_lat,
        l_world_forward_mm=-(l_foot[1] - stand_l_y) * 1000.0,
        r_world_forward_mm=-(r_foot[1] - stand_r_y) * 1000.0,
        l_clearance_mm=(l_foot[2] - FLOOR_Z) * 1000.0,
        l_hip_cmd=float(data.ctrl[IDX_L_HIP_PITCH]),
        l_knee_cmd=float(data.ctrl[IDX_L_KNEE]),
        l_ankle_cmd=float(data.ctrl[IDX_L_ANKLE_P]),
        l_hip_qpos=float(data.qpos[QPOS_L_HIP]),
        l_knee_qpos=float(data.qpos[QPOS_L_KNEE]),
        l_ankle_qpos=float(data.qpos[QPOS_L_ANKLE]),
    )


def _run_dynamic(env: BipedalWalkEnv) -> TrajSummary:
    from dynamic_forward_step_test import (
        Phase,
        _lerp_ctrl,
        _reset_pose_only,
        build_trajectory,
    )

    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset_pose_only(model, data)
    ctrl = DEFAULT_POSE.copy()
    summary = TrajSummary(name="dynamic_forward_step_test")
    stand_l_y = stand_r_y = 0.0
    stand_r_xy = np.zeros(2)
    step_i = 0
    airborne_seen = False
    swing_phases = {
        Phase.LEFT_LIFT.value,
        Phase.RIGHT_PUSH.value,
        Phase.LEFT_SWING.value,
    }

    for phase, target, n_steps, linear in build_trajectory():
        for s in range(n_steps):
            t = (s + 1) / n_steps
            alpha = t if linear else _smooth(t)
            ctrl = _lerp_ctrl(ctrl, target, alpha, cr)
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)
            if phase == Phase.STAND and s == n_steps - 1:
                stand_l_y = _foot_pos(model, data, "L")[1]
                stand_r_y = _foot_pos(model, data, "R")[1]
                stand_r_xy = _foot_pos(model, data, "R")[:2].copy()
            rec = _record_frame(
                env, model, data, summary.name, phase.value, step_i,
                stand_l_y, stand_r_y, stand_r_xy,
            )
            summary.records.append(rec)
            summary.peak_l_clearance_mm = max(summary.peak_l_clearance_mm, rec.l_clearance_mm)
            summary.peak_l_rel_r_forward_mm = max(
                summary.peak_l_rel_r_forward_mm, rec.l_rel_r_forward_mm,
            )
            if phase.value in swing_phases:
                summary.min_l_rel_r_forward_swing_mm = min(
                    summary.min_l_rel_r_forward_swing_mm, rec.l_rel_r_forward_mm,
                )
                summary.r_max_drift_swing_mm = max(
                    summary.r_max_drift_swing_mm,
                    float(np.linalg.norm(rec.r_foot_xyz[:2] - stand_r_xy) * 1000.0),
                )
            airborne = rec.l_contact == 0 and rec.l_foot_xyz[2] > FOOT_CONTACT_Z
            if airborne and phase.value in swing_phases:
                airborne_seen = True
                summary.peak_l_rel_r_forward_airborne_mm = max(
                    summary.peak_l_rel_r_forward_airborne_mm,
                    rec.l_rel_r_forward_mm,
                )
            if (
                airborne_seen
                and rec.l_contact > 0
                and summary.tilt_at_first_touchdown is None
                and phase.value in swing_phases | {Phase.LEFT_TOUCHDOWN.value}
            ):
                summary.tilt_at_first_touchdown = rec.torso_tilt_rad
                summary.l_rel_r_at_first_touchdown_mm = rec.l_rel_r_forward_mm
            step_i += 1

    peak_world_air = max(
        (r.l_world_forward_mm for r in summary.records
         if r.l_contact == 0 and r.l_foot_xyz[2] > FOOT_CONTACT_Z),
        default=0.0,
    )
    summary.misleading_world_success = (
        peak_world_air > 40.0
        and summary.peak_l_rel_r_forward_airborne_mm < 20.0
    )
    return summary


def _run_manual(env: BipedalWalkEnv) -> TrajSummary:
    from manual_forward_step import (
        Phase,
        _lerp_ctrl,
        _reset_pose_only,
        build_manual_trajectory,
    )

    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset_pose_only(model, data)
    ctrl = DEFAULT_POSE.copy()
    summary = TrajSummary(name="manual_forward_step")
    stand_l_y = stand_r_y = 0.0
    stand_r_xy = np.zeros(2)
    step_i = 0
    airborne_seen = False
    swing_phases = {
        Phase.LIFT_SHALLOW.value,
        Phase.LIFT_DEEP.value,
        Phase.LIFT_FULL.value,
        Phase.FORWARD_SWING.value,
    }

    for phase, target, n_steps in build_manual_trajectory():
        for s in range(n_steps):
            alpha = 1.0 if phase == Phase.STANCE_LOCK else _smooth((s + 1) / n_steps)
            ctrl = _lerp_ctrl(ctrl, target, alpha, cr)
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)
            if phase == Phase.STAND and s == n_steps - 1:
                stand_l_y = _foot_pos(model, data, "L")[1]
                stand_r_y = _foot_pos(model, data, "R")[1]
                stand_r_xy = _foot_pos(model, data, "R")[:2].copy()
            rec = _record_frame(
                env, model, data, summary.name, phase.value, step_i,
                stand_l_y, stand_r_y, stand_r_xy,
            )
            summary.records.append(rec)
            summary.peak_l_clearance_mm = max(summary.peak_l_clearance_mm, rec.l_clearance_mm)
            summary.peak_l_rel_r_forward_mm = max(
                summary.peak_l_rel_r_forward_mm, rec.l_rel_r_forward_mm,
            )
            if phase.value in swing_phases:
                summary.min_l_rel_r_forward_swing_mm = min(
                    summary.min_l_rel_r_forward_swing_mm, rec.l_rel_r_forward_mm,
                )
                summary.r_max_drift_swing_mm = max(
                    summary.r_max_drift_swing_mm,
                    float(np.linalg.norm(rec.r_foot_xyz[:2] - stand_r_xy) * 1000.0),
                )
            airborne = rec.l_contact == 0 and rec.l_foot_xyz[2] > FOOT_CONTACT_Z
            if airborne and phase.value in swing_phases:
                airborne_seen = True
                summary.peak_l_rel_r_forward_airborne_mm = max(
                    summary.peak_l_rel_r_forward_airborne_mm,
                    rec.l_rel_r_forward_mm,
                )
            if (
                airborne_seen
                and rec.l_contact > 0
                and summary.tilt_at_first_touchdown is None
                and phase.value in swing_phases | {Phase.PLACE.value}
            ):
                summary.tilt_at_first_touchdown = rec.torso_tilt_rad
                summary.l_rel_r_at_first_touchdown_mm = rec.l_rel_r_forward_mm
            step_i += 1
    return summary


def _phase_boundary_records(summary: TrajSummary) -> list[FrameRecord]:
    out: list[FrameRecord] = []
    seen: set[str] = set()
    for rec in summary.records:
        key = rec.phase
        if key in seen:
            continue
        seen.add(key)
        # last record of each phase
    last_by_phase: dict[str, FrameRecord] = {}
    for rec in summary.records:
        last_by_phase[rec.phase] = rec
    return list(last_by_phase.values())


def _print_phase_table(summary: TrajSummary) -> None:
    print(f"\n{'=' * 90}")
    print(f"PHASE BOUNDARIES: {summary.name}")
    print(f"{'=' * 90}")
    hdr = (
        f"{'phase':<22} {'tilt':>6} {'pitch':>6} "
        f"{'L-R fwd':>8} {'L-R lat':>8} {'L w-fwd':>8} {'R w-fwd':>8} "
        f"{'L cl':>6} {'Lc':>3} {'Rc':>3} "
        f"{'hip c/q':>12} {'knee c/q':>12} {'ank c/q':>12}"
    )
    print(hdr)
    for rec in _phase_boundary_records(summary):
        print(
            f"{rec.phase:<22} {rec.torso_tilt_rad:6.3f} {rec.pelvis_pitch_rad:6.3f} "
            f"{rec.l_rel_r_forward_mm:8.1f} {rec.l_rel_r_lateral_mm:8.1f} "
            f"{rec.l_world_forward_mm:8.1f} {rec.r_world_forward_mm:8.1f} "
            f"{rec.l_clearance_mm:6.1f} {rec.l_contact:3d} {rec.r_contact:3d} "
            f"{rec.l_hip_cmd:+.2f}/{rec.l_hip_qpos:+.2f} "
            f"{rec.l_knee_cmd:+.2f}/{rec.l_knee_qpos:+.2f} "
            f"{rec.l_ankle_cmd:+.2f}/{rec.l_ankle_qpos:+.2f}"
        )


def _print_swing_detail(summary: TrajSummary) -> None:
    swing_recs = [
        r for r in summary.records
        if "SWING" in r.phase or "PUSH" in r.phase or "LIFT" in r.phase
    ]
    if not swing_recs:
        return
    print(f"\n--- Swing-phase extrema: {summary.name} ---")
    min_rec = min(swing_recs, key=lambda r: r.l_rel_r_forward_mm)
    max_rec = max(swing_recs, key=lambda r: r.l_rel_r_forward_mm)
    print(
        f"  L-R forward MIN {min_rec.l_rel_r_forward_mm:+.1f} mm "
        f"at {min_rec.phase} step {min_rec.step} "
        f"(L world fwd {min_rec.l_world_forward_mm:+.1f}, pitch {min_rec.pelvis_pitch_rad:+.3f})"
    )
    print(
        f"  L-R forward MAX {max_rec.l_rel_r_forward_mm:+.1f} mm "
        f"at {max_rec.phase} step {max_rec.step}"
    )


def print_comparison(dyn: TrajSummary, man: TrajSummary) -> None:
    _print_phase_table(dyn)
    _print_swing_detail(dyn)
    _print_phase_table(man)
    _print_swing_detail(man)

    print(f"\n{'=' * 90}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 90}")
    rows = [
        ("peak L clearance (mm)", dyn.peak_l_clearance_mm, man.peak_l_clearance_mm),
        ("peak L-R forward (mm)", dyn.peak_l_rel_r_forward_mm, man.peak_l_rel_r_forward_mm),
        ("peak L-R forward airborne (mm)", dyn.peak_l_rel_r_forward_airborne_mm, man.peak_l_rel_r_forward_airborne_mm),
        ("min L-R forward during swing (mm)", dyn.min_l_rel_r_forward_swing_mm, man.min_l_rel_r_forward_swing_mm),
        ("R drift during swing (mm)", dyn.r_max_drift_swing_mm, man.r_max_drift_swing_mm),
        ("tilt at first L touchdown (rad)", dyn.tilt_at_first_touchdown or float("nan"), man.tilt_at_first_touchdown or float("nan")),
        ("L-R forward at first touchdown (mm)", dyn.l_rel_r_at_first_touchdown_mm or float("nan"), man.l_rel_r_at_first_touchdown_mm or float("nan")),
    ]
    print(f"{'metric':<40} {'dynamic':>12} {'manual':>12}")
    for name, d, m in rows:
        print(f"{name:<40} {d:12.1f} {m:12.1f}")

    print("\n--- DIAGNOSIS ---")
    print(
        "1. Apparent backward L swing (dynamic): during RIGHT PUSH-OFF and early "
        "LEFT SWING, L_rel_R_forward goes NEGATIVE while L_world_forward may still "
        "look positive — the pelvis and R foot move in world -Y faster than L foot."
    )
    print(
        "2. L vs R during dynamic swing: see min L-R forward during swing above. "
        "Negative = L foot is BEHIND R (anatomical backward relative to stance)."
    )
    print(
        f"3. R drift during dynamic swing: {dyn.r_max_drift_swing_mm:.1f} mm "
        f"(manual: {man.r_max_drift_swing_mm:.1f} mm)."
    )
    dyn_pitch_swing = next(
        (r for r in dyn.records if r.phase == "LEFT FORWARD SWING"),
        dyn.records[-1],
    )
    print(
        f"4. Pelvis pitch at dynamic swing phase end: "
        f"{dyn_pitch_swing.pelvis_pitch_rad:+.3f} rad, tilt {dyn_pitch_swing.torso_tilt_rad:.3f} rad."
    )
    print(
        "5. Misleading SUCCESS: dynamic script used world -Y foot displacement while "
        "airborne; pelvis rotation + R-foot skid created false forward progress in world "
        "frame without L moving ahead of R until late/collapsing touchdown."
    )
    print(
        "6. Manual vs dynamic: manual uses gradual unload (R hip pitch +0.07 only), "
        "staged lift (knee before hip), no R sagittal push-off; dynamic uses rapid "
        "lateral unload + aggressive knee lift + R push that pitches pelvis backward "
        "before L hip pitch is applied."
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze step trajectories.")
    parser.parse_args(argv)
    env = BipedalWalkEnv()
    dyn = _run_dynamic(env)
    man = _run_manual(env)
    print_comparison(dyn, man)


if __name__ == "__main__":
    main(sys.argv[1:])
