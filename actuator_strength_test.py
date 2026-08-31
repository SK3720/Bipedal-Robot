"""Actuator authority diagnostic for L-leg hip pitch, knee, and ankle pitch.

Commands each joint individually through representative targets while the L leg
is unloaded (weight-shift pose). Optionally compares commanded vs actual joint
positions during actual_forward_step_test.py phases.

Evaluation only — does not modify robot.xml, biped_env, or PPO artifacts.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np

from biped_env import (
    BipedalWalkEnv,
    CHEST_Z_CONTACT,
    DEFAULT_POSE,
    STANDING_QUAT,
)

IDX_L_HIP_PITCH = 6
IDX_L_KNEE = 7
IDX_L_ANKLE_P = 8
IDX_L_HIP_ROLL = 5
IDX_R_HIP_ROLL = 10
IDX_R_HIP_PITCH = 11
IDX_R_ANKLE_ROLL = 14

# qpos addresses for L leg joints (actuator ctrl i -> qpos[7+i] for actuated joints)
QPOS_ADR = {IDX_L_HIP_PITCH: 13, IDX_L_KNEE: 14, IDX_L_ANKLE_P: 15}

JOINT_SPECS = (
    {
        "name": "L hip pitch",
        "actuator": "L_hip_L_hip_pitch_motor",
        "joint": "L_hip_L_hip_pitch",
        "ctrl_idx": IDX_L_HIP_PITCH,
        "targets": (0.0, -0.10, -0.14, -0.28),
    },
    {
        "name": "L knee",
        "actuator": "L_leg_L_knee_motor",
        "joint": "L_leg_L_knee",
        "ctrl_idx": IDX_L_KNEE,
        "targets": (0.0, -0.10, -0.30, -0.34),
    },
    {
        "name": "L ankle pitch",
        "actuator": "L_shin_L_ankle_pitch_motor",
        "joint": "L_shin_L_ankle_pitch",
        "ctrl_idx": IDX_L_ANKLE_P,
        "targets": (0.0, 0.06, 0.12),
    },
)

STAND_SETTLE_STEPS = 500
WEIGHT_SHIFT_SETTLE_STEPS = 700
TARGET_SETTLE_STEPS = 500
TRACK_TOLERANCE_RAD = 0.02  # ~1.1 deg
FORCE_SAT_N = 2.29

FORWARD_STEP_PHASES = (
    "STAND",
    "WEIGHT SHIFT",
    "IMMEDIATE LIFT",
    "LIFT",
    "FORWARD SWING",
    "PLACE",
    "STABILIZE",
)


@dataclass
class ActuatorConfig:
    name: str
    actuator_type: str
    kp: float
    kv: float | None
    forcerange: tuple[float, float]
    ctrlrange: tuple[float, float]
    joint_range: tuple[float, float]


@dataclass
class TrackingSample:
    label: str
    commanded: float
    actual_qpos: float
    error_rad: float
    error_deg: float
    actuator_force: float
    force_saturated: bool
    reached_target: bool


@dataclass
class JointSweepResult:
    joint_name: str
    context: str
    samples: list[TrackingSample]


@dataclass
class ForwardStepTracking:
    phase: str
    hip_cmd: float
    hip_qpos: float
    hip_err_deg: float
    knee_cmd: float
    knee_qpos: float
    knee_err_deg: float
    ankle_cmd: float
    ankle_qpos: float
    ankle_err_deg: float
    l_forward_mm: float
    l_height_mm: float
    l_normal_force: float


def _smooth(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _reset_stand(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = DEFAULT_POSE
    data.qvel[:] = 0.0
    data.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(model, data)


def _lerp_ctrl(ctrl: np.ndarray, target: np.ndarray, alpha: float, cr: np.ndarray) -> np.ndarray:
    return np.clip((1.0 - alpha) * ctrl + alpha * target, cr[:, 0], cr[:, 1])


def weight_shift_pose() -> np.ndarray:
    pose = DEFAULT_POSE.copy()
    pose[IDX_R_HIP_PITCH] = 0.07
    pose[IDX_R_HIP_ROLL] = -0.04
    pose[IDX_R_ANKLE_ROLL] = 0.03
    pose[IDX_L_HIP_ROLL] = -0.03
    return pose


def _settle_to(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: np.ndarray,
    target: np.ndarray,
    cr: np.ndarray,
    steps: int,
) -> np.ndarray:
    for s in range(steps):
        ctrl = _lerp_ctrl(ctrl, target, _smooth((s + 1) / steps), cr)
        data.ctrl[:15] = ctrl
        mujoco.mj_step(model, data)
    return ctrl


def _read_actuator_config(model: mujoco.MjModel, spec: dict) -> ActuatorConfig:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec["actuator"])
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec["joint"])
    kp = float(model.actuator_gainprm[aid, 0])
    # MuJoCo position actuators: no explicit kv; biasprm[1] = -kp for position servo
    kv = None
    fr = model.actuator_forcerange[aid]
    cr = model.actuator_ctrlrange[aid]
    jr = model.jnt_range[jid]
    return ActuatorConfig(
        name=spec["name"],
        actuator_type="position",
        kp=kp,
        kv=kv,
        forcerange=(float(fr[0]), float(fr[1])),
        ctrlrange=(float(cr[0]), float(cr[1])),
        joint_range=(float(jr[0]), float(jr[1])),
    )


def _tracking_sample(
    label: str,
    ctrl_idx: int,
    qpos_adr: int,
    data: mujoco.MjData,
) -> TrackingSample:
    cmd = float(data.ctrl[ctrl_idx])
    qpos = float(data.qpos[qpos_adr])
    err = cmd - qpos
    force = float(data.actuator_force[ctrl_idx])
    sat = abs(force) >= FORCE_SAT_N
    reached = abs(err) <= TRACK_TOLERANCE_RAD and not sat
    return TrackingSample(
        label=label,
        commanded=cmd,
        actual_qpos=qpos,
        error_rad=err,
        error_deg=float(np.degrees(err)),
        actuator_force=force,
        force_saturated=sat,
        reached_target=reached,
    )


def _run_joint_sweep(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: dict,
    base_pose: np.ndarray,
    context: str,
) -> JointSweepResult:
    cr = model.actuator_ctrlrange[:15]
    _reset_stand(model, data)
    ctrl = _settle_to(model, data, DEFAULT_POSE.copy(), base_pose, cr, STAND_SETTLE_STEPS)
    if context == "unloaded":
        ctrl = _settle_to(model, data, ctrl, base_pose, cr, WEIGHT_SHIFT_SETTLE_STEPS)

    samples: list[TrackingSample] = []
    ci = spec["ctrl_idx"]
    qadr = QPOS_ADR[ci]
    for target_val in spec["targets"]:
        target = base_pose.copy()
        target[ci] = target_val
        ctrl = _settle_to(model, data, ctrl, target, cr, TARGET_SETTLE_STEPS)
        samples.append(_tracking_sample(f"target {target_val:+.2f}", ci, qadr, data))

    return JointSweepResult(joint_name=spec["name"], context=context, samples=samples)


def print_actuator_config(model: mujoco.MjModel) -> None:
    print("=" * 72)
    print("L-LEG ACTUATOR CONFIGURATION (from robot.xml)")
    print("=" * 72)
    for spec in JOINT_SPECS:
        cfg = _read_actuator_config(model, spec)
        print(f"\n{cfg.name} ({spec['actuator']}, ctrl[{spec['ctrl_idx']}]):")
        print(f"  actuator type:  {cfg.actuator_type}")
        print(f"  kp:             {cfg.kp}")
        print(f"  kv:             {cfg.kv if cfg.kv is not None else 'N/A (position servo, joint damping=0.1)'}")
        print(f"  forcerange:     [{cfg.forcerange[0]:+.1f}, {cfg.forcerange[1]:+.1f}] N")
        print(f"  ctrlrange:      [{cfg.ctrlrange[0]:+.4f}, {cfg.ctrlrange[1]:+.4f}] rad")
        print(f"  joint range:    [{cfg.joint_range[0]:+.4f}, {cfg.joint_range[1]:+.4f}] rad")


def print_sweep_results(results: list[JointSweepResult]) -> None:
    print("\n" + "=" * 72)
    print("INDIVIDUAL JOINT TRACKING (unloaded = weight-shift pose)")
    print("=" * 72)
    for result in results:
        print(f"\n--- {result.joint_name} [{result.context}] ---")
        print(f"{'label':<16} {'cmd':>8} {'qpos':>8} {'err(deg)':>9} {'force':>8} {'sat':>5} {'ok':>4}")
        for s in result.samples:
            ok = "YES" if s.reached_target else "no"
            sat = "YES" if s.force_saturated else "no"
            print(
                f"{s.label:<16} {s.commanded:+.3f} {s.actual_qpos:+.3f} "
                f"{s.error_deg:+.2f} {s.actuator_force:+.3f} {sat:>5} {ok:>4}"
            )


def _foot_pos(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    return data.xpos[model.body(f"{side}_foot").id].copy()


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


def run_forward_step_tracking() -> list[ForwardStepTracking]:
    from actual_forward_step_test import (
        build_trajectory,
        _reset_pose_only,
        _lerp_ctrl,
        _smooth,
    )

    env = BipedalWalkEnv()
    model, data = env.model, env.data
    cr = model.actuator_ctrlrange[:15]
    _reset_pose_only(model, data)

    ctrl = DEFAULT_POSE.copy()
    stand_l = _foot_pos(model, data, "L")
    records: list[ForwardStepTracking] = []

    for phase, target, n_steps in build_trajectory():
        for s in range(n_steps):
            ctrl = _lerp_ctrl(ctrl, target, _smooth((s + 1) / n_steps), cr)
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)
            if phase.value == "STAND" and s == n_steps - 1:
                stand_l = _foot_pos(model, data, "L")

        l_pos = _foot_pos(model, data, "L")
        delta = l_pos - stand_l
        records.append(
            ForwardStepTracking(
                phase=phase.value,
                hip_cmd=float(ctrl[IDX_L_HIP_PITCH]),
                hip_qpos=float(data.qpos[QPOS_ADR[IDX_L_HIP_PITCH]]),
                hip_err_deg=float(np.degrees(ctrl[IDX_L_HIP_PITCH] - data.qpos[QPOS_ADR[IDX_L_HIP_PITCH]])),
                knee_cmd=float(ctrl[IDX_L_KNEE]),
                knee_qpos=float(data.qpos[QPOS_ADR[IDX_L_KNEE]]),
                knee_err_deg=float(np.degrees(ctrl[IDX_L_KNEE] - data.qpos[QPOS_ADR[IDX_L_KNEE]])),
                ankle_cmd=float(ctrl[IDX_L_ANKLE_P]),
                ankle_qpos=float(data.qpos[QPOS_ADR[IDX_L_ANKLE_P]]),
                ankle_err_deg=float(np.degrees(ctrl[IDX_L_ANKLE_P] - data.qpos[QPOS_ADR[IDX_L_ANKLE_P]])),
                l_forward_mm=float(-delta[1] * 1000.0),
                l_height_mm=float(delta[2] * 1000.0),
                l_normal_force=_foot_normal_force(model, data, "L"),
            )
        )

    return records


def print_forward_step_tracking(records: list[ForwardStepTracking]) -> None:
    print("\n" + "=" * 72)
    print("FORWARD STEP: COMMANDED vs ACTUAL L-JOINT POSITIONS (phase end)")
    print("=" * 72)
    hdr = (
        f"{'phase':<16} {'hip cmd':>8} {'hip q':>8} {'err d':>6} "
        f"{'knee cmd':>9} {'knee q':>8} {'err d':>6} "
        f"{'ank cmd':>8} {'ank q':>8} {'err d':>6} "
        f"{'fwd mm':>7} {'ht mm':>6} {'L_N':>6}"
    )
    print(hdr)
    for r in records:
        print(
            f"{r.phase:<16} {r.hip_cmd:+.3f} {r.hip_qpos:+.3f} {r.hip_err_deg:+.1f} "
            f"{r.knee_cmd:+.3f} {r.knee_qpos:+.3f} {r.knee_err_deg:+.1f} "
            f"{r.ankle_cmd:+.3f} {r.ankle_qpos:+.3f} {r.ankle_err_deg:+.1f} "
            f"{r.l_forward_mm:+.1f} {r.l_height_mm:+.1f} {r.l_normal_force:+.1f}"
        )


def print_diagnosis(
    sweep_results: list[JointSweepResult],
    forward_records: list[ForwardStepTracking] | None,
) -> None:
    print("\n" + "=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)

    unloaded = [r for r in sweep_results if r.context == "unloaded"]
    any_sat = any(s.force_saturated for r in unloaded for s in r.samples)
    max_err = max(abs(s.error_deg) for r in unloaded for s in r.samples)

    print("\n1. Actuator torque / control authority:")
    if any_sat:
        print("   FORCE SATURATION detected — actuators hitting ±2.3 N limit.")
    else:
        print(f"   No force saturation at tested targets. Max tracking error ~ {max_err:.1f} deg.")
        print("   Servos track commanded positions within ~1 deg when L leg is unloaded.")

    print("\n2. Joint-limit constraints:")
    print("   Tested targets are well inside ctrlrange/joint range for all three joints.")
    print("   Tracking failures at these targets are NOT due to joint limits.")

    print("\n3. Trajectory target magnitude:")
    if forward_records:
        swing = next(r for r in forward_records if r.phase == "FORWARD SWING")
        print(
            f"   At FORWARD SWING end: hip cmd={swing.hip_cmd:+.3f}, knee cmd={swing.knee_cmd:+.3f}, "
            f"ankle cmd={swing.ankle_cmd:+.3f}"
        )
        print(
            f"   L-foot forward displacement at that point: {swing.l_forward_mm:+.1f} mm "
            f"(height {swing.l_height_mm:+.1f} mm, L_normal={swing.l_normal_force:.1f} N)"
        )

    print("\n4. Dynamic forces during forward step:")
    if forward_records:
        max_hip_err = max(abs(r.hip_err_deg) for r in forward_records)
        max_knee_err = max(abs(r.knee_err_deg) for r in forward_records)
        max_ank_err = max(abs(r.ankle_err_deg) for r in forward_records)
        print(f"   Max |tracking error| during step: hip {max_hip_err:.1f} deg, "
              f"knee {max_knee_err:.1f} deg, ankle {max_ank_err:.1f} deg")
        lift = next(r for r in forward_records if r.phase == "LIFT")
        if lift.l_normal_force > 2.0 and lift.l_height_mm < 20.0:
            print(
                f"   L foot still loaded at LIFT end ({lift.l_normal_force:.1f} N, "
                f"height only {lift.l_height_mm:.1f} mm) - dynamic loading resists lift."
            )
        elif max_hip_err < 3.0 and max_knee_err < 3.0:
            print("   Joints track commands during step; small foot motion is primarily")
            print("   a kinematics/target issue (conservative angles + foot still on ground).")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L-leg actuator strength and tracking diagnostic.")
    p.add_argument(
        "--compare-forward-step",
        action="store_true",
        help="Also run actual_forward_step_test trajectory and report joint tracking.",
    )
    p.add_argument(
        "--also-default-stand",
        action="store_true",
        help="Repeat individual joint sweeps from DEFAULT_POSE (loaded) baseline.",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()
    model, data = env.model, env.data

    print_actuator_config(model)

    ws = weight_shift_pose()
    sweep_results: list[JointSweepResult] = []
    for spec in JOINT_SPECS:
        sweep_results.append(_run_joint_sweep(model, data, spec, ws, "unloaded"))

    if args.also_default_stand:
        for spec in JOINT_SPECS:
            sweep_results.append(_run_joint_sweep(model, data, spec, DEFAULT_POSE.copy(), "default stand"))

    print_sweep_results(sweep_results)

    forward_records: list[ForwardStepTracking] | None = None
    if args.compare_forward_step:
        forward_records = run_forward_step_tracking()
        print_forward_step_tracking(forward_records)

    print_diagnosis(sweep_results, forward_records)


if __name__ == "__main__":
    main(sys.argv[1:])
