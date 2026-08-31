"""Isolated L ankle pitch vs roll axis diagnostic.

Stand the robot, freeze all joints except one L ankle DOF at a time.
Sweep through a large range and record physical foot/sole response.

Evaluation only — does not modify robot.xml or gait scripts.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np

from biped_env import BipedalWalkEnv, CHEST_Z_CONTACT, DEFAULT_POSE, STANDING_QUAT
from foot_geometry import legacy_foot_pitch_rad, sole_metrics

IDX_L_ANKLE_P = 8
IDX_L_ANKLE_R = 9
QPOS_ANKLE_P = 15
QPOS_ANKLE_R = 16

SETTLE_STEPS = 800
SWEEP_STEPS_PER_TARGET = 400

PITCH_TARGETS = np.linspace(-0.6, 1.0, 9)
ROLL_TARGETS = np.linspace(-0.5, 0.8, 9)


def _reset_standing(env: BipedalWalkEnv) -> None:
    model, data = env.model, env.data
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _sample(env: BipedalWalkEnv, label: str, ankle_p: float, ankle_r: float) -> dict:
    model, data = env.model, env.data
    m = sole_metrics(model, data)
    return {
        "label": label,
        "ankle_pitch_cmd": ankle_p,
        "ankle_roll_cmd": ankle_r,
        "ankle_pitch_qpos": float(data.qpos[QPOS_ANKLE_P]),
        "ankle_roll_qpos": float(data.qpos[QPOS_ANKLE_R]),
        "foot_body_pos": m["foot_body_pos"],
        "sole_angle_horiz_rad": m["sole_angle_from_horizontal_rad"],
        "sole_angle_vert_rad": m["sole_angle_from_vertical_rad"],
        "heel_toe_pitch_rad": m["heel_toe_pitch_rad"],
        "legacy_pitch_rad": legacy_foot_pitch_rad(model, data),
        "heel_clearance_mm": m["heel_clearance_mm"],
        "toe_clearance_mm": m["toe_clearance_mm"],
        "heel_y": float(m["heel_world"][1]),
        "toe_y": float(m["toe_world"][1]),
        "heel_z": float(m["heel_world"][2]),
        "toe_z": float(m["toe_world"][2]),
    }


def _sweep_joint(
    env: BipedalWalkEnv,
    ctrl_idx: int,
    targets: np.ndarray,
    fixed_ankle_p: float,
    fixed_ankle_r: float,
    viewer: mujoco.viewer.Handle | None,
    slow: bool,
) -> list[dict]:
    model, data = env.model, env.data
    results: list[dict] = []
    for tgt in targets:
        ctrl = DEFAULT_POSE.copy()
        ctrl[IDX_L_ANKLE_P] = fixed_ankle_p
        ctrl[IDX_L_ANKLE_R] = fixed_ankle_r
        ctrl[ctrl_idx] = float(tgt)
        for _ in range(SWEEP_STEPS_PER_TARGET):
            data.ctrl[:15] = ctrl
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                time.sleep(0.004 if slow else 0.0)
        label = f"{'pitch' if ctrl_idx == IDX_L_ANKLE_P else 'roll'}={tgt:.2f}"
        results.append(
            _sample(
                env,
                label,
                float(ctrl[IDX_L_ANKLE_P]),
                float(ctrl[IDX_L_ANKLE_R]),
            )
        )
    return results


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    print(
        "target   pitch_q  roll_q   sole_h°  heel_t°  legacy°  heel_z   toe_z    heel_clr toe_clr"
    )
    for r in rows:
        print(
            f"{r['label']:<8} "
            f"{r['ankle_pitch_qpos']:7.3f} "
            f"{r['ankle_roll_qpos']:7.3f} "
            f"{np.degrees(r['sole_angle_horiz_rad']):7.1f} "
            f"{np.degrees(r['heel_toe_pitch_rad']):7.1f} "
            f"{np.degrees(r['legacy_pitch_rad']):7.1f} "
            f"{r['heel_z']:7.4f} "
            f"{r['toe_z']:7.4f} "
            f"{r['heel_clearance_mm']:7.1f} "
            f"{r['toe_clearance_mm']:7.1f}"
        )


def _infer_mapping(pitch_rows: list[dict], roll_rows: list[dict]) -> None:
    p0 = pitch_rows[0]
    p1 = pitch_rows[-1]
    r0 = roll_rows[0]
    r1 = roll_rows[-1]

    dp = p1["sole_angle_horiz_rad"] - p0["sole_angle_horiz_rad"]
    dr = r1["sole_angle_horiz_rad"] - r0["sole_angle_horiz_rad"]
    dht_p = p1["heel_toe_pitch_rad"] - p0["heel_toe_pitch_rad"]
    dht_r = r1["heel_toe_pitch_rad"] - r0["heel_toe_pitch_rad"]

    print("\n--- PHYSICAL MAPPING (experimental) ---")
    print(
        f"L ankle PITCH qpos {p0['ankle_pitch_qpos']:.2f} -> {p1['ankle_pitch_qpos']:.2f}: "
        f"sole from horizontal {np.degrees(p0['sole_angle_horiz_rad']):.1f}° -> "
        f"{np.degrees(p1['sole_angle_horiz_rad']):.1f}° "
        f"(delta {np.degrees(dp):+.1f}°); "
        f"heel-toe pitch delta {np.degrees(dht_p):+.1f}°"
    )
    print(
        f"L ankle ROLL  qpos {r0['ankle_roll_qpos']:.2f} -> {r1['ankle_roll_qpos']:.2f}: "
        f"sole from horizontal {np.degrees(r0['sole_angle_horiz_rad']):.1f}° -> "
        f"{np.degrees(r1['sole_angle_horiz_rad']):.1f}° "
        f"(delta {np.degrees(dr):+.1f}°); "
        f"heel-toe pitch delta {np.degrees(dht_r):+.1f}°"
    )
    if abs(dp) > abs(dr):
        print("SOLE SAGITTAL FLATTENING: primarily L ankle PITCH")
    else:
        print("SOLE SAGITTAL FLATTENING: primarily L ankle ROLL (unexpected — verify visually)")
    if dp < 0:
        print("Increasing +ankle pitch qpos -> SOLE MORE PARALLEL TO GROUND (flatter)")
    else:
        print("Increasing +ankle pitch qpos -> SOLE MORE TILTED (steeper)")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolated L ankle axis diagnostic.")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = BipedalWalkEnv()
    model, data = env.model, env.data

    def run_diag(viewer: mujoco.viewer.Handle | None) -> None:
        _reset_standing(env)
        for _ in range(SETTLE_STEPS):
            data.ctrl[:15] = DEFAULT_POSE
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

        print("Isolated L ankle diagnostic — standing, all joints frozen except tested ankle DOF")
        pitch_rows = _sweep_joint(
            env, IDX_L_ANKLE_P, PITCH_TARGETS, 0.0, 0.0, viewer, args.slow
        )
        _reset_standing(env)
        for _ in range(SETTLE_STEPS):
            data.ctrl[:15] = DEFAULT_POSE
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
        roll_rows = _sweep_joint(
            env, IDX_L_ANKLE_R, ROLL_TARGETS, 0.0, 0.0, viewer, args.slow
        )
        _print_table("L ANKLE PITCH SWEEP (roll=0)", pitch_rows)
        _print_table("L ANKLE ROLL SWEEP (pitch=0)", roll_rows)
        _infer_mapping(pitch_rows, roll_rows)

    if args.headless:
        run_diag(None)
        return

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0.0, -0.05, 1.0]
        v.cam.distance = 1.4
        v.cam.azimuth = 88
        v.cam.elevation = -15
        run_diag(v)


if __name__ == "__main__":
    main(sys.argv[1:])
