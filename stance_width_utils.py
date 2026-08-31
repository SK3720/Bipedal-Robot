"""Helpers for evaluation-only stance-width variants without modifying robot/robot.xml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from biped_env import STANDING_QUAT

ROBOT_XML = Path("robot/robot.xml")
HIP_CENTER_X = 0.03493175
L_HIP_YZ = (-0.02575, 0.0452117)
R_HIP_YZ = (-0.02575, 0.0452634)

STANCE_WIDTHS_MM = (40, 80, 120)


def load_model_with_stance_width(target_width_m: float, xml_path: Path = ROBOT_XML) -> mujoco.MjModel:
    """Load a fresh MuJoCo model with symmetric hip spacing for the requested foot width."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    half = target_width_m / 2.0
    l_hip_id = model.body("L_hip").id
    r_hip_id = model.body("R_hip").id
    model.body_pos[l_hip_id, 0] = HIP_CENTER_X + half
    model.body_pos[l_hip_id, 1] = L_HIP_YZ[0]
    model.body_pos[l_hip_id, 2] = L_HIP_YZ[1]
    model.body_pos[r_hip_id, 0] = HIP_CENTER_X - half
    model.body_pos[r_hip_id, 1] = R_HIP_YZ[0]
    model.body_pos[r_hip_id, 2] = R_HIP_YZ[1]
    return model


def measure_foot_stance_width_m(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    l_foot_id = model.body("L_foot").id
    r_foot_id = model.body("R_foot").id
    return float(abs(data.xpos[l_foot_id, 0] - data.xpos[r_foot_id, 0]))


def quat_tilt_rad(data: mujoco.MjData) -> float:
    quat = data.qpos[3:7]
    cos_half_angle = np.clip(abs(np.dot(quat, STANDING_QUAT)), 0.0, 1.0)
    return float(2.0 * np.arccos(cos_half_angle))


@dataclass
class StanceStepEnv:
    """Minimal shim so dynamic_step_test._run_trial can run on alternate stance models."""

    model: mujoco.MjModel
    data: mujoco.MjData

    def _quat_tilt_rad(self) -> float:
        return quat_tilt_rad(self.data)
