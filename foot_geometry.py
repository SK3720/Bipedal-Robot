"""Verified foot/sole geometry metrics from MuJoCo model meshes.

Uses L_foot_collision mesh vertices — do not trust heuristic body-axis estimates.
"""

from __future__ import annotations

import mujoco
import numpy as np

FLOOR_Z = 1.0


def _geom_mesh_vertices_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_name: str,
) -> np.ndarray:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[gid]
    vert_adr = model.mesh_vertadr[mesh_id]
    vert_num = model.mesh_vertnum[mesh_id]
    local = model.mesh_vert[vert_adr : vert_adr + vert_num].reshape(-1, 3)
    pos = data.geom_xpos[gid]
    rot = data.geom_xmat[gid].reshape(3, 3)
    return (rot @ local.T).T + pos


def l_foot_mesh_vertices_world(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return _geom_mesh_vertices_world(model, data, "L_foot_collision")


def sole_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float | np.ndarray]:
    """Compute sole geometry from foot collision mesh vertices."""
    verts = l_foot_mesh_vertices_world(model, data)
    lowest_z = float(np.min(verts[:, 2]))
    tol = 0.002
    sole_pts = verts[verts[:, 2] <= lowest_z + tol]
    if len(sole_pts) < 3:
        sole_pts = verts

    centroid = sole_pts.mean(axis=0)
    centered = sole_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[2]
    if normal[2] < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)

    sole_angle_from_horizontal_rad = float(np.arccos(np.clip(normal[2], -1.0, 1.0)))
    sole_angle_from_vertical_rad = float(np.arctan2(np.linalg.norm(normal[:2]), normal[2]))

    # Forward = world -Y (anatomical forward in this project).
    fwd = np.array([0.0, -1.0, 0.0])
    heel_score = sole_pts @ fwd
    toe_score = -sole_pts @ fwd
    heel = sole_pts[int(np.argmax(heel_score))]
    toe = sole_pts[int(np.argmax(toe_score))]

    heel_toe = toe - heel
    ht_horiz = np.array([heel_toe[0], heel_toe[1], 0.0])
    ht_len = np.linalg.norm(ht_horiz)
    if ht_len > 1e-9:
        heel_toe_pitch_rad = float(
            np.arctan2(heel_toe[2], ht_len)
        )
    else:
        heel_toe_pitch_rad = 0.0

    foot_body = model.body("L_foot").id
    body_pos = data.xpos[foot_body].copy()
    body_rot = data.xmat[foot_body].reshape(3, 3)

    return {
        "sole_normal": normal,
        "sole_angle_from_horizontal_rad": sole_angle_from_horizontal_rad,
        "sole_angle_from_vertical_rad": sole_angle_from_vertical_rad,
        "heel_world": heel,
        "toe_world": toe,
        "heel_toe_pitch_rad": heel_toe_pitch_rad,
        "heel_clearance_mm": (heel[2] - FLOOR_Z) * 1000.0,
        "toe_clearance_mm": (toe[2] - FLOOR_Z) * 1000.0,
        "sole_lowest_z": lowest_z,
        "foot_body_pos": body_pos,
        "foot_body_euler_xyz_rad": _mat_to_euler_xyz(body_rot),
    }


def legacy_foot_pitch_rad(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Old heuristic used in prior experiments (may not equal sole angle)."""
    rot = data.xmat[model.body("L_foot").id].reshape(3, 3)
    candidates = [
        rot @ np.array([0.0, 0.0, -1.0]),
        rot @ np.array([0.0, 0.0, 1.0]),
        rot @ np.array([0.0, -1.0, 0.0]),
        rot @ np.array([0.0, 1.0, 0.0]),
    ]
    sole = max(candidates, key=lambda v: v[2])
    return float(np.arctan2(np.linalg.norm(sole[:2]), sole[2]))


def _mat_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy > 1e-9:
        x = np.arctan2(rot[2, 1], rot[2, 2])
        y = np.arctan2(-rot[2, 0], sy)
        z = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        x = np.arctan2(-rot[1, 2], rot[1, 1])
        y = np.arctan2(-rot[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])
