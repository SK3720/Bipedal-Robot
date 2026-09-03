"""Correct, signed balance metrics for push-recovery experiments.

Motivation (verified 2026-08-31 with _arrest_probe2.py):
  biped_env._quat_tilt_rad() = 2*acos|q . q_nominal| is the TOTAL rotation angle
  between the chest and the standing orientation. It conflates forward-lean,
  side-lean and YAW. In the golden staged_forward_catch continuation the chest
  yaws ~45 deg and side-leans ~-60 deg while chest_z barely moves; the conflated
  metric reports "tilt 1.4-1.9 rad = fallen" when the real sagittal lean is far
  smaller and the robot is actually spinning / toppling sideways on one foot.

This module returns the components separately plus a support-polygon / capture
-point view and an explicit "recovered" test, so experiments measure the thing
they claim to measure.

Conventions for this project:
  world -Y = anatomical forward   (matches _forward_vel / _forward_mm in the
             stepping scripts and the L/R hip-pitch swing direction)
  world +Z = up ;  floor plane at z = 1.0
  chest body id = 1 ; chest local +Y ~ world +Z when standing
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

FLOOR_Z = 1.0
CHEST_BODY = 1
G = 9.81
NOMINAL_CHEST_Z = 1.26


def _foot_xy_z(model, data, side):
    return data.xpos[model.body(f"{side}_foot").id].copy()


def _foot_contact(model, data, side):
    tag = f"{side}_foot_collision"
    for ci in range(data.ncon):
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[ci].geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[ci].geom2) or ""
        if tag in g1 or tag in g2:
            return True
    return False


def _foot_normal_force(model, data, side):
    tag = f"{side}_foot_collision"
    total = 0.0
    w = np.zeros(6)
    for ci in range(data.ncon):
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[ci].geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[ci].geom2) or ""
        if tag in g1 or tag in g2:
            mujoco.mj_contactForce(model, data, ci, w)
            total += max(0.0, float(w[0]))
    return total


@dataclass
class BalanceState:
    # orientation (degrees, signed)
    up_tilt_deg: float          # angle of chest 'up' axis from world vertical (unsigned)
    fwd_lean_deg: float         # + => leaning forward (-Y)
    side_lean_deg: float        # + => leaning toward +X (robot's left)
    yaw_deg: float              # heading change about vertical
    # translation
    chest_z: float
    com: np.ndarray             # whole-body CoM, world
    com_vel: np.ndarray         # whole-body CoM velocity, world
    com_fwd: float              # -com_y
    com_vfwd: float             # -com_vy
    com_speed_horiz: float
    # support
    l_contact: bool
    r_contact: bool
    l_nf: float
    r_nf: float
    support_fwd_min: float      # rear edge of support (min -Y foot), forward coord
    support_fwd_max: float      # front edge
    com_fwd_rel_support_mm: float     # >0 => CoM ahead of front support edge
    capture_fwd_rel_support_mm: float # >0 => capture point ahead of front edge (must step)
    pitch_rate: float           # chest angular vel about the left-right axis (rad/s, + fwd)


def chest_lean_yaw(model, data):
    R = data.xmat[CHEST_BODY].reshape(3, 3)
    up = R @ np.array([0.0, 1.0, 0.0])
    up_tilt = np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))
    fwd_lean = np.degrees(np.arctan2(-up[1], up[2]))
    side_lean = np.degrees(np.arctan2(up[0], up[2]))
    x_world = R @ np.array([1.0, 0.0, 0.0])
    yaw = np.degrees(np.arctan2(x_world[1], x_world[0]))
    return up_tilt, fwd_lean, side_lean, yaw


def sample_balance(model, data) -> BalanceState:
    mujoco.mj_subtreeVel(model, data)
    up_tilt, fwd_lean, side_lean, yaw = chest_lean_yaw(model, data)
    com = data.subtree_com[CHEST_BODY].copy()
    com_v = data.subtree_linvel[CHEST_BODY].copy()

    lc = _foot_contact(model, data, "L")
    rc = _foot_contact(model, data, "R")
    lf = _foot_xy_z(model, data, "L")
    rf = _foot_xy_z(model, data, "R")

    feet_fwd = []
    if lc:
        feet_fwd.append(-lf[1])
    if rc:
        feet_fwd.append(-rf[1])
    if not feet_fwd:  # airborne: use both nominal
        feet_fwd = [-lf[1], -rf[1]]
    s_min, s_max = min(feet_fwd), max(feet_fwd)

    com_fwd = -float(com[1])
    com_vfwd = -float(com_v[1])
    h = max(float(com[2]) - FLOOR_Z, 0.05)
    capture_fwd = com_fwd + com_vfwd * np.sqrt(h / G)

    # chest pitch rate about world left-right axis (world X); qvel[3:6] is LOCAL,
    # rotate to world.
    w_local = data.qvel[3:6].copy()
    w_world = data.xmat[CHEST_BODY].reshape(3, 3) @ w_local
    pitch_rate = -float(w_world[0])  # sign so + == pitching forward (toward -Y)

    return BalanceState(
        up_tilt_deg=up_tilt,
        fwd_lean_deg=fwd_lean,
        side_lean_deg=side_lean,
        yaw_deg=yaw,
        chest_z=float(data.qpos[2]),
        com=com,
        com_vel=com_v,
        com_fwd=com_fwd,
        com_vfwd=com_vfwd,
        com_speed_horiz=float(np.hypot(com_v[0], com_v[1])),
        l_contact=lc,
        r_contact=rc,
        l_nf=_foot_normal_force(model, data, "L"),
        r_nf=_foot_normal_force(model, data, "R"),
        support_fwd_min=s_min,
        support_fwd_max=s_max,
        com_fwd_rel_support_mm=(com_fwd - s_max) * 1000.0,
        capture_fwd_rel_support_mm=(capture_fwd - s_max) * 1000.0,
        pitch_rate=pitch_rate,
    )


# ---- outcome classification ---------------------------------------------------

@dataclass
class RecoveryVerdict:
    fell: bool
    fell_step: int | None
    fell_reason: str
    recovered: bool
    peak_up_tilt_deg: float
    peak_fwd_lean_deg: float
    peak_side_lean_deg_abs: float
    end_up_tilt_deg: float
    end_fwd_lean_deg: float
    end_side_lean_deg: float
    end_chest_z: float
    end_com_speed: float
    end_l_contact: bool
    end_r_contact: bool
    settle_mean_up_tilt_deg: float
    settle_mean_com_speed: float
    label: str


# thresholds
FALL_UP_TILT_DEG = 45.0        # chest 'up' axis > 45 deg from vertical => going down
FALL_CHEST_Z_DROP = 0.14       # m below nominal
REC_UP_TILT_DEG = 12.0
REC_LEAN_DEG = 10.0
REC_COM_SPEED = 0.10
REC_CHEST_Z_DROP = 0.05


def classify_run(samples: list[tuple[int, BalanceState]], settle_frac: float = 0.2) -> RecoveryVerdict:
    """samples: list of (step, BalanceState) over the full post-push window."""
    fell = False
    fell_step = None
    fell_reason = ""
    peak_up = 0.0
    peak_fwd = -1e9
    peak_side = 0.0
    for step, s in samples:
        peak_up = max(peak_up, s.up_tilt_deg)
        peak_fwd = max(peak_fwd, s.fwd_lean_deg)
        peak_side = max(peak_side, abs(s.side_lean_deg))
        if not fell and (
            s.up_tilt_deg > FALL_UP_TILT_DEG
            or s.chest_z < NOMINAL_CHEST_Z - FALL_CHEST_Z_DROP
        ):
            fell = True
            fell_step = step
            fell_reason = (
                "up_tilt>45deg" if s.up_tilt_deg > FALL_UP_TILT_DEG else "chest dropped"
            )

    n_settle = max(1, int(len(samples) * settle_frac))
    tail = [s for _, s in samples[-n_settle:]]
    end = tail[-1]
    settle_mean_up = float(np.mean([s.up_tilt_deg for s in tail]))
    settle_mean_v = float(np.mean([s.com_speed_horiz for s in tail]))

    recovered = (
        not fell
        and settle_mean_up < REC_UP_TILT_DEG
        and abs(end.fwd_lean_deg) < REC_LEAN_DEG
        and abs(end.side_lean_deg) < REC_LEAN_DEG
        and settle_mean_v < REC_COM_SPEED
        and end.chest_z > NOMINAL_CHEST_Z - REC_CHEST_Z_DROP
        and end.l_contact and end.r_contact
    )
    if fell:
        label = f"FELL ({fell_reason}) @ step {fell_step}"
    elif recovered:
        label = "RECOVERED (upright, still, both feet down)"
    else:
        label = "NO-FALL but not settled (leaning / drifting / one-foot)"

    return RecoveryVerdict(
        fell=fell, fell_step=fell_step, fell_reason=fell_reason,
        recovered=recovered,
        peak_up_tilt_deg=peak_up, peak_fwd_lean_deg=peak_fwd,
        peak_side_lean_deg_abs=peak_side,
        end_up_tilt_deg=end.up_tilt_deg, end_fwd_lean_deg=end.fwd_lean_deg,
        end_side_lean_deg=end.side_lean_deg, end_chest_z=end.chest_z,
        end_com_speed=end.com_speed_horiz,
        end_l_contact=end.l_contact, end_r_contact=end.r_contact,
        settle_mean_up_tilt_deg=settle_mean_up, settle_mean_com_speed=settle_mean_v,
        label=label,
    )
