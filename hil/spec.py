"""sim2real_v1 -- HARDWARE INTERFACE SPECIFICATION (single source of truth).

Everything the HIL / hardware code needs to know about the `runs/sim2real_v1`
policy, extracted verbatim from `biped_sim2real_env.py`, `biped_locomotion_env.py`
and `robot/_exp_hands_3x.xml` and cross-checked empirically (see
`hil/validate_stack.py`).  Do not edit numbers here by hand -- regenerate the
exported constants with `hil/export_consts.py` if the policy/model changes.

READ `hil/README.md` for the prose walkthrough.
"""
from __future__ import annotations

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
MODEL_XML = os.path.join(REPO, "robot", "_exp_hands_3x.xml")
POLICY_DIR = os.path.join(REPO, "runs", "sim2real_v1")
POLICY_FILE = os.path.join(POLICY_DIR, "policy_best.pth")        # == policy_dagger3
VECNORM_FILE = os.path.join(POLICY_DIR, "vecnormalize_best.pkl")  # obs mean/var
CONSTS_FILE = os.path.join(POLICY_DIR, "base_controller_consts.npz")
# torch-free copy of the policy MLP + obs mean/var (made by hil/export_pi_bundle.py);
# this is what the Raspberry Pi uses.  Preferred over the .pth/.pkl when present.
HIL_POLICY_FILE = os.path.join(POLICY_DIR, "hil_policy.npz")

# ----------------------------------------------------------------------------
# RATES
# ----------------------------------------------------------------------------
SIM_TIMESTEP = 0.001            # s   (MuJoCo)
FRAME_SKIP = 5                  #     mj_steps per control step
CONTROL_HZ = 200.0             # Hz  = 1 / (SIM_TIMESTEP * FRAME_SKIP)
CONTROL_DT = 1.0 / CONTROL_HZ  # s   = 0.005

GAIT_HZ = 0.80                 # gait cycles per second (one full L+R cycle)
# gait-clock advance PER CONTROL STEP.  On hardware advance _phase by
#   PHASE_RATE / actual_loop_hz   (NOT by PHASE_PER_STEP if you don't hit 200 Hz)
PHASE_RATE = 2.0 * np.pi * GAIT_HZ                      # rad / s     = 5.0265
PHASE_PER_CONTROL_STEP = PHASE_RATE * CONTROL_DT        # rad / step  = 0.025133
GAIT_CYCLE_S = 1.0 / GAIT_HZ                            # 1.25 s  (~250 control steps)

SPEED_TGT = 0.30              # m/s -- the value the policy was DISTILLED with
                              # (distill_s2r.py --speed default).  Fixed command.

# ----------------------------------------------------------------------------
# JOINTS  (MuJoCo qpos / qvel / actuator layout -- from the XML, in XML order)
# ----------------------------------------------------------------------------
#   qpos: [0:3] base xyz | [3:7] base quat (w,x,y,z) | [7:22] 15 joints
#   qvel: [0:3] base lin vel (world) | [3:6] base ang vel (BODY frame) | [6:21] 15 joints
#   nq=22  nv=21  nu=15
#
# The 15 joints / actuators, in MuJoCo index order (ctrl[i] -> joint i -> qpos[7+i]):
JOINTS_MJ = [
    "Chest_neck",            # 0
    "Chest_L_shoulder",      # 1
    "L_arm_L_elbow",         # 2
    "Chest_R_shoulder",      # 3
    "R_arm_R_elbow",         # 4
    "Chest_L_hip_roll",      # 5
    "L_hip_L_hip_pitch",     # 6
    "L_leg_L_knee",          # 7
    "L_shin_L_ankle_pitch",  # 8
    "L_ankle_L_ankle_roll",  # 9
    "Chest_R_hip_roll",      # 10
    "R_hip_R_hip_pitch",     # 11
    "R_leg_R_knee",          # 12
    "R_shin_R_ankle_pitch",  # 13
    "R_ankle_R_ankle_roll",  # 14
]

# All 15 actuators are POSITION servos:  tau = clip( kp*(ctrl - q), +/-FRC )
ACT_KP = 30.0
ACT_FORCE_LIMIT = 2.3        # N*m  (hard actuator torque limit -- do NOT exceed)

# ctrlrange per actuator == joint limit (from the XML), MuJoCo index order:
CTRL_RANGE_MJ = np.array([
    [-3.141593,  3.141593],  # 0  neck
    [-3.577925,  0.174533],  # 1  L_shoulder
    [-2.268928,  2.268928],  # 2  L_elbow
    [-0.174533,  3.577925],  # 3  R_shoulder
    [-2.268928,  2.268928],  # 4  R_elbow
    [-0.087266,  0.610865],  # 5  L_hip_roll
    [-2.007129,  2.094395],  # 6  L_hip_pitch
    [-2.748894,  1.178097],  # 7  L_knee
    [-2.487094,  1.788962],  # 8  L_ankle_pitch
    [-0.785398,  1.832596],  # 9  L_ankle_roll
    [-0.610865,  0.087266],  # 10 R_hip_roll
    [-2.094395,  2.007129],  # 11 R_hip_pitch
    [-1.178097,  2.748894],  # 12 R_knee
    [-1.788962,  2.487094],  # 13 R_ankle_pitch
    [-1.832596,  0.785398],  # 14 R_ankle_roll
], dtype=np.float64)

# ----------------------------------------------------------------------------
# POLICY OBSERVATION / ACTION ORDER
# ----------------------------------------------------------------------------
# The policy's 14-DOF "leg+arm" ordering  (== biped_locomotion_env.ACT_CTRL).
# It is used for BOTH the joint block of the observation AND the action vector.
#   index -> MuJoCo ctrl index
ACT_CTRL = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4]
# ...which is these joints, in this order:
POLICY_JOINT_ORDER = [
    "Chest_L_hip_roll",      # 0
    "L_hip_L_hip_pitch",     # 1
    "L_leg_L_knee",          # 2
    "L_shin_L_ankle_pitch",  # 3
    "L_ankle_L_ankle_roll",  # 4
    "Chest_R_hip_roll",      # 5
    "R_hip_R_hip_pitch",     # 6
    "R_leg_R_knee",          # 7
    "R_shin_R_ankle_pitch",  # 8
    "R_ankle_R_ankle_roll",  # 9
    "Chest_L_shoulder",      # 10
    "L_arm_L_elbow",         # 11
    "Chest_R_shoulder",      # 12
    "R_arm_R_elbow",         # 13
]
# The NECK (MuJoCo ctrl 0) is NOT in the policy -- it is always commanded to 0.

# per-joint scale (rad) applied to the policy output before it becomes a
# position residual:   residual = action * ACT_SCALE      (POLICY_JOINT_ORDER)
ACT_SCALE = np.array([0.20, 0.52, 0.55, 0.34, 0.20,
                      0.20, 0.52, 0.55, 0.34, 0.20,
                      0.55, 0.42, 0.55, 0.42], dtype=np.float64)

# ---- observation vector (217-D) layout ----------------------------------------
# obs = concat( frame[t-3], frame[t-2], frame[t-1], frame[t], extra )
#   4 stacked "sensor frames" (oldest first) of SFRAME_DIM each, then EXTRA_DIM.
SFRAME_DIM = 50
HIST = 4
EXTRA_DIM = 17
OBS_DIM = HIST * SFRAME_DIM + EXTRA_DIM      # 217

# one sensor frame (50-D), in order:
SFRAME_LAYOUT = [
    ("up",          3),   # chest local +Y axis, expressed in WORLD frame  (~[0,0,1] upright)
    ("fwd_xy",      2),   # chest local -Z axis in WORLD frame, x&y only    (~[0,1] heading ref)
    ("gyro",        3),   # R_chest.T @ omega_body   (see README -- NOT raw gyro)
    ("accel",       3),   # specific force in chest BODY frame, m/s^2       (~[0,9.81,0] at rest)
    ("joint_pos",  14),   # POLICY_JOINT_ORDER, rad
    ("joint_vel",  14),   # POLICY_JOINT_ORDER, rad/s
    ("contact",     2),   # [L_foot, R_foot]  1.0 = in contact, 0.0 = not
    ("v_est",       3),   # leg-odometry base linear-velocity estimate, chest BODY frame, m/s
    ("foot_L_rel",  3),   # L_foot position in the chest BODY frame (FK), m
    ("foot_R_rel",  3),   # R_foot position in the chest BODY frame (FK), m
]
assert sum(n for _, n in SFRAME_LAYOUT) == SFRAME_DIM

# the trailing "extra" block (17-D):
EXTRA_LAYOUT = [
    ("clock_sin",   1),   # sin(gait_phase)
    ("clock_cos",   1),   # cos(gait_phase)
    ("speed_tgt",   1),   # constant SPEED_TGT
    ("prev_action", 14),  # last policy output, clipped to [-1,1], POLICY_JOINT_ORDER
]
assert sum(n for _, n in EXTRA_LAYOUT) == EXTRA_DIM

# ----------------------------------------------------------------------------
# NORMALISATION
# ----------------------------------------------------------------------------
# The policy expects:   x_norm = clip( (obs - MEAN) / sqrt(VAR + 1e-8), -10, 10 )
# MEAN, VAR are 217-D arrays loaded from VECNORM_FILE (VecNormalize.obs_rms).
# They are FIXED (from the distillation dataset) -- never updated at run time.
NORM_EPS = 1e-8
NORM_CLIP = 10.0

# ----------------------------------------------------------------------------
# SENSOR-NOISE MODEL the policy was trained against (1-sigma).
# Use these as the *upper bound* on acceptable real-sensor noise; if your real
# sensors are noisier, widen these in biped_sim2real_env.py and re-distill.
# ----------------------------------------------------------------------------
TRAIN_NOISE = dict(
    imu_tilt_rad=0.015,      # on the `up` / `fwd` unit-vector components
    gyro_rad_s=0.02,
    gyro_bias_rad_s=0.03,    # constant per episode
    accel_m_s2=0.35,
    enc_pos_rad=0.004,
    enc_vel_rad_s=0.06,
    v_est_m_s=0.03,
    contact_dropout=0.03,    # P(contact bit flipped)
)

# ----------------------------------------------------------------------------
# SAFETY ENVELOPES  (used by hil/safety.py)
# ----------------------------------------------------------------------------
SAFE = dict(
    # sensor sanity
    accel_norm_min=2.0, accel_norm_max=60.0,      # m/s^2   (~1g rest; brief dips
                                                  # toward free-fall are normal mid-step)
    gyro_abs_max=25.0,                            # rad/s   per axis
    up_norm_min=0.5, up_norm_max=1.6,             # `up` should be ~unit length
    joint_vel_abs_max=30.0,                       # rad/s
    v_est_abs_max=3.0,                            # m/s
    # attitude / fall
    tilt_deg_warn=25.0, tilt_deg_abort=40.0,      # angle of `up` from world +Z
    # timing
    loop_hz_min=120.0,                            # abort if the loop drops below this
    loop_jitter_ms_max=6.0,
    # command
    cmd_margin_rad=0.02,                          # keep this far inside CTRL_RANGE
)


# ----------------------------------------------------------------------------
def load_consts():
    """K_att (15x6), qpos0, qvel0, ctrl0, ctrlrange, act_scale -- as a dict."""
    z = np.load(CONSTS_FILE)
    return {k: z[k] for k in z.files}


def load_norm():
    """(mean, var) 217-D arrays from the VecNormalize pickle."""
    import pickle
    with open(VECNORM_FILE, "rb") as f:
        vn = pickle.load(f)
    return np.asarray(vn.obs_rms.mean, np.float64), np.asarray(vn.obs_rms.var, np.float64)
