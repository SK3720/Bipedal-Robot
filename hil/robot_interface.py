"""Robot I/O abstraction for HIL.

`RobotInterface` is the contract the HIL loop uses.  Two implementations:

  SimRobot       -- wraps the MuJoCo model; used to exercise the ENTIRE HIL stack
                    (obs build -> policy -> base controller -> logging -> timing)
                    end to end with no hardware.  It DOES apply the command, so
                    `run_hil.py --sim` produces a full walking session.

  HardwareRobot  -- the real robot.  Every method that talks to a device is a
                    clearly marked  >>> WIRE UP <<<  stub.  `command()` is a
                    logging NO-OP unless `enable_motors=True` AND the stub is
                    implemented -- log-only is the default and safe path.

SensorReading fields are all in the CHEST BODY frame and POLICY_JOINT_ORDER,
i.e. exactly what ObservationBuilder / BaseController expect.  See README for the
IMU-mounting / joint-sign / units requirements the hardware side must satisfy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from hil import spec


@dataclass
class SensorReading:
    t_wall: float                 # time.perf_counter() when the read completed
    imu_quat_wxyz: np.ndarray     # (4,) chest orientation, body->world, unit
    imu_gyro: np.ndarray          # (3,) body-frame angular velocity, rad/s (bias-removed)
    imu_accel: np.ndarray         # (3,) body-frame specific force, m/s^2 (~[0,9.81,0] at rest)
    joint_pos: np.ndarray         # (14,) POLICY_JOINT_ORDER, rad
    joint_vel: np.ndarray         # (14,) POLICY_JOINT_ORDER, rad/s
    contact_L: bool
    contact_R: bool
    # optional raw extras for logging / debugging (may be None)
    raw: dict | None = None


class RobotInterface:
    def read(self) -> SensorReading:
        raise NotImplementedError

    def command(self, u15: np.ndarray) -> None:
        """u15: 15-D joint POSITION command, MuJoCo ctrl order (neck first)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


# ============================================================ SIM
class SimRobot(RobotInterface):
    """The MuJoCo model, stepping FRAME_SKIP mj_steps per `command()`.
    IMU signals are built to match biped_sim2real_env._sensor_frame exactly
    (noise optional).  Use for stack validation and dry-runs."""

    def __init__(self, noise=False, seed=0, start_standing=True):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(spec.MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.noise = bool(noise)
        self._qadr = np.array([7 + i for i in spec.ACT_CTRL])
        self._vadr = np.array([6 + i for i in spec.ACT_CTRL])
        self.b_chest = self.model.body("Chest").id
        self.b_lf = self.model.body("L_foot").id
        self.b_rf = self.model.body("R_foot").id
        c = spec.load_consts()
        self.data.qpos[:] = c["qpos0"]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:15] = c["ctrl0"]
        mujoco.mj_forward(self.model, self.data)
        if start_standing:                       # settle a few steps holding ctrl0
            for _ in range(20):
                mujoco.mj_step(self.model, self.data)
        self._prev_cvel = self.data.cvel[self.b_chest][3:6].copy()
        self._gyro_bias = (self.rng.normal(0, spec.TRAIN_NOISE["gyro_bias_rad_s"], 3)
                           if self.noise else np.zeros(3))

    def _foot_contact_force(self, side):
        tag = f"{side}_foot_collision"
        d, m = self.data, self.model
        tot = 0.0
        w = np.zeros(6)
        for ci in range(d.ncon):
            g1 = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_GEOM, d.contact[ci].geom1) or ""
            g2 = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_GEOM, d.contact[ci].geom2) or ""
            if tag in g1 or tag in g2:
                self.mj.mj_contactForce(m, d, ci, w)
                tot += max(0.0, float(w[0]))
        return tot

    def read(self) -> SensorReading:
        d = self.data
        R = d.xmat[self.b_chest].reshape(3, 3)
        n = self.rng.normal
        q = d.qpos[3:7].copy()
        gyro = d.qvel[3:6].copy()
        if self.noise:
            gyro = gyro + self._gyro_bias + n(0, spec.TRAIN_NOISE["gyro_rad_s"], 3)
        cvel = d.cvel[self.b_chest][3:6].copy()
        a_world = (cvel - self._prev_cvel) / spec.CONTROL_DT
        accel = R.T @ (a_world + np.array([0.0, 0.0, 9.81]))   # G == recovery_metrics.G
        if self.noise:
            accel = accel + n(0, spec.TRAIN_NOISE["accel_m_s2"], 3)
        jp = d.qpos[self._qadr].copy()
        jv = d.qvel[self._vadr].copy()
        if self.noise:
            jp = jp + n(0, spec.TRAIN_NOISE["enc_pos_rad"], 14)
            jv = jv + n(0, spec.TRAIN_NOISE["enc_vel_rad_s"], 14)
        cl = self._foot_contact_force("L") > 6.0
        cr = self._foot_contact_force("R") > 6.0
        if self.noise:
            if self.rng.random() < spec.TRAIN_NOISE["contact_dropout"]:
                cl = not cl
            if self.rng.random() < spec.TRAIN_NOISE["contact_dropout"]:
                cr = not cr
        return SensorReading(time.perf_counter(), q, gyro, accel, jp, jv, cl, cr,
                             raw=dict(chest_z=float(d.qpos[2]),
                                      up_world=(R @ np.array([0., 1., 0.])).copy()))

    def command(self, u15):
        self._prev_cvel = self.data.cvel[self.b_chest][3:6].copy()
        self.data.ctrl[:15] = np.asarray(u15, float)
        for _ in range(spec.FRAME_SKIP):
            self.mj.mj_step(self.model, self.data)

    def fell(self):
        d = self.data
        R = d.xmat[self.b_chest].reshape(3, 3)
        up = R @ np.array([0.0, 1.0, 0.0])
        tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
        return tilt > 48.0 or d.qpos[2] < 1.2692 - 0.30


# ============================================================ HARDWARE  (stubs)
class HardwareRobot(RobotInterface):
    """
    >>> NOTHING BELOW IS WIRED TO A DEVICE. <<<
    Fill in the four marked methods with your real drivers.  Until then `read()`
    raises and `command()` is a no-op, so run_hil.py can still import / dry-run.

    Requirements the wiring MUST satisfy (see README "Hardware requirements"):
      * IMU axes aligned to the MuJoCo chest body frame (X=chest-X, Y=chest-UP,
        Z=chest-Z); if not, set self.R_mount so  x_chest = R_mount @ x_imu.
      * IMU quaternion is body->world, unit, [w,x,y,z]; "world" zeroed at startup
        with the robot held in the nominal standing pose.
      * gyro in rad/s, gyro bias removed (calibrate at rest).
      * accel in m/s^2 specific force (gravity included; ~9.81 at rest).
      * encoders in rad, MuJoCo sign & zero (validate each joint against
        interactive_test_joints.py in sim).  Order = POLICY_JOINT_ORDER.
      * contact = boolean per foot.
    """

    def __init__(self, enable_motors=False, imu_mount_R=None):
        self.enable_motors = bool(enable_motors)
        self.R_mount = np.eye(3) if imu_mount_R is None else np.asarray(imu_mount_R, float)
        self._gyro_bias = np.zeros(3)
        # >>> WIRE UP <<<  open serial/CAN/I2C buses, spin up the IMU, etc.
        #   self.imu = ...
        #   self.servo_bus = ...
        #   self._gyro_bias = self._calibrate_gyro_at_rest()
        # Then DELETE the next line.
        raise NotImplementedError(
            "fill in __init__ / _read_imu / _read_encoders / _read_contacts / "
            "_write_positions with your drivers")

    # ---- >>> WIRE UP <<< : IMU ----
    def _read_imu(self):
        """return (quat_wxyz(4), gyro_body(3) rad/s, accel_body(3) m/s^2)."""
        raise NotImplementedError

    # ---- >>> WIRE UP <<< : encoders ----
    def _read_encoders(self):
        """return (joint_pos(14), joint_vel(14)) in POLICY_JOINT_ORDER, rad / rad/s."""
        raise NotImplementedError

    # ---- >>> WIRE UP <<< : foot contact ----
    def _read_contacts(self):
        """return (contact_L: bool, contact_R: bool)."""
        raise NotImplementedError

    # ---- >>> WIRE UP <<< : actuators ----
    def _write_positions(self, u15):
        """send the 15-D position command (MuJoCo ctrl order) to the servos."""
        raise NotImplementedError

    # ---- glue (usually no changes needed) ----
    def read(self) -> SensorReading:
        q, gyro, accel = self._read_imu()
        gyro = self.R_mount @ (np.asarray(gyro, float) - self._gyro_bias)
        accel = self.R_mount @ np.asarray(accel, float)
        jp, jv = self._read_encoders()
        cl, cr = self._read_contacts()
        return SensorReading(time.perf_counter(), np.asarray(q, float), gyro, accel,
                             np.asarray(jp, float), np.asarray(jv, float), bool(cl), bool(cr))

    def command(self, u15):
        if not self.enable_motors:
            return                       # LOG-ONLY: motors stay disabled
        self._write_positions(np.asarray(u15, float))
