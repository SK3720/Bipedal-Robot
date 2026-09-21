"""MPU-6050 -> chest-frame IMU signals the policy expects.

  raw sensor (accel m/s^2, gyro rad/s)
     -> accel: (a - offset) * scale   gyro: g - bias        (calibration, sensor frame)
     -> rotated sensor->chest frame by R_mount               (calibration)
     -> Mahony filter (6-axis, gravity only)                 -> quaternion [w,x,y,z] body->world
  yields (quat, gyro_chest, accel_chest) == what hil.robot_interface.SensorReading wants.

CHEST FRAME (MuJoCo "Chest" body; verified by walking the sim: the robot travels along +Z_chest):
    +X = robot LEFT,   +Y = UP,   +Z = FORWARD (walk direction)     (right-handed: X = Y x Z)
    world at the sim standing pose:  chest X = world +X, up = world +Z, walk direction = world -Y
NOTE the sim's `_fwd_local` = -Z_chest is therefore the BACKWARD axis (misnamed); the observation's
`fwd_xy` is just a yaw reference (~[0,+1] at standing) and is reproduced exactly as-is.

NO magnetometer: roll/pitch are gravity-referenced and stay correct; YAW is integrated
from the gyro only and DRIFTS (fwd_xy in the observation encodes yaw).  World yaw is set at
start-up so the chest heading equals the sim's standing heading.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from hil import spec
from hil.mpu6050 import G0

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpu_calib.json")


# ---------------------------------------------------------------- quaternion utils
def qmul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1, w0*x1 + x0*w1 + y0*z1 - z0*y1,
                     w0*y1 - x0*z1 + y0*w1 + z0*x1, w0*z1 + x0*y1 - y0*x1 + z0*w1])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def mat2quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2
        q = np.array([(R[2,1]-R[1,2])/s, .25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2
        q = np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, .25*s, (R[1,2]+R[2,1])/s])
    else:
        s = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2
        q = np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, .25*s])
    return q / np.linalg.norm(q)


def standing_heading_xy():
    """(x,y) of the observation's yaw reference -Z_chest (`fwd_xy`) in the world at the sim standing pose (~[0,1])."""
    R0 = quat_to_mat(spec.load_consts()["qpos0"][3:7])
    f = (R0 @ np.array([0.0, 0.0, -1.0]))[:2]
    return f / np.linalg.norm(f)


# ---------------------------------------------------------------- Mahony (6-axis)
class Mahony:
    """q = body->world quaternion; world +Z is up.  Rows of R(q) are the world axes
    expressed in body coordinates, so row 2 is world-up in the body frame -- exactly what
    an accelerometer at rest measures."""

    def __init__(self, kp=1.5, ki=0.02, gate=0.30):
        self.kp, self.ki, self.gate = kp, ki, gate
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.I = np.zeros(3)

    def init_from_accel(self, a_chest, fwd_xy=None):
        """Roll/pitch from gravity; yaw chosen so the obs yaw-reference axis (-Z_chest) projects
        onto `fwd_xy` in the world (default: the sim's standing value)."""
        a = np.asarray(a_chest, float)
        a = a / np.linalg.norm(a)
        f = standing_heading_xy() if fwd_xy is None else np.asarray(fwd_xy, float)
        fb = np.array([0.0, 0.0, -1.0])                      # obs yaw-reference axis, body coords
        h = fb - (fb @ a) * a                                # its horizontal part
        if np.linalg.norm(h) < 1e-3:                         # chest pointing straight up/down
            raise ValueError("chest is not near upright -- cannot fix start-up heading")
        h /= np.linalg.norm(h)
        s = np.cross(a, h)
        xb = f[0] * h - f[1] * s                             # world X axis, body coords
        yb = f[1] * h + f[0] * s                             # world Y axis, body coords
        self.q = mat2quat(np.vstack([xb, yb, a]))
        self.I[:] = 0.0

    def update(self, gyro, accel, dt):
        w = np.asarray(gyro, float).copy()
        an = np.linalg.norm(accel)
        if an > 1e-6 and abs(an / G0 - 1.0) < self.gate:     # skip when dynamic accel dominates
            a = np.asarray(accel, float) / an
            qw, qx, qy, qz = self.q
            v = np.array([2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)])
            e = np.cross(a, v)
            self.I += self.ki * e * dt
            w = w + self.kp * e + self.I
        self.q = self.q + 0.5 * dt * qmul(self.q, np.array([0.0, *w]))
        self.q /= np.linalg.norm(self.q)
        return self.q


# ---------------------------------------------------------------- calibration
class Calib:
    """accel_chest = R_mount @ ((a_raw - accel_offset) * accel_scale)
       gyro_chest  = R_mount @ (g_raw - gyro_bias)            (gyro_bias re-measured every run)"""

    def __init__(self, R_mount=None, gyro_bias=None, accel_offset=None, accel_scale=None):
        self.R_mount = np.eye(3) if R_mount is None else np.asarray(R_mount, float)
        self.gyro_bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, float)
        self.accel_offset = np.zeros(3) if accel_offset is None else np.asarray(accel_offset, float)
        self.accel_scale = np.ones(3) if accel_scale is None else np.broadcast_to(
            np.asarray(accel_scale, float), (3,)).copy()

    def fix_accel(self, a_raw):
        return (np.asarray(a_raw, float) - self.accel_offset) * self.accel_scale

    def to_dict(self):
        return dict(R_mount=self.R_mount.tolist(), gyro_bias=self.gyro_bias.tolist(),
                    accel_offset=self.accel_offset.tolist(), accel_scale=self.accel_scale.tolist())

    def save(self, path=CALIB_FILE):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path=CALIB_FILE, missing_ok=False):
        if not os.path.exists(path):
            if missing_ok:
                return Calib()
            raise FileNotFoundError(f"{path} -- run  python -m hil.calibrate_mpu  first")
        with open(path) as f:
            d = json.load(f)
        return Calib(d.get("R_mount"), d.get("gyro_bias"), d.get("accel_offset"),
                     d.get("accel_scale"))


def six_point_accel(poses):
    """poses: dict axis-> (a_plus_mean, a_minus_mean); a_plus = accel with sensor +axis UP
    (reads ~+1g on that axis), a_minus = with -axis up.  Returns (offset(3), scale(3))."""
    off, sc = np.zeros(3), np.ones(3)
    for i in range(3):
        ap, am = poses[i]
        off[i] = 0.5 * (ap[i] + am[i])
        sc[i] = 2.0 * G0 / (ap[i] - am[i])
    return off, sc


def mount_from_poses(a_up, a_lean):
    """Two still poses -> R_mount with  x_chest = R_mount @ x_sensor  (a_* already offset/scale-fixed).
      a_up   : mean accel with the chest in its nominal UPRIGHT pose      (= chest +Y in sensor axes)
      a_lean : mean accel with the chest tipped FORWARD by ~30-45 deg      (rotate about the left-right axis)
    Chest frame: +X left, +Y up, +Z forward (walk direction), right-handed."""
    y = np.asarray(a_up, float); y /= np.linalg.norm(y)
    d = np.asarray(a_lean, float); d /= np.linalg.norm(d)
    # leaning forward by th:  world-up in chest frame = cos(th)*y - sin(th)*z   (z = forward)
    z = -(d - (d @ y) * y)
    n = np.linalg.norm(z)
    if n < np.sin(np.radians(8)):
        raise ValueError(f"lean pose is only {np.degrees(np.arcsin(min(n, 1))):.1f} deg from "
                         "upright -- tip the chest forward by ~30-45 deg")
    z /= n
    x = np.cross(y, z)                       # right-handed:  X = Y x Z
    return np.vstack([x, y, z])


# ---------------------------------------------------------------- source
class MPUImu:
    """Reads the MPU, applies calibration, runs the filter.  read() -> (quat, gyro, accel)."""

    def __init__(self, mpu, calib=None, kp=1.5, ki=0.02):
        self.mpu = mpu
        self.cal = calib or Calib()
        self.filt = Mahony(kp, ki)
        self._t = None
        self.i2c_errors = 0
        self._consec = 0
        self._last = None
        self.n_reads = 0
        self.n_saturated = 0
        self.bias_used = np.zeros(3)

    def _raw(self):
        try:
            self._last = self.mpu.read()
            self._consec = 0
            self.n_reads += 1
            self.n_saturated += int(getattr(self.mpu, "saturated", False))
        except OSError:
            self.i2c_errors += 1
            self._consec += 1
            if self._last is None or self._consec > 5:
                raise RuntimeError("MPU6050 I2C read failed repeatedly -- check wiring/address")
        return self._last

    def start(self, settle_s=1.0, rate_hz=200.0):
        """Hold the IMU STILL in the nominal upright pose (same pose as the sim standing
        pose).  Measures the gyro bias and initialises orientation from gravity."""
        n = max(20, int(settle_s * rate_hz))
        A, Gy = [], []
        for _ in range(n):
            a, g, _ = self._raw()
            A.append(a)
            Gy.append(g)
            time.sleep(1.0 / rate_hz)
        self.bias_used = np.mean(Gy, 0)
        gstd = float(np.linalg.norm(np.std(Gy, 0)))
        self.cal.gyro_bias = self.bias_used
        a_chest = self.cal.R_mount @ self.cal.fix_accel(np.mean(A, 0))
        self.filt.init_from_accel(a_chest)
        self._t = time.perf_counter()
        R0 = quat_to_mat(self.filt.q)
        q0 = quat_to_mat(spec.load_consts()["qpos0"][3:7])
        up, up0 = R0 @ [0, 1, 0], q0 @ [0, 1, 0]
        return dict(gyro_bias=self.bias_used, gyro_std=gstd,
                    accel_norm=float(np.linalg.norm(a_chest)),
                    up_chest=a_chest / np.linalg.norm(a_chest),
                    tilt_from_sim_stand_deg=float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1)))))

    def read(self, dt=None):
        """dt: override the integration step (tests / replay); default = wall-clock delta."""
        a_s, g_s, _ = self._raw()
        t = time.perf_counter()
        if dt is None:
            dt = float(np.clip(t - self._t, 1e-3, 2e-2)) if self._t else 5e-3
        self._t = t
        R = self.cal.R_mount
        a = R @ self.cal.fix_accel(a_s)
        g = R @ (g_s - self.cal.gyro_bias)
        q = self.filt.update(g, a, dt)
        return q.copy(), g, a
