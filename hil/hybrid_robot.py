"""HybridRobot -- REAL IMU + SIMULATED robot body ("gantry" HIL).

  * IMU fields (quat / gyro / accel)  <- the real MPU-6050 via hil.imu_fusion.MPUImu
  * joints, contacts                  <- the MuJoCo sim, driven by the policy's own command
  * the sim torso is PINNED at the standing pose every step (a robot hanging on a
    gantry), so the sim legs move as commanded but the sim never falls over.

What this tests:  sensor conventions, real-IMU noise/latency, and the policy's
RESPONSE to a real IMU signal (tilt the board -> do the commanded hip/ankle moves
go the right way?).  What it can NOT test: balance -- the real IMU does not feel
the sim legs, and the sim torso does not follow the real IMU.

Nothing here drives any real actuator.  Existing SimRobot behaviour is untouched
(this only subclasses it).
"""
from __future__ import annotations

from hil import spec
from hil.robot_interface import SimRobot


class HybridRobot(SimRobot):
    def __init__(self, imu, pin_base=True, **kw):
        super().__init__(noise=False, **kw)
        self.imu = imu
        self.pin_base = bool(pin_base)
        self._base_q = spec.load_consts()["qpos0"][:7].copy()

    def read(self):
        sr = super().read()                       # sim joints + contacts (sim IMU discarded)
        sr.imu_quat_wxyz, sr.imu_gyro, sr.imu_accel = self.imu.read()   # REAL IMU
        sr.raw = None
        return sr

    def command(self, u15):
        super().command(u15)
        if self.pin_base:
            d = self.data
            d.qpos[:7] = self._base_q
            d.qvel[:6] = 0.0
            self.mj.mj_forward(self.model, d)

    def fell(self):
        return False            # torso is pinned; tilt comes from the real IMU

    def close(self):
        m = getattr(self.imu, "mpu", None)
        if m is not None and hasattr(m, "close"):
            m.close()
