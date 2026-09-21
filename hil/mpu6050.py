"""MPU-6050 driver (I2C) -- accel + gyro, in the SENSOR frame, SI units.

Register map / sensitivities are from the InvenSense MPU-6000/6050 datasheet:
  0x19 SMPLRT_DIV   0x1A CONFIG (DLPF)   0x1B GYRO_CONFIG   0x1C ACCEL_CONFIG
  0x3B ACCEL_XOUT_H (14 bytes: ax ay az temp gx gy gz, big-endian int16)
  0x6B PWR_MGMT_1   0x75 WHO_AM_I (0x68)
  gyro LSB per deg/s : 131 / 65.5 / 32.8 / 16.4   (+-250/500/1000/2000 dps)
  accel LSB per g    : 16384 / 8192 / 4096 / 2048 (+-2/4/8/16 g)

`smbus2` is imported lazily, so this module imports fine on the PC.  `FakeBus`
emulates the chip's register file so the whole chain can be tested with no
hardware (see hil/selftest_mpu.py).
"""
from __future__ import annotations

import struct
import time

import numpy as np

G0 = 9.80665
DEG2RAD = np.pi / 180.0
REG = dict(SMPLRT_DIV=0x19, CONFIG=0x1A, GYRO_CONFIG=0x1B, ACCEL_CONFIG=0x1C,
           ACCEL_XOUT_H=0x3B, PWR_MGMT_1=0x6B, WHO_AM_I=0x75)
GYRO_LSB = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
ACC_LSB = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
_GYRO_CODE = {250: 0, 500: 1, 1000: 2, 2000: 3}
_ACC_CODE = {2: 0, 4: 1, 8: 2, 16: 3}
_CODE_GYRO = {v: k for k, v in _GYRO_CODE.items()}
_CODE_ACC = {v: k for k, v in _ACC_CODE.items()}


class MPU6050:
    # Defaults = the settings validated on the bench (bus 1, 0x68, +-2 g, +-250 dps).
    # +-2 g / +-250 dps WILL SATURATE during real walking (sim accel spikes to ~4 g,
    # trunk rates to ~5 rad/s) -- use accel_g=8, gyro_dps=500 or higher for that; the
    # driver counts saturated samples so it is visible in the log.
    def __init__(self, bus=1, addr=0x68, accel_g=2, gyro_dps=250, dlpf=3,
                 rate_hz=1000, smbus=None):
        if smbus is None:
            try:
                from smbus2 import SMBus
            except ImportError as e:
                raise ImportError("smbus2 not installed:  pip install smbus2") from e
            smbus = SMBus(bus)
        self.bus, self.addr = smbus, addr
        self.saturated = False
        self.accel_g, self.gyro_dps = accel_g, gyro_dps
        self._alsb, self._glsb = ACC_LSB[accel_g], GYRO_LSB[gyro_dps]
        self.who = self.bus.read_byte_data(addr, REG["WHO_AM_I"])
        if self.who != 0x68:
            print(f"[mpu6050] WARNING WHO_AM_I=0x{self.who:02x} (expected 0x68) -- a clone/"
                  "MPU6500-family part?  Register map is the same for our purposes.")
        w = lambda r, v: self.bus.write_byte_data(addr, REG[r], v)
        w("PWR_MGMT_1", 0x80)          # device reset
        time.sleep(0.1)
        w("PWR_MGMT_1", 0x01)          # wake, clock = gyro-X PLL
        time.sleep(0.05)
        w("SMPLRT_DIV", max(0, int(round(1000.0 / rate_hz)) - 1))   # 1 kHz base (DLPF on)
        w("CONFIG", dlpf & 7)          # 3 -> ~44 Hz accel / 42 Hz gyro bandwidth
        w("GYRO_CONFIG", _GYRO_CODE[gyro_dps] << 3)
        w("ACCEL_CONFIG", _ACC_CODE[accel_g] << 3)
        time.sleep(0.05)

    def read(self):
        """-> (accel m/s^2 (3,), gyro rad/s (3,), temp_C), sensor frame, no bias removed."""
        d = bytes(self.bus.read_i2c_block_data(self.addr, REG["ACCEL_XOUT_H"], 14))
        ax, ay, az, t, gx, gy, gz = struct.unpack(">7h", d)
        self.saturated = any(abs(v) >= 32760 for v in (ax, ay, az, gx, gy, gz))
        acc =np.array([ax, ay, az], float) / self._alsb * G0
        gyr = np.array([gx, gy, gz], float) / self._glsb * DEG2RAD
        return acc, gyr, t / 340.0 + 36.53

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


class FakeBus:
    """Register-level emulation of an MPU-6050.  `source()` -> (accel m/s^2,
    gyro rad/s, temp_C) in the sensor frame; the bytes returned honour whatever
    ranges the driver programmed, so scaling/sign/endianness bugs would show."""

    def __init__(self, source, who=0x68):
        self.source, self.who, self.regs = source, who, {}

    def write_byte_data(self, addr, reg, val):
        self.regs[reg] = val

    def read_byte_data(self, addr, reg):
        return self.who if reg == REG["WHO_AM_I"] else self.regs.get(reg, 0)

    def read_i2c_block_data(self, addr, reg, n):
        assert reg == REG["ACCEL_XOUT_H"] and n == 14
        a, g, tc = self.source()
        alsb = ACC_LSB[_CODE_ACC[(self.regs.get(REG["ACCEL_CONFIG"], 0) >> 3) & 3]]
        glsb = GYRO_LSB[_CODE_GYRO[(self.regs.get(REG["GYRO_CONFIG"], 0) >> 3) & 3]]
        clip = lambda v: int(np.clip(round(v), -32768, 32767))
        vals = [clip(x / G0 * alsb) for x in a] + [clip((tc - 36.53) * 340.0)] \
            + [clip(x / DEG2RAD * glsb) for x in g]
        return list(struct.pack(">7h", *vals))

    def close(self):
        pass
