"""PCA9685 16-channel PWM driver (I2C) + a joint-space ServoBank on top of it.

Register map / formulas are from the NXP PCA9685 datasheet:
  MODE1 0x00 (bit4 SLEEP, bit5 AI auto-increment)   MODE2 0x01   PRESCALE 0xFE
  LEDn_ON_L = 0x06 + 4n  (ON_L, ON_H, OFF_L, OFF_H);  OFF_H bit4 = "full off"
  prescale = round(osc / (4096 * freq)) - 1,  osc nominally 25 MHz (chip-to-chip error ~ +-5 %)
  pulse counts = pulse_us / period_us * 4096

`smbus2` is imported lazily.  `FakePCA` records register writes so this can be tested
with no hardware (hil/selftest_servo.py).  NOTHING here runs unless you construct it.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from hil import spec

MODE1, MODE2, PRESCALE, LED0 = 0x00, 0x01, 0xFE, 0x06
SERVO_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servo_map.json")


class PCA9685:
    def __init__(self, bus=1, addr=0x40, freq_hz=50, osc_hz=25_000_000, smbus=None):
        if smbus is None:
            try:
                from smbus2 import SMBus
            except ImportError as e:
                raise ImportError("smbus2 not installed:  pip install smbus2") from e
            smbus = SMBus(bus)
        self.bus, self.addr, self.freq = smbus, addr, float(freq_hz)
        self.period_us = 1e6 / self.freq
        pre = int(round(osc_hz / (4096.0 * self.freq))) - 1
        old = self.bus.read_byte_data(addr, MODE1)
        self.bus.write_byte_data(addr, MODE1, (old & 0x7F) | 0x10)     # sleep (needed to set prescale)
        self.bus.write_byte_data(addr, PRESCALE, pre)
        self.bus.write_byte_data(addr, MODE1, 0x20)                    # wake, auto-increment
        time.sleep(0.005)
        self.bus.write_byte_data(addr, MODE1, 0xA0)                    # restart + AI
        self.bus.write_byte_data(addr, MODE2, 0x04)                    # totem-pole outputs
        self.prescale = pre

    def counts(self, pulse_us):
        return int(np.clip(round(pulse_us / self.period_us * 4096.0), 0, 4095))

    def set_pulse_us(self, ch, pulse_us):
        assert 0 <= ch <= 15
        off = self.counts(pulse_us)
        self.bus.write_i2c_block_data(self.addr, LED0 + 4 * ch, [0, 0, off & 0xFF, off >> 8])

    def release(self, ch):
        """Stop the PWM on a channel (servo goes limp / holds nothing)."""
        self.bus.write_i2c_block_data(self.addr, LED0 + 4 * ch, [0, 0, 0, 0x10])

    def release_all(self):
        for ch in range(16):
            self.release(ch)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


class FakePCA:
    """Register-file stand-in for the I2C bus (records the last 4 bytes per channel)."""

    def __init__(self):
        self.regs, self.led = {}, {}
        self.history = []

    def read_byte_data(self, addr, reg):
        return self.regs.get(reg, 0)

    def write_byte_data(self, addr, reg, val):
        self.regs[reg] = val

    def write_i2c_block_data(self, addr, reg, data):
        ch = (reg - LED0) // 4
        self.led[ch] = list(data)
        self.history.append((ch, list(data)))

    def pulse_us(self, ch, period_us=20000.0):
        on_l, on_h, off_l, off_h = self.led[ch]
        if off_h & 0x10:
            return None                                             # released
        return ((off_h << 8) | off_l) / 4096.0 * period_us

    def close(self):
        pass


# ---------------------------------------------------------------------------
def template_map():
    """Blank map for the 15 MuJoCo actuators.  EVERY number that depends on the servo /
    horn / linkage is null on purpose -- fill it in from the datasheet or a bench test."""
    return {
        "_notes": ("pulse_us = zero_us + direction * us_per_rad * angle_rad, clamped to [min_us, max_us]. "
                   "angle_rad is the MuJoCo joint angle (0 = the sim's zero).  channel = PCA9685 output 0-15.  "
                   "Nothing is moved for a joint whose fields are null."),
        "pca9685": {"bus": 1, "addr": 64, "freq_hz": 50},
        "max_slew_rad_s": 1.5,
        "joints": {n: {"channel": None, "direction": None, "zero_us": None, "us_per_rad": None,
                       "min_us": None, "max_us": None} for n in spec.JOINTS_MJ},
    }


class ServoBank:
    """15-D MuJoCo-order position command -> PCA9685 pulses, with per-joint slew limiting and
    hard pulse clamps.  Joints whose map entry is incomplete are skipped (never driven)."""

    def __init__(self, pca, cfg, dt=spec.CONTROL_DT):
        self.pca, self.cfg, self.dt = pca, cfg, dt
        self.max_step = float(cfg.get("max_slew_rad_s", 1.5)) * dt
        self.j = {}
        for i, n in enumerate(spec.JOINTS_MJ):
            e = cfg["joints"][n]
            if all(e.get(k) is not None for k in ("channel", "direction", "zero_us", "us_per_rad", "min_us", "max_us")):
                assert e["direction"] in (-1, 1) and e["min_us"] < e["max_us"]
                self.j[i] = e
        self.state = {i: None for i in self.j}                     # last commanded angle (rad)

    def pulse_for(self, i, angle):
        e = self.j[i]
        return float(np.clip(e["zero_us"] + e["direction"] * e["us_per_rad"] * angle,
                             e["min_us"], e["max_us"]))

    def write(self, u15):
        u = np.clip(np.asarray(u15, float), spec.CTRL_RANGE_MJ[:, 0], spec.CTRL_RANGE_MJ[:, 1])
        for i in self.j:
            cur = self.state[i]
            tgt = u[i] if cur is None else cur + np.clip(u[i] - cur, -self.max_step, self.max_step)
            self.state[i] = float(tgt)
            self.pca.set_pulse_us(self.j[i]["channel"], self.pulse_for(i, tgt))

    def release(self):
        for i in self.j:
            self.pca.release(self.j[i]["channel"])

    def close(self):
        self.release()
        self.pca.close()


def load_map(path=SERVO_MAP_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing -- run  python -m hil.servo_test --write-template")
    with open(path) as f:
        return json.load(f)
