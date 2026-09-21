"""Safety / sanity checks for the HIL loop.

Every check returns a list of (level, code, message).  level:
  "info"  -- logged only
  "warn"  -- logged + printed; keep running
  "abort" -- the loop must stop (in log-only mode: stop reading; with motors:
             latch a safe hold and cut torque)

Thresholds live in hil.spec.SAFE.
"""
from __future__ import annotations

import numpy as np

from hil import spec

S = spec.SAFE


def check_reading(sr) -> list[tuple[str, str, str]]:
    out = []
    # --- NaN / shape ---
    for name, v, n in (("imu_quat", sr.imu_quat_wxyz, 4), ("imu_gyro", sr.imu_gyro, 3),
                       ("imu_accel", sr.imu_accel, 3), ("joint_pos", sr.joint_pos, 14),
                       ("joint_vel", sr.joint_vel, 14)):
        a = np.asarray(v)
        if a.shape != (n,):
            out.append(("abort", "shape", f"{name} shape {a.shape} != ({n},)"))
        elif not np.all(np.isfinite(a)):
            out.append(("abort", "nan", f"{name} has non-finite values"))
    if out:
        return out

    # --- IMU sanity ---
    qn = np.linalg.norm(sr.imu_quat_wxyz)
    if abs(qn - 1.0) > 0.05:
        out.append(("warn", "quat_norm", f"|imu_quat| = {qn:.3f} (should be ~1)"))
    an = np.linalg.norm(sr.imu_accel)
    if not (S["accel_norm_min"] <= an <= S["accel_norm_max"]):
        out.append(("warn", "accel_norm", f"|accel| = {an:.1f} m/s^2 outside "
                    f"[{S['accel_norm_min']}, {S['accel_norm_max']}]"))
    if np.any(np.abs(sr.imu_gyro) > S["gyro_abs_max"]):
        out.append(("warn", "gyro_big", f"gyro max |{np.max(np.abs(sr.imu_gyro)):.1f}| "
                    f"> {S['gyro_abs_max']} rad/s"))

    # --- joints ---
    if np.any(np.abs(sr.joint_vel) > S["joint_vel_abs_max"]):
        j = int(np.argmax(np.abs(sr.joint_vel)))
        out.append(("warn", "jvel_big", f"joint_vel[{j}] ({spec.POLICY_JOINT_ORDER[j]}) = "
                    f"{sr.joint_vel[j]:+.1f} rad/s"))
    lo = spec.CTRL_RANGE_MJ[spec.ACT_CTRL, 0]
    hi = spec.CTRL_RANGE_MJ[spec.ACT_CTRL, 1]
    over = (sr.joint_pos < lo - 0.05) | (sr.joint_pos > hi + 0.05)
    if np.any(over):
        j = int(np.argmax(over))
        out.append(("warn", "jpos_range", f"joint_pos[{j}] ({spec.POLICY_JOINT_ORDER[j]}) = "
                    f"{sr.joint_pos[j]:+.3f} outside [{lo[j]:+.3f}, {hi[j]:+.3f}]"))
    return out


def check_attitude(obs) -> list[tuple[str, str, str]]:
    """`up` = obs[0:3] of the newest sensor frame == obs[3*50 : 3*50+3]."""
    up = np.asarray(obs[spec.SFRAME_DIM * (spec.HIST - 1):
                       spec.SFRAME_DIM * (spec.HIST - 1) + 3], float)
    n = np.linalg.norm(up)
    out = []
    if not (S["up_norm_min"] <= n <= S["up_norm_max"]):
        out.append(("warn", "up_norm", f"|up| = {n:.2f}"))
    tilt = np.degrees(np.arccos(np.clip(up[2] / max(n, 1e-6), -1, 1)))
    if tilt > S["tilt_deg_abort"]:
        out.append(("abort", "fell", f"chest tilt {tilt:.0f} deg > {S['tilt_deg_abort']}"))
    elif tilt > S["tilt_deg_warn"]:
        out.append(("warn", "tilt", f"chest tilt {tilt:.0f} deg"))
    return out, tilt


def check_command(u15) -> list[tuple[str, str, str]]:
    u = np.asarray(u15, float)
    out = []
    if u.shape != (15,):
        return [("abort", "cmd_shape", f"command shape {u.shape} != (15,)")]
    if not np.all(np.isfinite(u)):
        return [("abort", "cmd_nan", "command has non-finite values")]
    lo = spec.CTRL_RANGE_MJ[:, 0] + S["cmd_margin_rad"]
    hi = spec.CTRL_RANGE_MJ[:, 1] - S["cmd_margin_rad"]
    at_lim = (u <= lo) | (u >= hi)
    if np.any(at_lim):
        js = np.where(at_lim)[0]
        out.append(("info", "cmd_at_limit",
                    f"command at joint limit: {[spec.JOINTS_MJ[j] for j in js]}"))
    if abs(u[spec.JOINTS_MJ.index("Chest_neck") if "Chest_neck" in spec.JOINTS_MJ else 0]) > 1e-6:
        out.append(("warn", "neck_nonzero", "neck command should be 0"))
    return out


class TimingMonitor:
    def __init__(self, target_hz=spec.CONTROL_HZ, window=200):
        self.target_dt = 1.0 / target_hz
        self.window = window
        self.dts: list[float] = []
        self._last = None

    def tick(self, t_now):
        if self._last is not None:
            self.dts.append(t_now - self._last)
            if len(self.dts) > self.window:
                self.dts.pop(0)
        self._last = t_now

    def stats(self):
        if len(self.dts) < 5:
            return dict(hz=float("nan"), jitter_ms=float("nan"), n=len(self.dts))
        a = np.asarray(self.dts)
        return dict(hz=1.0 / a.mean(), jitter_ms=a.std() * 1e3,
                    dt_min_ms=a.min() * 1e3, dt_max_ms=a.max() * 1e3, n=len(a))

    def check(self):
        s = self.stats()
        out = []
        if s["n"] >= 20:
            if s["hz"] < S["loop_hz_min"]:
                out.append(("abort", "slow_loop",
                            f"loop {s['hz']:.0f} Hz < {S['loop_hz_min']} Hz"))
            if s["jitter_ms"] > S["loop_jitter_ms_max"]:
                out.append(("warn", "jitter", f"loop jitter {s['jitter_ms']:.1f} ms"))
        return out, s
