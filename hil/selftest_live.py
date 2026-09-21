"""Hardware-free test of the live link (imu_stream -> UDP loopback -> live_sim).  python -m hil.selftest_live

A FakeBus MPU-6050 (register level) is swayed +-17 deg about the chest X axis (forward/back tip) after 3 s of
stillness (the stream itself spends the first 1 s settling); hil.imu_stream sends real UDP packets to 127.0.0.1; hil.live_sim (headless) receives them and drives the
sim.  Checks: packets arrive with few drops, the recorded quaternion follows the sway, the sim torso follows the
real quaternion, and the policy's command changes with the tilt.
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil import spec                                                     # noqa: E402
from hil import imu_stream, live_sim                                     # noqa: E402
from hil.mpu6050 import MPU6050, FakeBus, G0                             # noqa: E402

PORT = 5077
AMP = 0.30                                                               # rad (17 deg)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return bool(ok)


def main():
    ok = True
    t0 = time.perf_counter()

    def source():
        t = time.perf_counter() - t0
        if t < 3.0:
            th, thd = 0.0, 0.0
        else:
            ph = 2 * np.pi * 0.5 * (t - 3.0)
            th, thd = AMP * np.sin(ph), AMP * 2 * np.pi * 0.5 * np.cos(ph)
        return G0 * np.array([0.0, np.cos(th), -np.sin(th)]), np.array([thd, 0.0, 0.0]), 30.0

    tmp = os.path.join(tempfile.mkdtemp(), "no_calib.json")             # nonexistent -> identity mount, never the real file
    th = threading.Thread(target=imu_stream.main, kwargs=dict(
        argv=["--host", "127.0.0.1", "--port", str(PORT), "--seconds", "8", "--calib", tmp],
        mpu_factory=lambda **kw: MPU6050(smbus=FakeBus(source), **kw)), daemon=True)
    th.start()
    before = set(glob.glob(os.path.join(live_sim.LOG_DIR, "live_*.npz")))
    rc = live_sim.main(["--port", str(PORT), "--seconds", "5", "--no-viewer", "--wait", "10"])
    th.join(timeout=6)
    ok &= check("live_sim exits 0", rc == 0, f"rc={rc}")
    new = sorted(set(glob.glob(os.path.join(live_sim.LOG_DIR, "live_*.npz"))) - before)
    ok &= check("log written", len(new) == 1)
    z = np.load(new[-1])
    q, cmd, age = z["quat"], z["command"], z["age_ms"]
    up = np.array([[2 * (x * y - w * zz), 1 - 2 * (x * x + zz * zz), 2 * (y * zz + w * x)] for w, x, y, zz in q])
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    ok &= check("packets flowing at ~loop rate", len(q) > 700, f"{len(q)} steps in {z['t'][-1]:.1f} s")
    ok &= check("packet age small (loopback)", np.median(age) < 15.0, f"median {np.median(age):.1f} ms, max {age.max():.0f} ms")
    ok &= check("sway reached over the link", 12.0 < tilt.max() < 22.0, f"max tilt {tilt.max():.1f} deg (sent {np.degrees(AMP):.1f})")
    ok &= check("still at start", tilt[:200].max() < 2.5, f"first 1 s max tilt {tilt[:200].max():.2f} deg")
    still, tilted = cmd[tilt < 1.0], cmd[tilt > 0.8 * tilt.max()]
    diff = np.abs(np.median(tilted, 0) - np.median(still, 0)).max() if len(still) and len(tilted) else 0.0
    ok &= check("policy command responds to the real tilt", diff > 0.03, f"max joint change {diff:.2f} rad")

    # the sim torso must follow the real quaternion exactly
    class One:
        def __init__(self): self.q = np.array([np.cos(0.15), np.sin(0.15), 0.0, 0.0])
        def read(self): return self.q.copy(), np.zeros(3), np.array([0.0, G0, 0.0])
    imu = One()
    rob = live_sim.LiveRobot(imu, follow=True)
    rob.read(); rob.command(spec.load_consts()["ctrl0"])
    ok &= check("sim torso quaternion == real quaternion", np.allclose(rob.data.qpos[3:7], imu.q, atol=1e-9),
                f"{np.round(rob.data.qpos[3:7], 3)}")
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
