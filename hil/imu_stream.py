"""Pi side of the live link: stream the calibrated MPU-6050 (orientation, gyro, accel) over UDP.

  python -m hil.imu_stream --host <PC-IP> [--port 5005] [--accel-g 8 --gyro-dps 1000]

Runs the SAME chain as `run_hil --mpu` (hil/mpu6050.py -> calibration hil/mpu_calib.json -> Mahony AHRS),
so the packets are exactly the IMU channels the policy consumes: chest-frame quaternion [w,x,y,z]
(body->world, yaw matched to the sim standing pose), bias-removed gyro (rad/s) and accel (m/s^2).
Hold the board STILL and in the nominal upright pose for the first second (gyro bias + orientation init).

The receiver is `python -m hil.live_sim` on the PC.  Nothing here drives any actuator.
"""
from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil.imu_fusion import Calib, MPUImu, CALIB_FILE                     # noqa: E402

# seq (u32), pi wall time (f64), quat wxyz (4f), gyro (3f), accel (3f), saturated-sample count (u32)
PKT = struct.Struct("<Id10fI")


def main(argv=None, mpu_factory=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="PC IP address (where hil.live_sim runs)")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until Ctrl-C")
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=0x68)
    ap.add_argument("--accel-g", type=int, default=8, choices=[2, 4, 8, 16],
                    help="default +-8 g (hand motion / walking); +-2 g clips on quick moves")
    ap.add_argument("--gyro-dps", type=int, default=1000, choices=[250, 500, 1000, 2000])
    ap.add_argument("--calib", default=CALIB_FILE)
    a = ap.parse_args(argv)

    from hil.mpu6050 import MPU6050
    mk = mpu_factory or MPU6050
    try:
        mpu = mk(bus=a.bus, addr=a.addr, accel_g=a.accel_g, gyro_dps=a.gyro_dps)
    except (ImportError, OSError) as e:
        print(f"[imu_stream] cannot open the MPU-6050 (bus {a.bus} addr 0x{a.addr:02x}): {e}")
        return 2
    calib_ok = os.path.exists(a.calib)
    imu = MPUImu(mpu, Calib.load(a.calib, missing_ok=True))
    print(f"[imu_stream] MPU-6050 +-{a.accel_g} g +-{a.gyro_dps} dps -> udp://{a.host}:{a.port} @ {a.rate:.0f} Hz")
    if not calib_ok:
        print(f"  [warn] no {a.calib} -- axes NOT mapped / accel uncalibrated: run  python -m hil.calibrate_mpu mount")
    print("  hold the board STILL and upright for 1 s ...")
    info = imu.start(settle_s=1.0, rate_hz=a.rate)
    print(f"  gyro bias {np.round(np.degrees(info['gyro_bias']), 2)} deg/s   |a| {info['accel_norm']:.3f} m/s^2   "
          f"tilt vs sim standing pose {info['tilt_from_sim_stand_deg']:.1f} deg   -- streaming (Ctrl-C to stop)")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (a.host, a.port)
    dt = 1.0 / a.rate
    t0 = time.perf_counter()
    nxt = t0
    seq = 0
    t_last, seq_last = t0, 0
    try:
        while True:
            q, g, ac = imu.read()
            sock.sendto(PKT.pack(seq, time.time(), *q, *g, *ac, imu.n_saturated), dest)
            seq += 1
            now = time.perf_counter()
            if now - t_last >= 2.0:
                print(f"  sent {seq:7d}  {(seq - seq_last) / (now - t_last):6.1f} Hz   i2c errors {imu.i2c_errors}   "
                      f"saturated {imu.n_saturated}", end="\r")
                t_last, seq_last = now, seq
            if a.seconds and now - t0 >= a.seconds:
                break
            nxt += dt
            slp = nxt - time.perf_counter()
            if slp > 0:
                time.sleep(slp)
            elif slp < -0.05:                      # fell far behind: resync instead of bursting
                nxt = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if hasattr(mpu, "close"):
            mpu.close()
    print(f"\n[imu_stream] stopped after {seq} packets; i2c errors {imu.i2c_errors}, saturated {imu.n_saturated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
