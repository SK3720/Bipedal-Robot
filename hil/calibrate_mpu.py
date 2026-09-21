"""MPU-6050 calibration + live check (run ON THE PI, IMU wired, robot/board hand-held or fixed).

  python -m hil.calibrate_mpu stationary    # 1) gyro bias + accel norm, board still (any pose)
  python -m hil.calibrate_mpu six-point     # 2) per-axis accel offset & scale (6 still poses)
  python -m hil.calibrate_mpu mount         # 3) sensor->chest axis mapping (upright + lean-forward)
  python -m hil.calibrate_mpu live          # 4) live chest-frame view for sign/axis checks

Results are merged into hil/mpu_calib.json (per-machine, git-ignored).  Steps 2 and 3 are
independent of each other; do 2 before 3.  Gyro bias is ALSO re-measured at the start of every
run_hil session (the MPU-6050 gyro bias drifts with temperature), so step 1 is a health check.

Chest frame: +X = robot LEFT, +Y = UP, +Z = FORWARD (walk direction).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil.mpu6050 import MPU6050, G0                                     # noqa: E402
from hil import imu_fusion as F                                          # noqa: E402


def collect(mpu, seconds=2.0, rate=200.0):
    A, Gy = [], []
    for _ in range(int(seconds * rate)):
        a, g, _ = mpu.read()
        A.append(a)
        Gy.append(g)
        time.sleep(1.0 / rate)
    A, Gy = np.array(A), np.array(Gy)
    return A.mean(0), Gy.mean(0), A.std(0), Gy.std(0)


def wait(msg):
    input(f"\n>> {msg}\n   hold STILL, then press Enter... ")


def make_mpu(a):
    return MPU6050(bus=a.bus, addr=a.addr, accel_g=a.accel_g, gyro_dps=a.gyro_dps)


def cmd_stationary(a, mpu, cal):
    wait("Place the board still (any pose).")
    am, gm, asd, gsd = collect(mpu, a.seconds)
    print(f"accel mean {am} m/s^2   std {asd}\n  |a| = {np.linalg.norm(am):.3f} m/s^2 "
          f"= {np.linalg.norm(am)/G0:.3f} g   (ideal 1.000)")
    print(f"gyro  mean {np.degrees(gm)} deg/s   std {np.degrees(gsd)} deg/s")
    ok = np.linalg.norm(gsd) < 0.02
    print("  gyro noise " + ("OK (still)" if ok else "HIGH -- board moved during capture, repeat"))
    if abs(np.linalg.norm(am) / G0 - 1) > 0.03:
        print("  |a| is >3% off 1 g -> run `six-point` to fix accel offset/scale")
    cal.gyro_bias = gm
    cal.save(a.calib)
    print(f"saved gyro bias {np.degrees(gm)} deg/s")


def cmd_six(a, mpu, cal):
    names = ["X", "Y", "Z"]
    poses = {}
    for i in range(3):
        pair = []
        for sgn in (+1, -1):
            wait(f"Rest the board so the sensor {'+' if sgn > 0 else '-'}{names[i]} axis points straight UP "
                 f"(reads {'+' if sgn > 0 else '-'}1 g on {names[i]}).")
            am, _, asd, _ = collect(mpu, a.seconds)
            print(f"   mean {am}  std {asd}")
            if abs(am[i]) < 0.7 * G0 or np.sign(am[i]) != sgn:
                print("   !! that axis is not the one pointing up -- repeat this pose")
                return 1
            pair.append(am)
        poses[i] = tuple(pair)
    off, sc = F.six_point_accel(poses)
    cal.accel_offset, cal.accel_scale = off, sc
    cal.save(a.calib)
    print(f"\naccel offset {off} m/s^2\naccel scale  {sc}\nsaved.")
    for i in range(3):
        ap, am = poses[i]
        print(f"  axis {names[i]}: fixed +/-  {(ap[i]-off[i])*sc[i]/G0:+.3f} g / {(am[i]-off[i])*sc[i]/G0:+.3f} g")


def cmd_mount(a, mpu, cal):
    print("Mounting: define the chest axes on the board the way it is (or will be) mounted on the robot.\n"
          "  chest: +X = robot left, +Y = up, +Z = forward (walk direction).")
    wait("Hold the board in the NOMINAL UPRIGHT pose (chest upright, exactly how it sits when the robot stands).")
    a_up = cal.fix_accel(collect(mpu, a.seconds)[0])
    wait("Now tip the WHOLE board FORWARD ~30-45 deg (rotate about the left-right axis; robot 'leans forward'), "
         "keep the left-right axis level.")
    a_lean = cal.fix_accel(collect(mpu, a.seconds)[0])
    R = F.mount_from_poses(a_up, a_lean)
    cal.R_mount = R
    cal.save(a.calib)
    print("\nR_mount (x_chest = R_mount @ x_sensor):\n", np.round(R, 3))
    print("chest +Y(up) in sensor axes :", np.round(R[1], 3))
    print("chest +Z(forward) in sensor :", np.round(R[2], 3))
    print("chest +X(left) in sensor    :", np.round(R[0], 3))
    print("saved.  Verify with `live`.")


def cmd_live(a, mpu, cal):
    imu = F.MPUImu(mpu, cal)
    info = imu.start(settle_s=1.0)
    print("start-up:  gyro bias %s deg/s   gyro std %.4f rad/s   |a| %.2f m/s^2   "
          "tilt vs sim standing pose %.1f deg" % (np.round(np.degrees(info["gyro_bias"]), 2),
                                                  info["gyro_std"], info["accel_norm"],
                                                  info["tilt_from_sim_stand_deg"]))
    print("Expected signs (chest frame; right-handed, +X left, +Y up, +Z forward):\n"
          "  upright, still     : up_world ~ [0 0 1]   accel ~ [0 +9.8 0]   gyro ~ 0\n"
          "  tip FORWARD        : up_world y < 0,  accel z < 0,  gyro x > 0 while moving\n"
          "  tip to robot RIGHT : up_world x < 0,  accel x > 0,  gyro z > 0 while moving\n"
          "  spin about up axis : gyro y != 0, up_world unchanged\n(Ctrl-C to stop)\n")
    t_end = time.time() + a.seconds_live
    n = 0
    try:
        while time.time() < t_end:
            q, g, ac = imu.read()
            R = F.quat_to_mat(q)
            up = R @ [0, 1, 0]
            fwd = R @ [0, 0, -1]        # the obs yaw-reference axis (-Z_chest), what fwd_xy encodes
            n += 1
            if n % 10 == 0:
                tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
                print(f"up_world {up[0]:+.2f} {up[1]:+.2f} {up[2]:+.2f} | fwd_xy {fwd[0]:+.2f} {fwd[1]:+.2f} "
                      f"| tilt {tilt:5.1f} deg | gyro {g[0]:+6.2f} {g[1]:+6.2f} {g[2]:+6.2f} rad/s "
                      f"| accel {ac[0]:+6.2f} {ac[1]:+6.2f} {ac[2]:+6.2f}", end="\r")
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    print(f"\nsaturated samples: {imu.n_saturated}/{imu.n_reads}   i2c errors: {imu.i2c_errors}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stationary", "six-point", "mount", "live"])
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=0x68)
    ap.add_argument("--accel-g", type=int, default=2, choices=[2, 4, 8, 16])
    ap.add_argument("--gyro-dps", type=int, default=250, choices=[250, 500, 1000, 2000])
    ap.add_argument("--calib", default=F.CALIB_FILE, help="calibration json (read + written)")
    ap.add_argument("--seconds", type=float, default=2.0, help="capture time per pose")
    ap.add_argument("--seconds-live", type=float, default=600.0)
    a = ap.parse_args()
    mpu = make_mpu(a)
    print(f"[mpu6050] bus {a.bus} addr 0x{a.addr:02x}  WHO_AM_I=0x{mpu.who:02x}  "
          f"+-{a.accel_g} g  +-{a.gyro_dps} dps")
    cal = F.Calib.load(a.calib, missing_ok=True)
    fn = dict(stationary=cmd_stationary, **{"six-point": cmd_six}, mount=cmd_mount, live=cmd_live)[a.what]
    rc = fn(a, mpu, cal)
    mpu.close()
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
