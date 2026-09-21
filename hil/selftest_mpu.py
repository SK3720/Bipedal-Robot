"""Hardware-free self-test of the MPU-6050 chain (runs on the PC or the Pi).

A FakeBus emulates the MPU-6050 register file; its "sensor" is built from the SIM chest
motion, passed through an arbitrary mounting rotation, gyro bias and accel offset/scale
errors (like the ones seen on the bench).  The real driver -> calibration -> Mahony ->
chest-frame chain must undo all of that.  Then `run_hil --mpu` is run end-to-end on the
fake bus.

  python -m hil.selftest_mpu
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil import spec                                                     # noqa: E402
from hil import imu_fusion as F                                          # noqa: E402
from hil.mpu6050 import MPU6050, FakeBus, G0                             # noqa: E402

rng = np.random.default_rng(3)


def rand_rot():
    q = rng.normal(size=4)
    return F.quat_to_mat(q / np.linalg.norm(q))


class Truth:
    """sensor = R_true^T @ chest,  with bias / offset / scale errors."""

    def __init__(self):
        self.R = rand_rot()                                    # x_chest = R @ x_sensor
        self.gbias = np.radians([-1.7, -2.0, -0.4])            # rad/s (bench values)
        self.aoff = np.array([0.35, -0.25, 0.60])              # m/s^2
        self.ascl_true = np.array([1.06, 0.97, 1.09])          # sensor reads scale*true
        self.f_c = np.array([0.0, G0, 0.0])                    # chest-frame specific force
        self.w_c = np.zeros(3)

    def source(self):
        a = self.ascl_true * (self.R.T @ self.f_c) + self.aoff
        g = self.R.T @ self.w_c + self.gbias
        return a, g, 30.0


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


def main():
    ok = True
    T = Truth()
    mpu = MPU6050(smbus=FakeBus(T.source), accel_g=8, gyro_dps=500)
    print("== calibration procedure on the fake sensor ==")
    # six-point
    poses = {}
    for i in range(3):
        pair = []
        for sgn in (+1, -1):
            e = np.zeros(3); e[i] = sgn * G0                    # sensor +/-axis UP
            T.f_c = T.R @ e
            pair.append(np.mean([mpu.read()[0] for _ in range(20)], 0))
        poses[i] = tuple(pair)
    off, sc = F.six_point_accel(poses)
    ok &= check("six-point offset", np.allclose(off, T.aoff, atol=0.03), f"{np.round(off,3)} vs {T.aoff}")
    ok &= check("six-point scale", np.allclose(sc, 1 / T.ascl_true, atol=2e-3), f"{np.round(sc,4)} vs {np.round(1/T.ascl_true,4)}")
    cal = F.Calib(accel_offset=off, accel_scale=sc)
    # mount
    th = np.radians(35)
    T.f_c = np.array([0.0, G0, 0.0])
    a_up = cal.fix_accel(np.mean([mpu.read()[0] for _ in range(20)], 0))
    T.f_c = G0 * np.array([0.0, np.cos(th), -np.sin(th)])      # tipped FORWARD (+Z_chest = walk direction)
    a_lean = cal.fix_accel(np.mean([mpu.read()[0] for _ in range(20)], 0))
    cal.R_mount = F.mount_from_poses(a_up, a_lean)
    ok &= check("mount rotation", np.allclose(cal.R_mount, T.R, atol=2e-3),
                f"max err {np.abs(cal.R_mount - T.R).max():.1e}")

    print("== dynamic chain vs the simulator's own IMU (walking session) ==")
    from hil.robot_interface import SimRobot
    from hil.observation import ObservationBuilder
    from hil.policy import Policy
    from hil.base_controller import BaseController
    sim = SimRobot(noise=False, seed=0)
    imu = F.MPUImu(MPU6050(smbus=FakeBus(T.source), accel_g=16, gyro_dps=2000), cal)   # unsaturated range
    T.f_c, T.w_c = np.array([0, G0, 0.0]), np.zeros(3)
    sr = sim.read()
    T.f_c, T.w_c = np.array([0, G0, 0.0]), np.zeros(3)         # held STILL upright during start-up (bias + init)
    imu.start(settle_s=0.15, rate_hz=200)
    pol, bc = Policy(), BaseController()
    ob = ObservationBuilder(control_dt=spec.CONTROL_DT)
    ob.reset(sr.imu_quat_wxyz, sr.imu_gyro, sr.imu_accel, sr.joint_pos, sr.joint_vel,
             sr.contact_L, sr.contact_R, phase0=0.0)
    prev = np.zeros(14)
    up_err, gy_err, ac_err, yaw_err = [], [], [], []
    pk_a, pk_g = 0.0, 0.0
    for k in range(600):
        sr = sim.read()
        T.f_c, T.w_c = sr.imu_accel, sr.imu_gyro
        q, g, a = imu.read(dt=spec.CONTROL_DT)
        pk_a, pk_g = max(pk_a, np.abs(sr.imu_accel).max()), max(pk_g, np.abs(sr.imu_gyro).max())
        sat = imu.mpu.saturated
        Rs, Rh = F.quat_to_mat(sr.imu_quat_wxyz), F.quat_to_mat(q)
        up_err.append(np.degrees(np.arccos(np.clip((Rs @ [0, 1, 0]) @ (Rh @ [0, 1, 0]), -1, 1))))
        fs, fh = (Rs @ [0, 0, -1])[:2], (Rh @ [0, 0, -1])[:2]
        yaw_err.append(np.degrees(np.arctan2(fs[0]*fh[1]-fs[1]*fh[0], fs @ fh)))
        if not sat:                                            # clipped samples are the sensor range, not a bug
            gy_err.append(np.abs(g - sr.imu_gyro).max())
            ac_err.append(np.abs(a - sr.imu_accel).max())
        obs = ob.step(q, g, a, sr.joint_pos, sr.joint_vel, sr.contact_L, sr.contact_R, prev)
        act = pol.act(obs)
        u = bc.compose(act, ob.phase, q, g, ob.kin.foot_L_rel, ob.kin.foot_R_rel, ob.kin.v_est,
                       sr.contact_L, sr.contact_R)
        sim.command(u)
        prev = act
    up_err, yaw_err = np.array(up_err), np.array(yaw_err)
    ok &= check("gyro recovered", max(gy_err) < 0.02, f"max |err| {max(gy_err):.4f} rad/s (quantisation)")
    ok &= check("accel recovered", max(ac_err) < 0.05, f"max |err| {max(ac_err):.4f} m/s^2")
    ok &= check("tilt (up) vs sim quat", np.percentile(up_err, 95) < 6.0,
                f"p95 {np.percentile(up_err,95):.2f} deg  max {up_err.max():.2f}  (Mahony, walking, gated)")
    ok &= check("yaw error after 3 s of walking", abs(yaw_err[-1]) < 10.0,
                f"{yaw_err[-1]:+.2f} deg  (sim impact spikes are under-sampled at 200 Hz; informational bound)")
    print(f"  sim IMU peaks during 3 s of walking: |accel| {pk_a:.1f} m/s^2 ({pk_a/G0:.1f} g), "
          f"|gyro| {pk_g:.1f} rad/s ({np.degrees(pk_g):.0f} dps); saturated at +-16g/2000dps: "
          f"{imu.n_saturated}/{imu.n_reads} samples")

    print("  (at +-8 g / +-500 dps the sim's foot-strike spikes clip and the clipped gyro integrates into yaw drift -- "
          "use the wider range for anything dynamic)")
    print("== sign conventions the bench `live` view tells you to expect ==")
    def up_after(f_c):
        T.f_c, T.w_c = np.array(f_c) * G0, np.zeros(3)
        im = F.MPUImu(MPU6050(smbus=FakeBus(T.source), accel_g=8, gyro_dps=500), cal)
        im.start(settle_s=0.1, rate_hz=200)
        return F.quat_to_mat(im.filt.q) @ [0, 1, 0]
    t = np.radians(20)
    up_f = up_after([0, np.cos(t), -np.sin(t)])                # forward tip: world-up gets -Z_chest component
    up_r = up_after([np.sin(t), np.cos(t), 0])                 # tip to robot RIGHT: world-up gets +X_chest component
    ok &= check("tip forward -> up_world.y < 0 (walk dir = world -Y)", up_f[1] < -0.3 and abs(up_f[0]) < 0.05, f"up {np.round(up_f,2)}")
    ok &= check("tip right   -> up_world.x < 0", up_r[0] < -0.3 and abs(up_r[1]) < 0.05, f"up {np.round(up_r,2)}")
    print("== yaw integration: smooth spin about the up axis (known answer) ==")
    T.f_c, T.w_c = np.array([0, G0, 0.0]), np.zeros(3)
    imu3 = F.MPUImu(MPU6050(smbus=FakeBus(T.source), accel_g=8, gyro_dps=500), cal)
    imu3.start(settle_s=0.1, rate_hz=200)
    f0 = (F.quat_to_mat(imu3.filt.q) @ [0, 0, -1])[:2]
    T.w_c = np.array([0.0, 0.5, 0.0])                          # +0.5 rad/s about chest +Y (up)
    for _ in range(600):
        q3, *_ = imu3.read(dt=0.005)
    f1 = (F.quat_to_mat(q3) @ [0, 0, -1])[:2]
    turned = np.degrees(np.arctan2(f0[0]*f1[1]-f0[1]*f1[0], f0 @ f1))
    ok &= check("yaw integrates the gyro", abs(turned - np.degrees(1.5)) < 1.0,
                f"turned {turned:.2f} deg, expected {np.degrees(1.5):.2f} (about +Y=up, CCW seen from above)")
    print("== ranges: what happens at the bench-validated +-2 g / +-250 dps ==")
    imu2 = F.MPUImu(MPU6050(smbus=FakeBus(T.source), accel_g=2, gyro_dps=250), cal)
    T.f_c, T.w_c = np.array([0, G0, 0.0]), np.zeros(3)
    imu2.start(settle_s=0.1, rate_hz=200)
    T.f_c = np.array([0, 2.5 * G0, 0.0])
    imu2.read(dt=0.005)
    check("saturation detected at 2.5 g", imu2.n_saturated >= 1, f"count {imu2.n_saturated}")

    print("== run_hil --mpu end to end on the fake bus ==")
    import hil.mpu6050 as m6
    T.f_c, T.w_c = np.array([0, G0, 0.0]), np.zeros(3)
    real = m6.MPU6050
    m6.MPU6050 = lambda **kw: real(smbus=FakeBus(T.source), **kw)
    import hil.run_hil as rh
    old = sys.argv
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "calib.json")
    cal.save(tmp)
    sys.argv = ["run_hil", "--mpu", "--seconds", "2", "--accel-g", "8", "--gyro-dps", "500", "--calib", tmp]
    try:
        rc = rh.main()
    finally:
        sys.argv, m6.MPU6050 = old, real
    ok &= check("run_hil --mpu exits 0", rc == 0, f"rc={rc}")
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
