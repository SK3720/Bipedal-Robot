"""HIL loop for runs/sim2real_v1 -- LOG-ONLY (motors disabled by default).

Each control step:
  1. read sensors (IMU + encoders + contacts)
  2. safety-check the raw reading
  3. build the 217-D observation
  4. run policy inference  -> 14-D action
  5. compose the 15-D base-controller position command (what WOULD be sent)
  6. safety-check the command + attitude + timing
  7. log EVERYTHING (raw reading, obs, action, command, latencies, safety events)
  8. `robot.command(u)`  -- a NO-OP for HardwareRobot unless --enable-motors AND
     the driver stub is implemented.  SimRobot always applies it (so `--sim`
     gives a full walking session to inspect).

  python -m hil.run_hil --sim  --seconds 8            # dry-run the whole stack
  python -m hil.run_hil --sim  --noise                # + sim sensor noise
  python -m hil.run_hil --hardware --seconds 20       # real robot, log-only
  python -m hil.run_hil --hardware --enable-motors    # ONLY after bench checks
  python -m hil.run_hil --mpu  --seconds 30           # REAL MPU-6050 + SIM body (Pi bench HIL, log-only)

Logs -> hil/logs/hil_<timestamp>.npz  (+ a .txt summary).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil import spec                                    # noqa: E402
from hil.observation import ObservationBuilder          # noqa: E402
from hil.base_controller import BaseController          # noqa: E402
from hil.policy import Policy                           # noqa: E402
from hil import safety                                  # noqa: E402
from hil.robot_interface import SimRobot, HardwareRobot  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def _p95(a):
    return float(np.percentile(a, 95)) if len(a) else float("nan")


def _cmd_limit_report(cmd):
    """cmd: (N,15).  Per-joint fraction of steps within cmd_margin of either limit."""
    lo = spec.CTRL_RANGE_MJ[:, 0] + spec.SAFE["cmd_margin_rad"]
    hi = spec.CTRL_RANGE_MJ[:, 1] - spec.SAFE["cmd_margin_rad"]
    at = (cmd <= lo) | (cmd >= hi)
    frac = at.mean(0)
    hot = [f"{spec.JOINTS_MJ[j]}:{frac[j]*100:.0f}%" for j in np.argsort(-frac) if frac[j] > 0.02][:6]
    return f"  cmd saturating joints  : {', '.join(hot) if hot else 'none'}"


def _sleep_to(target_t):
    """hybrid sleep -- coarse sleep then short spin (Windows sleep granularity ~1 ms)."""
    while True:
        rem = target_t - time.perf_counter()
        if rem <= 0:
            return
        if rem > 0.0015:
            time.sleep(rem - 0.0012)
        # else spin


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim", action="store_true", help="drive the MuJoCo sim (stack dry-run)")
    src.add_argument("--hardware", action="store_true", help="the real robot")
    src.add_argument("--mpu", action="store_true",
                     help="REAL MPU-6050 IMU + simulated joints/contacts (torso pinned; no actuators)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--rate", type=float, default=spec.CONTROL_HZ, help="target loop Hz")
    ap.add_argument("--noise", action="store_true", help="(sim only) add training sensor noise")
    ap.add_argument("--enable-motors", action="store_true",
                    help="(hardware) actually drive the servos -- DEFAULT OFF")
    ap.add_argument("--no-throttle", action="store_true",
                    help="run as fast as possible (measures max achievable rate)")
    ap.add_argument("--mpu-bus", type=int, default=1)
    ap.add_argument("--mpu-addr", type=lambda v: int(v, 0), default=0x68)
    ap.add_argument("--accel-g", type=int, default=2, choices=[2, 4, 8, 16],
                    help="(--mpu) accel full-scale; default = bench-validated +-2 g")
    ap.add_argument("--gyro-dps", type=int, default=250, choices=[250, 500, 1000, 2000],
                    help="(--mpu) gyro full-scale; default = bench-validated +-250 dps")
    ap.add_argument("--calib", default=None,
                    help="(--mpu) calibration json (default hil/mpu_calib.json)")
    ap.add_argument("--no-pin", action="store_true",
                    help="(--mpu) do NOT pin the sim torso (sim will fall -- not useful)")
    a = ap.parse_args()
    src_name = "sim" if a.sim else ("mpu+sim" if a.mpu else "hardware")

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = os.path.join(LOG_DIR, f"hil_{stamp}")

    # --- robot ---
    if a.sim:
        robot = SimRobot(noise=a.noise, seed=0)
        print(f"[hil] SimRobot  (noise={a.noise})")
    elif a.mpu:
        from hil.mpu6050 import MPU6050
        from hil.imu_fusion import Calib, MPUImu
        from hil.hybrid_robot import HybridRobot
        try:
            mpu = MPU6050(bus=a.mpu_bus, addr=a.mpu_addr, accel_g=a.accel_g, gyro_dps=a.gyro_dps)
        except (ImportError, OSError) as e:
            print(f"\n[hil] cannot open the MPU-6050 on I2C bus {a.mpu_bus} addr 0x{a.mpu_addr:02x}: {e}\n"
                  "  On the Pi:  pip install smbus2 ; enable I2C ; check  sudo i2cdetect -y 1")
            return 2
        calib_path = a.calib or os.path.join(os.path.dirname(__file__), "mpu_calib.json")
        cal = Calib.load(calib_path, missing_ok=True)
        imu = MPUImu(mpu, cal)
        print(f"[hil] MPU-6050 bus {a.mpu_bus} 0x{a.mpu_addr:02x} +-{a.accel_g} g +-{a.gyro_dps} dps; "
              "keep the IMU STILL in the nominal upright pose for 1 s (bias + orientation init)...")
        info = imu.start(settle_s=1.0, rate_hz=a.rate)
        print(f"  gyro bias {np.round(np.degrees(info['gyro_bias']), 2)} deg/s (std {info['gyro_std']:.4f} rad/s)  "
              f"|a| {info['accel_norm']:.2f} m/s^2  tilt vs sim standing pose {info['tilt_from_sim_stand_deg']:.1f} deg")
        if info["gyro_std"] > 0.05:
            print("  [warn] IMU was not still during start-up -- gyro bias estimate is poor")
        if not os.path.exists(calib_path):
            print(f"  [warn] no {calib_path} -- axes NOT mapped (identity) and accel uncalibrated; "
                  "run  python -m hil.calibrate_mpu mount")
        robot = HybridRobot(imu, pin_base=not a.no_pin)
        print("[hil] HybridRobot  real IMU + sim joints; NO actuators are driven")
    else:
        try:
            robot = HardwareRobot(enable_motors=a.enable_motors)
        except NotImplementedError as e:
            print(f"\n[hil] HardwareRobot is not wired to devices yet.\n\n  {e}\n\n"
                  "  Implement the 4 stub methods in hil/robot_interface.py, then re-run.\n"
                  "  Meanwhile use  `python -m hil.run_hil --sim`  to exercise the stack.")
            return 2
        print(f"[hil] HardwareRobot  motors={'ENABLED' if a.enable_motors else 'disabled (log-only)'}")
        if a.enable_motors:
            input("  --enable-motors set.  Robot on a gantry / spotter?  Enter to continue, Ctrl-C to abort. ")

    dt = 1.0 / a.rate
    obs_b = ObservationBuilder(control_dt=dt)
    pol = Policy()
    bc = BaseController()
    tim = safety.TimingMonitor(target_hz=a.rate)

    # --- prime ---
    sr0 = robot.read()
    ev0 = safety.check_reading(sr0)
    for lvl, code, msg in ev0:
        print(f"  [{lvl}] {code}: {msg}")
    if any(lvl == "abort" for lvl, _, _ in ev0):
        print("[hil] aborting: bad initial sensor reading")
        return 1
    obs = obs_b.reset(sr0.imu_quat_wxyz, sr0.imu_gyro, sr0.imu_accel,
                      sr0.joint_pos, sr0.joint_vel, sr0.contact_L, sr0.contact_R, phase0=0.0)
    prev_action = np.zeros(14)

    n_steps = int(a.seconds * a.rate)
    L = dict(t=[], read_ms=[], obs_ms=[], infer_ms=[], compose_ms=[], loop_ms=[],
             obs=[], action=[], command=[], tilt=[],
             imu_quat=[], imu_gyro=[], imu_accel=[], joint_pos=[], joint_vel=[],
             contact=[], v_est=[], phase=[])
    events = []
    last_warn = {}
    t_session = time.perf_counter()
    aborted = None
    next_t = t_session

    for k in range(n_steps):
        loop_t0 = time.perf_counter()
        next_t += dt

        t0 = time.perf_counter()
        sr = robot.read()
        t_read = time.perf_counter()

        ev = safety.check_reading(sr)
        obs = obs_b.step(sr.imu_quat_wxyz, sr.imu_gyro, sr.imu_accel,
                         sr.joint_pos, sr.joint_vel, sr.contact_L, sr.contact_R, prev_action)
        t_obs = time.perf_counter()

        if obs.shape != (spec.OBS_DIM,):
            ev.append(("abort", "obs_dim", f"obs shape {obs.shape} != ({spec.OBS_DIM},)"))
        if not np.all(np.isfinite(obs)):
            ev.append(("abort", "obs_nan", "observation has non-finite values"))

        action = pol.act(obs)
        t_infer = time.perf_counter()

        u = bc.compose(action, obs_b.phase, sr.imu_quat_wxyz, sr.imu_gyro,
                       obs_b.kin.foot_L_rel, obs_b.kin.foot_R_rel, obs_b.kin.v_est,
                       sr.contact_L, sr.contact_R)
        t_compose = time.perf_counter()

        att_ev, tilt = safety.check_attitude(obs)
        if a.mpu:                       # hand-tilting the board is the point of this test
            att_ev = [e for e in att_ev if e[1] != "fell"]
        ev += att_ev
        ev += safety.check_command(u)
        tim.tick(loop_t0)
        tim_ev, _ = tim.check()
        ev += tim_ev

        # log
        L["t"].append(loop_t0 - t_session)
        L["read_ms"].append((t_read - t0) * 1e3)
        L["obs_ms"].append((t_obs - t_read) * 1e3)
        L["infer_ms"].append((t_infer - t_obs) * 1e3)
        L["compose_ms"].append((t_compose - t_infer) * 1e3)
        L["obs"].append(obs.astype(np.float32))
        L["action"].append(action.astype(np.float32))
        L["command"].append(u.astype(np.float32))
        L["tilt"].append(tilt)
        L["imu_quat"].append(np.asarray(sr.imu_quat_wxyz, np.float32))
        L["imu_gyro"].append(np.asarray(sr.imu_gyro, np.float32))
        L["imu_accel"].append(np.asarray(sr.imu_accel, np.float32))
        L["joint_pos"].append(np.asarray(sr.joint_pos, np.float32))
        L["joint_vel"].append(np.asarray(sr.joint_vel, np.float32))
        L["contact"].append(np.array([sr.contact_L, sr.contact_R], np.float32))
        L["v_est"].append(obs_b.kin.v_est.astype(np.float32))
        L["phase"].append(obs_b.phase)

        for lvl, code, msg in ev:
            if lvl != "info":
                events.append((k, lvl, code, msg))
            if lvl == "warn" and (not a.mpu or k - last_warn.get(code, -10**9) >= a.rate):
                last_warn[code] = k
                print(f"  step {k:5d}  [warn] {code}: {msg}")
            if lvl == "abort":
                aborted = (k, code, msg)

        # command (no-op on hardware unless --enable-motors + stub implemented)
        robot.command(u)
        prev_action = action.copy()
        L["loop_ms"].append((time.perf_counter() - loop_t0) * 1e3)

        if aborted:
            print(f"\n[hil] ABORT at step {k}: {aborted[1]} -- {aborted[2]}")
            break
        if a.sim and hasattr(robot, "fell") and robot.fell():
            print(f"\n[hil] (sim) robot fell at step {k} -- expected in log-only; continuing not useful")
            break
        if not a.no_throttle:
            _sleep_to(next_t)

    robot.close()

    # ---- summary ----
    arr = {k: np.asarray(v) for k, v in L.items()}
    np.savez_compressed(logpath + ".npz", events=np.array(events, dtype=object),
                        target_hz=a.rate, source=src_name,
                        motors=bool(a.sim or a.enable_motors), **arr)
    s = tim.stats()
    lines = [
        f"HIL session {stamp}   source={src_name}   "
        f"motors={'on' if (a.sim or a.enable_motors) else 'OFF (log-only)'}",
        f"steps logged        : {len(arr['t'])}   ({arr['t'][-1] if len(arr['t']) else 0:.2f} s)",
        f"target rate         : {a.rate:.0f} Hz",
        f"achieved loop rate  : {s['hz']:.1f} Hz   jitter {s['jitter_ms']:.2f} ms   "
        f"(dt {s.get('dt_min_ms', float('nan')):.2f}-{s.get('dt_max_ms', float('nan')):.2f} ms)",
        "",
        "latency per step (ms)      mean   p95    max",
        f"  sensor read            {arr['read_ms'].mean():6.2f} {_p95(arr['read_ms']):6.2f} {arr['read_ms'].max():6.2f}",
        f"  observation build      {arr['obs_ms'].mean():6.2f} {_p95(arr['obs_ms']):6.2f} {arr['obs_ms'].max():6.2f}",
        f"  policy inference       {arr['infer_ms'].mean():6.2f} {_p95(arr['infer_ms']):6.2f} {arr['infer_ms'].max():6.2f}",
        f"  base-ctrl compose      {arr['compose_ms'].mean():6.2f} {_p95(arr['compose_ms']):6.2f} {arr['compose_ms'].max():6.2f}",
        f"  full loop              {arr['loop_ms'].mean():6.2f} {_p95(arr['loop_ms']):6.2f} {arr['loop_ms'].max():6.2f}",
        f"  sensor->action total   {(arr['read_ms']+arr['obs_ms']+arr['infer_ms']).mean():6.2f} "
        f"{_p95(arr['read_ms']+arr['obs_ms']+arr['infer_ms']):6.2f} "
        f"{(arr['read_ms']+arr['obs_ms']+arr['infer_ms']).max():6.2f}",
        "",
        f"observation            : dim {arr['obs'].shape[1]}   "
        f"range [{arr['obs'].min():+.2f}, {arr['obs'].max():+.2f}]   "
        f"non-finite: {int((~np.isfinite(arr['obs'])).sum())}",
        f"policy action          : range [{arr['action'].min():+.2f}, {arr['action'].max():+.2f}]   "
        f"|a| mean {np.abs(arr['action']).mean():.3f}",
        f"base command           : range [{arr['command'].min():+.2f}, {arr['command'].max():+.2f}] rad",
        _cmd_limit_report(arr["command"]),
        f"chest tilt (deg)       : mean {arr['tilt'].mean():.1f}   max {arr['tilt'].max():.1f}",
        f"|accel| (m/s^2)        : mean {np.linalg.norm(arr['imu_accel'],axis=1).mean():.2f}   "
        f"[{np.linalg.norm(arr['imu_accel'],axis=1).min():.2f}, {np.linalg.norm(arr['imu_accel'],axis=1).max():.2f}]",
        f"|gyro| (rad/s)         : mean {np.linalg.norm(arr['imu_gyro'],axis=1).mean():.2f}   "
        f"max {np.linalg.norm(arr['imu_gyro'],axis=1).max():.2f}",
        f"contact L/R fraction   : {arr['contact'][:,0].mean():.2f} / {arr['contact'][:,1].mean():.2f}",
        "",
        f"safety events (non-info): {len(events)}",
    ]
    from collections import Counter
    for (code, cnt) in Counter(e[2] for e in events).most_common():
        ex = next(e for e in events if e[2] == code)
        lines.append(f"  {ex[1]:5s} {code:14s} x{cnt:<4d}  e.g. step {ex[0]}: {ex[3]}")
    if a.mpu:
        lines += ["",
                  f"IMU (MPU-6050)         : reads {imu.n_reads}   i2c errors {imu.i2c_errors}   "
                  f"saturated samples {imu.n_saturated} ({100.0*imu.n_saturated/max(1, imu.n_reads):.1f}%)",
                  "  (sim joints/contacts + torso pinned; only the IMU channels are real)"]
        if imu.n_saturated:
            lines.append(f"  [warn] IMU range +-{a.accel_g} g / +-{a.gyro_dps} dps saturated -- "
                         "raise --accel-g/--gyro-dps")
    if aborted:
        lines.append(f"\nABORTED at step {aborted[0]}: {aborted[1]} -- {aborted[2]}")

    txt = "\n".join(lines)
    print("\n" + txt)
    with open(logpath + ".txt", "w") as f:
        f.write(txt + "\n")
    print(f"\n[hil] wrote {logpath}.npz  and  {logpath}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
