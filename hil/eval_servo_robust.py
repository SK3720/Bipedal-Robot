"""How much does sim2real_v1 tolerate hobby-servo behaviour?  (EVAL ONLY -- no training, no
existing file touched.)

Runs the DEPLOYMENT stack (hil.ObservationBuilder + hil.Policy + hil.BaseController, the same
code that runs on the Pi) against MuJoCo with a servo model between the command and the sim:

  command -> transport delay -> deadband -> rate (slew) limit -> 1st-order lag -> MuJoCo position actuator

and two joint-feedback modes:
  enc=true : joint_pos/vel are the true sim joints (a robot WITH encoders)
  enc=cmd  : joint_pos = the last COMMANDED angle, joint_vel = its finite difference
             (a hobby-servo robot with NO position feedback; FK / leg-odometry are then computed
              from commanded angles too, exactly as the Pi would)

It also measures what the policy DEMANDS of each actuator (command rate, torque) so it can be
compared with a real servo's datasheet.  The servo parameters here are a SWEEP, not assumed values
for the TD-7120MG -- when you know the real numbers, read them off the table.

  python -m hil.eval_servo_robust --quick          # 4 episodes / config
  python -m hil.eval_servo_robust --n 20           # full sweep (parallel)

Noise profile (--noise): 'hw' (default) = IMU noise MEASURED on the user's calibrated MPU-6050 from the
Pi log hil_20260920_214508 (gyro white 0.0007 rad/s, accel 0.016 m/s^2, residual gyro bias 0.0005 rad/s),
no encoder noise (enc=cmd reads the command; enc=true assumes ideal encoders), no contact dropout;
'train' = the (much larger) noise the policy was distilled against.

Caveat: the HIL stack runs the base controller once per 200 Hz step (the env runs it at 1 kHz on ground-truth
state), so even the 'ideal' row is the number to compare against, not the env's 97 %.  The '1 kHz base'
rows emulate the env's controller rate for comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil import spec                                                     # noqa: E402

EP_SECONDS = 7.0
DT_SIM = spec.SIM_TIMESTEP

CONFIGS = [
    dict(name="ideal (reference)",               enc="true"),
    dict(name="no encoders (cmd feedback)",      enc="cmd"),
    dict(name="lag 40 ms",                       enc="true", tau_ms=40),
    dict(name="lag 80 ms",                       enc="true", tau_ms=80),
    dict(name="rate limit 6 rad/s",              enc="true", rate=6.0),
    dict(name="rate limit 3 rad/s",              enc="true", rate=3.0),
    dict(name="rate limit 1.5 rad/s",            enc="true", rate=1.5),
    dict(name="delay 20 ms",                     enc="true", delay_ms=20),
    dict(name="delay 40 ms",                     enc="true", delay_ms=40),
    dict(name="deadband 0.02 rad",               enc="true", deadband=0.02),
    dict(name="cmd refresh 100 Hz (PWM frame)",  enc="true", hold_hz=100),
    dict(name="cmd refresh 50 Hz (PWM frame)",   enc="true", hold_hz=50),
    dict(name="combo A (lag40+6r/s+d10+db.01)",  enc="true", tau_ms=40, rate=6.0, delay_ms=10, deadband=0.01),
    dict(name="combo A + no encoders",           enc="cmd",  tau_ms=40, rate=6.0, delay_ms=10, deadband=0.01),
    dict(name="combo C (lag80+3r/s+d20+db.02)+no enc", enc="cmd", tau_ms=80, rate=3.0, delay_ms=20, deadband=0.02),
    dict(name="1 kHz base ctrl (env-like), ideal",           enc="true", bc_hz=1000),
    dict(name="1 kHz base + rate 6 r/s",                     enc="true", bc_hz=1000, rate=6.0),
    dict(name="1 kHz base + combo A + no encoders",          enc="cmd",  bc_hz=1000, tau_ms=40, rate=6.0, delay_ms=10, deadband=0.01),
    # ---- boundary search (200 Hz deployment stack)
    dict(name="lag 5 ms",                        enc="true", tau_ms=5),
    dict(name="lag 10 ms",                       enc="true", tau_ms=10),
    dict(name="lag 20 ms",                       enc="true", tau_ms=20),
    dict(name="rate limit 50 rad/s",             enc="true", rate=50.0),
    dict(name="rate limit 25 rad/s",             enc="true", rate=25.0),
    dict(name="rate limit 12 rad/s",             enc="true", rate=12.0),
    dict(name="delay 5 ms",                      enc="true", delay_ms=5),
    dict(name="delay 10 ms",                     enc="true", delay_ms=10),
    # ---- boundary search (1 kHz base controller, env-like)
    dict(name="1 kHz base + lag 10 ms",          enc="true", bc_hz=1000, tau_ms=10),
    dict(name="1 kHz base + lag 20 ms",          enc="true", bc_hz=1000, tau_ms=20),
    dict(name="1 kHz base + rate 25 r/s",        enc="true", bc_hz=1000, rate=25.0),
    dict(name="1 kHz base + rate 12 r/s",        enc="true", bc_hz=1000, rate=12.0),
    # ---- no-encoder observers: joint feedback = low-passed command (ideal actuators)
    dict(name="no enc: smoothed cmd, tau 15 ms", enc="cmdlp", obs_tau_ms=15),
    dict(name="no enc: smoothed cmd, tau 30 ms", enc="cmdlp", obs_tau_ms=30),
    dict(name="no enc: smoothed cmd, tau 60 ms", enc="cmdlp", obs_tau_ms=60),
]


class ServoModel:
    """Per-joint servo proxy, advanced once per 1 ms sim substep (all 15 joints, vectorised)."""

    def __init__(self, delay_ms=0, deadband=0.0, rate=np.inf, tau_ms=0.0, hold_hz=0.0, n=15):
        self.hold_n = int(round(1.0 / (hold_hz * DT_SIM))) if hold_hz else 1        # sim ticks per command refresh
        self.delay = int(round(delay_ms))
        self.db, self.rate, self.tau = float(deadband), float(rate), float(tau_ms) / 1e3
        self.n = n
        self.reset(np.zeros(n))
        self.rate_hits = 0
        self.steps = 0

    def reset(self, u0):
        u0 = np.asarray(u0, float).copy()
        self.buf = deque([u0.copy() for _ in range(self.delay + 1)])
        self.tgt, self.mid, self.out = u0.copy(), u0.copy(), u0.copy()
        self.held, self._tick = u0.copy(), 0

    def step(self, u):
        if self._tick % self.hold_n == 0:                                # sample-and-hold (PWM frame rate)
            self.held = np.asarray(u, float)
        self._tick += 1
        self.buf.append(self.held)
        v = self.buf.popleft()                                           # transport delay
        self.tgt = np.where(np.abs(v - self.tgt) > self.db, v, self.tgt)  # deadband
        lim = self.rate * DT_SIM
        delta = self.tgt - self.mid
        self.rate_hits += int(np.sum(np.abs(delta) > lim)) if np.isfinite(lim) else 0
        self.steps += 1
        self.mid = self.mid + np.clip(delta, -lim, lim)                  # slew limit
        self.out = self.mid if self.tau <= 0 else self.out + (self.mid - self.out) * (DT_SIM / self.tau)
        return self.out


def make_robot(cfg, seed):
    from hil.robot_interface import SimRobot

    class ServoSimRobot(SimRobot):
        def __init__(self):
            super().__init__(noise=True, seed=seed)
            c = spec.load_consts()
            self.srv = None
            if any(k in cfg for k in ("tau_ms", "rate", "delay_ms", "deadband", "hold_hz")):
                self.srv = ServoModel(cfg.get("delay_ms", 0), cfg.get("deadband", 0.0),
                                      cfg.get("rate", np.inf), cfg.get("tau_ms", 0.0), cfg.get("hold_hz", 0.0))
                self.srv.reset(self.data.ctrl[:15])
            self.cmd_mode = cfg["enc"] in ("cmd", "cmdlp")
            self._lp = c["ctrl0"].copy()
            self._lp_prev = self._lp.copy()
            self._lp_a = 1.0 - np.exp(-spec.CONTROL_DT / (cfg.get("obs_tau_ms", 0.0) / 1e3)) if cfg["enc"] == "cmdlp" else 1.0
            self._cmd = c["ctrl0"].copy()
            self._cmd_prev = self._cmd.copy()
            self.max_rate = np.zeros(15)
            self.rates = []
            self.torque = []

        def read(self):
            sr = super().read()
            if self.cmd_mode:
                # raw command ('cmd') or its low-passed dead-reckoned version ('cmdlp'; alpha=1 -> raw)
                sr.joint_pos = self._lp[spec.ACT_CTRL].copy()
                sr.joint_vel = ((self._lp - self._lp_prev) / spec.CONTROL_DT)[spec.ACT_CTRL]
            return sr

        def _track(self):
            self._lp_prev = self._lp.copy()
            self._lp = self._lp + self._lp_a * (self._cmd - self._lp)

        def command(self, u15):
            d = self.data
            u = np.asarray(u15, float)
            self._prev_cvel = d.cvel[self.b_chest][3:6].copy()
            self._cmd_prev, self._cmd = self._cmd, u.copy()
            self.rates.append(np.abs(self._cmd - self._cmd_prev) / spec.CONTROL_DT)
            self._track()
            peak = np.zeros(15)
            for _ in range(spec.FRAME_SKIP):
                d.ctrl[:15] = self.srv.step(u) if self.srv is not None else u
                self.mj.mj_step(self.model, d)
                peak = np.maximum(peak, np.abs(d.actuator_force[:15]))
            self.torque.append(peak)

        def command_fast(self, bc, a, ob):
            """Emulate the env: base controller recomputed every 1 ms substep from fresh sim state."""
            d = self.data
            self._prev_cvel = d.cvel[self.b_chest][3:6].copy()
            u = None
            peak = np.zeros(15)
            for j in range(spec.FRAME_SKIP):
                R = d.xmat[self.b_chest].reshape(3, 3)
                base = d.xpos[self.b_chest]
                fl, fr = R.T @ (d.xpos[self.b_lf] - base), R.T @ (d.xpos[self.b_rf] - base)
                Ll, Lr = self._foot_contact_force("L"), self._foot_contact_force("R")
                u = bc.compose(a, ob.phase + 2 * np.pi * spec.GAIT_HZ * DT_SIM * j, d.qpos[3:7].copy(),
                               d.qvel[3:6].copy(), fl, fr, ob.kin.v_est, Ll > 6.0, Lr > 6.0,
                               foot_load_L=Ll, foot_load_R=Lr)
                d.ctrl[:15] = self.srv.step(u) if self.srv is not None else u
                self.mj.mj_step(self.model, d)
                peak = np.maximum(peak, np.abs(d.actuator_force[:15]))
            self._cmd_prev, self._cmd = self._cmd, u.copy()
            self.rates.append(np.abs(self._cmd - self._cmd_prev) / spec.CONTROL_DT)
            self._track()
            self.torque.append(peak)

    return ServoSimRobot()


NOISE_HW = dict(gyro_rad_s=0.0007, accel_m_s2=0.016, gyro_bias_rad_s=0.0005,
                enc_pos_rad=0.0, enc_vel_rad_s=0.0, contact_dropout=0.0)


def run_episode(args):
    ci, seed, noise = args
    if noise == "hw":
        spec.TRAIN_NOISE.update(NOISE_HW)
    from hil.observation import ObservationBuilder
    from hil.base_controller import BaseController
    from hil.policy import Policy
    cfg = CONFIGS[ci]
    robot = make_robot(cfg, seed)
    pol, bc = Policy(), BaseController()
    ob = ObservationBuilder(control_dt=spec.CONTROL_DT)
    sr = robot.read()
    ob.reset(sr.imu_quat_wxyz, sr.imu_gyro, sr.imu_accel, sr.joint_pos, sr.joint_vel,
             sr.contact_L, sr.contact_R, phase0=0.0)
    prev = np.zeros(14)
    y0 = float(robot.data.qpos[1])
    n = int(EP_SECONDS * spec.CONTROL_HZ)
    survived = True
    for k in range(n):
        sr = robot.read()
        obs = ob.step(sr.imu_quat_wxyz, sr.imu_gyro, sr.imu_accel, sr.joint_pos, sr.joint_vel,
                      sr.contact_L, sr.contact_R, prev)
        a = pol.act(obs)
        if cfg.get("bc_hz") == 1000:
            robot.command_fast(bc, a, ob)
        else:
            u = bc.compose(a, ob.phase, sr.imu_quat_wxyz, sr.imu_gyro, ob.kin.foot_L_rel,
                           ob.kin.foot_R_rel, ob.kin.v_est, sr.contact_L, sr.contact_R)
            robot.command(u)
        prev = a
        if robot.fell():
            survived = False
            break
    dist = y0 - float(robot.data.qpos[1])                                 # walk direction = world -Y
    R = np.array(robot.rates)
    T = np.array(robot.torque)
    return dict(ci=ci, seed=seed, survived=survived, steps=k + 1, dist=dist,
                rate_p95=np.percentile(R, 95, axis=0).tolist(), rate_max=R.max(0).tolist(),
                tq_p95=np.percentile(T, 95, axis=0).tolist(), tq_max=T.max(0).tolist(),
                rate_hit=(robot.srv.rate_hits / max(1, robot.srv.steps * 15)) if robot.srv else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="episodes per config")
    ap.add_argument("--quick", action="store_true", help="4 episodes per config")
    ap.add_argument("--noise", choices=["hw", "train"], default="hw")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    n = 4 if a.quick else a.n
    tasks = [(ci, 7000 + s, a.noise) for ci in range(len(CONFIGS)) for s in range(n)]
    t0 = time.time()
    print(f"[servo-robust] {len(CONFIGS)} configs x {n} episodes = {len(tasks)} runs, {a.workers} workers, noise={a.noise}")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(run_episode, tasks, chunksize=1))
    print(f"  done in {time.time()-t0:.0f} s")

    lines = [f"sim2real_v1 vs servo behaviour  ({n} episodes/config, {EP_SECONDS:.0f} s each, deployment "
             f"stack, noise profile '{a.noise}'; NOT the env's own eval)", "",
             f"{'config':44s} {'survive':>8s} {'dist m':>7s} {'m/s':>6s} {'rate-limited':>12s}"]
    for ci, cfg in enumerate(CONFIGS):
        r = [x for x in res if x["ci"] == ci]
        sv = np.mean([x["survived"] for x in r]) * 100
        dist = np.mean([x["dist"] for x in r])
        spd = np.mean([x["dist"] / (x["steps"] / spec.CONTROL_HZ) for x in r])
        rh = np.mean([x["rate_hit"] for x in r]) * 100
        lines.append(f"{cfg['name']:44s} {sv:7.0f}% {dist:7.2f} {spd:6.2f} {rh:11.1f}%")

    ref = [x for x in res if x["ci"] == 0]
    fast = [x for x in res if x["ci"] == 15]
    lines += ["", "What the policy demands of each actuator (ideal actuators; rate = |d command/dt| at the 200 Hz step, "
              "torque = per-step peak; sim torque limit 2.3 N*m):",
              f"{'joint':22s} | {'200 Hz base: rate p95':>21s} {'max':>6s} | {'1 kHz base: rate p95':>20s} {'max':>6s} | "
              f"{'torque p95':>10s} {'max':>5s}"]
    P95 = np.mean([x["rate_p95"] for x in ref], 0); PMAX = np.max([x["rate_max"] for x in ref], 0)
    F95 = np.mean([x["rate_p95"] for x in fast], 0); FMAX = np.max([x["rate_max"] for x in fast], 0)
    Q95 = np.mean([x["tq_p95"] for x in ref], 0); QMAX = np.max([x["tq_max"] for x in ref], 0)
    for i, nme in enumerate(spec.JOINTS_MJ):
        lines.append(f"{nme:22s} | {P95[i]:21.1f} {PMAX[i]:6.0f} | {F95[i]:20.1f} {FMAX[i]:6.0f} | {Q95[i]:10.2f} {QMAX[i]:5.2f}")
    txt = "\n".join(lines)
    print("\n" + txt)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
    base = os.path.join(os.path.dirname(__file__), "logs", f"servo_robust_{stamp}")
    open(base + ".txt", "w").write(txt + "\n")
    json.dump(res, open(base + ".json", "w"))
    print(f"\nwrote {base}.txt / .json")


if __name__ == "__main__":
    main()
