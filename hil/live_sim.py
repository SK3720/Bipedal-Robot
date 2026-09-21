"""PC side of the live link: the REAL MPU-6050 (streamed from the Pi) driving the SIM, in a MuJoCo viewer.

  python -m hil.live_sim                       # listen on UDP 5005, open the viewer
  python -m hil.live_sim --no-viewer --seconds 20

What is real and what is simulated:
  * IMU (orientation / gyro / accel)   REAL  -- from `python -m hil.imu_stream` on the Pi
  * sim torso orientation              PUPPETED to the real quaternion every step (tilt the board -> the sim
                                       robot tilts); position pinned at the standing pose (a robot on a gantry)
  * joints / foot contacts             SIM, driven by the policy's own commands
  * observation -> sim2real_v1 policy -> base controller -> command   the deployment stack, unchanged
Nothing is sent to any actuator.  The torso is kinematic, so this shows the policy's RESPONSE to a real IMU
signal (do the hips/ankles/knees move the right way?) -- it is not a balance test, and with the torso tilted the
sim feet push into / lift off the ground, which is expected.

Windows Firewall may ask to allow Python on the private network the first time it binds the UDP port: allow it.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hil import spec, safety                                             # noqa: E402
from hil.imu_stream import PKT                                           # noqa: E402
from hil.observation import ObservationBuilder                           # noqa: E402
from hil.base_controller import BaseController                           # noqa: E402
from hil.policy import Policy                                            # noqa: E402
from hil.hybrid_robot import HybridRobot                                 # noqa: E402
from hil.robot_interface import SimRobot                                 # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


class NetImu:
    """Threaded UDP receiver; read() returns the latest (quat, gyro, accel)."""

    def __init__(self, port, bind="0.0.0.0"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind, port))
        self.sock.settimeout(0.2)
        self.lock = threading.Lock()
        self.latest = None
        self.t_recv = 0.0
        self.n = 0
        self.drops = 0
        self._seq = None
        self.saturated = 0
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                return
            if len(data) != PKT.size:
                continue
            f = PKT.unpack(data)
            seq, q, g, a, sat = f[0], np.array(f[2:6]), np.array(f[6:9]), np.array(f[9:12]), f[12]
            n = np.linalg.norm(q)
            if not np.all(np.isfinite(q)) or n < 0.5:
                continue
            with self.lock:
                if self._seq is not None and seq > self._seq + 1:
                    self.drops += seq - self._seq - 1
                self._seq = seq
                self.latest = (q / n, g, a)
                self.t_recv = time.perf_counter()
                self.saturated = sat
                self.n += 1

    def wait_first(self, timeout):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.latest is not None:
                return True
            time.sleep(0.02)
        return False

    def age(self):
        return time.perf_counter() - self.t_recv

    def read(self):
        with self.lock:
            q, g, a = self.latest
        return q.copy(), g.copy(), a.copy()

    def close(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


class LiveRobot(HybridRobot):
    """HybridRobot whose sim torso follows the REAL quaternion instead of standing upright."""

    def __init__(self, imu, follow=True):
        super().__init__(imu, pin_base=True)
        self.follow = follow
        self._q_last = self._base_q[3:7].copy()

    def read(self):
        sr = super().read()
        self._q_last = np.asarray(sr.imu_quat_wxyz, float).copy()
        return sr

    def command(self, u15):
        SimRobot.command(self, u15)          # 5 physics substeps with the legs driven by the policy command
        d = self.data
        d.qpos[:3] = self._base_q[:3]        # gantry: position pinned ...
        d.qpos[3:7] = self._q_last if self.follow else self._base_q[3:7]   # ... orientation = the real IMU's
        d.qvel[:6] = 0.0
        self.mj.mj_forward(self.model, d)


def _sleep_to(t):
    while True:
        rem = t - time.perf_counter()
        if rem <= 0:
            return
        if rem > 0.0015:
            time.sleep(rem - 0.0012)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until the viewer is closed / Ctrl-C")
    ap.add_argument("--rate", type=float, default=spec.CONTROL_HZ)
    ap.add_argument("--viz", choices=["stick", "mujoco", "none"], default="stick",
                    help="stick = tkinter stick figure (no OpenGL needed, default); mujoco = MuJoCo's OpenGL viewer "
                         "(needs a working GPU driver); none = headless")
    ap.add_argument("--no-viewer", action="store_true", help="same as --viz none")
    ap.add_argument("--snapshot", default=None, help="(stick) save a PNG of the view after ~3.5 s (tests / docs)")
    ap.add_argument("--no-follow", action="store_true", help="keep the sim torso upright (do not puppet it)")
    ap.add_argument("--wait", type=float, default=180.0, help="seconds to wait for the first packet")
    a = ap.parse_args(argv)
    if a.no_viewer:
        a.viz = "none"

    net = NetImu(a.port)
    print(f"[live_sim] listening on UDP {a.port} ... start  `python -m hil.imu_stream --host <this PC's IP>`  on the Pi")
    if not net.wait_first(a.wait):
        print("[live_sim] no packets received. Check: the Pi is running hil.imu_stream, --host is THIS PC's IP, "
              "both are on the same network, and Windows Firewall allows Python (private network).")
        net.close()
        return 2
    print("[live_sim] receiving IMU packets")

    robot = LiveRobot(net, follow=not a.no_follow)
    dt = 1.0 / a.rate
    ob, pol, bc = ObservationBuilder(control_dt=dt), Policy(), BaseController()
    sr = robot.read()
    ob.reset(sr.imu_quat_wxyz, sr.imu_gyro, sr.imu_accel, sr.joint_pos, sr.joint_vel,
             sr.contact_L, sr.contact_R, phase0=0.0)
    prev = np.zeros(14)

    viewer = viz = None
    if a.viz == "mujoco":
        import mujoco.viewer                      # hard-exits the process if the GPU driver has no OpenGL
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
    elif a.viz == "stick":
        from hil.stick_viz import StickViz
        viz = StickViz(robot.model)
    shot_done = False

    L = dict(t=[], quat=[], gyro=[], accel=[], action=[], command=[], age_ms=[], loop_ms=[])
    t0 = time.perf_counter()
    nxt = t0
    k = 0
    last_print = t0
    stale_warned = 0.0
    try:
        while True:
            t_it = time.perf_counter()
            nxt += dt
            sr = robot.read()
            ev = safety.check_reading(sr)
            if any(l == "abort" for l, _, _ in ev):
                print(f"\n[live_sim] bad IMU reading: {[e for e in ev if e[0] == 'abort']}")
                break
            obs = ob.step(sr.imu_quat_wxyz, sr.imu_gyro, sr.imu_accel, sr.joint_pos, sr.joint_vel,
                          sr.contact_L, sr.contact_R, prev)
            act = pol.act(obs)
            u = bc.compose(act, ob.phase, sr.imu_quat_wxyz, sr.imu_gyro, ob.kin.foot_L_rel,
                           ob.kin.foot_R_rel, ob.kin.v_est, sr.contact_L, sr.contact_R)
            robot.command(u)
            prev = act
            k += 1
            L["t"].append(t_it - t0); L["quat"].append(sr.imu_quat_wxyz.astype(np.float32))
            L["gyro"].append(sr.imu_gyro.astype(np.float32)); L["accel"].append(sr.imu_accel.astype(np.float32))
            L["action"].append(act.astype(np.float32)); L["command"].append(u.astype(np.float32))
            L["age_ms"].append(net.age() * 1e3)
            if viewer is not None:
                if not viewer.is_running():
                    break
                if k % 3 == 0:
                    viewer.sync()
            now = time.perf_counter()
            if viz is not None:
                if not viz.alive():
                    break
                if k % 6 == 0:                                              # ~33 fps
                    w, x, y, z = sr.imu_quat_wxyz
                    tilt = np.degrees(np.arccos(np.clip(2 * (y * z + w * x), -1, 1)))    # angle of chest-up from world-up
                    viz.draw(robot.data, [
                        f"REAL MPU-6050 -> sim      tilt {tilt:5.1f} deg      |gyro| {np.linalg.norm(sr.imu_gyro):5.2f} rad/s",
                        f"loop {k / max(now - t0, 1e-9):5.1f} Hz     packets {net.n} (dropped {net.drops})     "
                        f"age {net.age() * 1e3:4.0f} ms     saturated {net.saturated}",
                        f"policy |action| mean {np.abs(act).mean():.2f}     contacts  L {int(sr.contact_L)}  R {int(sr.contact_R)}"],
                        sr.contact_L, sr.contact_R)
                if a.snapshot and not shot_done and now - t0 > 3.5:
                    viz.snapshot(a.snapshot)
                    shot_done = True
            if net.age() > 0.25 and now - stale_warned > 1.0:
                print(f"\n[live_sim] WARNING: no IMU packet for {net.age():.2f} s (holding last value)")
                stale_warned = now
            if now - last_print > 0.5:
                w, x, y, z = sr.imu_quat_wxyz
                upv = np.array([2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)])   # chest +Y (up) in the world
                tilt = np.degrees(np.arccos(np.clip(upv[2], -1, 1)))
                print(f"  t {now - t0:6.1f}s  loop {k / (now - t0):5.1f} Hz  tilt {tilt:5.1f} deg  "
                      f"gyro {np.linalg.norm(sr.imu_gyro):5.2f} rad/s  pkts {net.n} (drop {net.drops})  "
                      f"age {net.age()*1e3:4.0f} ms  |a_cmd| {np.abs(act).mean():.2f}", end="\r")
                last_print = now
            if a.seconds and now - t0 >= a.seconds:
                break
            _sleep_to(nxt)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        if viz is not None:
            viz._close()
        net.close()
    arr = {kk: np.asarray(v) for kk, v in L.items()}
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
    if len(arr["t"]):
        np.savez_compressed(path, **arr)
    n = max(1, len(arr["t"]))
    dur = arr["t"][-1] if len(arr["t"]) else 0.0
    print(f"\n[live_sim] {n} steps in {dur:.1f} s ({n / max(dur, 1e-9):.1f} Hz); packets {net.n}, dropped {net.drops}, "
          f"saturated samples {net.saturated}; mean packet age {arr['age_ms'].mean() if len(arr['t']) else 0:.1f} ms")
    if len(arr["t"]):
        print(f"           command range [{arr['command'].min():+.2f}, {arr['command'].max():+.2f}] rad; log {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
