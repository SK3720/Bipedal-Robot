"""Latency-robust standing LQR - drop-in replacement for
standing_balance_lqr.StandingLQR.

Why this exists
--------------
validate_recovery.py showed the recovery controller collapses under a few ms of
control latency.  Root cause: the original StandingLQR (Q = orientation 400,
orientation-rate 25, R = 3) has a closed loop whose balance poles sit at only
-1.4 .. -3.6 rad/s - so sluggish and so close to the unit circle that ~2-3 ms of
delay destabilises it even with NO disturbance.  A marginally-stable base
controller will not survive contact with real hardware.

This version re-weights Q toward heavy ORIENTATION-RATE damping (phase lead ->
delay margin) with a higher R (lower loop gain -> gain margin).  Measured: it
stands with NO disturbance up to ~8 ms control latency (vs ~2 ms), a ~4x margin,
while still rejecting the small pushes the recovery controller is built for.

Same interface as StandingLQR: .qpos0 .qvel0 .ctrl0 .K .control(model, data).

    python standing_lqr_robust.py        # self-test: latency + push rejection
"""
from __future__ import annotations

import numpy as np
import mujoco

from biped_env import DEFAULT_POSE
from standing_balance_lqr import _dare, settle_standing

# tuned by search in validate_recovery / scratch - see module docstring
Q_ORI = 600.0
Q_ORI_RATE = 400.0        # <- the key term: rate damping buys the delay margin
Q_HEIGHT = 40.0
Q_JOINT = 4.0
Q_JOINT_RATE = 4.0
Q_BASE_XY = 3.0
Q_BASE_XY_RATE = 6.0
R_SCALE = 10.0
DESIGN_LATENCY_MS = 8      # documented tested no-disturbance latency margin


class RobustStandingLQR:
    def __init__(self, model, data, verbose=True):
        self.model = model
        self.nv = model.nv
        self.nu = 15
        self.qpos0, self.qvel0 = settle_standing(model, data)
        self.ctrl0 = DEFAULT_POSE.copy()

        data.qpos[:] = self.qpos0
        data.qvel[:] = self.qvel0
        data.ctrl[:15] = self.ctrl0
        mujoco.mj_forward(model, data)

        n2 = 2 * self.nv
        A = np.zeros((n2, n2))
        B = np.zeros((n2, self.nu))
        mujoco.mjd_transitionFD(model, data, 1e-6, 1, A, B, None, None)
        self.A, self.B = A, B

        qp = np.ones(self.nv) * 2.0
        qp[0:2] = Q_BASE_XY
        qp[2] = Q_HEIGHT
        qp[3:6] = Q_ORI
        qp[6:21] = Q_JOINT
        qv = np.ones(self.nv) * 1.0
        qv[0:3] = Q_BASE_XY_RATE
        qv[3:6] = Q_ORI_RATE
        qv[6:21] = Q_JOINT_RATE
        Q = np.diag(np.concatenate([qp, qv]))
        R = np.eye(self.nu) * R_SCALE

        self.K, _, it = _dare(A, B, Q, R, iters=5000)
        if verbose:
            rho = np.sort(np.abs(np.linalg.eigvals(A - B @ self.K)))
            print(f"[RobustStandingLQR] DARE it={it}  balance-mode rho={rho[-4]:.4f}  "
                  f"max|K|={np.abs(self.K).max():.1f}  design latency margin ~{DESIGN_LATENCY_MS} ms")

    def state_error(self, data):
        dq = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.model, dq, 1.0, self.qpos0, data.qpos)
        return np.concatenate([dq, data.qvel - self.qvel0])

    def control(self, model, data, k=None, bs=None):
        u = self.ctrl0 - self.K @ self.state_error(data)
        return np.clip(u, model.actuator_ctrlrange[:15, 0],
                       model.actuator_ctrlrange[:15, 1])


# make it usable as `from standing_lqr_robust import StandingLQR`
StandingLQR = RobustStandingLQR


def _selftest():
    from recovery_metrics import sample_balance, CHEST_BODY
    from biped_env import PUSH_DURATION_STEPS
    from standing_balance_lqr import StandingLQR as BaseLQR

    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)

    def latency_scan(lqr, tag):
        print(f"\n  {tag}: no-disturbance latency tolerance")
        for lat in (0, 2, 4, 6, 8, 10, 12, 16, 20):
            d.qpos[:] = lqr.qpos0; d.qvel[:] = lqr.qvel0; d.act[:] = 0.0
            mujoco.mj_forward(m, d)
            buf = []
            fell = None
            for kk in range(3500):
                u = lqr.control(m, d)
                buf.append(u)
                uu = buf.pop(0) if len(buf) > lat else buf[0]
                d.ctrl[:15] = uu
                mujoco.mj_step(m, d)
                if sample_balance(m, d).up_tilt_deg > 25:
                    fell = kk
                    break
            print(f"    {lat:2d} ms: {'FELL @ ' + str(fell) if fell else 'stood 3.5 s'}")

    latency_scan(BaseLQR(m, d, verbose=False), "ORIGINAL StandingLQR")
    latency_scan(RobustStandingLQR(m, d, verbose=True), "RobustStandingLQR")

    # push rejection at 6 ms latency
    lqr = RobustStandingLQR(m, d, verbose=False)
    print("\n  RobustStandingLQR push rejection @ 6 ms latency (no step):")
    for pn in (80, 100, 115, 130, 145):
        d.qpos[:] = lqr.qpos0; d.qvel[:] = lqr.qvel0; d.act[:] = 0.0
        mujoco.mj_forward(m, d)
        buf = []
        fell = False
        for kk in range(2500):
            d.xfrc_applied[CHEST_BODY, :] = 0.0
            if 20 <= kk < 20 + PUSH_DURATION_STEPS:
                d.xfrc_applied[CHEST_BODY, 0:2] = [0.0, -pn]
            u = lqr.control(m, d)
            buf.append(u)
            uu = buf.pop(0) if len(buf) > 6 else buf[0]
            d.ctrl[:15] = uu
            mujoco.mj_step(m, d)
            if sample_balance(m, d).up_tilt_deg > 25:
                fell = True
                break
        b = sample_balance(m, d)
        print(f"    {pn:3d} N: {'FELL' if fell else f'held (lean {b.fwd_lean_deg:+.1f} deg)'}")


if __name__ == "__main__":
    _selftest()
