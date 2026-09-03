"""Standing balance via LQR about the standing equilibrium.

THE BOTTLENECK (established experimentally, 2026-08-31):
  forward_push_recovery_test.py + _push_probe.py show that with the default
  position-servo "hold DEFAULT_POSE" control the robot recovers a forward push
  only up to ~60-70 N. From ~80-130 N it REBOUNDS and topples BACKWARD (stiff
  undamped ankle acting as a spring); from ~140 N it topples forward. There is
  no standing balance feedback controller anywhere in the repo - the position
  servos are a stiff spring, not a stabiliser. Every downstream experiment
  (golden catch / arrest / r-recovery) is built on this missing foundation,
  which is why they diverge into 3-D topple + spin rather than a clean sagittal
  fall.

  Also: control->balance coupling is messy (ankle-pitch commands induce roll,
  hip-pitch commands induce large roll) - see _sign_probe.py - so a hand-tuned
  SISO stabiliser is fragile. LQR on the full finite-difference linearisation
  handles the cross-coupling directly.

This module builds an infinite-horizon discrete LQR gain about the settled
standing state using mujoco.mjd_transitionFD, and exposes it as a controller
(target = ctrl0 - K (x - x0)) for forward_push_recovery_test.py.

Does not modify robot.xml / biped_env / any golden experiment.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

import mujoco
import numpy as np

from biped_env import CHEST_Z_CONTACT, DEFAULT_POSE, SETTLE_STEPS, STANDING_QUAT


def settle_standing(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, CHEST_Z_CONTACT]
    data.qpos[3:7] = STANDING_QUAT
    data.qpos[7:22] = DEFAULT_POSE
    data.qvel[:] = 0.0
    data.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(model, data)
    for _ in range(SETTLE_STEPS):
        data.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


def _dare(A, B, Q, R, iters=4000, tol=1e-10):
    """Discrete algebraic Riccati via fixed-point iteration (no scipy).

    The iteration converges quickly on the controllable subspace; the cap is
    normally only reached because a few lightly-weighted, effectively
    uncontrollable base modes (x, y, yaw) leave a residual. That does not affect
    the stabilising gain on the modes we care about (orientation / height).
    """
    P = Q.copy()
    for i in range(iters):
        BtP = B.T @ P
        S = R + BtP @ B
        K = np.linalg.solve(S, BtP @ A)
        Pn = Q + A.T @ P @ A - (A.T @ P @ B) @ K
        Pn = 0.5 * (Pn + Pn.T)
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    BtP = B.T @ P
    K = np.linalg.solve(R + BtP @ B, BtP @ A)
    return K, P, i


class StandingLQR:
    def __init__(self, model, data, verbose=True):
        self.model = model
        self.nv = model.nv
        self.nu = 15
        self.qpos0, self.qvel0 = settle_standing(model, data)
        self.ctrl0 = DEFAULT_POSE.copy()

        # linearise about the settled state at ctrl0
        data.qpos[:] = self.qpos0
        data.qvel[:] = self.qvel0
        data.ctrl[:15] = self.ctrl0
        mujoco.mj_forward(model, data)

        n2 = 2 * self.nv
        A = np.zeros((n2, n2))
        B = np.zeros((n2, self.nu))
        mujoco.mjd_transitionFD(model, data, 1e-6, 1, A, B, None, None)
        self.A, self.B = A, B

        # ---- cost ----
        # state order: [dq(nv)] then [dv(nv)];  dof layout:
        #   0:3 base translation, 3:6 base rotation, 6:21 joints
        q_pos = np.ones(self.nv) * 2.0
        q_pos[0:2] = 3.0        # base x,y drift  (low-ish; stepping owns big drift)
        q_pos[2] = 40.0         # base height
        q_pos[3:6] = 400.0      # base orientation (pitch/roll/yaw)  <-- dominate
        q_pos[6:21] = 1.0       # joint deviations - cheap
        q_vel = np.ones(self.nv) * 1.0
        q_vel[0:3] = 6.0        # base linear vel
        q_vel[3:6] = 25.0       # base angular vel
        q_vel[6:21] = 0.4
        Q = np.diag(np.concatenate([q_pos, q_vel]))
        R = np.diag(np.ones(self.nu) * 3.0)

        self.K, _, it = _dare(A, B, Q, R)
        if verbose:
            eig = np.linalg.eigvals(A - B @ self.K)
            print(f"[StandingLQR] linearised nv={self.nv}, DARE iters={it}, "
                  f"closed-loop spectral radius = {np.max(np.abs(eig)):.4f}")

    def state_error(self, data):
        dq = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.model, dq, 1.0, self.qpos0, data.qpos)
        dv = data.qvel - self.qvel0
        return np.concatenate([dq, dv])

    def control(self, model, data, k=None, bs=None):
        dx = self.state_error(data)
        u = self.ctrl0 - self.K @ dx
        return np.clip(u, model.actuator_ctrlrange[:15, 0], model.actuator_ctrlrange[:15, 1])


_CACHE: dict[int, StandingLQR] = {}


def get_controller(model, data):
    """Cached factory keyed on model id (rebuild is ~cheap but not free)."""
    key = id(model)
    if key not in _CACHE:
        _CACHE[key] = StandingLQR(model, data, verbose=True)
    return _CACHE[key].control


def _selftest(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=6.0)
    args = p.parse_args(list(argv) if argv is not None else None)

    model = mujoco.MjModel.from_xml_path("robot/robot.xml")
    data = mujoco.MjData(model)
    lqr = StandingLQR(model, data)

    from recovery_metrics import chest_lean_yaw

    data.qpos[:] = lqr.qpos0
    data.qvel[:] = lqr.qvel0
    mujoco.mj_forward(model, data)
    n = int(args.seconds * 1000)
    print(f"\nHold standing for {args.seconds}s under LQR (no push):")
    for k in range(n):
        u = lqr.control(model, data)
        data.ctrl[:15] = u
        mujoco.mj_step(model, data)
        if k % 500 == 0:
            ut, fl, sl, yaw = chest_lean_yaw(model, data)
            print(f"  t={k/1000:4.1f}s  up_tilt={ut:5.2f}  fwd={fl:+5.2f}  side={sl:+5.2f}  "
                  f"yaw={yaw:+5.2f}  chestZ={data.qpos[2]:.4f}  |u-u0|={np.linalg.norm(u-lqr.ctrl0):.3f}")
    ut, fl, sl, yaw = chest_lean_yaw(model, data)
    ok = ut < 3.0 and data.qpos[2] > CHEST_Z_CONTACT - 0.03
    print(f"\n  final up_tilt {ut:.2f} deg, chestZ {data.qpos[2]:.4f}  -> {'STABLE' if ok else 'NOT STABLE'}")


if __name__ == "__main__":
    _selftest(sys.argv[1:])
