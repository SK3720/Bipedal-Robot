"""iLQR trajectory optimisation + TVLQR tracking for a single forward-push
recovery STEP on the baseline biped (robot/robot.xml, real geometry).

Rationale (see project memory / push_recovery_step.py):
  every working controller so far is a FIXED-POINT LQR (standing, single-support
  from rest).  Every failure is the TRANSITION between them while the state has
  momentum - switching K_stand -> K_ss off-nominal makes K_ss slam toward a
  nominal that is wrong for the live state.  The fix is to linearise ALONG a
  dynamically feasible reference maneuver: reference trajectory (x*,u*) from
  iLQR through the true MuJoCo dynamics, tracked online with the time-varying
  feedback gains K* that the iLQR backward pass already produces.

Contact schedule is NOT optimised: the cost fixes the swing / plant windows
(double support -> single support -> double support); the physics contacts are
whatever MuJoCo computes.

Nothing here modifies robot.xml / biped_env / any golden experiment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS, STANDING_QUAT
from recovery_metrics import (
    CHEST_BODY, NOMINAL_CHEST_Z, _foot_normal_force, _foot_xy_z, sample_balance,
)
from standing_balance_lqr import _dare, settle_standing

FORWARD_DIR_RAD = -np.pi / 2.0          # world -Y = anatomical forward
PUSH_AT_STEP = 20
G = 9.81

LEG_CTRL = {   # ctrl indices  (matches push_step_recovery_test.LEG_IDX)
    "L": dict(hip_roll=5, hip=6, knee=7, ankle=8, ankle_roll=9),
    "R": dict(hip_roll=10, hip=11, knee=12, ankle=13, ankle_roll=14),
}
AR_L_IDX, AR_R_IDX = 9, 14
AR_SIGN = {"R": -1.0, "L": +1.0}        # both-ankle-roll bias that unloads that foot
FWD_HIP_SIGN = {"L": -1.0, "R": +1.0}
KNEE_FLEX_SIGN = -1.0


def _smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * t))


# ----------------------------------------------------------------------------
# low-level MuJoCo helpers
# ----------------------------------------------------------------------------
class Sim:
    """Thin wrapper: deterministic set-state / step / linearise on one MjData."""

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self._d_lin = mujoco.MjData(model)      # scratch for linearisation only
        self.nv = model.nv
        self.nq = model.nq
        self.nu = 15
        self.ndx = 2 * self.nv
        self.ulo = model.actuator_ctrlrange[:15, 0].copy()
        self.uhi = model.actuator_ctrlrange[:15, 1].copy()

    # state = concat(qpos[nq], qvel[nv])
    def get_x(self):
        return np.concatenate([self.d.qpos.copy(), self.d.qvel.copy()])

    def set_x(self, x):
        self.d.qpos[:] = x[: self.nq]
        self.d.qvel[:] = x[self.nq :]
        self.d.qacc[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def dx(self, x_from, x_to):
        """tangent (x_to - x_from), dim ndx."""
        dq = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.m, dq, 1.0, x_from[: self.nq], x_to[: self.nq])
        dv = x_to[self.nq :] - x_from[self.nq :]
        return np.concatenate([dq, dv])

    def integrate_x(self, x, delta):
        """x (+) delta  in the manifold sense."""
        q = x[: self.nq].copy()
        mujoco.mj_integratePos(self.m, q, delta[: self.nv], 1.0)
        v = x[self.nq :] + delta[self.nv :]
        return np.concatenate([q, v])

    def step_hold(self, x0, u, H, xfrc=None):
        """apply constant ctrl u for H mj_steps from state x0; return x_H."""
        self.set_x(x0)
        uc = np.clip(u, self.ulo, self.uhi)
        for _ in range(H):
            if xfrc is not None:
                self.d.xfrc_applied[CHEST_BODY, :] = 0.0
                self.d.xfrc_applied[CHEST_BODY, 0:2] = xfrc
            self.d.ctrl[:15] = uc
            mujoco.mj_step(self.m, self.d)
        if xfrc is not None:
            self.d.xfrc_applied[CHEST_BODY, :] = 0.0
        return self.get_x()

    def linearize_hold(self, x0, u, H, eps=1e-6):
        """A,B of the H-step transition x_{k+1}=f(x_k,u_k) in tangent coords.

        Runs on a SEPARATE scratch MjData - `mjd_transitionFD` + `mj_step`
        mutate the data they run on, so linearising must never touch the live
        `self.d` (doing so teleported the running simulation - a real bug that
        produced a fake 'stable' state)."""
        dl = self._d_lin
        A = np.eye(self.ndx)
        B = np.zeros((self.ndx, self.nu))
        Ak = np.zeros((self.ndx, self.ndx))
        Bk = np.zeros((self.ndx, self.nu))
        dl.qpos[:] = x0[: self.nq]
        dl.qvel[:] = x0[self.nq:]
        dl.qacc[:] = 0.0
        dl.act[:] = 0.0
        dl.time = 0.0
        dl.xfrc_applied[:] = 0.0
        uc = np.clip(u, self.ulo, self.uhi)
        dl.ctrl[:15] = uc
        mujoco.mj_forward(self.m, dl)
        for _ in range(H):
            mujoco.mjd_transitionFD(self.m, dl, eps, 1, Ak, Bk, None, None)
            A = Ak @ A
            B = Ak @ B + Bk
            dl.ctrl[:15] = uc
            mujoco.mj_step(self.m, dl)
        return A, B


# ----------------------------------------------------------------------------
# residual features used by the cost  (all differentiable-ish via FD)
# ----------------------------------------------------------------------------
def features(sim: Sim, x, swing):
    """kinematic/dynamic features at state x."""
    sim.set_x(x)
    d, m = sim.d, sim.m
    mujoco.mj_subtreeVel(m, d)
    R = d.xmat[CHEST_BODY].reshape(3, 3)
    up = R @ np.array([0.0, 1.0, 0.0])
    tilt = np.array([up[0], -up[1]])           # [side, fwd]  (0 when upright)
    com = d.subtree_com[CHEST_BODY].copy()
    comv = d.subtree_linvel[CHEST_BODY].copy()
    sf = _foot_xy_z(m, d, swing)
    stf = _foot_xy_z(m, d, "L" if swing == "R" else "R")
    return dict(
        tilt=tilt,                              # 2
        chest_z=np.array([d.qpos[2]]),          # 1
        com_xy=com[:2].copy(),                  # 2  (world x,y)
        com_v=comv[:2].copy(),                  # 2
        swing_foot=sf.copy(),                   # 3  (x,y,z world)
        stance_foot=stf.copy(),                 # 3
    )


# ----------------------------------------------------------------------------
# the optimal-control problem
# ----------------------------------------------------------------------------
@dataclass
class Weights:
    w_tilt: float = 40.0
    w_tilt_T: float = 400.0
    w_chz: float = 30.0
    w_chz_T: float = 120.0
    w_com: float = 8.0
    w_com_T: float = 60.0
    w_comv: float = 2.0
    w_comv_T: float = 40.0
    w_comv_back: float = 12.0        # EXTRA penalty on backward CoM velocity (no rock-back)
    w_swing: float = 40.0           # swing-foot tracking during swing window
    w_plant: float = 70.0           # swing-foot on target after plant
    w_foot_down: float = 60.0       # swing-foot z -> floor after plant window
    w_stance: float = 8.0           # stance foot stays where it started
    w_u: float = 0.5               # deviation from nominal ctrl
    w_du: float = 2.0              # ctrl rate
    w_posture: float = 0.15        # joints toward default (regulariser)
    w_posture_T: float = 3.0       # joints toward default at the end


@dataclass
class StepPlan:
    swing: str = "R"
    H: int = 10                     # mj_steps per control knot
    N: int = 130                    # control knots  (=> 1.3 s maneuver)
    push_n: float = 160.0
    unload_frac: float = 0.26       # 0..this : weight-shift window (~matches SS-LQR 340-step ramp)
    swing_frac: float = 0.40        # unload_end..this : swing window (~140-step demo swing)
    # plant window = swing_frac..1.0
    step_len_margin_mm: float = 60.0   # target footfall = capture point + margin
    foot_clear_mm: float = 65.0        # peak swing-foot lift target
    capture_gain: float = 1.0          # scale the x0 capture-point estimate
    foot_target_mm: float = 0.0        # >0 overrides the computed footfall (mm ahead of trailing foot)
    ankle_roll_amp: float = 0.12       # weight-shift lean magnitude (single_support_step_demo value)
    step_out_mm: float = 0.0
    stance_brake: float = 0.0
    # single_support_step_demo swing params (these provably hold SS from rest);
    # here executed on a faster clock.
    swing_hip_rad: float = 0.45
    swing_knee_peak_rad: float = 0.25
    swing_ankle_dorsi_rad: float = 0.20
    # --- terminal descent (make the foot actually PLANT, not skate) ---
    descend_frac: float = 0.45      # of the plant window: active foot-descent, then hold
    plant_hip_retract: float = 0.12   # rad: pull the swing hip BACK from its peak (decelerate foot)
    plant_ankle_pf: float = 0.38      # rad: swing-ankle plantarflexion to press the sole down
    stance_knee_bend: float = 0.30    # rad: bend the STANCE knee to LOWER the pelvis toward the foot
    plant_lean_keep: float = 0.30     # keep this fraction of the ankle-roll lean during descent


class RecoveryILQR:
    def __init__(self, model, plan: StepPlan, weights: Weights | None = None,
                 verbose=True):
        self.sim = Sim(model)
        self.m = model
        self.plan = plan
        self.w = weights or Weights()
        self.swing = plan.swing
        self.verbose = verbose
        self.N, self.H = plan.N, plan.H
        self.ndx, self.nu = self.sim.ndx, self.sim.nu

        # nominal standing / post-push initial state -------------------------
        q0, v0 = settle_standing(model, self.sim.d)
        self.q_stand, self.v_stand = q0, v0
        self.x_stand = np.concatenate([q0, v0])
        self.u_nom = DEFAULT_POSE.copy()

        # post-push x0 : settle, then apply the impulse
        self.x0 = self._post_push_state(plan.push_n)
        f0 = features(self.sim, self.x0, self.swing)
        self.com0 = f0["com_xy"].copy()
        self.comv0 = f0["com_v"].copy()

        # capture-point based footfall target (state-dependent placement) ----
        self.stance_foot_xy = f0["stance_foot"][:2].copy()
        h = max(NOMINAL_CHEST_Z - 1.0, 0.05)
        cp_fwd = -self.com0[1] + plan.capture_gain * (-self.comv0[1]) * np.sqrt(h / G)
        if plan.foot_target_mm > 0:
            self.footfall_fwd = -self.stance_foot_xy[1] + plan.foot_target_mm / 1000.0
        else:
            self.footfall_fwd = cp_fwd + plan.step_len_margin_mm / 1000.0
        # swing-foot target world (x keeps its nominal, y = -footfall_fwd)
        self.swing_target = np.array([f0["swing_foot"][0], -self.footfall_fwd, 1.0])

        # reference com path : nominal -> over the eventual support midpoint,
        # velocity bled to zero by the end
        mid_xy = 0.5 * (self.stance_foot_xy + self.swing_target[:2])
        self.com_ref = np.zeros((self.N + 1, 2))
        self.comv_ref = np.zeros((self.N + 1, 2))
        for k in range(self.N + 1):
            s = _smooth(k / self.N)
            self.com_ref[k] = (1 - s) * self.com0 + s * mid_xy
        # swing-foot reference arc over the swing window
        self.k_unload = int(plan.unload_frac * self.N)
        self.k_swing_end = int(plan.swing_frac * self.N)
        self.foot_ref = np.zeros((self.N + 1, 3))
        f_sw0 = f0["swing_foot"].copy()
        for k in range(self.N + 1):
            if k <= self.k_unload:
                self.foot_ref[k] = f_sw0
            elif k <= self.k_swing_end:
                w = (k - self.k_unload) / max(1, self.k_swing_end - self.k_unload)
                arc = np.sin(np.pi * w)
                self.foot_ref[k] = (1 - _smooth(w)) * f_sw0 + _smooth(w) * self.swing_target
                self.foot_ref[k, 2] = 1.0 + (plan.foot_clear_mm / 1000.0) * arc
            else:
                self.foot_ref[k] = self.swing_target

        # terminal stabiliser: feet-together standing LQR.  The maneuver ends
        # with a short step (footfall ~90-120 mm), close enough that freezing
        # the ACHIEVED joint pose as the reference and letting this K damp +
        # right the torso is what actually settles a step (proven in
        # push_step_recovery_test's settle phase).
        self._build_terminal_lqr()

    # kept name for the driver; now just the feet-together standing LQR
    def _build_terminal_lqr(self):
        s = self.sim
        A, B = s.linearize_hold(self.x_stand, self.u_nom, 1)
        qp = np.ones(s.nv) * 2.0
        qp[0:2] = 3.0; qp[2] = 40.0; qp[3:6] = 400.0; qp[6:21] = 1.0
        qv = np.ones(s.nv); qv[0:3] = 6.0; qv[1] = 25.0; qv[3:6] = 25.0; qv[6:21] = 0.4
        Q = np.diag(np.concatenate([qp, qv])); R = np.diag(np.ones(15) * 3.0)
        K, _, _ = _dare(A, B, Q, R)
        self.K_term = K
        self.u_term = self.u_nom.copy()
        self.x_term = self.x_stand.copy()          # posture reference for iLQR terminal
        if self.verbose:
            print(f"[terminal standing-LQR] rho="
                  f"{np.max(np.abs(np.linalg.eigvals(A - B @ K))):.4f}  "
                  f"footfall_fwd={self.footfall_fwd*1000:.0f}mm  "
                  f"post-push CoM v_fwd={-self.comv0[1]:.2f} m/s")

    # ------------------------------------------------------------------ setup
    def _post_push_state(self, push_n):
        s = self.sim
        s.set_x(self.x_stand)
        fxy = push_n * np.array([np.cos(FORWARD_DIR_RAD), np.sin(FORWARD_DIR_RAD)])
        for k in range(PUSH_AT_STEP + PUSH_DURATION_STEPS + 3):
            s.d.xfrc_applied[CHEST_BODY, :] = 0.0
            if PUSH_AT_STEP <= k < PUSH_AT_STEP + PUSH_DURATION_STEPS:
                s.d.xfrc_applied[CHEST_BODY, 0:2] = fxy
            s.d.ctrl[:15] = self.u_nom
            mujoco.mj_step(s.m, s.d)
        s.d.xfrc_applied[CHEST_BODY, :] = 0.0
        return s.get_x()

    # ------------------------------------------------------------- seed guess
    def _build_ss_lqr(self):
        """single-support LQR about the LEANED one-foot state (reached by an
        ankle-roll bias under the standing LQR).  Reused from
        push_step_recovery_test - this is the controller that actually holds
        single support through a leg swing."""
        from push_step_recovery_test import (
            _LQRAbout, _SingleSupportLQR, StepConfig,
        )
        cfg = StepConfig(swing=self.swing, ankle_roll_amp_rad=self.plan.ankle_roll_amp,
                         ankle_roll_ramp=60, ss_reach_steps=340)
        stand = _LQRAbout(self.m, self.sim.d, DEFAULT_POSE.copy(), tag="stand",
                          verbose=self.verbose)
        ss = _SingleSupportLQR(self.m, self.sim.d, stand, cfg, verbose=self.verbose)
        self._stand_lqr = stand
        self._ss_lqr = ss
        self._ss_cfg = cfg

    def seed_controls(self):
        """iLQR initial guess: a CLOSED-LOOP rollout using the controllers that
        already work in isolation -
          balance/unload : standing LQR + ramped ankle-roll weight shift
          swing/plant    : single-support (leaned) LQR + feed-forward swing leg
        Holds the torso upright through the swing; the weight transfer / plant
        back to double support is what it does NOT close (that is iLQR's job)."""
        if not hasattr(self, "_ss_lqr"):
            self._build_ss_lqr()
        s = self.sim
        s.set_x(self.x0)
        U = np.zeros((self.N, 15))
        for k in range(self.N):
            q = s.d.qpos.copy(); v = s.d.qvel.copy()
            u = self.seed_ctrl(k, q, v)
            U[k] = u
            s.step_hold(np.concatenate([q, v]), u, self.H)
        return U

    # cap on the LQR feedback CORRECTION (rad) - so a diverged state can never
    # make the SS-/standing-K command a joint absurdly far from its reference
    # (that saturation drove the stance leg to its limits: the 'curl'/'pretzel').
    FB_CLAMP = 1.2

    def seed_ctrl(self, kf, q, v):
        """Unload/swing/plant control law at fractional knot position kf.
        The swing phase is the single_support_step_demo law verbatim (it
        provably holds single support through the leg swing from rest); here it
        runs on a faster clock and starts from a post-push state.  Live-callable
        (closed-loop) and used to build the iLQR seed tape."""
        if not hasattr(self, "_ss_lqr"):
            self._build_ss_lqr()
        s = self.sim
        stand, ss = self._stand_lqr, self._ss_lqr
        sw = LEG_CTRL[self.swing]
        hf = FWD_HIP_SIGN[self.swing]
        ss_idx = (sw["hip"], sw["knee"], sw["ankle"])
        ar_amp = AR_SIGN[self.swing] * self.plan.ankle_roll_amp
        nv = s.nv
        p = self.plan

        if self.k_unload < kf <= self.k_swing_end:                 # SWING (demo law)
            w = min(1.0, (kf - self.k_unload) / max(1, self.k_swing_end - self.k_unload))
            sm = _smooth(w); bump = np.sin(np.pi * w)
            hip = hf * p.swing_hip_rad * sm
            knee = -(0.06 + (p.swing_knee_peak_rad - 0.06) * bump)
            ankle = -hf * p.swing_ankle_dorsi_rad * bump
            qref = ss.qpos0.copy(); cref = ss.ctrl0.copy()
            for ci, val in zip(ss_idx, (hip, knee, ankle)):
                qref[7 + ci] = val; cref[ci] = val
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(s.m, dq, 1.0, qref, q)
            dx = np.concatenate([dq, v - ss.qvel0])
            for ci in ss_idx:
                dx[6 + ci] = 0.0; dx[nv + 6 + ci] = 0.0
            u = cref - np.clip(ss.K @ dx, -self.FB_CLAMP, self.FB_CLAMP)
            for ci in ss_idx:
                u[ci] = cref[ci]
        elif kf > self.k_swing_end:                                # PLANT: descend + load
            # w: 0 at swing end -> 1 at descend end -> stays 1 (hold) after.
            dur = max(1.0, p.descend_frac * (self.N - self.k_swing_end))
            w = min(1.0, (kf - self.k_swing_end) / dur)
            sm = _smooth(w)
            st_idx = LEG_CTRL["L" if self.swing == "R" else "R"]
            # swing leg: retract the hip a touch (decelerate the foot), extend the
            # knee toward straight, plantarflex hard so the SOLE reaches the floor.
            hip = hf * (p.swing_hip_rad - p.plant_hip_retract * sm)
            knee = -(0.06 + (p.swing_knee_peak_rad - 0.06) * (1 - sm)) * 0.4
            ankle = hf * (0.02 + p.plant_ankle_pf * sm)
            # stance leg: bend the knee to LOWER the pelvis toward the lead foot
            st_knee = ss.qpos0[7 + st_idx["knee"]] + KNEE_FLEX_SIGN * p.stance_knee_bend * sm
            qref = ss.qpos0.copy(); cref = ss.ctrl0.copy()
            for ci, val in zip(ss_idx, (hip, knee, ankle)):
                qref[7 + ci] = val; cref[ci] = val
            qref[7 + st_idx["knee"]] = st_knee; cref[st_idx["knee"]] = st_knee
            # ease the lean toward (but not to) centre so the pelvis drops but
            # the stance foot stays loaded
            lean = ar_amp * (1 - (1 - p.plant_lean_keep) * sm)
            for ci in (AR_L_IDX, AR_R_IDX):
                cref[ci] = lean; qref[7 + ci] = lean
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(s.m, dq, 1.0, qref, q)
            dx = np.concatenate([dq, v - ss.qvel0])
            for ci in ss_idx:                       # swing leg: pure feed-forward
                dx[6 + ci] = 0.0; dx[nv + 6 + ci] = 0.0
            u = cref - np.clip(ss.K @ dx, -self.FB_CLAMP, self.FB_CLAMP)
            for ci in ss_idx:
                u[ci] = cref[ci]
            u[st_idx["knee"]] = cref[st_idx["knee"]]
            u[AR_L_IDX] = cref[AR_L_IDX]; u[AR_R_IDX] = cref[AR_R_IDX]
        else:                                                       # BALANCE + UNLOAD
            ar = ar_amp * _smooth(min(1.0, kf / max(1, self.k_unload)))
            qref = stand.qpos0.copy(); cref = stand.ctrl0.copy()
            for ci in (AR_L_IDX, AR_R_IDX):
                qref[7 + ci] += ar; cref[ci] += ar
            dq = np.zeros(nv)
            mujoco.mj_differentiatePos(s.m, dq, 1.0, qref, q)
            dx = np.concatenate([dq, v - stand.qvel0])
            u = cref - np.clip(stand.K @ dx, -self.FB_CLAMP, self.FB_CLAMP)
            u[AR_L_IDX] = cref[AR_L_IDX]; u[AR_R_IDX] = cref[AR_R_IDX]
        return np.clip(u, s.ulo, s.uhi)

    # --------------------------------------------------------------- rollout
    def rollout(self, U, x0=None):
        s = self.sim
        X = np.zeros((self.N + 1, len(self.x0)))
        X[0] = self.x0 if x0 is None else x0
        for k in range(self.N):
            X[k + 1] = s.step_hold(X[k], U[k], self.H)
        return X

    # ------------------------------------------------------------------ cost
    def _running_res(self, k, x, u, u_prev):
        """weighted residual vector r and its weight (as sqrt) for GN."""
        f = features(self.sim, x, self.swing)
        terminal = k == self.N
        w = self.w
        wt = w.w_tilt_T if terminal else w.w_tilt
        wz = w.w_chz_T if terminal else w.w_chz
        wc = w.w_com_T if terminal else w.w_com
        wv = w.w_comv_T if terminal else w.w_comv
        kf = min(k, self.N)
        wp = w.w_posture_T if terminal else w.w_posture
        cv = f["com_v"]
        # backward CoM velocity = +y component (forward is -y); smooth relu so
        # there is no derivative kink at cv_y = 0
        cv_back = np.array([0.5 * (cv[1] + np.sqrt(cv[1] * cv[1] + 1e-4))])
        parts = [
            (np.sqrt(wt), f["tilt"]),
            (np.sqrt(wz), f["chest_z"] - NOMINAL_CHEST_Z),
            (np.sqrt(wc), f["com_xy"] - self.com_ref[kf]),
            (np.sqrt(wv * (0.2 + 0.8 * k / self.N)), cv),
            (np.sqrt(w.w_comv_back), cv_back),
            (np.sqrt(wp), (x[7:22] - self.q_stand[7:22])),
            (np.sqrt(w.w_stance), f["stance_foot"][:2] - self.stance_foot_xy),
        ]
        # swing-foot tracking
        if k <= self.k_unload:
            pass
        elif k <= self.k_swing_end:
            parts.append((np.sqrt(w.w_swing), f["swing_foot"] - self.foot_ref[kf]))
        else:
            parts.append((np.sqrt(w.w_plant),
                          (f["swing_foot"][:2] - self.foot_ref[kf][:2])))
            parts.append((np.sqrt(w.w_foot_down),
                          np.array([f["swing_foot"][2] - 1.0])))
        r = np.concatenate([wgt * p for wgt, p in parts])
        # control residuals (linear, handled separately for exact derivs)
        return r

    def _ctrl_res(self, k, u, u_prev):
        w = self.w
        r_u = np.sqrt(w.w_u) * (u - self.u_nom)
        r_du = np.sqrt(w.w_du) * (u - u_prev)
        return r_u, r_du

    def cost(self, X, U):
        J = 0.0
        for k in range(self.N):
            r = self._running_res(k, X[k], U[k], U[k - 1] if k > 0 else self.u_nom)
            r_u, r_du = self._ctrl_res(k, U[k], U[k - 1] if k > 0 else self.u_nom)
            J += 0.5 * (r @ r + r_u @ r_u + r_du @ r_du)
        rT = self._running_res(self.N, X[self.N], self.u_nom, self.u_nom)
        J += 0.5 * (rT @ rT)
        return J if np.isfinite(J) else np.inf

    def _res_jac_x(self, k, x, eps=1e-5):
        """FD Jacobian of the running residual wrt tangent state (ndx)."""
        r0 = self._running_res(k, x, self.u_nom, self.u_nom)
        Jx = np.zeros((len(r0), self.ndx))
        for i in range(self.ndx):
            e = np.zeros(self.ndx); e[i] = eps
            xp = self.sim.integrate_x(x, e)
            xm = self.sim.integrate_x(x, -e)
            rp = self._running_res(k, xp, self.u_nom, self.u_nom)
            rm = self._running_res(k, xm, self.u_nom, self.u_nom)
            Jx[:, i] = (rp - rm) / (2 * eps)
        return r0, Jx

    # ----------------------------------------------------------- backward pass
    def backward(self, X, U, AB, reg):
        ndx, nu, N = self.ndx, self.nu, self.N
        # terminal
        rT, JxT = self._res_jac_x(N, X[N])
        Vx = JxT.T @ rT
        Vxx = JxT.T @ JxT
        k_ff = np.zeros((N, nu))
        K_fb = np.zeros((N, nu, ndx))
        dV = 0.0
        w = self.w
        for k in range(N - 1, -1, -1):
            A, B = AB[k]
            u_prev = U[k - 1] if k > 0 else self.u_nom
            r, Jx = self._res_jac_x(k, X[k])
            lx = Jx.T @ r
            lxx = Jx.T @ Jx
            # control cost (exact)
            lu = w.w_u * (U[k] - self.u_nom) + w.w_du * (U[k] - u_prev)
            luu = np.eye(nu) * (w.w_u + w.w_du)
            lux = np.zeros((nu, ndx))

            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            Quu_reg = Quu + reg * np.eye(nu)
            try:
                np.linalg.cholesky(Quu_reg)
            except np.linalg.LinAlgError:
                return None
            kff = -np.linalg.solve(Quu_reg, Qu)
            Kfb = -np.linalg.solve(Quu_reg, Qux)
            # trust region: physics near marginal single support has cost cliffs;
            # cap the per-knot feed-forward and feedback so one bad DOF cannot
            # blow up the forward rollout.
            kff = np.clip(kff, -0.20, 0.20)
            kn = np.linalg.norm(Kfb)
            if kn > 250.0:
                Kfb = Kfb * (250.0 / kn)
            k_ff[k] = kff
            K_fb[k] = Kfb
            dV += kff @ Qu + 0.5 * kff @ Quu @ kff
            Vx = Qx + Kfb.T @ Quu @ kff + Kfb.T @ Qu + Qux.T @ kff
            Vxx = Qxx + Kfb.T @ Quu @ Kfb + Kfb.T @ Qux + Qux.T @ Kfb
            Vxx = 0.5 * (Vxx + Vxx.T)
        return k_ff, K_fb, dV

    def forward(self, X, U, k_ff, K_fb, alpha):
        s = self.sim
        Xn = np.zeros_like(X); Un = np.zeros_like(U)
        Xn[0] = X[0]
        for k in range(self.N):
            dx = s.dx(X[k], Xn[k])
            du = alpha * k_ff[k] + K_fb[k] @ dx
            Un[k] = np.clip(U[k] + du, s.ulo, s.uhi)
            Xn[k + 1] = s.step_hold(Xn[k], Un[k], self.H)
        return Xn, Un

    def linearize_all(self, X, U):
        return [self.sim.linearize_hold(X[k], U[k], self.H) for k in range(self.N)]

    def optimize(self, U0=None, iters=60, reg0=1e-2):
        U = self.seed_controls() if U0 is None else U0.copy()
        X = self.rollout(U)
        J = self.cost(X, U)
        reg = max(reg0, 5e-2)
        hist = [J]
        stall = 0
        alphas = (1.0, 0.6, 0.36, 0.2, 0.1, 0.05, 0.02, 0.008, 0.003)
        if self.verbose:
            print(f"[iLQR] seed cost {J:.3f}")
        for it in range(iters):
            t0 = time.time()
            AB = self.linearize_all(X, U)
            bp = self.backward(X, U, AB, reg)
            if bp is None:
                reg = min(reg * 3, 1e7)
                if self.verbose:
                    print(f"  it{it:02d}  backward non-PD, reg->{reg:.1e}")
                stall += 1
                if stall > 8:
                    break
                continue
            k_ff, K_fb, dV = bp
            improved = False
            for alpha in alphas:
                Xn, Un = self.forward(X, U, k_ff, K_fb, alpha)
                Jn = self.cost(Xn, Un)
                # Armijo for big steps; any strict decrease for small steps
                # (crawls past the contact cliffs)
                thresh = J - 1e-4 * abs(dV) * alpha if alpha > 0.05 else J - 1e-6
                if np.isfinite(Jn) and Jn < thresh:
                    X, U, J = Xn, Un, Jn
                    reg = max(reg / 1.6, 1e-5)
                    improved = True
                    break
            hist.append(J)
            dt = time.time() - t0
            if self.verbose:
                print(f"  it{it:02d}  J={J:.3f}  reg={reg:.1e}  a={alpha if improved else 0}"
                      f"  {dt:.1f}s")
            if improved:
                stall = 0
            else:
                reg = min(reg * 3, 1e7)
                stall += 1
                if stall > 8:
                    break
            if improved and len(hist) > 5 and abs(hist[-2] - hist[-1]) < 5e-4 * abs(hist[-1]):
                break
        # final gains
        AB = self.linearize_all(X, U)
        bp = self.backward(X, U, AB, max(reg, 1e-3))
        K_fb = bp[1] if bp is not None else np.zeros((self.N, self.nu, self.ndx))
        self.X_ref, self.U_ref, self.K_ref = X, U, K_fb
        return X, U, K_fb


# ----------------------------------------------------------------------------
# closed-loop execution with TVLQR tracking + terminal DS-LQR
# ----------------------------------------------------------------------------
@dataclass
class ExecResult:
    fell: bool = False
    fell_step: int | None = None
    fell_reason: str = ""
    recovered: bool = False
    unload_ms: int | None = None
    min_swing_nf: float = 1e9
    swing_fwd_mm: float = 0.0
    swing_clear_mm: float = 0.0
    ss_peak_tilt_deg: float = 0.0
    plant_step: int | None = None
    plant_nf_at_detect: float = 0.0
    foot_sep_fwd_at_plant_mm: float = 0.0
    post_plant_peak_tilt_deg: float = 0.0
    residual_com_vfwd: float = 0.0
    end_up_tilt_deg: float = 0.0
    end_side_lean_deg: float = 0.0
    end_chest_z: float = 0.0
    end_double_support: bool = False
    end_com_speed: float = 0.0
    diagnosis: str = ""
    samples: list = field(default_factory=list)


def execute(prob: RecoveryILQR, X, U, K, settle_steps=2500, viewer=None,
            slow=False, verbose=True):
    s = prob.sim
    m, d = s.m, s.d
    swing = prob.swing
    H, N = prob.H, prob.N
    s.set_x(prob.x0)
    res = ExecResult()
    samples = []
    swing_y0 = _foot_xy_z(m, d, swing)[1]
    total = N * H + settle_steps
    step_i = 0
    unl_streak = 0
    plant_streak = 0

    def rec(tag=""):
        b = sample_balance(m, d)
        samples.append((step_i, b, tag))
        return b

    for k in range(N):
        xref = X[k]; uref = U[k]; Kk = K[k]
        # TVLQR tracking: feedback each ms against the reference trajectory
        # linearly interpolated (in tangent space) across the knot from X[k] to
        # X[k+1], so the gains correct deviation FROM THE PATH, not from a frozen
        # knot-start snapshot.
        dx_nom = s.dx(xref, X[k + 1])
        for j in range(H):
            err = s.dx(xref, s.get_x()) - (j / H) * dx_nom
            u = np.clip(uref - Kk @ err, s.ulo, s.uhi)
            d.ctrl[:15] = u
            mujoco.mj_step(m, d)
            step_i += 1
            b = rec("maneuver")
            nf = _foot_normal_force(m, d, swing)
            res.min_swing_nf = min(res.min_swing_nf, nf)
            if res.unload_ms is None:
                unl_streak = unl_streak + 1 if nf < 3.0 else 0
                if unl_streak >= 15:
                    res.unload_ms = step_i
            f = _foot_xy_z(m, d, swing)
            res.swing_fwd_mm = max(res.swing_fwd_mm, -(f[1] - swing_y0) * 1000.0)
            res.swing_clear_mm = max(res.swing_clear_mm, (f[2] - 1.0) * 1000.0)
            if k > prob.k_unload:
                res.ss_peak_tilt_deg = max(res.ss_peak_tilt_deg, b.up_tilt_deg)
            if k > prob.k_swing_end and res.plant_step is None:
                sc = getattr(b, f"{swing.lower()}_contact")
                plant_streak = plant_streak + 1 if (sc and nf > 8.0) else 0
                if plant_streak >= 10:
                    res.plant_step = step_i
                    res.plant_nf_at_detect = nf
                    sfy = _foot_xy_z(m, d, swing)[1]
                    stfy = _foot_xy_z(m, d, "L" if swing == "R" else "R")[1]
                    res.foot_sep_fwd_at_plant_mm = -(sfy - stfy) * 1000.0
            if res.plant_step is not None:
                res.post_plant_peak_tilt_deg = max(res.post_plant_peak_tilt_deg, b.up_tilt_deg)
            if b.up_tilt_deg > 50 or b.chest_z < NOMINAL_CHEST_Z - 0.22:
                res.fell = True; res.fell_step = step_i
                res.fell_reason = "tilt>50" if b.up_tilt_deg > 50 else "chest dropped"
            if viewer is not None:
                if not viewer.is_running():
                    return _finish(prob, res, samples, verbose)
                viewer.sync(); time.sleep(0.02 if slow else 0.002)
            if res.fell:
                return _finish(prob, res, samples, verbose)

    # terminal handoff: freeze the ACHIEVED joint pose (torso forced upright) as
    # the reference and let the feet-together standing K damp + right the torso.
    q_freeze = s.get_x()[: s.nq].copy()
    q_freeze[3:7] = STANDING_QUAT
    u_freeze = np.clip(d.ctrl[:15].copy(), s.ulo, s.uhi)
    x_freeze = np.concatenate([q_freeze, np.zeros(s.nv)])
    for _ in range(settle_steps):
        dx = s.dx(x_freeze, s.get_x())
        u = np.clip(u_freeze - prob.K_term @ dx, s.ulo, s.uhi)
        d.ctrl[:15] = u
        mujoco.mj_step(m, d)
        step_i += 1
        b = rec("settle")
        res.post_plant_peak_tilt_deg = max(res.post_plant_peak_tilt_deg, b.up_tilt_deg)
        if b.up_tilt_deg > 50 or b.chest_z < NOMINAL_CHEST_Z - 0.22:
            res.fell = True; res.fell_step = step_i
            res.fell_reason = "tilt>50 (settle)" if b.up_tilt_deg > 50 else "chest dropped (settle)"
            return _finish(prob, res, samples, verbose)
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync(); time.sleep(0.02 if slow else 0.002)
    return _finish(prob, res, samples, verbose)


def _finish(prob, res: ExecResult, samples, verbose):
    m, d = prob.sim.m, prob.sim.d
    b = sample_balance(m, d)
    res.samples = samples
    res.end_up_tilt_deg = b.up_tilt_deg
    res.end_side_lean_deg = b.side_lean_deg
    res.end_chest_z = b.chest_z
    res.end_double_support = b.l_contact and b.r_contact
    res.end_com_speed = b.com_speed_horiz
    res.residual_com_vfwd = b.com_vfwd
    tail = [sm[1] for sm in samples[-400:]]
    settle_up = float(np.mean([t.up_tilt_deg for t in tail])) if tail else 99
    settle_v = float(np.mean([t.com_speed_horiz for t in tail])) if tail else 99
    res.recovered = (not res.fell and settle_up < 12 and abs(b.side_lean_deg) < 12
                     and res.end_double_support and settle_v < 0.12
                     and b.chest_z > NOMINAL_CHEST_Z - 0.10)
    res.diagnosis = _diagnose(prob, res)
    if verbose:
        _print_result(prob, res)
    return res


def _diagnose(prob, r: ExecResult) -> str:
    if r.recovered:
        return "RECOVERED — push -> step -> plant -> double support -> LQR"
    if r.unload_ms is None:
        return ("UNLOAD failed: swing foot never went below 3 N "
                f"(min {r.min_swing_nf:.1f} N). Weight shift not achieved.")
    if r.swing_fwd_mm < 40:
        return (f"SWING failed: foot only moved {r.swing_fwd_mm:.0f} mm forward "
                f"(unloaded at {r.unload_ms} ms, SS peak tilt {r.ss_peak_tilt_deg:.0f} deg). "
                "Swing-leg drive / SS balance during swing.")
    if r.ss_peak_tilt_deg > 25 and r.plant_step is None:
        return (f"SINGLE-SUPPORT balance failed: tilt reached {r.ss_peak_tilt_deg:.0f} deg "
                f"during swing (foot got {r.swing_fwd_mm:.0f} mm fwd) before any plant.")
    if r.plant_step is None:
        return (f"PLANT failed: swing foot reached {r.swing_fwd_mm:.0f} mm fwd, "
                f"{r.swing_clear_mm:.0f} mm clearance, but never loaded >8 N. "
                "Foot placement too high / short, or timing.")
    if r.fell and r.fell_step is not None and r.plant_step is not None and \
            r.fell_step - r.plant_step < 400:
        return (f"POST-PLANT stabilisation failed: planted at {r.plant_step} ms "
                f"(sep {r.foot_sep_fwd_at_plant_mm:.0f} mm, residual CoM v_fwd "
                f"{r.residual_com_vfwd:.2f} m/s), fell {r.fell_step - r.plant_step} ms later "
                f"(tilt {r.post_plant_peak_tilt_deg:.0f} deg). "
                "Momentum not arrested by the new stance / terminal LQR mismatch.")
    if r.fell:
        return (f"FELL @ {r.fell_step} ms ({r.fell_reason}) after plant@{r.plant_step}. "
                f"post-plant peak tilt {r.post_plant_peak_tilt_deg:.0f} deg, "
                f"residual CoM v_fwd {r.residual_com_vfwd:.2f} m/s.")
    return (f"NO FALL but not settled: end tilt {r.end_up_tilt_deg:.0f} deg, "
            f"side {r.end_side_lean_deg:+.0f}, DS={r.end_double_support}, "
            f"CoM speed {r.end_com_speed:.2f}.")


def _print_result(prob, r: ExecResult):
    print("\n" + "=" * 72)
    print(f"push {prob.plan.push_n:.0f} N   swing {prob.swing}   "
          f"post-push CoM v_fwd {-prob.comv0[1]:.2f} m/s   "
          f"footfall target {prob.footfall_fwd*1000:.0f} mm")
    print(f"  unload:      {'%d ms' % r.unload_ms if r.unload_ms else 'FAILED'}"
          f"   min swing NF {r.min_swing_nf:.1f} N")
    print(f"  swing:       {r.swing_fwd_mm:.0f} mm fwd, {r.swing_clear_mm:.0f} mm clearance"
          f"   SS peak tilt {r.ss_peak_tilt_deg:.0f} deg")
    print(f"  plant:       {'%d ms' % r.plant_step if r.plant_step else 'FAILED'}"
          f"   NF {r.plant_nf_at_detect:.0f} N   foot sep fwd {r.foot_sep_fwd_at_plant_mm:.0f} mm")
    print(f"  post-plant:  peak tilt {r.post_plant_peak_tilt_deg:.0f} deg"
          f"   residual CoM v_fwd {r.residual_com_vfwd:.2f} m/s")
    print(f"  end:         up_tilt {r.end_up_tilt_deg:.1f}  side {r.end_side_lean_deg:+.1f}"
          f"  chestZ {r.end_chest_z:.3f}  DS {r.end_double_support}  CoM speed {r.end_com_speed:.3f}")
    if r.fell:
        print(f"  FELL @ {r.fell_step} ms ({r.fell_reason})")
    print(f"\n  >>> {r.diagnosis}")
    print("=" * 72)
