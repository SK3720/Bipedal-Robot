"""Single-support balance + forward leg swing, from standing (no push).

This is the capability that was declared impossible last turn. It works:

  stand (LQR)
    -> ramp a small both-ankle-roll bias toward the L foot  (weight shift)
    -> the R foot unloads to ~0 N; the robot holds ~8 deg lean  (SINGLE SUPPORT)
    -> switch to an LQR linearised about that leaned single-support state
    -> swing the R hip forward ~80-150 mm  (RECOVERY STEP, no hop)
    -> stays balanced, ~8 deg lean throughout

What still does NOT close is doing this fast enough, and catching the extra
forward momentum, when it is triggered mid-fall from a real >=150 N forward push
(the swing hops and the residual CoM velocity re-topples it) - see
push_step_recovery_test.py.

Run headless for numbers, or:
    python single_support_step_demo.py --slow      # watch it in the viewer
"""

from __future__ import annotations

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from biped_env import DEFAULT_POSE, STANDING_QUAT
from push_step_recovery_test import (
    AR_L_IDX, AR_R_IDX, AR_SIGN, LEG_IDX, StepConfig,
    _LQRAbout, _SingleSupportLQR, _pos_error, _smooth,
)
from recovery_metrics import _foot_normal_force, _foot_xy_z, sample_balance

NORMAL_SLEEP_S = 0.003
SLOW_SLEEP_S = 0.02


def run(model, data, swing="R", viewer=None, slow=False):
    cfg = StepConfig(swing=swing, ankle_roll_amp_rad=0.12, step_hip_fwd_rad=0.45,
                     swing_knee_peak_rad=0.25, swing_ankle_dorsi_rad=0.20, swing_steps=140)
    stand = _LQRAbout(model, data, DEFAULT_POSE.copy(), tag="stand", verbose=True)
    ss = _SingleSupportLQR(model, data, stand, cfg, verbose=True)
    idx = LEG_IDX[swing]
    ss_idx = (idx["hip"], idx["knee"], idx["ankle"])
    hf = {"R": 1.0, "L": -1.0}[swing]

    data.qpos[:] = stand.qpos0
    data.qvel[:] = stand.qvel0
    mujoco.mj_forward(model, data)

    HOLD, SWING, PLANT_HOLD = 120, cfg.swing_steps, 600
    ar = AR_SIGN[swing] * cfg.ankle_roll_amp_rad
    y0 = None
    peak_clear = peak_fwd = 0.0
    print("\n  phase          t(s)  up_tilt side_lean  swing_nf  swing_fwd  clear")

    def report(tag, k):
        b = sample_balance(model, data)
        nf = _foot_normal_force(model, data, swing)
        f = _foot_xy_z(model, data, swing)
        fwd = 0.0 if y0 is None else -(f[1] - y0) * 1000
        print(f"  {tag:<14} {k/1000:4.2f}  {b.up_tilt_deg:6.1f}  {b.side_lean_deg:+7.1f}  "
              f"{nf:7.1f}  {fwd:8.0f}  {(f[2]-1.0)*1000:6.0f}")

    k = 0
    def step(u):
        nonlocal k, peak_clear, peak_fwd
        data.ctrl[:15] = np.clip(u, model.actuator_ctrlrange[:15, 0], model.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(model, data)
        k += 1
        if y0 is not None:
            f = _foot_xy_z(model, data, swing)
            peak_fwd = max(peak_fwd, -(f[1] - y0) * 1000)
            peak_clear = max(peak_clear, (f[2] - 1.0) * 1000)
        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)

    # 1) plain standing
    for _ in range(HOLD):
        step(stand.ctrl0 - stand.K @ np.concatenate(
            [_pos_error(model, stand.qpos0, data.qpos), data.qvel - stand.qvel0]))
    report("STAND", k)

    # 2) ankle-roll weight shift -> single support.  Identical to the sequence
    #    _SingleSupportLQR used to define ss.qpos0, so the state at swing start
    #    matches the linearisation point exactly.
    for j in range(cfg.ss_reach_steps):
        a = ar * min(1.0, j / cfg.ankle_roll_ramp)
        qr = stand.qpos0.copy(); cr = stand.ctrl0.copy()
        for ci in (AR_L_IDX, AR_R_IDX):
            qr[7 + ci] += a; cr[ci] += a
        u = cr - stand.K @ np.concatenate([_pos_error(model, qr, data.qpos), data.qvel - stand.qvel0])
        u[AR_L_IDX] = cr[AR_L_IDX]; u[AR_R_IDX] = cr[AR_R_IDX]
        step(u)
    report("SINGLE-SUPPORT", k)

    # 3) swing the free leg forward and hold it there, under the SS LQR.
    #    (Lowering it back onto the ground to finish in stepped double support is
    #    NOT solved - the weight-shift relax + K hand-off destabilises. This demo
    #    shows the part that works: hold single support AND move the free leg.)
    y0 = _foot_xy_z(model, data, swing)[1]
    for j in range(SWING + PLANT_HOLD):
        w = min(1.0, j / SWING); s = _smooth(w); bump = np.sin(np.pi * w)
        hip = hf * cfg.step_hip_fwd_rad * s
        knee = -(0.06 + (cfg.swing_knee_peak_rad - 0.06) * bump)
        ankle = -hf * cfg.swing_ankle_dorsi_rad * bump
        qref = ss.qpos0.copy(); cref = ss.ctrl0.copy()
        for ci, v in zip(ss_idx, (hip, knee, ankle)):
            qref[7 + ci] = v; cref[ci] = v
        dx = np.concatenate([_pos_error(model, qref, data.qpos), data.qvel - ss.qvel0])
        for ci in ss_idx:
            dx[6 + ci] = 0.0; dx[model.nv + 6 + ci] = 0.0
        u = cref - ss.K @ dx
        for ci in ss_idx:
            u[ci] = cref[ci]
        step(u)
        if j == SWING:
            report("SWING peak", k)
    report("HELD 0.6s", k)

    b = sample_balance(model, data)
    stance = "L" if swing == "R" else "R"
    ok = b.up_tilt_deg < 18 and _foot_normal_force(model, data, stance) > 3
    print(f"\n  peak swing-foot forward = {peak_fwd:.0f} mm, peak clearance = {peak_clear:.0f} mm")
    print(f"  final: up_tilt = {b.up_tilt_deg:.1f} deg, stance foot load = "
          f"{_foot_normal_force(model, data, stance):.0f} N")
    print(f"  -> {'HELD single support through the leg swing' if ok else 'lost balance'}")
    if viewer is not None:
        print("\n  close viewer to exit")
        while viewer.is_running():
            viewer.sync(); time.sleep(SLOW_SLEEP_S if slow else NORMAL_SLEEP_S)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--swing", choices=["L", "R"], default="R")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    a = p.parse_args(argv)
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)
    if a.headless:
        run(m, d, a.swing, None, False)
        return
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, -0.05, 1.05]
        v.cam.distance = 1.9
        v.cam.azimuth = 90
        v.cam.elevation = -8
        try:
            run(m, d, a.swing, v, a.slow)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main(sys.argv[1:])
