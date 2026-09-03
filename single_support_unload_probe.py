"""Dynamic (transient) lateral weight throw: does a fast ankle-roll pulse briefly
unload the swing foot, and for how long a window?"""
import mujoco, numpy as np
from standing_balance_lqr import StandingLQR
from push_step_recovery_test import _LQRAbout, _pos_error
from recovery_metrics import sample_balance, _foot_normal_force, CHEST_BODY, _foot_xy_z
from biped_env import DEFAULT_POSE

m = mujoco.MjModel.from_xml_path("robot/robot.xml"); d = mujoco.MjData(m)
stand = _LQRAbout(m, d, DEFAULT_POSE.copy(), tag="stand", verbose=False)
# unload R  ->  ankle rolls negative (verified in _shift_probe2 era)
# ctrl idx: L ankle roll = 9, R ankle roll = 14

for roll_amp, ramp in [(-0.15, 60), (-0.25, 60), (-0.35, 50), (-0.5, 40), (-0.35, 25)]:
    d.qpos[:] = stand.qpos0; d.qvel[:] = stand.qvel0; mujoco.mj_forward(m, d)
    win_lo = win_hi = None
    rnf_min = 1e9
    peak_side = 0.0
    fell = False
    for k in range(500):
        qr = stand.qpos0.copy(); cr = stand.ctrl0.copy()
        a = min(1.0, k / ramp)
        cr[9] += roll_amp * a
        cr[14] += roll_amp * a
        qr[7 + 9] += roll_amp * a
        qr[7 + 14] += roll_amp * a
        dx = np.concatenate([_pos_error(m, qr, d.qpos), d.qvel - stand.qvel0])
        u = cr - stand.K @ dx
        # let ankle rolls be pure feed-forward (don't let K fight the pulse)
        u[9] = cr[9]; u[14] = cr[14]
        d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(m, d)
        rnf = _foot_normal_force(m, d, "R")
        rnf_min = min(rnf_min, rnf)
        bs = sample_balance(m, d)
        peak_side = max(peak_side, abs(bs.side_lean_deg))
        if rnf < 2.0 and win_lo is None:
            win_lo = k
        if win_lo is not None and win_hi is None and rnf > 4.0 and k > win_lo + 3:
            win_hi = k
        if bs.up_tilt_deg > 35:
            fell = True; break
    wl = win_lo if win_lo is not None else -1
    wh = win_hi if win_hi is not None else (k if win_lo is not None else -1)
    print(f"roll={roll_amp:+.2f} ramp={ramp:3d}: Rnf_min={rnf_min:5.1f}N  "
          f"unload window steps {wl}..{wh} ({(wh-wl) if wl>=0 else 0} ms)  "
          f"peak_side={peak_side:5.1f}deg  {'FELL' if fell else 'ok'}")
