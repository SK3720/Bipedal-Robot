"""Characterise the LQR forward-fall: timing window, weight distribution, foot state."""
import mujoco, numpy as np
from standing_balance_lqr import StandingLQR
from recovery_metrics import sample_balance, chest_lean_yaw
from biped_env import PUSH_DURATION_STEPS
from recovery_metrics import CHEST_BODY, _foot_normal_force, _foot_contact

m = mujoco.MjModel.from_xml_path("robot/robot.xml"); d = mujoco.MjData(m)
lqr = StandingLQR(m, d)
FWD = -np.pi/2
for pn in (140.0, 170.0, 200.0):
    d.qpos[:] = lqr.qpos0; d.qvel[:] = lqr.qvel0; mujoco.mj_forward(m, d)
    fxy = pn*np.array([np.cos(FWD), np.sin(FWD)])
    print(f"\n=== LQR + {pn:.0f} N forward push ===")
    print(f"{'t':>5} {'capt-supp':>9} {'com-supp':>9} {'fwd_lean':>8} {'pitchR':>7} {'Lnf':>6} {'Rnf':>6} {'Lc':>3} {'Rc':>3} {'comZ':>6}")
    trig_t = None
    for k in range(1400):
        d.xfrc_applied[CHEST_BODY,:] = 0.0
        if 5 <= k < 5+PUSH_DURATION_STEPS:
            d.xfrc_applied[CHEST_BODY,0:2] = fxy
        d.ctrl[:15] = lqr.control(m, d)
        mujoco.mj_step(m, d)
        bs = sample_balance(m, d)
        if trig_t is None and bs.capture_fwd_rel_support_mm > 15:
            trig_t = k
        if k % 25 == 0 or (trig_t and k < trig_t+10):
            print(f"{k/1000:5.3f} {bs.capture_fwd_rel_support_mm:9.1f} {bs.com_fwd_rel_support_mm:9.1f} "
                  f"{bs.fwd_lean_deg:8.1f} {bs.pitch_rate:7.2f} {bs.l_nf:6.1f} {bs.r_nf:6.1f} "
                  f"{int(bs.l_contact):3d} {int(bs.r_contact):3d} {bs.com[2]:6.3f}")
        if bs.com[2] < 1.10:
            print(f"  -> chest collapsed at t={k/1000:.3f}s (trigger crossed +15mm at t={None if trig_t is None else trig_t/1000:.3f}s)")
            break
