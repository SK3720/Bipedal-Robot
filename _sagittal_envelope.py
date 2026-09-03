"""Sagittal-only envelope: how big a PURE forward push can the standing LQR hold
in place, and just above that, is a STEP genuinely required (capture point leaves
the support polygon) and how much TIME is there to execute it?

No stepping here - this only characterises the standing-LQR baseline so we can
pick a small push in the regime: 'standing fails, step needed, plenty of time'.

python _sagittal_envelope.py
"""
import numpy as np
import mujoco

from standing_balance_lqr import StandingLQR
from recovery_metrics import CHEST_BODY, NOMINAL_CHEST_Z, sample_balance
from biped_env import PUSH_DURATION_STEPS

FWD = -np.pi / 2.0
PUSH_AT = 20


def run_push(m, d, lqr, pn, horizon=4000):
    d.qpos[:] = lqr.qpos0
    d.qvel[:] = lqr.qvel0
    d.act[:] = 0.0
    d.ctrl[:15] = lqr.ctrl0
    mujoco.mj_forward(m, d)
    fxy = pn * np.array([np.cos(FWD), np.sin(FWD)])

    peak_lean = 0.0
    peak_capt = -1e9
    peak_com_rel = -1e9
    v_after_push = None
    t_capt_out = None          # first ms capture point past front support edge
    t_com_out = None           # first ms CoM past front support edge
    fell = False
    t_fell = None
    for k in range(horizon):
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT <= k < PUSH_AT + PUSH_DURATION_STEPS:
            d.xfrc_applied[CHEST_BODY, 0:2] = fxy
        d.ctrl[:15] = lqr.control(m, d)
        mujoco.mj_step(m, d)
        b = sample_balance(m, d)
        if k == PUSH_AT + PUSH_DURATION_STEPS + 1:
            v_after_push = b.com_vfwd
        peak_lean = max(peak_lean, b.fwd_lean_deg)
        peak_capt = max(peak_capt, b.capture_fwd_rel_support_mm)
        peak_com_rel = max(peak_com_rel, b.com_fwd_rel_support_mm)
        if t_capt_out is None and b.capture_fwd_rel_support_mm > 0:
            t_capt_out = k - PUSH_AT
        if t_com_out is None and b.com_fwd_rel_support_mm > 0:
            t_com_out = k - PUSH_AT
        if b.up_tilt_deg > 50 or b.chest_z < NOMINAL_CHEST_Z - 0.22:
            fell = True
            t_fell = k - PUSH_AT
            break
    b = sample_balance(m, d)
    settled = (not fell and abs(b.fwd_lean_deg) < 6 and b.com_speed_horiz < 0.08)
    return dict(pn=pn, v_after_push=v_after_push, peak_lean=peak_lean,
               peak_capt=peak_capt, peak_com_rel=peak_com_rel,
               t_capt_out=t_capt_out, t_com_out=t_com_out,
               fell=fell, t_fell=t_fell, settled=settled,
               end_lean=b.fwd_lean_deg, end_v=b.com_speed_horiz)


def main():
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)
    lqr = StandingLQR(m, d, verbose=True)
    print(f"\n{'push':>5} {'v+':>6} {'pk_lean':>8} {'pk_capt':>8} {'pk_com':>8} "
          f"{'t_capt':>7} {'t_com':>6} {'t_fell':>7} {'result':>10}")
    for pn in (40, 55, 70, 80, 90, 100, 110, 120, 130, 140, 150, 165, 180, 200):
        r = run_push(m, d, lqr, float(pn))
        res = "settled" if r["settled"] else ("FELL" if r["fell"] else "drift")
        print(f"{r['pn']:5.0f} {r['v_after_push'] or 0:6.2f} {r['peak_lean']:8.1f} "
              f"{r['peak_capt']:8.0f} {r['peak_com_rel']:8.0f} "
              f"{str(r['t_capt_out']):>7} {str(r['t_com_out']):>6} "
              f"{str(r['t_fell']):>7} {res:>10}")


if __name__ == "__main__":
    main()
