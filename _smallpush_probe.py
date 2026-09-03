"""Characterise the SMALL-push regime for step recovery (robot.xml, orig geometry).

The milestone is: small push -> 1-2 small forward steps -> stable standing.
NOT "clean 144 N forward fall".  So find where the standing LQR *eventually*
loses it, how much TIME there is, how far the CoM/capture point actually
escapes, and which way it goes.

python _smallpush_probe.py
"""
import numpy as np
import mujoco

from standing_balance_lqr import StandingLQR
from recovery_metrics import CHEST_BODY, NOMINAL_CHEST_Z, sample_balance
from biped_env import PUSH_DURATION_STEPS

FWD = -np.pi / 2.0
PUSH_AT = 20


def characterise(m, d, lqr, pn, horizon=4000):
    d.qpos[:] = lqr.qpos0
    d.qvel[:] = lqr.qvel0
    d.act[:] = 0.0
    d.ctrl[:15] = lqr.ctrl0
    mujoco.mj_forward(m, d)
    fxy = pn * np.array([np.cos(FWD), np.sin(FWD)])

    v_post = None
    pk_lean = -1e9
    pk_capt = -1e9
    pk_com_rel = -1e9
    min_lean = 1e9
    t_capt_pos = None       # first ms capture point > +2 mm past front edge
    t_capt_5 = None         # first ms capture point > +5 mm
    t_lean10 = None         # first ms fwd_lean > 10 deg
    t_fell = None
    fell_dir = ""
    n_capt_out = 0          # ms with capture point past the edge (any)
    for k in range(horizon):
        d.xfrc_applied[CHEST_BODY, :] = 0.0
        if PUSH_AT <= k < PUSH_AT + PUSH_DURATION_STEPS:
            d.xfrc_applied[CHEST_BODY, 0:2] = fxy
        d.ctrl[:15] = lqr.control(m, d)
        mujoco.mj_step(m, d)
        b = sample_balance(m, d)
        t = k - PUSH_AT
        if k == PUSH_AT + PUSH_DURATION_STEPS + 1:
            v_post = b.com_vfwd
        pk_lean = max(pk_lean, b.fwd_lean_deg)
        min_lean = min(min_lean, b.fwd_lean_deg)
        pk_capt = max(pk_capt, b.capture_fwd_rel_support_mm)
        pk_com_rel = max(pk_com_rel, b.com_fwd_rel_support_mm)
        if b.capture_fwd_rel_support_mm > 0:
            n_capt_out += 1
        if t_capt_pos is None and b.capture_fwd_rel_support_mm > 2:
            t_capt_pos = t
        if t_capt_5 is None and b.capture_fwd_rel_support_mm > 5:
            t_capt_5 = t
        if t_lean10 is None and b.fwd_lean_deg > 10:
            t_lean10 = t
        if b.up_tilt_deg > 45 or b.chest_z < NOMINAL_CHEST_Z - 0.20:
            t_fell = t
            fell_dir = ("fwd" if b.fwd_lean_deg > 20 else
                        "back" if b.fwd_lean_deg < -20 else "side")
            break
    b = sample_balance(m, d)
    settled = (t_fell is None and abs(b.fwd_lean_deg) < 5
               and b.com_speed_horiz < 0.05)
    # window: from capture-point-out to lean>10 (time a step has to act)
    win = (t_lean10 - t_capt_5) if (t_lean10 and t_capt_5) else None
    return dict(pn=pn, v_post=v_post or 0, pk_lean=pk_lean, min_lean=min_lean,
                pk_capt=pk_capt, pk_com_rel=pk_com_rel, n_capt_out=n_capt_out,
                t_capt_pos=t_capt_pos, t_capt_5=t_capt_5, t_lean10=t_lean10,
                t_fell=t_fell, fell_dir=fell_dir, settled=settled, win=win)


def main():
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)
    lqr = StandingLQR(m, d, verbose=True)
    print(f"\n{'push':>5} {'v+':>5} {'pkLean':>7} {'minLean':>7} {'pkCapt':>7} "
          f"{'pkComR':>7} {'captOutMs':>9} {'t_capt5':>8} {'t_lean10':>9} "
          f"{'window':>7} {'t_fell':>7} {'result':>10}")
    for pn in (90, 100, 108, 115, 120, 124, 128, 130, 133, 136, 139, 142, 145, 150):
        r = characterise(m, d, lqr, float(pn))
        res = ("settled" if r["settled"]
               else (f"FELL/{r['fell_dir']}" if r["t_fell"] else "drift"))
        print(f"{pn:5.0f} {r['v_post']:5.2f} {r['pk_lean']:7.1f} {r['min_lean']:7.1f} "
              f"{r['pk_capt']:7.0f} {r['pk_com_rel']:7.0f} {r['n_capt_out']:9d} "
              f"{str(r['t_capt_5']):>8} {str(r['t_lean10']):>9} {str(r['win']):>7} "
              f"{str(r['t_fell']):>7} {res:>10}")


if __name__ == "__main__":
    main()
