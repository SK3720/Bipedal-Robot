"""Frontal-plane dynamics probe for the DYNAMIC-transfer recovery step.

Answers, by measurement on robot/robot.xml (original geometry):
  1. foot lateral extent / support-polygon edges in world X
  2. a brief both-ankle-roll IMPULSE toward the stance foot: how big / how long
     to briefly unload the swing foot (normal force -> ~0)
  3. the lateral CoM velocity that impulse injects
  4. the TIME WINDOW: from release, how long until the lateral CoM would pass the
     far (outer) edge of the stance foot  -> that is the swing time budget
  5. whether a matched opposite counter-impulse rocks the CoM back to centre
     (the "ride it back" fallback)

No stepping here.  python _frontal_lipm_probe.py
"""
import numpy as np
import mujoco

from standing_balance_lqr import StandingLQR
from recovery_metrics import CHEST_BODY, _foot_normal_force, _foot_xy_z, sample_balance
from push_step_recovery_test import AR_L_IDX, AR_R_IDX, AR_SIGN, _pos_error

G = 9.81


def foot_x_extent(model, data, foot):
    """min/max world-X of that foot's collision geom AABB."""
    gid = model.geom(f"{foot}_foot_collision").id
    mujoco.mj_forward(model, data)
    c = data.geom_xpos[gid].copy()
    # geom_rbound is a bounding sphere radius; for a mesh use aabb if available
    try:
        aabb = model.geom_aabb[gid].reshape(2, 3)      # center(3), halfsize(3)
        R = data.geom_xmat[gid].reshape(3, 3)
        half = np.abs(R) @ aabb[1]
        cx = c[0] + (R @ aabb[0])[0]
        return cx - half[0], cx + half[0]
    except Exception:
        r = model.geom_rbound[gid]
        return c[0] - r, c[0] + r


def run_impulse(model, data, lqr, swing, amp, dur, counter=0.0, counter_at=0,
                horizon=1400, verbose=False):
    """Ramp a both-ankle-roll bias to `amp*AR_SIGN` over ~8 ms, hold `dur` ms,
    release to 0 (optionally a `counter` bias `counter_at` ms after release).
    Under StandingLQR the whole time.  Log lateral CoM."""
    data.qpos[:] = lqr.qpos0
    data.qvel[:] = lqr.qvel0
    data.act[:] = 0.0
    data.ctrl[:15] = lqr.ctrl0
    mujoco.mj_forward(model, data)

    stance = "L" if swing == "R" else "R"
    st_lo, st_hi = foot_x_extent(model, data, stance)
    outer_edge = st_hi if AR_SIGN[swing] < 0 else st_lo   # edge the CoM heads for

    com0 = data.subtree_com[CHEST_BODY][0]
    sgn = AR_SIGN[swing]
    rel_k = None
    t_unload = None
    t_edge = None
    vpeak = 0.0
    xs, vs = [], []
    for k in range(horizon):
        # ankle-roll schedule
        if k < 8:
            a = sgn * amp * (k / 8.0)
        elif k < 8 + dur:
            a = sgn * amp
        else:
            a = 0.0
            if rel_k is None:
                rel_k = k
        if rel_k is not None and counter and (k - rel_k) >= counter_at:
            a = -sgn * counter

        qr = lqr.qpos0.copy(); cr = lqr.ctrl0.copy()
        for ci in (AR_L_IDX, AR_R_IDX):
            qr[7 + ci] += a; cr[ci] += a
        dx = np.concatenate([_pos_error(model, qr, data.qpos), data.qvel - lqr.qvel0])
        dx[0] = dx[model.nv + 0] = 0.0        # don't let K chase the lateral drift
        u = cr - lqr.K @ dx
        u[AR_L_IDX] = cr[AR_L_IDX]; u[AR_R_IDX] = cr[AR_R_IDX]
        data.ctrl[:15] = np.clip(u, model.actuator_ctrlrange[:15, 0],
                                 model.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(model, data)

        mujoco.mj_subtreeVel(model, data)
        cx = data.subtree_com[CHEST_BODY][0]
        cvx = data.subtree_linvel[CHEST_BODY][0]
        xs.append(cx - com0); vs.append(cvx)
        b = sample_balance(model, data)
        sw_nf = _foot_normal_force(model, data, swing)
        if t_unload is None and sw_nf < 5.0 and k > 8:
            t_unload = k
        # heading +X (sgn<0) -> past outer edge when cx > outer_edge
        past = (cx > outer_edge) if sgn < 0 else (cx < outer_edge)
        if rel_k is not None and t_edge is None and past:
            t_edge = k - rel_k
        if abs(cvx) > abs(vpeak):
            vpeak = cvx
        if b.up_tilt_deg > 45:
            break

    xs = np.array(xs); vs = np.array(vs)
    b = sample_balance(model, data)
    settled = abs(b.side_lean_deg) < 4 and b.com_speed_horiz < 0.05
    return dict(amp=amp, dur=dur, counter=counter, rel_k=rel_k,
               t_unload=t_unload, t_edge=t_edge, v_at_release=(vs[rel_k] if rel_k and rel_k < len(vs) else 0.0),
               vpeak=vpeak, x_max=float(np.max(np.abs(xs)) * 1000),
               fell=b.up_tilt_deg > 45, end_side=b.side_lean_deg, settled=settled,
               outer_edge_mm=(outer_edge - com0) * 1000)


def main():
    m = mujoco.MjModel.from_xml_path("robot/robot.xml")
    d = mujoco.MjData(m)
    lqr = StandingLQR(m, d, verbose=True)

    d.qpos[:] = lqr.qpos0; d.qvel[:] = lqr.qvel0
    mujoco.mj_forward(m, d)
    com = d.subtree_com[CHEST_BODY]
    lL, hL = foot_x_extent(m, d, "L")
    lR, hR = foot_x_extent(m, d, "R")
    print(f"\nCoM x = {com[0]*1000:+.1f} mm   CoM height = {com[2]:.3f} m")
    print(f"L foot X: [{lL*1000:+.0f}, {hL*1000:+.0f}] mm   R foot X: [{lR*1000:+.0f}, {hR*1000:+.0f}] mm")
    print(f"LIPM omega = sqrt(g/h) = {np.sqrt(G/(com[2]-1.0)):.2f} rad/s   "
          f"time-const 1/omega = {1000/np.sqrt(G/(com[2]-1.0)):.0f} ms")

    print(f"\n--- swing=R : impulse toward L (stance) foot ---")
    print(f"{'amp':>5} {'dur':>4} {'t_unload':>9} {'v@rel':>7} {'vpeak':>7} "
          f"{'x_max':>7} {'t_edge':>7} {'end':>16}")
    for amp in (0.10, 0.15, 0.20, 0.25, 0.30):
        for dur in (25, 45, 70):
            r = run_impulse(m, d, lqr, "R", amp, dur)
            tag = "FELL" if r["fell"] else ("settled" if r["settled"] else f"side{r['end_side']:+.0f}")
            print(f"{amp:5.2f} {dur:4d} {str(r['t_unload']):>9} {r['v_at_release']*1000:7.0f} "
                  f"{r['vpeak']*1000:7.0f} {r['x_max']:7.1f} {str(r['t_edge']):>7} {tag:>16}")

    print(f"\n--- with matched counter-impulse (ride-back fallback), amp 0.20 dur 45 ---")
    for cf, cat in ((0.12, 60), (0.18, 60), (0.18, 90), (0.25, 90)):
        r = run_impulse(m, d, lqr, "R", 0.20, 45, counter=cf, counter_at=cat)
        tag = "FELL" if r["fell"] else ("settled" if r["settled"] else f"side{r['end_side']:+.0f}")
        print(f"  counter {cf:.2f} @ +{cat}ms  -> x_max {r['x_max']:.1f} mm  end {tag}")


if __name__ == "__main__":
    main()
