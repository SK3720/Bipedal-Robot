import numpy as np, mujoco
from _envelope2 import build
from standing_balance_lqr import settle_standing
from recovery_metrics import sample_balance, _foot_normal_force

for fs in (1.0, 1.4, 1.6, 2.0):
    m = build(foot_scale=None if fs == 1.0 else fs)
    d = mujoco.MjData(m)
    q0, v0 = settle_standing(m, d)
    b = sample_balance(m, d)
    # foot geom AABB via geom_aabb (local) -> use rbound + xpos
    fids = [gi for gi in range(m.ngeom)
            if "foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or "")]
    print(f"\nfoot_scale={fs}")
    for gi in fids:
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)
        lo = m.geom_aabb[gi, :3] - m.geom_aabb[gi, 3:]
        hi = m.geom_aabb[gi, :3] + m.geom_aabb[gi, 3:]
        print(f"  {nm:16} local AABB x[{lo[0]*1000:+.0f},{hi[0]*1000:+.0f}] "
              f"y[{lo[1]*1000:+.0f},{hi[1]*1000:+.0f}] z[{lo[2]*1000:+.0f},{hi[2]*1000:+.0f}] mm  "
              f"world x={d.geom_xpos[gi][0]*1000:+.0f}")
    # inter-foot contacts?
    ff = 0
    for ci in range(d.ncon):
        c = d.contact[ci]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if "foot" in g1 and "foot" in g2:
            ff += 1
    print(f"  settled: chestZ={d.qpos[2]:.3f} up_tilt={b.up_tilt_deg:.2f} "
          f"Lnf={_foot_normal_force(m,d,'L'):.0f} Rnf={_foot_normal_force(m,d,'R'):.0f} "
          f"foot-foot contacts={ff} ncon={d.ncon}")
