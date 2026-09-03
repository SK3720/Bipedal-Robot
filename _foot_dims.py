"""World-frame sole footprint (fore-aft = Y, lateral = X) for the baseline and
for mesh-scaled variants, plus L/R sole gap.  Also: which MESH-LOCAL axis is
fore-aft, so a variant can lengthen ONLY the toe direction."""
import numpy as np, mujoco, re
from pathlib import Path
from standing_balance_lqr import settle_standing

BASE = Path("robot/robot.xml").read_text()


def build(sx=1.0, sy=1.0, sz=1.0):
    s = BASE
    for L in ("L", "R"):
        s = s.replace(
            f'mesh name="{L}_foot" content_type="model/stl" file="meshes/{L}_foot.stl" scale="0.001 0.001 0.001"',
            f'mesh name="{L}_foot" content_type="model/stl" file="meshes/{L}_foot.stl" scale="{0.001*sx} {0.001*sy} {0.001*sz}"')
    p = Path("robot/_fd.xml"); p.write_text(s)
    try:
        return mujoco.MjModel.from_xml_path(str(p))
    finally:
        p.unlink(missing_ok=True)


def sole(m, d, name):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    mid = m.geom_dataid[gid]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    loc = m.mesh_vert[va:va + vn].reshape(-1, 3)
    rot = d.geom_xmat[gid].reshape(3, 3)
    w = (rot @ loc.T).T + d.geom_xpos[gid]
    low = w[:, 2].min()
    sp = w[w[:, 2] <= low + 0.002]
    return sp, loc, rot


for tag, kw in [("baseline", {}), ("x1.6", dict(sx=1.6)), ("y1.6", dict(sy=1.6)),
                ("z1.6", dict(sz=1.6)), ("y2.2", dict(sy=2.2))]:
    m = build(**kw); d = mujoco.MjData(m)
    settle_standing(m, d)
    lsp, lloc, lrot = sole(m, d, "L_foot_collision")
    rsp, _, _ = sole(m, d, "R_foot_collision")
    faL = (lsp[:, 1].min() * 1000, lsp[:, 1].max() * 1000)
    latL = (lsp[:, 0].min() * 1000, lsp[:, 0].max() * 1000)
    gap = rsp[:, 0].min() * 1000 - latL[1]        # R is at smaller x; gap = Rmin_x - Lmax_x  (want >0)
    # which mesh-local axis -> world Y (fore-aft)?
    ax = np.argmax(np.abs(lrot @ np.eye(3) @ np.array([0, 1, 0])))  # crude
    col = np.abs(lrot[1, :])  # world-Y row of rot: contribution of each local axis
    print(f"{tag:9} L sole fore-aft Y[{faL[0]:+.0f},{faL[1]:+.0f}] len={faL[1]-faL[0]:.0f}mm  "
          f"lateral X[{latL[0]:+.0f},{latL[1]:+.0f}] w={latL[1]-latL[0]:.0f}mm  "
          f"L-R sole gap={gap:+.0f}mm  local->worldY weights={col.round(2)}")
