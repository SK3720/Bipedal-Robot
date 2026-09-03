"""Visual: same forward push, baseline feet vs fore-aft-lengthened feet, standing
LQR only (no step).  Shows the support-polygon lever directly.

  python _longfoot_demo.py --push 200 --slow            # long feet (sz 2.0) recover
  python _longfoot_demo.py --push 200 --slow --sz 1.0   # baseline feet fall
"""
import argparse, sys, time
import mujoco, mujoco.viewer, numpy as np
from pathlib import Path
from standing_balance_lqr import settle_standing, _dare
from recovery_metrics import sample_balance, NOMINAL_CHEST_Z
from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS

BASE = Path("robot/robot.xml").read_text()


def build(sz):
    s = BASE
    for L in ("L", "R"):
        s = s.replace(
            f'mesh name="{L}_foot" content_type="model/stl" file="meshes/{L}_foot.stl" scale="0.001 0.001 0.001"',
            f'mesh name="{L}_foot" content_type="model/stl" file="meshes/{L}_foot.stl" scale="0.001 0.001 {0.001*sz}"')
    p = Path("robot/_lfd.xml"); p.write_text(s)
    try:
        return mujoco.MjModel.from_xml_path(str(p))
    finally:
        p.unlink(missing_ok=True)


def make_K(m, d, q0, v0):
    d.qpos[:] = q0; d.qvel[:] = v0; d.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(m, d)
    A = np.zeros((2 * m.nv, 2 * m.nv)); B = np.zeros((2 * m.nv, 15))
    mujoco.mjd_transitionFD(m, d, 1e-6, 1, A, B, None, None)
    qp = np.ones(m.nv) * 2.0; qp[0] = 3.0; qp[1] = 3.0; qp[2] = 40.0
    qp[3:6] = 300.0; qp[6:21] = 1.0
    qv = np.ones(m.nv); qv[0] = 6.0; qv[1] = 40.0; qv[2] = 6.0; qv[3:6] = 25.0; qv[6:21] = 0.4
    Q = np.diag(np.concatenate([qp, qv])); R = np.diag(np.ones(15) * 3.0)
    K, _, _ = _dare(A, B, Q, R)
    return K


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--push", type=float, default=200.0)
    p.add_argument("--sz", type=float, default=2.0)
    p.add_argument("--slow", action="store_true")
    p.add_argument("--headless", action="store_true")
    a = p.parse_args(argv)

    m = build(a.sz); d = mujoco.MjData(m)
    q0, v0 = settle_standing(m, d)
    K = make_K(m, d, q0, v0)
    print(f"foot fore-aft ~{88*a.sz:.0f} mm   push {a.push:.0f} N forward")

    def step_sim(viewer=None):
        d.qpos[:] = q0; d.qvel[:] = v0; d.ctrl[:15] = DEFAULT_POSE
        mujoco.mj_forward(m, d)
        fxy = a.push * np.array([0.0, -1.0])
        peak_tilt = 0.0
        for k in range(4000):
            d.xfrc_applied[1, :] = 0
            if 20 <= k < 20 + PUSH_DURATION_STEPS:
                d.xfrc_applied[1, 0:2] = fxy
            dq = np.zeros(m.nv); mujoco.mj_differentiatePos(m, dq, 1.0, q0, d.qpos)
            u = DEFAULT_POSE - K @ np.concatenate([dq, d.qvel - v0])
            d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
            mujoco.mj_step(m, d)
            b = sample_balance(m, d)
            peak_tilt = max(peak_tilt, b.up_tilt_deg)
            if viewer is not None:
                if not viewer.is_running():
                    return
                viewer.sync()
                time.sleep(0.02 if a.slow else 0.002)
            if b.up_tilt_deg > 50 or d.qpos[2] < NOMINAL_CHEST_Z - 0.20:
                print(f"  FELL @ step {k}"); return
        b = sample_balance(m, d)
        print(f"  held: end up_tilt {b.up_tilt_deg:.1f} deg  peak {peak_tilt:.1f} deg  "
              f"chestZ {d.qpos[2]:.3f}  DS={b.l_contact and b.r_contact}")

    if a.headless:
        step_sim()
        return
    with mujoco.viewer.launch_passive(m, d) as vw:
        vw.cam.lookat[:] = [0.0, -0.1, 1.05]; vw.cam.distance = 2.2
        vw.cam.azimuth = 90; vw.cam.elevation = -8
        step_sim(vw)
        print("  close viewer to exit")
        while vw.is_running():
            vw.sync(); time.sleep(0.02 if a.slow else 0.002)


if __name__ == "__main__":
    main(sys.argv[1:])
