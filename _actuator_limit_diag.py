"""Is the forward in-place recovery limited by TORQUE AUTHORITY or by GEOMETRY?

At the standing-LQR forward ceiling (~124 N on baseline), log:
  - per-leg-actuator torque vs its +/-2.3 saturation  (is torque the limit?)
  - CoP / capture point vs the front edge of the feet  (is footprint the limit?)
  - stance-ankle-pitch angle vs its ctrlrange           (is joint range the limit?)

If actuators are NOT pinned at saturation when the robot loses balance, then
stronger actuators / torque-mode actuators cannot help - the limit is the
support-polygon geometry.
"""
import numpy as np, mujoco
from standing_balance_lqr import StandingLQR, settle_standing
from recovery_metrics import sample_balance, _foot_normal_force, NOMINAL_CHEST_Z
from biped_env import DEFAULT_POSE, PUSH_DURATION_STEPS

m = mujoco.MjModel.from_xml_path("robot/robot.xml")
d = mujoco.MjData(m)
settle_standing(m, d)
lqr = StandingLQR(m, d, verbose=False)

LEG = [i for i in range(m.nu)
       if any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
              for k in ("hip", "knee", "ankle"))]
LEG_NAMES = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in LEG]
AP = m.actuator("L_shin_L_ankle_pitch_motor").id


def run(push_n, verbose=False):
    d.qpos[:] = lqr.qpos0; d.qvel[:] = lqr.qvel0; d.ctrl[:15] = lqr.ctrl0
    mujoco.mj_forward(m, d)
    fxy = push_n * np.array([0.0, -1.0])
    max_frac = np.zeros(len(LEG)); n_sat = np.zeros(len(LEG))
    fell_at = None
    for k in range(3500):
        d.xfrc_applied[1, :] = 0
        if 5 <= k < 5 + PUSH_DURATION_STEPS:
            d.xfrc_applied[1, 0:2] = fxy
        dq = np.zeros(m.nv); mujoco.mj_differentiatePos(m, dq, 1.0, lqr.qpos0, d.qpos)
        u = lqr.ctrl0 - lqr.K @ np.concatenate([dq, d.qvel - lqr.qvel0])
        d.ctrl[:15] = np.clip(u, m.actuator_ctrlrange[:15, 0], m.actuator_ctrlrange[:15, 1])
        mujoco.mj_step(m, d)
        af = np.abs(d.actuator_force[LEG]) / 2.3
        max_frac = np.maximum(max_frac, af)
        n_sat += (af > 0.98)
        b = sample_balance(m, d)
        if verbose and k % 25 == 0:
            print(f"  k={k:3d} upT={b.up_tilt_deg:5.1f} capF_mm={_capture_front_mm(b):+6.1f} "
                  f"ankleP_frc={d.actuator_force[AP]:+5.2f} ankleP_q={d.qpos[7+8]:+.3f} "
                  f"maxleg_frc%={af.max()*100:3.0f}")
        if fell_at is None and (b.up_tilt_deg > 50 or d.qpos[2] < NOMINAL_CHEST_Z - 0.20):
            fell_at = k
            break
    return max_frac, n_sat, fell_at


def _capture_front_mm(b):
    h = NOMINAL_CHEST_Z; g = 9.81
    # capture point forward (world -Y) position relative to front-most foot edge
    cp_y = b.com[1] + b.com_vel[1] * np.sqrt(h / g)
    # front edge = most negative y of foot geoms
    return (cp_y - _front_edge_y()) * 1000.0


def _front_edge_y():
    ys = []
    for gi in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if "foot" in nm:
            ys.append(d.geom_xpos[gi][1] - m.geom_rbound[gi])
    return min(ys) if ys else 0.0


for pn in (100, 120, 124, 128, 140):
    mf, ns, fa = run(pn)
    tag = f"FELL@{fa}" if fa else "held"
    print(f"push {pn:3d}N  {tag:>9}  peak leg-actuator use: "
          + "  ".join(f"{n.split('_')[1][:3]}:{f*100:3.0f}%" for n, f in zip(LEG_NAMES, mf)))

print("\n--- detailed trace at 128N (just over ceiling) ---")
run(128, verbose=True)
