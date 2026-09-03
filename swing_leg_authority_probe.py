"""Can the R hip actuator swing an UNLOADED leg forward? (pin the chest, command hip)"""
import mujoco, numpy as np
from biped_env import DEFAULT_POSE, STANDING_QUAT
from recovery_metrics import _foot_xy_z

m = mujoco.MjModel.from_xml_path("robot/robot.xml"); d = mujoco.MjData(m)
# put chest high so feet are off the ground; zero gravity effect on translation by
# pinning: we simply hold qpos[0:7] fixed each step (kinematic base).
base = np.array([0,0,1.9] + list(STANDING_QUAT))
for hip_tgt in (0.3, 0.5, 0.7, 0.9):
    mujoco.mj_resetData(m, d)
    d.qpos[0:7] = base; d.qpos[7:22] = DEFAULT_POSE; d.qvel[:] = 0
    d.ctrl[:15] = DEFAULT_POSE
    mujoco.mj_forward(m, d)
    y0 = _foot_xy_z(m, d, "R")[1]
    for k in range(600):
        a = min(1.0, k/200)
        c = DEFAULT_POSE.copy()
        c[11] = hip_tgt * a          # R hip pitch
        c[12] = -(0.06 + 0.30*a)     # a little knee flex like the real swing
        d.ctrl[:15] = c
        d.qpos[0:7] = base; d.qvel[0:6] = 0   # pin the base (kinematic)
        mujoco.mj_step(m, d)
    y1 = _foot_xy_z(m, d, "R")[1]
    print(f"R hip target {hip_tgt:.2f} rad -> foot forward travel = {-(y1-y0)*1000:+6.1f} mm  "
          f"(qpos hip = {d.qpos[18]:.3f})")
