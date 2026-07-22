import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco/robot.xml")
data = mujoco.MjData(model)

data.qpos[2] = 0.5
data.qpos[3:7] = [1, 0, 0, 0]

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # bend knee slowly
        data.qpos[9] = 0.5 * np.sin(time.time())

        mujoco.mj_forward(model, data)

        viewer.sync()
        time.sleep(0.01)
