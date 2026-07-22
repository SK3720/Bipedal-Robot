import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("mujoco/robot.xml")
data = mujoco.MjData(model)

data.qpos[0:3] = [0, 0, 2.0]
data.qpos[3:7] = [1, 0, 0, 0]

mujoco.mj_forward(model, data)
print("Number of joints:", model.njnt)

for i in range(model.njnt):
    print(i, model.joint(i).name)

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.lookat[:] = [0, 0, 1]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    while viewer.is_running():

        mujoco.mj_step(model, data)

        # IMPORTANT
        viewer.sync()

        time.sleep(0.01)
