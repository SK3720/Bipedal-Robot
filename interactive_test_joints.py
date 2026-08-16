import mujoco
import mujoco.viewer

from biped_env import BipedalWalkEnv

env = BipedalWalkEnv()

# Start from neutral pose
obs, info = env.reset()

print("\nCONTROLS")
print("--------------------------------")
print("1 / 2 = Left hip pitch")
print("3 / 4 = Left knee")
print("5 / 6 = Left ankle pitch")
print()
print("7 / 8 = Right hip pitch")
print("9 / 0 = Right knee")
print("- / = = Right ankle pitch")
print()
print("ESC = quit")
print("--------------------------------\n")


# Joint qpos indices
JOINTS = {
    "l_hip": 13,
    "l_knee": 14,
    "l_ankle": 15,
    "r_hip": 18,
    "r_knee": 19,
    "r_ankle": 20,
}

STEP = 0.1


def key_callback(key):
    # Left hip
    if key == ord("1"):
        env.data.qpos[JOINTS["l_hip"]] += STEP
    elif key == ord("2"):
        env.data.qpos[JOINTS["l_hip"]] -= STEP

    # Left knee
    elif key == ord("3"):
        env.data.qpos[JOINTS["l_knee"]] += STEP
    elif key == ord("4"):
        env.data.qpos[JOINTS["l_knee"]] -= STEP

    # Left ankle
    elif key == ord("5"):
        env.data.qpos[JOINTS["l_ankle"]] += STEP
    elif key == ord("6"):
        env.data.qpos[JOINTS["l_ankle"]] -= STEP

    # Right hip
    elif key == ord("7"):
        env.data.qpos[JOINTS["r_hip"]] += STEP
    elif key == ord("8"):
        env.data.qpos[JOINTS["r_hip"]] -= STEP

    # Right knee
    elif key == ord("9"):
        env.data.qpos[JOINTS["r_knee"]] += STEP
    elif key == ord("0"):
        env.data.qpos[JOINTS["r_knee"]] -= STEP

    # Right ankle
    elif key == ord("-"):
        env.data.qpos[JOINTS["r_ankle"]] += STEP
    elif key == ord("="):
        env.data.qpos[JOINTS["r_ankle"]] -= STEP

    mujoco.mj_forward(env.model, env.data)

    print("qpos:", env.data.qpos)


with mujoco.viewer.launch_passive(
    env.model, env.data, key_callback=key_callback
) as viewer:

    viewer.cam.lookat[:] = [0, 0, 1]
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -10

    while viewer.is_running():

        # Freeze physics while we find the pose
        env.data.qvel[:] = 0
        env.data.qacc[:] = 0

        mujoco.mj_forward(env.model, env.data)

        viewer.sync()
