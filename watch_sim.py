import time
import mujoco.viewer
from biped_env import BipedalWalkEnv
from stable_baselines3 import PPO

# Create environment
env = BipedalWalkEnv()
obs, _ = env.reset()

# Create a fresh PPO model with the same architecture
model = PPO("MlpPolicy", env, verbose=0, device="cpu")

# Load the saved policy weights directly
model.policy.load_state_dict(
    __import__("torch").load("model_test/policy.pth", weights_only=True)
)

print("Policy weights loaded successfully!")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    viewer.cam.lookat[:] = [0, 0, 1]
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -20

    while viewer.is_running():

        # Let the trained neural network choose the action
        action, _ = model.predict(obs, deterministic=True)

        # Apply action to MuJoCo
        obs, reward, terminated, truncated, info = env.step(action)

        # Update viewer
        viewer.sync()

        time.sleep(0.01)

        # Reset if robot falls
        if terminated or truncated:
            obs, _ = env.reset()
            print("qpos:", env.data.qpos)
