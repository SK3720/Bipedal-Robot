import time
import mujoco.viewer
from biped_env import BipedalWalkEnv
from stable_baselines3 import PPO

# 1. Load the environment
env = BipedalWalkEnv()
obs, _ = env.reset()

# (Optional) Load your trained neural network brain once training is done
# model = PPO.load("ppo_biped_walking_model")

# 2. Launch the MuJoCo viewer
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    
    # Set up a good camera angle
    viewer.cam.lookat[:] = [0, 0, 1]
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -20
    
    while viewer.is_running():
        # 3. Choose an action
        # Use this to see random flailing before training is complete:
        action = env.action_space.sample() 
        
        # UNCOMMENT this to watch your TRAINED brain control the robot:
        # action, _states = model.predict(obs, deterministic=True)
        
        # 4. Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 5. Sync the visualizer to the physics engine
        viewer.sync()
        time.sleep(0.01) # Match real-world time speed
        
        # 6. Reset the simulation if the robot falls over
        if terminated or truncated:
            obs, _ = env.reset()