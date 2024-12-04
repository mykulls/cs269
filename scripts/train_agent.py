import robosuite as suite
from robosuite.environments.base import register_env
from stable_baselines3 import PPO  # Assuming you're using PPO for training
from stable_baselines3.common.vec_env import DummyVecEnv
from robosuite.wrappers import GymWrapper
from envs.bottle_flip import BottleFlipTask

register_env(BottleFlipTask)

# Create the environment using gym.make()
env = GymWrapper(
    suite.make(
        env_name="BottleFlipTask",  # Your custom environment name
        robots=["Panda"],  # Robot to use
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
)

# Wrap in DummyVecEnv for Stable-Baselines3 compatibility
vec_env = DummyVecEnv([lambda: env])

# Create the model (using PPO as an example)
model = PPO("MlpPolicy", vec_env, verbose=1)
# # Create the model (using PPO as an example)
# model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_bottle_flip")

# Test the trained agent
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()