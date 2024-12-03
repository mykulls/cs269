import gym
import envs # This imports envs/__init__.py and triggers registration
from stable_baselines3 import PPO  # Assuming you're using PPO for training
from stable_baselines3.common.vec_env import DummyVecEnv

# Create the environment using gym.make()
env = gym.make("BottleFlipTask-v0")  # This uses the environment registered in envs/__init__.py

# Wrap the environment for stable-baselines3 (if using RL libraries like stable-baselines3)
env = DummyVecEnv([lambda: env])

# Create the model (using PPO as an example)
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_bottle_flip")

# Evaluate the trained model (optional)
obs = env.reset()
for _ in range(1000):
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()