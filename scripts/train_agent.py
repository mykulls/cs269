import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym.envs.registration import register
from envs.bottle_flip import BottleFlipTask  # Updated import

# Register the environment
register(
    id="BottleFlipTask-v0",  # Updated ID
    entry_point="envs.bottle_flip:BottleFlipTask",  # Updated entry point
)

# Load the environment
env = gym.make("BottleFlipTask-v0")

# Ensure compatibility
check_env(env)

# Wrap the environment for training
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env(lambda: env, n_envs=1)

# Train using PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Save the model
model.save("ppo_bottle_flip")

# Test the trained agent
env = gym.make("BottleFlipTask-v0")
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()