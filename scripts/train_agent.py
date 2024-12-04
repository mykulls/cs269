import robosuite as suite
from robosuite.environments.base import register_env
from stable_baselines3 import PPO  # Assuming you're using PPO for training
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robosuite.wrappers import GymWrapper
from envs.bottle_flip import BottleFlipTask
from config import BOTTLE_FLIP_TASK_ARGS

def wrap_env(env):
    wrapped_env = Monitor(env)                          # Needed for extracting eprewmean and eplenmean
    wrapped_env = DummyVecEnv([lambda : wrapped_env])   # Needed for all environments (e.g. used for mulit-processing)
    wrapped_env = VecNormalize(wrapped_env)             # Needed for improving training when using MuJoCo envs?
    return wrapped_env

register_env(BottleFlipTask)

# Create the environment using gym.make()
env = GymWrapper(
    suite.make(
        has_renderer=False,
        **BOTTLE_FLIP_TASK_ARGS,
    )
)

env = wrap_env(env)

# Create the model (using PPO as an example)
model = PPO("MlpPolicy", env, verbose=1)
# # Create the model (using PPO as an example)
# model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model_path = "./models/"
filename = "ppo_bottle_flip"
model.save(model_path + filename)
env.save(model_path + "vec_normalize_" + filename + '.pkl')
