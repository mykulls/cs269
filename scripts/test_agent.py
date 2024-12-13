import robosuite as suite
from robosuite.environments.base import register_env
from stable_baselines3 import PPO, TD3 # Assuming you're using PPO for training
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import SAC 
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robosuite.wrappers import GymWrapper
from envs.bottle_flip import BottleFlipTask
from config import BOTTLE_FLIP_TASK_ARGS, MODEL_PATH, FILENAME
import torch
# import RLkit

# model = torch.load("params.pkl")
# model.eval() 
register_env(BottleFlipTask)

env = GymWrapper(
    suite.make(
        has_renderer=True,
        **BOTTLE_FLIP_TASK_ARGS,
    )
)

model_path = "./models/"
filename = "ppo_5mil_bottle_top_lift"
model = PPO.load(model_path + filename)
# # filename = "td3_1_5_mil_bottle_flip"
# # model = TD3.load(MODEL_PATH + filename)

# Test the trained agent
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()

    if done:
        obs, _ = env.reset()
        break
