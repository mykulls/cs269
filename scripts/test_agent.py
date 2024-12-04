import os

import robosuite as suite
from robosuite.environments.base import register_env
from stable_baselines3 import PPO  # Assuming you're using PPO for training
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robosuite.wrappers import GymWrapper
from envs.bottle_flip import BottleFlipTask

register_env(BottleFlipTask)

env_render = GymWrapper(
    suite.make(
        env_name="BottleFlipTask",  # Your custom environment name
        robots=["Panda"],  # Robot to use
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        render_camera=None,
        control_freq=20,
    )
)

root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
model_path = root_path + "/models/"
filename = "ppo_bottle_flip"
model = PPO.load(model_path + filename)
env_test = DummyVecEnv([lambda : env_render])
env_test = VecNormalize.load(model_path + "vec_normalize_" + filename + '.pkl', env_test)
env_test.training = False
env_test.norm_reward = False

# Test the trained agent
obs = env_test.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env_test.step(action)
    env_render.render()

    if done:
        obs = env_test.reset()
