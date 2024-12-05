import robosuite as suite
from robosuite.environments.base import register_env
from stable_baselines3 import SAC 
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robosuite.wrappers import GymWrapper
from envs.bottle_flip import BottleFlipTask
from config import BOTTLE_FLIP_TASK_ARGS, MODEL_PATH, FILENAME

register_env(BottleFlipTask)

env = GymWrapper(
    suite.make(
        has_renderer=True,
        **BOTTLE_FLIP_TASK_ARGS,
    )
)

model = SAC.load(MODEL_PATH + FILENAME)

# Test the trained agent
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()

    if done:
        obs, _ = env.reset()
        break
