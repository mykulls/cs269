from envs.bottle_flip import BottleFlipTask
import robosuite as suite
import numpy as np

env = suite.make(
    env_name="BottleFlipTask",  # Your custom environment name
    robots=["Panda"],  # Robot to use
    has_renderer=True,
    has_offscreen_renderer=False,
    ignore_done=True,
    use_camera_obs=False,
    control_freq=20,
)

# Test the environment
obs = env.reset()

# get action range
action_min, action_max = env.action_spec
assert action_min.shape == action_max.shape

# Get robot prefix
pr = env.robots[0].robot_model.naming_prefix

# run 10 random actions
for _ in range(1000):

    # assert pr + "proprio-state" in obs
    # assert obs[pr + "proprio-state"].ndim == 1

    # assert "agentview_image" in obs
    # assert obs["agentview_image"].shape == (84, 84, 3)

    # assert "object-state" not in obs
    env.render()
    # For debugging: this pauses the simulation until u press enter
    # input("press enter")
    action = np.random.uniform(action_min, action_max)
    obs, reward, done, info = env.step(action)

# env.close()