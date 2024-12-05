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
    use_object_obs=True,
    control_freq=20,
    render_camera=None,
)

# Test the environment
obs = env.reset()

# run 10 random actions
for _ in range(1000):

    # assert pr + "proprio-state" in obs
    # assert obs[pr + "proprio-state"].ndim == 1

    # assert "agentview_image" in obs
    # assert obs["agentview_image"].shape == (84, 84, 3)

    # assert "object-state" not in obs
    env.render()
    action = [0., 0., 0., 0., 0., 0., 0.]
    obs, reward, done, info = env.step(action)

# env.close()