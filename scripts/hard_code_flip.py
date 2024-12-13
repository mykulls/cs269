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
    reward_shaping=True
)

# Test the environment
obs = env.reset()

def run_actions(actions, type, pause=True):
    # print(f'Running {type} actions')
    for i in range(len(actions)):
        env.render()
        # if pause:
            # input("Press enter to continue")
        # action = [0., 0., 0., 0., 0., 0., 0.]
        action = actions[i]
        # print("action: ",action)
        # print("gripper pos: ", env.sim.data.get_site_xpos("gripper0_right_grip_site"))
        # print("bottle pos: ", env.sim.data.get_site_xpos("bottle_default_site"))

        obs, reward, done, info = env.step(action)

x_vec =np.array([1., 0., 0., 0., 0., 0., 0.])
y_vec = np.array([0., 1., 0., 0., 0., 0., 0.])
z_vec = np.array([0., 0., 1., 0., 0., 0., 0.])
open_grip_vec = np.array([0., 0., 0., 0., 0., 0., -1.])
close_grip_vec = np.array([0., 0., 0., 0., 0., 0., 1.])

init_actions = [open_grip_vec for i in range(12)]
x_actions = [x_vec+open_grip_vec for i in range(2)]
lower_actions = [-1*z_vec+close_grip_vec for i in range(3)]
grab_actions = [close_grip_vec for i in range(8)]
lift_actions = [z_vec+close_grip_vec for i in range(6)]
toss_actions = [50000000*z_vec + 5000000*x_vec +close_grip_vec for i in range(4)]
release_actions = [50000000*z_vec + 5000000*x_vec +open_grip_vec for i in range(10)]

run_actions(init_actions, "initial", pause=False)
run_actions(x_actions, "x", pause=False)
run_actions(lower_actions, "lower", pause=False)
run_actions(grab_actions, "grab", pause=False)
run_actions(lift_actions, "lift", pause=False)
run_actions(toss_actions, "toss", pause=False)
run_actions(release_actions, "release", pause=False)

# run 10 random actions
# for i in range(len(actions)):

#     # assert pr + "proprio-state" in obs
#     # assert obs[pr + "proprio-state"].ndim == 1

#     # assert "agentview_image" in obs
#     # assert obs["agentview_image"].shape == (84, 84, 3)

#     # assert "object-state" not in obs
#     env.render()
#     input("Press enter to continue")
#     # action = [0., 0., 0., 0., 0., 0., 0.]
#     action = actions[i]
#     print(action)
#     obs, reward, done, info = env.step(action)

# env.close()
