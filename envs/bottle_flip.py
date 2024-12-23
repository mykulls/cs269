from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
# from robosuite.models.objects import BoxObject
from robosuite.models.objects import BottleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from scipy.spatial.transform import Rotation as R


# import pybullet as p

class BottleFlipTask(ManipulationEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (bottle) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
        # prev position used for velocity calculation
        self.prev_bottle_bottom_pos = self.sim.data.get_site_xpos("bottle_default_site")
        self.prev_bottle_top_pos = self.get_bottle_top_pos()
        self.prev_gripper_center_pos = self.sim.data.get_site_xpos("gripper0_right_grip_site")
        self.lifted = False
        self.flipped = False
        self.landed = False
        self.max_reward = 6
        self.success = False

    def get_bottle_top_pos(self):
        # Get position of top of bottle
        bottle_quaternion = self.sim.data.get_body_xquat(self.bottle.root_body)
        magnitude = np.sum(np.sqrt(bottle_quaternion[1:]**2))
        # print("nromalized: ",bottle_quaternion[1:]/magnitude)
        bottle_height = 0.085
        bottle_top_offset = bottle_quaternion[1:]/magnitude * bottle_height # 0.085 is the height of bottle
        # print("Offset: ", bottle_top_offset)
        bottle_bottom_pos = self.sim.data.get_site_xpos("bottle_default_site")
        bottle_top_pos = bottle_bottom_pos + bottle_top_offset
        return bottle_top_pos
    
    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the bottle is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the bottle
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the bottle
            - Lifting: in {0, 1}, non-zero if arm has lifted the bottle

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0
        # print("Action: action")

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:
            # Reward for distance to bottle: max of 1
            bottle_top_pos = self.get_bottle_top_pos()
            # print("bottle bottom: ", bottle_bottom_pos)
            right_gripper_pos = self.sim.data.get_site_xpos("gripper0_right_grip_site")
            dist_to_top = np.linalg.norm(right_gripper_pos - bottle_top_pos)
            # print("Dist to bottle: ",dist_to_top)
            reaching_reward = 1 - np.tanh(10.0 * dist_to_top)
            reward += reaching_reward
            # print("Gripper: ", right_gripper_pos)
            # print("bottle pos: ", bottle_top_pos)

            joint1_qpos = self.sim.data.get_joint_qpos("gripper0_right_finger_joint1")
            joint2_qpos = self.sim.data.get_joint_qpos("gripper0_right_finger_joint2")
            gripper_width = abs(joint1_qpos - joint2_qpos)
            # print("Gripper width: ",gripper_width)

            # Penalize fully closed gripper: max penalty of -0.25
            if gripper_width < 0.02:
                fully_closed = True
                fully_closed_penalty = 0.25 - np.tanh(120 * gripper_width)
                reward -= fully_closed_penalty
                # print("Closed penalty: ", fully_closed_penalty)
            else:
                fully_closed = False

            fully_opened = True if gripper_width > 0.06 else False
    
            # grasping reward only if gripper isn't fully closed 
            # ensure the bottle is inbetween the gripper
            grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.bottle)
            if not fully_closed and grasped:
                reward += 0.25
                # print("GRASP DETECTED")
            
            # make smooth reward function for lift
            # smooth concave up curve from 0 to 1 when dist is from 0 to 0.015
            if grasped:
                amount_lifted = self.get_bottle_lift()
                # print("Lifted: ",amount_lifted)
                reward += min(np.tanh(200*(amount_lifted - 0.015)) + 1, 1)

            if self.lifted:
                reward = 2.25
                reward += self.flip_reward(
                    fully_closed,
                    grasped,
                    fully_opened,
                    bottle_top_pos,
                    right_gripper_pos
                )
                

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / self.max_reward
        # print("Reward: ", reward)
        return reward
    
    def flip_reward(self, fully_closed, grasped, fully_opened, bottle_top_pos, gripper_center_pos):
        bottle_bottom_pos = self.sim.data.get_site_xpos("bottle_default_site")
        bottle_vel = bottle_bottom_pos - self.prev_bottle_bottom_pos
        # gripper_center_pos = self.sim.data.get_site_xpos("gripper0_right_grip_site")
        gripper_vel = gripper_center_pos - self.prev_gripper_center_pos

        reward = 0.0

        bottle_z_diff = bottle_bottom_pos - bottle_top_pos
        if self.flipped:
            reward += 1
        elif grasped:
            # upward velocity gripper reward
            # max of 0.3
            important_vel = np.sqrt(gripper_vel[2]**2 + gripper_vel[0]**2)
            # want reward from 0 to 0.3 with velocity from 0 to 1? 0.015 is considered lifted
            # we want really fast so velocity of 1 is like moving 7 times the distance in one frame
            reward += min(0.3 * (np.tanh(2(important_vel-1))+1), 0.3)
            
            # bottle rotation while grasped reward
            # reward range 0 to 0.7
            # bottle_z_diff range -0.085 to 0.085
            # bottle_z_diff = bottle_bottom_pos - bottle_top_pos
            # we want to reward for going upside down so we want this to be positive and max value of bottle height
            # range of bottle_z_diff is -0.085 to 0.085
            reward += min(0.7* (np.tanh(15*(bottle_z_diff-0.085))+1), 0.7)
        
        if bottle_z_diff > 0.08: # flip can occur after release also
                self.flipped = True
        

        if fully_opened and self.bottle_is_above_table():
            # airborne velocity reward
            vel_mag = np.sum(np.sqrt(bottle_vel**2))
            # more reward than when grasped
            # velocity range probably from 0 to 1
            reward += min(0.3* (np.tanh(2(vel_mag-1))+1), 0.3)
            if self.flipped:
            # airbone rotation reward
            # maybe also do top and bottom diff? but the other way????
            # reward range 0 to 1
                bottle_z_reverse_diff =  bottle_top_pos - bottle_bottom_pos
                reward += min(0.7 * (np.tanh(15*(bottle_z_reverse_diff-0.085))+1), 0.7)

        # landing on table orientation reward and contact with table
        if self.flipped and self.bottle_on_table():
            bottle_quaternion = self.sim.data.get_body_xquat(self.bottle.root_body)
            magnitude = np.sum(np.sqrt(bottle_quaternion[1:]**2))
            norm_bottle_vec = bottle_quaternion[1:] / magnitude
            z_vec = np.array([0, 0, 1])
            diff_from_vertical = np.dot(z_vec, norm_bottle_vec)
            # input from 0 to 1
            # reward from 0 to 2.75
            reward += min(2.75*(np.tanh(4(diff_from_vertical-1))+1), 2.75)

            if self.landed and diff_from_vertical > 0.95:
                self.success = True
                reward = self.max_reward
            self.landed = True

        return reward

    def bottle_on_table(self):
        table_height = self.model.mujoco_arena.table_offset[2]

        bottle_pos = self.sim.data.get_site_xpos("bottle_default_site")
        x = bottle_pos[0]
        y = bottle_pos[1]
        z = bottle_pos[2]
        return x < 0.4 and x > -0.4 and y < 0.4 and y > -0.4 and z > table_height


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        # tex_attrib = {
        #     "type": "bottle",
        # }
        # mat_attrib = {
        #     "texrepeat": "1 1",
        #     "specular": "0.4",
        #     "shininess": "0.1",
        # }
        # redwood = CustomMaterial(
        #     texture="WoodRed",
        #     tex_name="redwood",
        #     mat_name="redwood_mat",
        #     tex_attrib=tex_attrib,
        #     mat_attrib=mat_attrib,
        # )
        self.bottle = BottleObject(
            name="bottle"
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            # Reset the placement initializer
            self.placement_initializer.reset()
            # Directly place the object at (0, 0)
            self.placement_initializer.add_objects(self.bottle)
        else:
            # Manually place the bottle at (0, 0)
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.bottle,
                x_range=[0, 0],  # No range, fixed at 0
                y_range=[0, 0],  # No range, fixed at 0
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.bottle,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.bottle_body_id = self.sim.model.body_name2id(self.bottle.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # bottle-related observables
            @sensor(modality=modality)
            def bottle_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.bottle_body_id])

            @sensor(modality=modality)
            def bottle_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.bottle_body_id]), to="xyzw")

            sensors = [bottle_pos, bottle_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to bottle position sensor; one for each arm
            sensors += [
                self._get_obj_eef_sensor(full_pf, "bottle_pos", f"{arm_pf}gripper_to_bottle_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        
        # Define desired joint positions for the robot arm
        desired_joint_positions = [0, 1.7, 0, 0.9, 0, 1.8, 0.75]
        gripper_closed_qpos = 0.2

        # # print(desired_joint_positions)
        for i, joint_name in enumerate(self.sim.model.joint_names):
            if "gripper" not in joint_name and "robot" in joint_name:  # Avoid gripper joints
                # print(joint_name)
                self.sim.data.set_joint_qpos(joint_name, desired_joint_positions[i])

        # Set the gripper joint positions (e.g., fingers closed)
        self.sim.data.set_joint_qpos("gripper0_right_finger_joint1", gripper_closed_qpos)
        self.sim.data.set_joint_qpos("gripper0_right_finger_joint2", gripper_closed_qpos)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the bottle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the bottle
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.bottle)

    def _check_success(self):
        """
        Check if bottle has been lifted.

        Returns:
            bool: True if bottle has been lifted
        """
        return self.success
    
    def bottle_is_above_table(self):
        """
        Check if bottle has been lifted.

        Returns:
            bool: True if bottle has been lifted
        """
        bottle_height = self.sim.data.body_xpos[self.bottle_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # bottle is higher than the table top above a margin
        return bottle_height > table_height + 0.08
    
    def get_bottle_lift(self):
        """
        Returns:
            float: height bottle has been lifted
        """
        bottle_height = self.sim.data.body_xpos[self.bottle_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        table_thickness = 0.0646
        # bottle is higher than the table top above a margin
        return (bottle_height - table_height) - table_thickness