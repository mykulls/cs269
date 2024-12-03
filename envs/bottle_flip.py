from robosuite.models.objects import BottleObject
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.environments.robot_env import RobotEnv
from robosuite.utils.placement_samplers import UniformRandomSampler

class BottleFlipTask(RobotEnv):
    def __init__(self, robots=["Panda"], table_full_size=(0.8, 0.8, 0.05), **kwargs):
        """
        Initialize the BottleFlipTask environment.
        
        Args:
            robots (list): A list of robot types to be used (default: ["Panda"])
            table_full_size (tuple): The full size of the table (defualt: (0.8, 0.8, 0.05))
        """
        self.table_full_size = table_full_size
        # Create other objects in the environment (e.g., table, bottle)
        self.arena = TableArena(table_full_size=table_full_size)
        self.arena.set_origin([0, 0, 0])  # Set origin of the arena
        self.bottle = BottleObject(name="bottle")  # Define the bottle

        # Initialize the parent RobotEnv class with the robots argument
        super().__init__(robots=robots, **kwargs)  # Pass robots list to RobotEnv constructor

    def _load_model(self):
        """
        Loads an XML model, puts it in self.model
        """
        super()._load_model()  # Call the parent class's _load_model method

        # Adjust the robot's base position
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])  # Adjust for the table size
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Optionally set the camera view for better visualization (if needed)
        self.arena.set_camera(
            camera_name="agentview", 
            pos=[0.5, -1.5, 1.0],  # Adjust camera position for better view
            quat=[0.5, 0.5, 0.5, 0.5]  # Adjust camera orientation if needed
        )

        # Task includes arena, robot, and bottle object
        self.model = ManipulationTask(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],  # List of robots in the task
            mujoco_objects=self.bottle,  # Include bottle object
        )

    # def _reset_internal(self):
    #     """Reset the environment to its initial state."""
    #     super()._reset_internal()  # Call the base reset method to reset the robot
    #     # Reset bottle position
    #     self.sim.data.set_joint_qpos("bottle_joint", [0, 0, 0.1, 0, 0, 0, 1])

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the door handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the door handle
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.door.important_sites["handle"], target_type="site"
            )

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment

        Returns:
            set: All components that can be individually visualized for this environment
        """
        vis_set = super()._visualizations
        vis_set.add("grippers")
        return vis_set

    def reward(self, action=None):
        """Defines the reward function."""
        # bottle_pos = self.sim.data.get_body_xpos("bottle")
        # bottle_orientation = self.sim.data.get_body_xquat("bottle")
        
        # # Reward based on height of bottle and upright orientation
        # upright = bottle_orientation[-1]  # Extracting the "w" component of quaternion
        return 1

    def step(self, action):
        """Takes a step in the environment."""
        # Apply robot action using the base class method
        obs, reward, done, info = super().step(action)

        # # Define termination condition (e.g., if the bottle falls off the table)
        # if self.sim.data.get_body_xpos("bottle")[2] < 0.05:
        #     done = True
        # done = True

        return obs, reward, done, info
    
    def _check_robot_configuration(self, robots):
        return len(robots) == 1
