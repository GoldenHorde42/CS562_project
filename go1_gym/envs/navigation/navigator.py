#Currently not being used. The wall creation happens in the legged robot class. This will be only used for downstream tasks.

# from go1_gym.envs.base.legged_robot import LeggedRobot
# # from navigator_config import NavigatorConfig
# from isaacgym import gymtorch, gymapi, gymutil
# from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

# class Navigator(VelocityTrackingEasyEnv):
#     def __init__(self, sim_device, headless, cfg):
#         super().__init__(sim_device, headless, cfg=cfg)
#         self.cfg = cfg
#         # Set up the navigation specific environment
#         self.create_navigation_environments()

#     def create_navigation_environments(self):
#         # Common setup for the wall asset
#         box_asset_options = gymapi.AssetOptions()
#         box_asset_options.density = 0  # High density for static behavior
#         box_asset_options.fix_base_link = True
#         wall_length = 4.0  # Length of the wall
#         wall_height = 1.0  # Height of the wall
#         wall_thickness = 0.1  # Thickness of the wall

#         # Loop through each environment
#         for i in range(self.num_envs):
#             env_handle = self.envs[i]
#             # Additional environment-specific setup can go here

#             # Create box asset for each environment
#             box_asset = self.gym.create_box(self.sim, wall_thickness, wall_height, wall_length, box_asset_options)

#             # Function to add a wall to a specific environment
#             def add_wall(env, position):
#                 pose = gymapi.Transform()
#                 pose.p = gymapi.Vec3(position[0], position[1], position[2])
#                 pose.r = gymapi.Quat(0, 0, 0, 1)
#                 wall_handle = self.gym.create_actor(env_handle, box_asset, pose, "wall", 0, 1)
#                 # Set additional properties if needed

#             # Add walls to the current environment
#             add_wall(env_handle, (0.0, -1.0, 0.5))  # Left wall
#             add_wall(env_handle, (0.0, 1.0, 0.5))   # Right wall

            

#     def step(self, actions):
#         # Override the step function if needed, or use LeggedRobot's step
#         return super().step(actions)

#     # Include other methods as necessary, such as for resetting the environment, computing rewards, etc.
