#Currently not being used. The wall creation happens in the legged robot class. This will be only used for downstream tasks.

from go1_gym.envs.base.legged_robot import LeggedRobot
# from navigator_config import NavigatorConfig
from isaacgym import gymtorch, gymapi, gymutil
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

class Navigator(VelocityTrackingEasyEnv):
    def __init__(self, sim_device, headless, cfg):
        super().__init__(sim_device, headless, cfg=cfg)
        self.cfg = cfg
        # Set up the navigation specific environment      

    def step(self, actions):
        # Override the step function if needed, or use LeggedRobot's step
        return super().step(actions)

    # Include other methods as necessary, such as for resetting the environment, computing rewards, etc.
