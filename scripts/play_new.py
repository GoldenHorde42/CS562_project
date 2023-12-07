import isaacgym
import torch
import numpy as np
import glob
import pickle as pkl
from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.navigation.navigator import Navigator
from tqdm import tqdm
from go1_gym.envs.navigation.ActorCritic import ActorCriticNetwork
from go1_gym.envs.navigation.PPOtrainer import PPOTrainer
from ml_logger import logger
from pathlib import Path
from go1_gym import MINI_GYM_ROOT_DIR
import glob
import os
import torch
import cv2

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.record_video = True
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    Cfg.terrain.mesh_type = 'plane'
    Cfg.terrain.teleport_robots = False
    Cfg.env.env_spacing = 5.25
    Cfg.viewer.pos = [3, 0, 8]
    Cfg.viewer.lookat = [3., 1, 0.]
    Cfg.init_state.pos = [0.5, 0.0, 0.5]
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)
    # env = Navigator(sim_device='cuda:0', headless=False, cfg=Cfg)
    # env = HistoryWrapper(env)
    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic
    policy = load_policy(logdir)
    return env, policy

# def calculate_reward(state_tensor, goal_position, wall_positions, wall_threshold, step):
#     state_array = state_tensor.cpu().numpy()
#     robot_position = state_array[:3]
#     goal_distance = np.linalg.norm(robot_position - np.array(goal_position))

#     # Use an inverse function of the distance to make it more rewarding as the agent gets closer
#     # Adding a small constant to avoid division by zero
#     reward = -1.0 / (goal_distance + 0.01)

#     # Penalty for walls
#     for wall_position in wall_positions:
#         wall_distance = np.linalg.norm(robot_position - np.array(wall_position))
#         if wall_distance < wall_threshold:
#             reward -= 10  # Penalty for being too close to a wall

#     return reward

# def calculate_reward(state_tensor, goal_position, wall_positions, wall_threshold, step):
#     # Ensure state_tensor is on CPU and converted to numpy for calculations
#     state_array = state_tensor.cpu().numpy()

#     # Extract robot position from the state tensor
#     # Assuming the first 3 elements are the robot's position
#     robot_position = state_array[:3]

#     # Reward for getting closer to the goal
#     goal_distance = np.linalg.norm(robot_position - np.array(goal_position))
#     if step%75 == 0:
#         print(robot_position, goal_position)
#     reward = -goal_distance  # Negative distance to the goal

#     # Penalty for getting too close to walls
#     for wall_position in wall_positions:
#         wall_distance = np.linalg.norm(robot_position - np.array(wall_position))
#         if wall_distance < wall_threshold:
#             reward -= 10  # Large penalty for being too close to a wall

#     return reward

# def calculate_reward(state_tensor, goal_position, wall_positions, wall_threshold, step):
#     state_array = state_tensor.cpu().numpy()
#     robot_position = state_array[:3]

#     # Reward for getting closer to the goal
#     goal_distance = np.linalg.norm(robot_position - np.array(goal_position))
#     if step%75 == 0:
#         print(robot_position, goal_position)
#     reward = -10.0 * goal_distance  # Significantly increased negative impact for distance to the goal

#     return reward

"""
Below reward function also promotes getting closer to the goal
"""

def calculate_reward(state_tensor, goal_position, wall_positions, wall_threshold, step):
    state_array = state_tensor.cpu().numpy()
    robot_position = state_array[:3]

    # Scaled distance to the goal
    goal_distance = np.linalg.norm(robot_position - np.array(goal_position))
    
    # Reward for moving towards the goal
    # This could be a large negative value that becomes less negative (or even positive) as the robot approaches the goal
    reward = -100 * goal_distance  # Increased scale factor for goal distance

    # Penalize touching or being too close to the walls
    for wall_position in wall_positions:
        wall_distance = np.linalg.norm(robot_position - np.array(wall_position))
        if wall_distance < wall_threshold:
            # A very large negative reward for being too close to or touching a wall
            reward -= 5000  # Large penalty for touching walls

    # Small step penalty to encourage efficiency
    reward -= 0.01  # Small penalty for each step taken

    # Large reward for reaching the goal
    if goal_distance < 0.2:  # Threshold for reaching the goal
        reward += 5000  # Large reward for reaching the goal

    # Debugging information
    print(goal_distance)
    if step % 75 == 0:
        print(f"Step: {step}, Position: {robot_position}, Goal: {goal_position}, Reward: {reward}")
        

    return reward


def map_continuous_action_to_velocities(action, step):
    
    max_linear_velocity = 1.0  
    max_angular_velocity = 1.0  
    action = np.clip(action, -1, 1)
    x_vel_cmd = action[0] * max_linear_velocity
    y_vel_cmd = action[1] * max_linear_velocity
    yaw_vel_cmd = action[2] * max_angular_velocity
    if step%75 == 0:
        print("Velocities", x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
    return x_vel_cmd, y_vel_cmd, yaw_vel_cmd

state_dim = 16 #(pos (3), quat(4), lin vel(3), ang vel(3))
action_dim = 3 

# Load the trained high-level policy model
high_level_policy = ActorCriticNetwork(state_dim, action_dim)
checkpoint = torch.load("model_ep20.pth")
high_level_policy.load_state_dict(checkpoint['model_state_dict'])


# Load the environment and low-level policy
env, low_level_policy = load_env("gait-conditioned-agility/2023-10-17/train", headless=False)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame = env.render(mode="rgb_array")
height, width, layers = frame.shape
out = cv2.VideoWriter('new_output.mp4', fourcc, 20.0, (width, height))
# Goal position and wall positions
goal_position = [4.5, 0.0, 0.1]
wall_positions = [
    [2.25, -1.0, 0.1],
    [2.25, 1.0, 0.1],
    [-0.25, 0.0, 0.1],
    [4.75, 0.0, 0.1]
]

wall_threshold = 0.5
def normalize_state_values(state):
    # Normalize the state values to a consistent scale
    # For example, you might scale positions and velocities to a [-1, 1] range
    # Implement the normalization logic
    return state
def play_episode(max_steps):
    obs = env.reset()
    cumulative_reward = 0

    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    for step in range(max_steps):
        # High-level policy action
        robot_state = env.get_robot_state()
        state_vector = np.concatenate([component.cpu().numpy().flatten() for component in robot_state])
        relative_goal_position = np.array(goal_position) - state_vector[:3]  # Assuming first 3 elements are position
        augmented_state_vector = np.concatenate([state_vector, relative_goal_position])
        normalized_state_vector = normalize_state_values(augmented_state_vector)
        state_tensor = torch.tensor(normalized_state_vector, dtype=torch.float32)
        with torch.no_grad():
            action, _ = high_level_policy.act(state_tensor)

        # Map action to velocities and set commands
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = map_continuous_action_to_velocities(action.numpy(), step)
        # Execute actions using low-level policy and environment
        with torch.no_grad():
            actions = low_level_policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        # env.commands[:, 0] = 1.0
        # env.commands[:, 1] = 0.0
        # env.commands[:, 2] = 0.0
        # print(env.commands[:, 0],env.commands[:, 1],env.commands[:, 2])
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd

        
        obs, rew, done, info = env.step(actions)
        frame = env.render(mode="rgb_array")
        frame = frame[:, :, :3]
        out.write(frame)
        new_robot_state = env.get_robot_state()
        new_state_vector = np.concatenate([component.cpu().numpy().flatten() for component in new_robot_state])
        with torch.no_grad():
            new_state_tensor = torch.tensor(new_state_vector, dtype=torch.float32)
        # Reward calculation and logging
        reward = calculate_reward(state_tensor, goal_position, wall_positions, wall_threshold, step)
        cumulative_reward += reward
        # print(f"Step: {step}, Position: {new_state_tensor}, Reward: {reward}")

        if done:
            break

    print(f"Total Reward: {cumulative_reward}")
    out.release()
# Play for one episode
max_steps_per_episode = 750
play_episode(max_steps_per_episode)
