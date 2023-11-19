import isaacgym

assert isaacgym
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

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=Cfg)
    env = HistoryWrapper(env)

    # env = Navigator(sim_device='cuda:0', headless=False, cfg=Cfg)
    # env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy

state_dim = 13 #(pos (3), quat(4), lin vel(3), ang vel(3))
action_dim = 3  

policy_lr = 0.0003
value_lr = 0.001

gamma = 0.99
eps_clip = 0.2
k_epochs = 4

# Initialize high-level Actor-Critic Network and PPOTrainer
high_level_policy = ActorCriticNetwork(state_dim, action_dim)
ppo_trainer = PPOTrainer(high_level_policy, policy_lr, value_lr, gamma, eps_clip, k_epochs)

# Load low-level locomotion policy

label = "gait-conditioned-agility/2023-10-17/train"
env, low_level_policy = load_env(label, headless=True)

import numpy as np

def calculate_reward(state_tensor, goal_position, wall_positions, wall_threshold, step):
    # Ensure state_tensor is on CPU and converted to numpy for calculations
    state_array = state_tensor.cpu().numpy()

    # Extract robot position from the state tensor
    # Assuming the first 3 elements are the robot's position
    robot_position = state_array[:3]

    # Reward for getting closer to the goal
    goal_distance = np.linalg.norm(robot_position - np.array(goal_position))
    if step%75 == 0:
        print(robot_position, goal_position)
    reward = -goal_distance  # Negative distance to the goal

    # Penalty for getting too close to walls
    for wall_position in wall_positions:
        wall_distance = np.linalg.norm(robot_position - np.array(wall_position))
        if wall_distance < wall_threshold:
            reward -= 10  # Large penalty for being too close to a wall

    return reward


def map_continuous_action_to_velocities(action, step):
    
    max_linear_velocity = 1.0  # Adjust as needed
    max_angular_velocity = 1.0  # Adjust as needed

    x_vel_cmd = action[0] * max_linear_velocity
    y_vel_cmd = action[1] * max_linear_velocity
    yaw_vel_cmd = action[2] * max_angular_velocity
    if step%75 == 0:
        print("Velocities", x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
    return x_vel_cmd, y_vel_cmd, yaw_vel_cmd

def train_high_level_policy(num_episodes, max_steps_per_episode, goal_position, wall_positions, wall_threshold):
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
    jj =0
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()  # Initial reset of the environment
        cumulative_reward = 0

        for step in range(max_steps_per_episode):
            # Get the state from the environment
            robot_state = env.get_robot_state()  # Adjust this based on your environment's state representation
            if jj == 0:
                print(robot_state)
                jj =1
            state_vector = np.concatenate([component.cpu().numpy().flatten() for component in robot_state])


           
            
            # High-level policy decides the action
            with torch.no_grad():
                state_tensor = torch.tensor(state_vector, dtype=torch.float32)
                action, _ = high_level_policy.act(state_tensor)
                x_vel_cmd, y_vel_cmd, yaw_vel_cmd = map_continuous_action_to_velocities(action.numpy(), step)

            # Set velocity commands in the environment
            env.commands[:, 0] = x_vel_cmd
            env.commands[:, 1] = y_vel_cmd
            env.commands[:, 2] = yaw_vel_cmd
            env.commands[:, 3] = body_height_cmd
            env.commands[:, 4] = step_frequency_cmd
            env.commands[:, 5:8] = gait
            env.commands[:, 8] = 0.5
            env.commands[:, 9] = footswing_height_cmd
            env.commands[:, 10] = pitch_cmd
            env.commands[:, 11] = roll_cmd
            env.commands[:, 12] = stance_width_cmd
            # Use low-level policy with observations only
            with torch.no_grad():
                actions = low_level_policy(obs)

            # Execute actions in the environment
            next_obs, rew, done, info = env.step(actions)
            new_robot_state = env.get_robot_state()
            new_state_vector = np.concatenate([component.cpu().numpy().flatten() for component in robot_state])
            with torch.no_grad():
                new_state_tensor = torch.tensor(state_vector, dtype=torch.float32)
            # Calculate reward and store transitions
            reward = calculate_reward(state_tensor, goal_position, wall_positions, wall_threshold,step)
            cumulative_reward += reward
            ppo_trainer.store_transition(state_tensor, action, reward, new_state_tensor, done)

            obs = next_obs  # Update the observation

            if done:
                break

        # Update high-level policy at the end of each episode
        ppo_trainer.train()

        print(f"Episode: {episode}, Total Reward: {cumulative_reward}")




robot_initial_position = [0.0, 0.0, 0.0]  # Assuming the origin as the initial position
goal_position = [4.5, 0.0, 0.1]  # Adjusted goal position

# Define wall positions based on their offsets
wall_positions = [
    [2.25, -1.0, 0.1],  # Left wall
    [2.25, 1.0, 0.1],   # Right wall
    [-0.25, 0.0, 0.1],  # Front wall
    [4.75, 0.0, 0.1]    # Back wall
]

wall_threshold = 0.5
# Train the high-level policy
num_episodes = 1000
max_steps_per_episode = 750
train_high_level_policy(num_episodes, max_steps_per_episode, goal_position, wall_positions, wall_threshold)
