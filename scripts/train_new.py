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

state_dim = 16 #(pos (3), quat(4), lin vel(3), ang vel(3), goal pos(3))
action_dim = 3  

policy_lr = 0.0003
value_lr = 0.001
gamma = 0.99
eps_clip = 0.05
k_epochs = 8

# Initialize high-level Actor-Critic Network and PPOTrainer
# high_level_policy = ActorCriticNetwork(state_dim, action_dim)

high_level_policy = ActorCriticNetwork(state_dim, action_dim)
# checkpoint = torch.load("model_ep0.pth")
# high_level_policy.load_state_dict(checkpoint['model_state_dict'])
ppo_trainer = PPOTrainer(high_level_policy, policy_lr, value_lr, gamma, eps_clip, k_epochs)

# Load low-level locomotion policy

label = "gait-conditioned-agility/2023-10-17/train"
env, low_level_policy = load_env(label, headless=True)

import numpy as np

def calculate_reward(state_tensor, goal_position, wall_positions, wall_threshold, step):
    touchwall = 0
    state_array = state_tensor.cpu().numpy()
    robot_position = state_array[:3]

    # Scaled distance to the goal
    goal_distance = np.linalg.norm(robot_position - np.array(goal_position))

    # Increase reward as robot gets closer to the goal
    reward = 100 * np.exp(-goal_distance)  # Exponential increase in reward

    # Exponential penalty for being too close to walls
    for wall_position in wall_positions:
        wall_distance = np.linalg.norm(robot_position - np.array(wall_position))
        if wall_distance < wall_threshold:
            reward -= 5000 * np.exp(-wall_distance)  # Exponential penalty for being close to walls
            touchwall = 1

    # Exponential step penalty to encourage efficiency
    step_penalty = 2 * np.exp(0.005 * step)  # The exponential factor can be adjusted
    reward -= step_penalty

    # Large reward for reaching the goal
    if goal_distance < 0.5:  # Threshold for reaching the goal
        reward += 5000  # Significant reward for reaching the goal

    # Debugging information
    if step % 75 == 0:
        print(f"Step: {step}, Goal Distance: {goal_distance}, Reward: {reward}, Step Penalty: {step_penalty}")

    return reward, touchwall






def map_continuous_action_to_velocities(action, step):
    
    max_linear_velocity = 1.0  
    max_angular_velocity = 1.0  
    action = np.clip(action, -0.5, 0.5)
    x_vel_cmd = action[0] * max_linear_velocity
    y_vel_cmd = action[1] * max_linear_velocity
    yaw_vel_cmd = action[2] * max_angular_velocity
    if step%75 == 0:
        print("Velocities", x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
    return x_vel_cmd, y_vel_cmd, yaw_vel_cmd

def save_model(model, optimizer, epoch, filename="ppo_model.pth"):
    """ Save the model state. """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)
# Function to simplify quaternion to Euler angles (or another representation)
def quaternion_to_euler(quaternion):
    # Assuming quaternion is in the format [x, y, z, w]
    # Implement the conversion to Euler angles or another simpler format
    # Return the simplified orientation representation
    pass

# Function to normalize state values
def normalize_state_values(state):
    # Normalize the state values to a consistent scale
    # For example, you might scale positions and velocities to a [-1, 1] range
    # Implement the normalization logic
    return state
def train_high_level_policy(num_episodes, max_steps_per_episode, goal_position, wall_positions, wall_threshold, batch_size=256):
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
    update_counter = 0
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()  
        cumulative_reward = 0

        for step in range(max_steps_per_episode):
            # Get the state from the environment
            robot_state = env.get_robot_state()  
            state_vector = np.concatenate([component.cpu().numpy().flatten() for component in robot_state])

            # Calculate relative position from the goal
            relative_goal_position = np.array(goal_position) - state_vector[:3]  # Assuming first 3 elements are position
            augmented_state_vector = np.concatenate([state_vector, relative_goal_position])
            normalized_state_vector = normalize_state_values(augmented_state_vector)

            # High-level policy decides the action
            with torch.no_grad():
                state_tensor = torch.tensor(normalized_state_vector, dtype=torch.float32)
                if step == 1:
                    start_state_tensor = state_tensor
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
            with torch.no_grad():
                new_robot_state = env.get_robot_state()
                
                new_state_vector = np.concatenate([component.cpu().numpy().flatten() for component in new_robot_state])
                relative_goal_position = np.array(goal_position) - new_state_vector[:3]  # Assuming first 3 elements are position
                augmented_state_vector = np.concatenate([new_state_vector, relative_goal_position])
                normalized_state_vector = normalize_state_values(augmented_state_vector)
                new_state_tensor = torch.tensor(normalized_state_vector, dtype=torch.float32)
            # Calculate reward and store transitions
            reward, touchwall = calculate_reward(new_state_tensor, goal_position, wall_positions, wall_threshold,step)
            cumulative_reward += reward
            ppo_trainer.store_transition(state_tensor, action, reward, new_state_tensor, done)

            update_counter += 1
            if update_counter >= batch_size:
                ppo_trainer.train()  # Train using the internal memory
                ppo_trainer.memory.clear()  # Clear the memory after training
                update_counter = 0  # Reset the counter

            obs = next_obs  # Update the observation
            # if touchwall == 1:
            #     done = 1
            if done:
                break

        # Update high-level policy at the end of each episode
        # ppo_trainer.train()
        if update_counter > 0:
            ppo_trainer.train()
            ppo_trainer.memory.clear()
            update_counter = 0
        
        if episode % 1 == 0:  # Save every episode
            save_model(high_level_policy, ppo_trainer.optimizer, episode, filename=f"model_ep{episode}.pth")

        print(f"Episode: {episode}, Total Reward: {cumulative_reward}")
    return (start_state_tensor)




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
num_episodes = 50
max_steps_per_episode = 750
start_state = train_high_level_policy(num_episodes, max_steps_per_episode, goal_position, wall_positions, wall_threshold, 256)
print(start_state)
