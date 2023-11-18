import torch
from tqdm import tqdm
from go1_gym.envs.navigation.ActorCritic import ActorCriticNetwork
from go1_gym.envs.navigation.PPOtrainer import PPOTrainer
from ml_logger import logger
from pathlib import Path
from go1_gym import MINI_GYM_ROOT_DIR
import glob
import os
from ..scripts.play import load_env

ACTION_MAPPING = {
    0: "move_forward",
    1: "move_backward",
    2: "turn_left",
    3: "turn_right",
    4: "stop",
    5: "move_forward_left",
    6: "move_forward_right",
    7: "move_backward_left",
    8: "move_backward_right"
}

state_dim = 5  # Example: position (2), orientation (1), goal position (2)
action_dim = 3  # Example: linear velocity (2), angular velocity (1)

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
env, low_level_policy = load_env(label, headless=False)

import numpy as np

def calculate_reward(robot_state, goal_position, wall_positions, wall_threshold):

    # Extract robot position from the state
    robot_position = robot_state['position']  # Adjust this based on how your state is defined

    # Reward for getting closer to the goal
    goal_distance = np.linalg.norm(np.array(robot_position) - np.array(goal_position))
    reward = -goal_distance  # Negative distance to the goal

    # Penalty for getting too close to walls
    for wall_position in wall_positions:
        wall_distance = np.linalg.norm(np.array(robot_position) - np.array(wall_position))
        if wall_distance < wall_threshold:
            reward -= 10  # Large penalty for being too close to a wall

    return reward

def map_action_to_velocities(action_index):
    """
    Map a discrete action index to velocity commands.
    :param action_index: Index of the action in the discrete action space.
    :return: (x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
    """
    # Define the velocity commands for each action
    # Adjust these values based on your robot's capabilities and the needs of your task
    if ACTION_MAPPING[action_index] == "move_forward":
        return (0.5, 0.0, 0.0)
    elif ACTION_MAPPING[action_index] == "move_backward":
        return (-0.5, 0.0, 0.0)
    elif ACTION_MAPPING[action_index] == "turn_left":
        return (0.0, 0.0, 0.5)
    elif ACTION_MAPPING[action_index] == "turn_right":
        return (0.0, 0.0, -0.5)
    # ... define other actions similarly
    else:
        return (0.0, 0.0, 0.0)  # Default to stop


def train_high_level_policy(num_episodes, max_steps_per_episode, robot_initial_position, goal_position, wall_positions, wall_threshold):
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        cumulative_reward = 0
        robot_current_position = robot_initial_position.copy()
        for step in range(max_steps_per_episode):
            
            # High-level policy decides the action
            with torch.no_grad():
                policy_output = high_level_policy.policy(torch.from_numpy(obs).float())
                action_index = np.argmax(policy_output.numpy())  # Choose the action with the highest probability

            # Map high-level action to velocity commands
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = map_action_to_velocities(action_index)

            # Set commands for low-level policy and execute
            # Here, integrate these commands with your environment or low-level policy
            actions = low_level_policy(obs, x_vel_cmd, y_vel_cmd, yaw_vel_cmd)
            next_obs, rew, done, info = env.step(actions)
            robot_current_position = env.get_robot_position()

            # Calculate reward and store transitions
            reward = calculate_reward(robot_current_position, goal_position, wall_positions, wall_threshold)
            cumulative_reward += reward
            ppo_trainer.store_transition(obs, action_index, reward, next_obs, done)

            obs = next_obs

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
max_steps_per_episode = 200
train_high_level_policy(num_episodes, max_steps_per_episode, robot_initial_position, goal_position, wall_positions, wall_threshold)
