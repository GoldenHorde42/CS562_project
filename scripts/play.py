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
from gym.wrappers import RecordVideo
import os
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
    Cfg.viewer.pos = [3, 0, 8]
    Cfg.viewer.lookat = [3., 1, 0.]

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


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    # label = "gait-conditioned-agility/pretrain-v0/train"
    label = "gait-conditioned-agility/2023-10-17/train"

    env, policy = load_env(label, headless=headless)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = env.render(mode="rgb_array")
    height, width, layers = frame.shape
    out = cv2.VideoWriter('new_output.mp4', fourcc, 20.0, (width, height))
    num_eval_steps = 500
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    # x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    commands_sequence = [
        (0.4, 0, 0.0),     # Forward
        (0, -0.3, 0.0),   # Left
        (-0.3, 0, 0.0),   # Backward
        (0, 0.5, 0.0),    # Right
        (0, 0, -0.5)   # Left rotation
    ]

    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    # measured_x_vels = np.zeros(num_eval_steps)
    # target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    measured_lin_vels = np.zeros((num_eval_steps, 3))
    target_lin_vels = np.zeros((num_eval_steps, 3))
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    cmd_idx = 0
    print(env.unwrapped.metadata)
    try:
        for i in tqdm(range(num_eval_steps)):
            if i % 100 == 0 and i > 0:
                cmd_idx += 1
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = commands_sequence[cmd_idx]
            with torch.no_grad():
                actions = policy(obs)
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
            obs, rew, done, info = env.step(actions)
            frame = env.render(mode="rgb_array")
            frame = frame[:, :, :3]
            out.write(frame)
            # measured_x_vels[i] = env.base_lin_vel[0, 0]
            measured_lin_vels[i] = env.base_lin_vel[0, :3].cpu()
            target_lin_vels[i] = [x_vel_cmd, y_vel_cmd, yaw_vel_cmd]
            joint_positions[i] = env.dof_pos[0, :].cpu()
    finally:
        out.release()   
   
    from matplotlib import pyplot as plt

    time_array = np.linspace(0, num_eval_steps * env.dt, num_eval_steps)

    directions = ["Forward-Backward", "Left-Right", "Rotation"]
    for idx, direction in enumerate(directions):
        plt.figure(figsize=(12, 5))
        plt.plot(time_array, measured_lin_vels[:, idx], color='black', linestyle="-", label="Measured")
        plt.plot(time_array, target_lin_vels[:, idx], color='black', linestyle="--", label="Desired")
        plt.title(f"{direction} Linear Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)" if idx != 2 else "Velocity (rad/s)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    # Plotting joint positions
    plt.figure(figsize=(12, 8))
    for j in range(12):
        plt.plot(time_array, joint_positions[:, j], label=f"Joint {j + 1}")
    plt.title("Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Position (rad)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
