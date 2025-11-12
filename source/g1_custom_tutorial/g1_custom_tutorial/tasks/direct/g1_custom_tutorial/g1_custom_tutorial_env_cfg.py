# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets import G1_MINIMAL_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class G1CustomTutorialEnvCfg(DirectRLEnvCfg):
    """Direct workflow configuration for Unitree G1 velocity tracking on flat terrain."""

    # env timing
    decimation = 4
    episode_length_s = 20.0
    # spaces (filled at runtime once the robot USD is parsed)
    action_space = 0
    observation_space = 0
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.5, replicate_physics=True)

    # policy/action parameters
    action_scale = 0.5  # radian offset applied on top of default pose
    action_clip = 1.0

    # observation scaling
    obs_scales = {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "joint_vel": 0.05,
    }

    # command sampling (x, y, yaw in body-yaw frame)
    command_time_range = (1.0, 3.0)
    command_ranges = {
        "lin_vel_x": (0.0, 1.2),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-1.0, 1.0),
    }
    command_smoothing = 0.2

    # reward/penalty scales (all multiplied inside the environment)
    reward_scales = {
        "track_lin_vel_xy": 2.0,
        "track_ang_vel_z": 1.5,
        "lin_vel_z": -1.0,
        "flat_orientation": -2.0,
        "action_rate": -0.01,
        "torques": -3.0e-5,
        "joint_vel": -0.05,
        "joint_pos_limits": -0.2,
        "joint_acc": -1.0e-4,
        "alive": 1.0,
        "termination": -5.0,
    }
    lin_vel_tracking_std = 0.5
    ang_vel_tracking_std = 0.5

    # reset noise
    reset_noise = {
        "pos": 0.5,
        "yaw": 3.14,
        "roll": 0.05,
        "pitch": 0.05,
        "lin_vel": 0.05,
        "ang_vel": 0.05,
        "joint_pos": 0.2,
        "joint_vel": 0.2,
    }
    init_base_height = 0.93

    # termination thresholds
    termination_height = 0.6
    max_tilt = 0.8  # [rad]
