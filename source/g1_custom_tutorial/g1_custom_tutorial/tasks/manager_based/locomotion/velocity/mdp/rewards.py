# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
学習環境の報酬を定義するための共通関数。

これらの関数は :class:`isaaclab.managers.RewardTermCfg` に渡して、
報酬関数およびそのパラメータを指定できます。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _log_reward_term(env: "ManagerBasedRLEnv", key: str, value: torch.Tensor) -> None:
    """Utility to append per-step reward diagnostics to env.extras['log'].

    - Creates the extras/log dicts if missing.
    - Accepts tensors shaped [num_envs] and stores them directly so the
      downstream runner (RSL-RL / RL-Games) can aggregate means.
    """
    # ensure extras/log structure exists
    if not hasattr(env, "extras") or env.extras is None:
        try:
            env.extras = {}
        except Exception:
            # if env doesn't allow attribute set, just return silently
            return
    log = env.extras.get("log")
    if log is None or not isinstance(log, dict):
        log = {}
        env.extras["log"] = log
    # store tensor on CPU-friendly dtype as-is; the runner will average it
    log[key] = value


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
    足の空中時間を報いる（L2カーネル）。

    足が一定しきい値より長く空中にあるステップを奨励します。これにより足上げ歩行を促進します。
    報酬は各足の空中時間の総和として計算されます。

    コマンドが小さい（歩行を求められていない）場合は報酬は0になります。
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    二足歩行向けの空中時間報酬。

    指定しきい値までの空中時間を報酬とし、さらに片脚支持（片方だけ接地）を維持することを奨励します。
    コマンドが小さい場合は報酬は0になります。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    足のスリップをペナルティ。

    接地中の足の水平速度ノルムに基づいてペナルティを与えます（接触時のみ適用）。
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    重力整列フレーム（Yawのみ）での並進速度（xy）指令の追従度を指数カーネルで評価。
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    ワールド座標系におけるヨー角速度指令の追従度を指数カーネルで評価。
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    コマンドがほぼゼロのとき、デフォルト関節位置からのずれをペナルティ。
    """
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


# -----------------------------------------------------------------------------
# Height tracking + height-aware speed tracking
# -----------------------------------------------------------------------------

def track_base_height_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """高さコマンドに追従するための指数型報酬。

    r = exp(- (z - z_cmd)^2 / std^2)
    """
    asset = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    # Read commanded height from env buffer (set by mdp.commands helpers)
    z_cmd = env._height_command[:, 0]
    err2 = torch.square(base_z - z_cmd)
    reward = torch.exp(-err2 / (std**2))
    # log instantaneous per-env reward and related values for diagnostics
    _log_reward_term(env, "Rewards/track_base_height_exp", reward)
    _log_reward_term(env, "Commands/base_height", z_cmd)
    _log_reward_term(env, "State/base_height", base_z)
    return reward


def track_lin_vel_xy_yaw_frame_exp_height_scaled(
    env: "ManagerBasedRLEnv",
    vel_cmd: str,
    std: float,
    height_z_min: float,
    height_z_ref: float,
    p_lin: float = 1.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ヨー整列フレームでの xy 並進速度を、高さに応じてスケールしたコマンドに対して追従。

    低い高さ姿勢での非現実的な目標を避けるため、速度コマンドは高さコマンドの関数として
    事前にスケーリングされます。
    """
    from .commands import height_scaled_velocity_commands  # local import

    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    cmd_scaled = height_scaled_velocity_commands(
        env, vel_cmd_name=vel_cmd, z_min=height_z_min, z_ref=height_z_ref, p_lin=p_lin, p_ang=1.0
    )
    err2 = torch.sum(torch.square(cmd_scaled[:, :2] - vel_yaw[:, :2]), dim=1)
    reward = torch.exp(-err2 / (std**2))
    _log_reward_term(env, "Rewards/track_lin_vel_xy_yaw_frame_exp_height_scaled", reward)
    return reward


def track_ang_vel_z_world_exp_height_scaled(
    env: "ManagerBasedRLEnv",
    vel_cmd: str,
    std: float,
    height_z_min: float,
    height_z_ref: float,
    p_ang: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """高さに応じてスケールした角速度コマンドに対する、ワールド座標系ヨー角速度の追従度。"""
    from .commands import height_scaled_velocity_commands  # local import

    asset = env.scene[asset_cfg.name]
    cmd_scaled = height_scaled_velocity_commands(
        env, vel_cmd_name=vel_cmd, z_min=height_z_min, z_ref=height_z_ref, p_lin=1.0, p_ang=p_ang
    )
    err2 = torch.square(cmd_scaled[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-err2 / (std**2))
    _log_reward_term(env, "Rewards/track_ang_vel_z_world_exp_height_scaled", reward)
    return reward
