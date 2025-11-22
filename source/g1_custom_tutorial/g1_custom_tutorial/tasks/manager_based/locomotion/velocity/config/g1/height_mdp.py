"""G1 向けの高さ関連ユーティリティを集約。

- 高さコマンドのバッファ管理（サンプリングは一様）
- Pelvis 高さの追従報酬
- 高さに応じた速度コマンドのスケーリング
- 高さ幅カリキュラムの拡大
"""

from __future__ import annotations

from typing import Sequence

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def ensure_height_buffers(env, center: float = 0.7, width: float = 0.0) -> None:
    """高さコマンド用バッファを確保。"""
    device = getattr(env, "device", "cpu")
    num_envs = getattr(env, "num_envs", 1)
    if not hasattr(env, "_height_command"):
        env._height_command = torch.full((num_envs, 1), float(center), device=device, dtype=torch.float32)  # type: ignore[attr-defined]
    if not hasattr(env, "_height_center"):
        env._height_center = torch.full((num_envs,), float(center), device=device, dtype=torch.float32)  # type: ignore[attr-defined]
    if not hasattr(env, "_height_width"):
        env._height_width = torch.full((num_envs,), float(width), device=device, dtype=torch.float32)  # type: ignore[attr-defined]


def init_height_center_width(
    env, env_ids, center: float = 0.7, width: float = 0.0, min_height: float = 0.1, max_height: float = 0.7
) -> None:
    """初期の高さコマンドを設定してサンプル。"""
    ensure_height_buffers(env, center=center, width=width)
    env._height_center.fill_(float(center))  # type: ignore[attr-defined]
    env._height_width.fill_(float(width))  # type: ignore[attr-defined]
    resample_height_command(env, None, min_height=min_height, max_height=max_height, center_default=center, width_default=width)


def resample_height_command(
    env,
    env_ids,
    min_height: float = 0.1,
    max_height: float = 0.7,
    center_default: float = 0.7,
    width_default: float = 0.0,
) -> None:
    """center±width から一様サンプルし、高さコマンドを更新。"""
    ensure_height_buffers(env, center=center_default, width=width_default)
    center = env._height_center  # type: ignore[attr-defined]
    width = env._height_width  # type: ignore[attr-defined]
    low = torch.clamp(center - width, min=min_height, max=max_height)
    high = torch.clamp(center + width, min=min_height, max=max_height)
    low, high = torch.minimum(low, high), torch.maximum(low, high)
    if env_ids is None or env_ids == slice(None):
        rand = torch.rand_like(center)
        samples = low + (high - low) * rand
        env._height_command[:, 0] = samples  # type: ignore[attr-defined]
    else:
        env_ids_tensor = torch.as_tensor(env_ids, device=center.device)
        rand = torch.rand_like(center[env_ids_tensor])
        samples = low[env_ids_tensor] + (high[env_ids_tensor] - low[env_ids_tensor]) * rand
        env._height_command[env_ids_tensor, 0] = samples  # type: ignore[attr-defined]


def height_command(env) -> torch.Tensor:
    """現在の高さコマンド [num_envs,1] を返す。"""
    ensure_height_buffers(env)
    return env._height_command  # type: ignore[attr-defined]


def _smooth_scale_from_height(z_cmd: torch.Tensor, z_min: float, z_ref: float, power: float) -> torch.Tensor:
    denom = max(z_ref - z_min, 1e-6)
    s = (z_cmd - z_min) / denom
    s = torch.clamp(s, 0.0, 1.0)
    if power != 1.0:
        s = torch.pow(s, power)
    return s


def height_scaled_velocity_commands(
    env,
    vel_cmd_name: str = "base_velocity",
    z_min: float = 0.1,
    z_ref: float = 0.7,
    p_lin: float = 1.2,
    p_ang: float = 1.0,
) -> torch.Tensor:
    """高さに応じて速度コマンドをスケーリング（低いほど抑制）。"""
    ensure_height_buffers(env)
    cmd = env.command_manager.get_command(vel_cmd_name)
    z = env._height_command[:, 0]  # type: ignore[attr-defined]
    s_lin = _smooth_scale_from_height(z, z_min, z_ref, p_lin).unsqueeze(1)
    s_ang = _smooth_scale_from_height(z, z_min, z_ref, p_ang).unsqueeze(1)
    out = cmd.clone()
    if out.shape[1] >= 2:
        out[:, 0:2] = out[:, 0:2] * s_lin
    if out.shape[1] >= 3:
        out[:, 2:3] = out[:, 2:3] * s_ang
    return out


def track_lin_vel_xy_yaw_frame_exp_height_scaled(
    env,
    vel_cmd: str,
    std: float,
    height_z_min: float,
    height_z_ref: float,
    p_lin: float = 1.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """高さスケール済みコマンドに対する並進速度追従。"""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    cmd_scaled = height_scaled_velocity_commands(
        env, vel_cmd_name=vel_cmd, z_min=height_z_min, z_ref=height_z_ref, p_lin=p_lin, p_ang=1.0
    )
    err2 = torch.sum(torch.square(cmd_scaled[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-err2 / (std**2))


def track_ang_vel_z_world_exp_height_scaled(
    env,
    vel_cmd: str,
    std: float,
    height_z_min: float,
    height_z_ref: float,
    p_ang: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """高さスケール済みコマンドに対するヨー角速度追従。"""
    asset = env.scene[asset_cfg.name]
    cmd_scaled = height_scaled_velocity_commands(
        env, vel_cmd_name=vel_cmd, z_min=height_z_min, z_ref=height_z_ref, p_lin=1.0, p_ang=p_ang
    )
    err2 = torch.square(cmd_scaled[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-err2 / (std**2))


def track_pelvis_height_exp(
    env,
    command_name: str,
    std: float,
    pelvis_name: str = "pelvis",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """pelvis 高さの指数追従。"""
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(pelvis_name)
    if len(body_ids) > 0:
        base_z = asset.data.body_pos_w[:, int(body_ids[0]), 2] - env.scene.env_origins[:, 2]
    else:
        base_z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    z_cmd = env._height_command[:, 0]
    err2 = torch.square(base_z - z_cmd)
    return torch.exp(-err2 / (std**2))


def expand_height_sampling(
    env,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    center: float = 0.7,
    width_min: float = 0.0,
    width_max: float = 0.6,
    widen_step: float = 0.05,
    success_scale: float = 0.5,
    min_height: float = 0.1,
    max_height: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """高さ幅を徐々に拡大（成功環境のみ）。"""
    ensure_height_buffers(env, center=center, width=width_min)
    width = env._height_width  # type: ignore[attr-defined]
    asset = env.scene[asset_cfg.name]
    dist = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    cmd_scaled = height_scaled_velocity_commands(
        env, vel_cmd_name=command_name, z_min=min_height, z_ref=max_height, p_lin=1.2, p_ang=1.0
    )
    vxy = torch.norm(cmd_scaled[env_ids, :2], dim=1)
    expected = vxy * env.max_episode_length_s * float(success_scale)
    success = dist >= expected
    new_width = width.clone()
    if len(env_ids) > 0:
        new_width[env_ids] = torch.where(
            success, torch.clamp(width[env_ids] + widen_step, min=width_min, max=width_max), width[env_ids]
        )
    env._height_width = new_width  # type: ignore[attr-defined]
    env._height_center.fill_(float(center))  # type: ignore[attr-defined]
    resample_height_command(
        env, env_ids, min_height=min_height, max_height=max_height, center_default=center, width_default=width_min
    )
    return torch.mean(env._height_width)  # type: ignore[attr-defined]
