"""G1 向け高さ・スクワット関連のユーティリティ。"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


# -----------------------------------------------------------------------------#
# 高さバッファとコマンド
# -----------------------------------------------------------------------------#

def ensure_height_buffers(env, center: float = 0.74, width: float = 0.0) -> None:
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
    env, env_ids, center: float = 0.74, width: float = 0.0, min_height: float = 0.24, max_height: float = 0.74
) -> None:
    """初期の高さコマンドを設定してサンプル。"""
    ensure_height_buffers(env, center=center, width=width)
    env._height_center.fill_(float(center))  # type: ignore[attr-defined]
    env._height_width.fill_(float(width))  # type: ignore[attr-defined]
    resample_height_command(env, None, min_height=min_height, max_height=max_height, center_default=center, width_default=width)


def resample_height_command(
    env,
    env_ids,
    min_height: float = 0.24,
    max_height: float = 0.74,
    center_default: float = 0.74,
    width_default: float = 0.0,
    squat_ratio: float = 1.0 / 3.0,
) -> None:
    """スクワット/歩行の混合サンプリングで高さコマンドを更新。"""
    ensure_height_buffers(env, center=center_default, width=width_default)
    if env_ids is None or env_ids == slice(None):
        env_ids_tensor = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids_tensor = torch.as_tensor(env_ids, device=env.device)
    n = env_ids_tensor.numel()
    if n == 0:
        return

    perm = torch.randperm(n, device=env.device)
    squat_count = int(n * float(squat_ratio))
    squat_ids = env_ids_tensor[perm[:squat_count]]
    walk_ids = env_ids_tensor[perm[squat_count:]]

    if squat_ids.numel() > 0:
        squat_height = torch.empty_like(squat_ids, dtype=torch.float32, device=env.device).uniform_(
            float(min_height), float(center_default)
        )
        env._height_command[squat_ids, 0] = torch.clamp(squat_height, min=min_height, max=max_height)  # type: ignore[attr-defined]
    if walk_ids.numel() > 0:
        walk_height = center_default + 0.02 * torch.randn_like(walk_ids, dtype=torch.float32, device=env.device)
        walk_height = torch.clamp(walk_height, min=min_height, max=max_height)
        env._height_command[walk_ids, 0] = walk_height  # type: ignore[attr-defined]

    env._height_center.fill_(float(center_default))  # type: ignore[attr-defined]
    env._height_width.fill_(float(width_default))  # type: ignore[attr-defined]


def height_command(env) -> torch.Tensor:
    """現在の高さコマンド [num_envs,1] を返す。"""
    ensure_height_buffers(env)
    return env._height_command  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------#
# コマンドスケーリング
# -----------------------------------------------------------------------------#

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
    z_min: float = 0.24,
    z_ref: float = 0.74,
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


# -----------------------------------------------------------------------------#
# 高さ追従・スクワット報酬
# -----------------------------------------------------------------------------#

def get_link_height(env, env_ids, asset_cfg: SceneEntityCfg, link_name: str) -> torch.Tensor:
    """指定リンクのワールド Z を返す。"""
    sim = env.scene[asset_cfg.name]
    body_idx = sim.body_names.index(link_name)
    ids = env_ids if env_ids is not None else slice(None)
    body_state = sim.data.body_state_w[ids, body_idx]
    return body_state[..., 2]


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


def track_torso_height_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    link_name: str = "torso_link",
):
    """torso 高さの指数追従。"""
    z_cmd = env._height_command[:, 0]  # type: ignore[attr-defined]
    h = get_link_height(env, None, asset_cfg, link_name=link_name)
    diff = h - z_cmd
    return torch.exp(-0.5 * (diff / std) ** 2)


def squat_knee_reward(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    knee_joint_names,
    h_std: float = 0.02,
    link_name: str = "torso_link",
):
    """HOMIE にならった膝報酬（簡略版）。"""
    z_cmd = env._height_command[:, 0]  # type: ignore[attr-defined]
    h = get_link_height(env, None, asset_cfg, link_name=link_name)
    h_err = h - z_cmd

    robot = env.scene[asset_cfg.name]
    knee_indices = [robot.joint_names.index(n) for n in knee_joint_names]
    q = robot.data.joint_pos[:, knee_indices]
    # そのステップの膝角度レンジで正規化（簡略化）
    jmin = q.min(dim=0, keepdim=True).values
    jmax = q.max(dim=0, keepdim=True).values
    q_norm = (q - jmin) / (jmax - jmin + 1e-6)

    too_high = (h_err > h_std).float().unsqueeze(-1)
    too_low = (h_err < -h_std).float().unsqueeze(-1)
    r_flex = q_norm
    r_extend = 1.0 - q_norm
    r = (too_high * r_flex + too_low * r_extend).mean(dim=-1)
    return r
