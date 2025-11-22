"""G1 向けの高さカリキュラム。"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from . import custom_rewards


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
    """成功環境だけ高さ幅を段階的に広げる。"""
    custom_rewards.ensure_height_buffers(env, center=center, width=width_min)
    width = env._height_width  # type: ignore[attr-defined]
    asset: Articulation = env.scene[asset_cfg.name]
    dist = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    cmd_scaled = custom_rewards.height_scaled_velocity_commands(
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
    custom_rewards.resample_height_command(
        env, env_ids, min_height=min_height, max_height=max_height, center_default=center, width_default=width_min
    )
    return torch.mean(env._height_width)  # type: ignore[attr-defined]
