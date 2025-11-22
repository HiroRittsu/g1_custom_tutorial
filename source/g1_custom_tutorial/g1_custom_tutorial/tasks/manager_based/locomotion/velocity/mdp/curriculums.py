# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""学習環境のカリキュラムを構成するための共通関数。

これらの関数は :class:`isaaclab.managers.CurriculumTermCfg` に渡すことで、
対応するカリキュラム制御を有効にできます。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    所望速度コマンド下での移動距離に基づくカリキュラム制御。

    十分に歩けた環境は難易度（地形レベル）を上げ、必要距離の半分未満しか歩けない環境は
    難易度を下げます。

    .. note::
        本項目は地形タイプが ``generator`` の場合のみ使用できます。
        地形タイプの詳細は :class:`isaaclab.terrains.TerrainImporter` を参照してください。

    戻り値:
        与えられた環境IDに対する平均地形レベル。
    """
    # 型ヒントのために使用するデータを取り出す
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # 走破距離を計算
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # 十分に歩けたロボットは難しい地形へ
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # 必要距離の半分未満は簡単な地形へ
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # 地形レベルを更新
    terrain.update_env_origins(env_ids, move_up, move_down)
    # 平均地形レベルを返す
    return torch.mean(terrain.terrain_levels.float())


def expand_height_sampling(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    center: float = 0.74,
    width_min: float = 0.0,
    width_max: float = 0.08,
    widen_step: float = 0.01,
    success_scale: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """カリキュラム: 成功した環境について高さサンプリング幅を拡大。

    成功判定: 走行距離 >= ||v_cmd_xy|| * エピソード時間 * success_scale。
    成功した環境のみ、幅を width_max まで段階的に増加させ、その他は現状維持とします。
    center はブロードキャスト（簡便のため一定）。
    """
    from .commands import _ensure_height_buffers  # internal util

    _ensure_height_buffers(env, center=center, width=width_min)
    # Read current width per-env
    width = env._height_width  # type: ignore[attr-defined]
    # Compute success for given env_ids
    asset: Articulation = env.scene[asset_cfg.name]
    # distance traveled in XY from origin
    dist = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # expected distance based on commanded velocity magnitude
    # use height-scaled command magnitude for expected distance
    from .commands import height_scaled_velocity_commands
    cmd_scaled = height_scaled_velocity_commands(env, vel_cmd_name=command_name)
    vxy = torch.norm(cmd_scaled[env_ids, :2], dim=1)
    expected = vxy * env.max_episode_length_s * float(success_scale)
    success = dist >= expected
    # Update widths for successful envs
    new_width = width.clone()
    if len(env_ids) > 0:
        new_width[env_ids] = torch.where(success, torch.clamp(width[env_ids] + widen_step, min=width_min, max=width_max), width[env_ids])
    env._height_width = new_width  # type: ignore[attr-defined]
    # Broadcast center (kept constant); this simplifies observation consistency
    env._height_center.fill_(float(center))  # type: ignore[attr-defined]
    # Return current mean width for logging
    return torch.mean(env._height_width)  # type: ignore[attr-defined]
