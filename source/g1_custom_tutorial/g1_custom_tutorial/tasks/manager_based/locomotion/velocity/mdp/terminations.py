# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
特定の終了条件（Done）を有効化するための共通関数。

これらの関数は :class:`isaaclab.managers.TerminationTermCfg` に渡すことで、
該当する終了条件を有効にできます。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """
    エージェントが地形の端に近づきすぎたときにエピソードを終了します。

    地形の端からの距離がしきい値（バッファ）より小さい場合に終了条件を発火します。
    距離は地形サイズとバッファから算出されます。
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # 平面地形は無限に広いとみなすため終了しない
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # サブ地形のサイズを取得
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # マップ全体のサイズを計算
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # 型ヒントのために使用するデータを取り出す
        asset: RigidObject = env.scene[asset_cfg.name]

        # 境界外かどうかを判定
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
