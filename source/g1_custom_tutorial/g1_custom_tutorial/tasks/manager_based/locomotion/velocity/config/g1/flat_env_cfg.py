# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # 親クラスの後処理初期化
        super().__post_init__()

        # 地形をフラットに変更
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # 高さスキャンなし
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # 地形カリキュラムなし
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # コマンド範囲
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # 親クラスの後処理初期化
        super().__post_init__()

        # 再生（デモ）用に小規模シーンに調整
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # デモではランダム化を無効化
        self.observations.policy.enable_corruption = False
        # ランダム外乱（プッシュ）を無効化
        self.events.base_external_force_torque = None
        self.events.push_robot = None
