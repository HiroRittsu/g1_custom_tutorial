# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

from ... import mdp
from ...velocity_env_cfg import EventCfg as BaseEventCfg
from ...velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg


@configclass
class G1Rewards(RewardsCfg):
    """G1 用の報酬セット"""

    # エピソード終了に対する大きなペナルティ
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # 高さスケール済みの並進速度追従
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_height_scaled,
        weight=1.0,
        params={
            "vel_cmd": "base_velocity",
            "std": math.sqrt(0.25),
            "height_z_min": 0.70,
            "height_z_ref": 0.74,
            "p_lin": 1.2,
        },
    )
    # 高さスケール済みの角速度追従
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp_height_scaled,
        weight=0.5,
        params={
            "vel_cmd": "base_velocity",
            "std": math.sqrt(0.25),
            "height_z_min": 0.70,
            "height_z_ref": 0.74,
            "p_ang": 1.0,
        },
    )
    # 高さコマンドそのものへの追従
    track_base_height_exp = RewTerm(
        func=mdp.track_base_height_exp,
        weight=0.4,
        params={"command_name": "base_height", "std": 0.04},
    )
    # 二足歩行向けの空中時間ボーナス
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    # 足のスリップ抑制
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # 足首の関節可動域超過をペナルティ
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # 歩行に必須でない関節のデフォルト姿勢からの逸脱を抑制
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    # 上半身（肩・肘）の姿勢逸脱を抑制
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    # 指の姿勢逸脱を抑制
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    # 胴体姿勢の逸脱を抑制
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )


@configclass
class EventCfgPlay(BaseEventCfg):
    """Play用のイベント構成。デバッグ矢印を追加し、更新を高速化。"""

    # デバッグ矢印（高さ同期）
    height_arrow_spawn = EventTerm(func=mdp.spawn_height_velocity_arrows, mode="startup")
    height_arrow_update = EventTerm(
        func=mdp.update_height_velocity_arrows, mode="interval", interval_range_s=(1.0 / 30.0, 1.0 / 30.0)
    )


@configclass
class G1CustomFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """rough/flat の内容を統合したカスタムフラット環境。"""

    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        super().__post_init__()

        # シーン構成
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # ランダム化
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # 平面地形に変更
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # 報酬調整
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
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

        # 終了条件
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


class G1CustomFlatEnvCfg_PLAY(G1CustomFlatEnvCfg):
    # Play ではイベント構成を差し替え（矢印の生成・更新を追加）
    events: EventCfgPlay = EventCfgPlay()

    def __post_init__(self) -> None:
        super().__post_init__()

        # 再生（デモ）用に小規模シーンに調整
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # デモではランダム化を無効化
        self.observations.policy.enable_corruption = False
        # ランダム外乱（プッシュ）を無効化
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 可視的に「高さ方向」にも変化が出るよう、高さコマンドを有効化して定期リサンプル
        # - 既定（学習用）は幅=0.0 で一定高さのため、デモでは幅を持たせて変化させる
        if getattr(self.events, "height_cmd_init", None) is not None:
            # 中央高さは据え置き、幅のみ付与
            self.events.height_cmd_init.params["center"] = 0.50
            # 変化を大きく（可視化重視）。例: ±50cm
            self.events.height_cmd_init.params["width"] = 0.5
        if getattr(self.events, "height_cmd_interval", None) is not None:
            # 変化を見やすく 2 秒ごとに再サンプル
            self.events.height_cmd_interval.interval_range_s = (2.0, 2.0)
        # デモではカリキュラムで幅が勝手に変わらないように固定
        if getattr(self.curriculum, "height_sampling", None) is not None:
            self.curriculum.height_sampling = None

        # 念のためここでもイベントを明示登録（上書き可能）
        self.events.height_arrow_spawn = EventTerm(func=mdp.spawn_height_velocity_arrows, mode="startup")
        self.events.height_arrow_update = EventTerm(
            func=mdp.update_height_velocity_arrows, mode="interval", interval_range_s=(1.0 / 30.0, 1.0 / 30.0)
        )
