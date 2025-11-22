"""不整地上での Unitree G1 速度追従用の Direct ワークフロー設定。

この設定は、上流の Manager ベース G1 ロコモーション速度タスク（地形・コマンド・観測・
報酬・終了条件）を反映し、Direct 環境でも同等の挙動となるようにします。
"""

from __future__ import annotations

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets import G1_MINIMAL_CFG


@configclass
class G1CustomTutorialEnvCfg(DirectRLEnvCfg):
    # 環境ステップのタイミング
    decimation = 4
    episode_length_s = 20.0

    # スペース（ロボット USD の解析とセンサ初期化後にランタイムで設定）
    action_space = 0
    observation_space = 0
    state_space = 0

    # 物理シミュレーション
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)

    # ロボット
    robot_cfg: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # シーン
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # ポリシー／アクション設定
    action_scale = 0.5  # 既定姿勢に対するラジアンオフセット
    action_clip = 1.0

    # コマンドのサンプリング（ボディのヨー整列フレームで x, y, yaw）
    command_time_range = (10.0, 10.0)
    command_ranges = {
        "lin_vel_x": (0.0, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.0, 1.0),
    }
    command_smoothing = 0.0

    # 観測スケーリング
    obs_scales = {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "joint_vel": 0.05,
    }

    # 報酬／ペナルティの重み（Manager ベース G1 不整地プリセットに合わせた値）
    reward_scales = {
        "track_lin_vel_xy": 1.0,
        "track_ang_vel_z": 2.0,
        "lin_vel_z": 0.0,
        "ang_vel_xy": -0.05,
        "flat_orientation": -1.0,
        "action_rate": -0.005,
        "torques": -1.5e-7,
        "joint_acc": -1.25e-7,
        "feet_air_time": 0.25,
        "feet_slide": -0.1,
        "joint_pos_limits": -1.0,
        "joint_deviation_hip": -0.1,
        "joint_deviation_arms": -0.1,
        "joint_deviation_fingers": -0.05,
        "joint_deviation_torso": -0.1,
        "termination": -200.0,
    }
    lin_vel_tracking_std = math.sqrt(0.25)
    ang_vel_tracking_std = math.sqrt(0.25)
    feet_air_time_threshold = 0.4
    base_contact_threshold = 1.0
    height_scan_offset = 0.5

    # リセット設定（G1 不整地環境に準拠：速度ゼロ、平面内ランダム姿勢）
    reset_pos_range_xy = 0.5
    reset_yaw_range = math.pi
    init_base_height = 0.74  # refined at runtime from asset default

    # 報酬／終了条件で使用するボディ／関節名パターン
    ankle_body_pattern = ".*_ankle_roll_link"
    torso_body_name = "torso_link"
    ankles_joints_patterns = [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
    hip_joints_patterns = [".*_hip_yaw_joint", ".*_hip_roll_joint"]
    arms_joints_patterns = [
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_pitch_joint",
        ".*_elbow_roll_joint",
    ]
    finger_joints_patterns = [
        ".*_five_joint",
        ".*_three_joint",
        ".*_six_joint",
        ".*_four_joint",
        ".*_zero_joint",
        ".*_one_joint",
        ".*_two_joint",
    ]
