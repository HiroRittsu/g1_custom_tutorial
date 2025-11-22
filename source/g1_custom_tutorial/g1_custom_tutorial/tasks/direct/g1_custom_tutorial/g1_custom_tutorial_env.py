"""Manager ベースの G1 ロコモーションタスクを Direct 方式で書き直した RL 環境。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from .g1_custom_tutorial_env_cfg import G1CustomTutorialEnvCfg


class G1CustomTutorialEnv(DirectRLEnv):
    """Unitree G1 を生成し、Manager ベースの G1 速度タスクを反映する Direct RL 環境。"""

    cfg: G1CustomTutorialEnvCfg

    def __init__(self, cfg: G1CustomTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 関節メタデータ
        self._dof_ids, _ = self.robot.find_joints(".*")
        self.num_dof = len(self._dof_ids)
        self.nominal_joint_pos = self.robot.data.default_joint_pos.clone()
        self.nominal_root_state = self.robot.data.default_root_state.clone()
        self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[0, self._dof_ids, 0]
        self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[0, self._dof_ids, 1]

        # ボディ／関節のインデックス群（アセット空間／センサ空間）
        # アセット（Articulation）側の ID 群
        self.asset_ankle_body_ids, _ = self.robot.find_bodies(self.cfg.ankle_body_pattern)
        asset_torso_ids, _ = self.robot.find_bodies(self.cfg.torso_body_name)
        self.asset_torso_body_id = asset_torso_ids[0] if len(asset_torso_ids) > 0 else 0
        # センサ（ContactSensor）側の ID 群
        self.sensor_ankle_body_ids, _ = self.contact_sensor.find_bodies(self.cfg.ankle_body_pattern)
        sensor_torso_ids, _ = self.contact_sensor.find_bodies(self.cfg.torso_body_name)
        self.sensor_torso_body_id = sensor_torso_ids[0] if len(sensor_torso_ids) > 0 else 0
        self.ankle_joint_ids, _ = self.robot.find_joints(self.cfg.ankles_joints_patterns)
        self.hip_joint_ids, _ = self.robot.find_joints(self.cfg.hip_joints_patterns)
        self.arm_joint_ids, _ = self.robot.find_joints(self.cfg.arms_joints_patterns)
        self.finger_joint_ids, _ = self.robot.find_joints(self.cfg.finger_joints_patterns)

        # 各ステップで再利用するバッファ
        self.actions = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.action_rate = torch.zeros_like(self.actions)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_time_left = torch.zeros(self.num_envs, device=self.device)

        self.base_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel_b = torch.zeros_like(self.base_lin_vel_b)
        self.projected_gravity = torch.zeros_like(self.base_lin_vel_b)
        self.base_lin_vel_yaw = torch.zeros_like(self.base_lin_vel_b)
        self.base_lin_vel_w = torch.zeros_like(self.base_lin_vel_b)
        self.base_ang_vel_w = torch.zeros_like(self.base_lin_vel_b)
        self.base_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_height = torch.zeros(self.num_envs, device=self.device)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.dof_acc = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.joint_pos_err = torch.zeros_like(self.actions)
        self._joint_pos_normalized = torch.zeros_like(self.actions)

        self.lin_vel_tracking_std = torch.tensor(self.cfg.lin_vel_tracking_std, device=self.device)
        self.ang_vel_tracking_std = torch.tensor(self.cfg.ang_vel_tracking_std, device=self.device)

    def _setup_scene(self):
        # 地形（不整地ジェネレータ）— env の原点を割り当て、/World/ground にメッシュを配置
        terrain_vis = sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        )
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            num_envs=self.cfg.scene.num_envs,
            env_spacing=self.cfg.scene.env_spacing,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=terrain_vis,
            debug_vis=False,
        )
        self.terrain = TerrainImporter(terrain_cfg)
        # シーンに地形を公開して、env_origins を地形から取得できるようにする
        self.scene._terrain = self.terrain  # type: ignore[attr-defined]

        # ロボット
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 複製・レプリケーション
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # センサ（高さスキャン／接触）
        self.height_scanner = RayCaster(
            RayCasterCfg(
                prim_path="/World/envs/env_.*/Robot/torso_link",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
                update_period=self.cfg.decimation * self.step_dt,
            )
        )
        self.contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*",
                history_length=3,
                track_air_time=True,
                update_period=self.step_dt,
            )
        )

        # ロボット構築後に各スペース次元を更新（センサのレイ本数は Play 後に取得可能）
        num_dof = self._infer_robot_dof_count()
        self.cfg.action_space = num_dof

        # 照明
        sky_light = sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        sky_light.func("/World/skyLight", sky_light)

        # 注意: ロボットの既定 Root 状態はシミュレーション開始まで取得不可。cfg.init_base_height を維持

    """
    コアとなるシミュレーション用フック。
    """

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        if actions is None:
            actions = torch.zeros_like(self.actions)
        self.actions = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
        self.action_rate = self.actions - self.prev_actions

        # コマンドのリサンプリングスケジューリング
        self.command_time_left -= self.step_dt
        env_ids = torch.nonzero(self.command_time_left <= 0.0, as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self._resample_commands(env_ids)

    def _apply_action(self) -> None:
        target = self.nominal_joint_pos + self.actions * self.cfg.action_scale
        lower = self.joint_lower_limits.unsqueeze(0)
        upper = self.joint_upper_limits.unsqueeze(0)
        target = torch.clamp(target, lower, upper)
        self.robot.set_joint_position_target(target, joint_ids=self._dof_ids)

    def _get_observations(self) -> dict:
        self._update_task_tensors()
        # 高さスキャン: pos_z - hit_z - オフセット
        hs = self.height_scanner.data
        height_scan = hs.pos_w[:, 2].unsqueeze(1) - hs.ray_hits_w[..., 2] - self.cfg.height_scan_offset

        scaled_lin_vel = self.base_lin_vel_b * self.cfg.obs_scales["lin_vel"]
        scaled_ang_vel = self.base_ang_vel_b * self.cfg.obs_scales["ang_vel"]
        scaled_joint_vel = self.dof_vel * self.cfg.obs_scales["joint_vel"]
        obs = torch.cat(
            (
                scaled_lin_vel,
                scaled_ang_vel,
                self.projected_gravity,
                self.commands,
                self.joint_pos_err,
                scaled_joint_vel,
                self.prev_actions,
                height_scan,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 追従系の報酬
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel_yaw[:, :2]), dim=-1)
        rew_lin_vel = torch.exp(-lin_vel_error / (self.lin_vel_tracking_std**2))
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel_w[:, 2])
        rew_ang_vel = torch.exp(-ang_vel_error / (self.ang_vel_tracking_std**2))

        # ペナルティ項目
        rew_lin_vel_z = torch.abs(self.base_lin_vel_yaw[:, 2])
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel_w[:, :2]), dim=-1)
        rew_flat_orientation = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        rew_action_rate = torch.sum(torch.square(self.action_rate), dim=-1)
        # トルク／加速度のペナルティは歩行用関節群に限定（Manager ベース設定に準拠）
        self.knee_joint_ids, _ = self.robot.find_joints([".*_knee_joint"]) if not hasattr(self, "knee_joint_ids") else (self.knee_joint_ids, None)
        torque_ids = list({*self.hip_joint_ids, *self.knee_joint_ids, *self.ankle_joint_ids})
        acc_ids = list({*self.hip_joint_ids, *self.knee_joint_ids})
        if len(torque_ids) > 0:
            rew_torques = torch.sum(torch.square(self.torques[:, torque_ids]), dim=-1)
        else:
            rew_torques = torch.sum(torch.square(self.torques), dim=-1)
        if len(acc_ids) > 0:
            rew_joint_acc = torch.sum(torch.square(self.dof_acc[:, acc_ids]), dim=-1)
        else:
            rew_joint_acc = torch.sum(torch.square(self.dof_acc), dim=-1)

        # ペナルティ用の関節グループ
        # 関節可動域（足首のみ）
        ankles_mask = torch.zeros(self.num_dof, dtype=torch.bool, device=self.device)
        ankles_mask[self.ankle_joint_ids] = True
        joint_norm = torch.where(
            ankles_mask.unsqueeze(0),
            torch.abs(self._joint_pos_normalized) - 1.0,
            torch.zeros_like(self._joint_pos_normalized),
        )
        rew_joint_limits = torch.sum(torch.clamp(joint_norm, min=0.0), dim=-1)

        # 関節群ごとの L1 ずれ
        def group_deviation(ids: list[int]) -> torch.Tensor:
            if len(ids) == 0:
                return torch.zeros(self.num_envs, device=self.device)
            err = torch.abs(self.joint_pos_err[:, ids])
            return torch.sum(err, dim=-1)

        rew_joint_dev_hip = group_deviation(self.hip_joint_ids)
        rew_joint_dev_arms = group_deviation(self.arm_joint_ids)
        rew_joint_dev_fingers = group_deviation(self.finger_joint_ids)
        rew_joint_dev_torso = group_deviation(self.robot.find_joints("torso_joint")[0])

        # 足の空中時間（2脚向けの正の報酬）
        cs = self.contact_sensor
        air_time = cs.data.current_air_time[:, self.sensor_ankle_body_ids]
        contact_time = cs.data.current_contact_time[:, self.sensor_ankle_body_ids]
        in_contact = contact_time > 0.0
        in_mode_time = torch.where(in_contact, contact_time, air_time)
        single_stance = torch.sum(in_contact.int(), dim=1) == 1
        rew_feet_air_time = torch.min(
            torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)), dim=1
        )[0]
        rew_feet_air_time = torch.clamp(rew_feet_air_time, max=self.cfg.feet_air_time_threshold)
        # no reward for zero command
        rew_feet_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.1

        # 足のスリップ（接地時のペナルティ）
        forces_hist = cs.data.net_forces_w_history
        if forces_hist is None:
            # 擬似的な時間次元を付け、現在の接触力を代用
            forces_hist = cs.data.net_forces_w.unsqueeze(1)
        contacts = (forces_hist[:, :, self.sensor_ankle_body_ids, :].norm(dim=-1).max(dim=1)[0] > self.cfg.base_contact_threshold)
        foot_lin_vel = self.robot.data.body_lin_vel_w[:, self.asset_ankle_body_ids, :2].norm(dim=-1)
        rew_feet_slide = torch.sum(foot_lin_vel * contacts, dim=1)

        # 終了ペナルティ（胴体の不正接触）
        forces_hist = cs.data.net_forces_w_history
        if forces_hist is None:
            forces_hist = cs.data.net_forces_w.unsqueeze(1)
        torso_contact = (
            forces_hist[:, :, self.sensor_torso_body_id, :].norm(dim=-1).max(dim=1)[0] > self.cfg.base_contact_threshold
        )
        rew_termination = torso_contact.float()

        rewards = (
            self.cfg.reward_scales["track_lin_vel_xy"] * rew_lin_vel
            + self.cfg.reward_scales["track_ang_vel_z"] * rew_ang_vel
            + self.cfg.reward_scales["lin_vel_z"] * rew_lin_vel_z
            + self.cfg.reward_scales["ang_vel_xy"] * rew_ang_vel_xy
            + self.cfg.reward_scales["flat_orientation"] * rew_flat_orientation
            + self.cfg.reward_scales["action_rate"] * rew_action_rate
            + self.cfg.reward_scales["torques"] * rew_torques
            + self.cfg.reward_scales["joint_acc"] * rew_joint_acc
            + self.cfg.reward_scales["feet_air_time"] * rew_feet_air_time
            + self.cfg.reward_scales["feet_slide"] * rew_feet_slide
            + self.cfg.reward_scales["joint_pos_limits"] * rew_joint_limits
            + self.cfg.reward_scales["joint_deviation_hip"] * rew_joint_dev_hip
            + self.cfg.reward_scales["joint_deviation_arms"] * rew_joint_dev_arms
            + self.cfg.reward_scales["joint_deviation_fingers"] * rew_joint_dev_fingers
            + self.cfg.reward_scales["joint_deviation_torso"] * rew_joint_dev_torso
            + self.cfg.reward_scales["termination"] * rew_termination
        )
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # illegal torso contact
        cs = self.contact_sensor
        forces_hist = cs.data.net_forces_w_history
        if forces_hist is None:
            forces_hist = cs.data.net_forces_w.unsqueeze(1)
        torso_contact = (
            forces_hist[:, :, self.sensor_torso_body_id, :].norm(dim=-1).max(dim=1)[0] > self.cfg.base_contact_threshold
        )
        return torso_contact, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_tensor = self.robot._ALL_INDICES.to(device=self.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids_list = env_ids_tensor.cpu().tolist()
        super()._reset_idx(env_ids_list)
        self.robot.reset(env_ids_tensor)  # type: ignore[arg-type]

        # joint state (set to default, zero vel)
        joint_pos = self.nominal_joint_pos[env_ids_tensor].clone()
        joint_vel = torch.zeros_like(joint_pos)

        # root pose (random planar position and yaw, zero velocities)
        root_state = self.nominal_root_state[env_ids_tensor].clone()
        root_state[:, :3] = self.scene.env_origins[env_ids_tensor]
        root_state[:, 2] += self.cfg.init_base_height
        root_state[:, 0] += sample_uniform(
            -self.cfg.reset_pos_range_xy, self.cfg.reset_pos_range_xy, (len(env_ids_list),), str(self.device)
        )
        root_state[:, 1] += sample_uniform(
            -self.cfg.reset_pos_range_xy, self.cfg.reset_pos_range_xy, (len(env_ids_list),), str(self.device)
        )
        yaw = sample_uniform(
            -self.cfg.reset_yaw_range, self.cfg.reset_yaw_range, (len(env_ids_list),), str(self.device)
        )
        quat_noise = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        root_state[:, 3:7] = math_utils.quat_mul(self.nominal_root_state[env_ids_tensor, 3:7], quat_noise)
        root_state[:, 7:13] = 0.0

        # 状態を反映
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids_tensor)  # type: ignore[arg-type]
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids_tensor)  # type: ignore[arg-type]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)  # type: ignore[arg-type]

        # バッファをリセット
        self.actions[env_ids_tensor] = 0.0
        self.prev_actions[env_ids_tensor] = 0.0
        self.action_rate[env_ids_tensor] = 0.0
        self.command_time_left[env_ids_tensor] = 0.0
        self._resample_commands(env_ids_tensor)
    """
    ヘルパー。
    """

    def _update_task_tensors(self):
        self.base_lin_vel_b = self.robot.data.root_lin_vel_b
        self.base_ang_vel_b = self.robot.data.root_ang_vel_b
        self.base_lin_vel_w = self.robot.data.root_lin_vel_w
        self.base_ang_vel_w = self.robot.data.root_ang_vel_w
        self.projected_gravity = self.robot.data.projected_gravity_b

        quat = self.robot.data.root_quat_w
        self.base_euler = torch.stack(math_utils.euler_xyz_from_quat(quat), dim=-1)
        self.base_height = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

        yaw_quat = math_utils.yaw_quat(quat)
        self.base_lin_vel_yaw = math_utils.quat_apply_inverse(yaw_quat, self.base_lin_vel_w)

        self.dof_pos = self.robot.data.joint_pos[:, self._dof_ids]
        self.dof_vel = self.robot.data.joint_vel[:, self._dof_ids]
        self.dof_acc = self.robot.data.joint_acc[:, self._dof_ids]
        self.torques = self.robot.data.applied_torque[:, self._dof_ids]
        self.joint_pos_err = self.dof_pos - self.nominal_joint_pos[:, self._dof_ids]
        self._joint_pos_normalized = math_utils.scale_transform(
            self.dof_pos,
            self.joint_lower_limits.unsqueeze(0),
            self.joint_upper_limits.unsqueeze(0),
        )

    def _resample_commands(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        num = env_ids.numel()
        lin_x = sample_uniform(self.cfg.command_ranges["lin_vel_x"][0], self.cfg.command_ranges["lin_vel_x"][1], (num,), str(self.device))
        lin_y = sample_uniform(self.cfg.command_ranges["lin_vel_y"][0], self.cfg.command_ranges["lin_vel_y"][1], (num,), str(self.device))
        ang_z = sample_uniform(self.cfg.command_ranges["ang_vel_z"][0], self.cfg.command_ranges["ang_vel_z"][1], (num,), str(self.device))
        new_cmd = torch.stack((lin_x, lin_y, ang_z), dim=-1)
        if self.cfg.command_smoothing > 0.0:
            alpha = self.cfg.command_smoothing
            self.commands[env_ids] = (1.0 - alpha) * self.commands[env_ids] + alpha * new_cmd
        else:
            self.commands[env_ids] = new_cmd

        self.command_time_left[env_ids] = sample_uniform(
            self.cfg.command_time_range[0], self.cfg.command_time_range[1], (num,), str(self.device)
        )

    def _infer_robot_dof_count(self) -> int:
        """Count articulated DOFs from the first spawned robot prim."""
        first_robot_prim = sim_utils.find_first_matching_prim(self.cfg.robot_cfg.prim_path)
        if first_robot_prim is None:
            raise RuntimeError(f"Failed to locate robot prim matching '{self.cfg.robot_cfg.prim_path}'.")
        count = 0
        stack = [first_robot_prim]
        from pxr import UsdPhysics  # local import to avoid cost when not needed
        while stack:
            prim = stack.pop()
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                count += 1
            stack.extend(list(prim.GetChildren()))
        if count == 0:
            raise RuntimeError("Detected zero DOFs for Unitree G1; verify the USD asset and prim path.")
        return count
