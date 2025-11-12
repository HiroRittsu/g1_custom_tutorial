from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import UsdPhysics

from .g1_custom_tutorial_env_cfg import G1CustomTutorialEnvCfg


def define_markers() -> VisualizationMarkers:
    """Create visualization markers for commanded and actual velocity directions."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/g1Markers",
        markers={
            "vel": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.4, 0.4, 0.8),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "cmd": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.4, 0.4, 0.8),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class G1CustomTutorialEnv(DirectRLEnv):
    """Direct RL environment that spawns the Unitree G1 and trains velocity-tracking locomotion."""

    cfg: G1CustomTutorialEnvCfg

    def __init__(self, cfg: G1CustomTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint meta-data
        self._dof_ids, _ = self.robot.find_joints(".*")
        self.num_dof = len(self._dof_ids)
        self.nominal_joint_pos = self.robot.data.default_joint_pos.clone()
        self.nominal_root_state = self.robot.data.default_root_state.clone()
        self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[0, self._dof_ids, 0]
        self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[0, self._dof_ids, 1]
        self.joint_limit_range = torch.clamp(self.joint_upper_limits - self.joint_lower_limits, min=1.0e-3)

        # buffers reused every step
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
        self.robot = Articulation(self.cfg.robot_cfg)
        # build terrain (flat plane)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot

        # update space dimensions now that we know the robot dof count
        num_dof = self._infer_robot_dof_count()
        self.cfg.action_space = num_dof
        self.cfg.observation_space = 12 + 3 * num_dof

        # lighting
        sky_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        sky_light.func("/World/Light", sky_light)

        # visualization markers for velocity vs command
        self.visualization_markers = define_markers()
        self._marker_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self._marker_offset[:, 2] = 0.8
        self._up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)

    """
    Core simulation hooks.
    """

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        if actions is None:
            actions = torch.zeros_like(self.actions)
        self.actions = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
        self.action_rate = self.actions - self.prev_actions

        # command scheduling
        self.command_time_left -= self.step_dt
        env_ids = torch.nonzero(self.command_time_left <= 0.0, as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self._resample_commands(env_ids)
        self._visualize_markers()

    def _apply_action(self) -> None:
        target = self.nominal_joint_pos + self.actions * self.cfg.action_scale
        lower = self.joint_lower_limits.unsqueeze(0)
        upper = self.joint_upper_limits.unsqueeze(0)
        target = torch.clamp(target, lower, upper)
        self.robot.set_joint_position_target(target, joint_ids=self._dof_ids)

    def _get_observations(self) -> dict:
        self._update_task_tensors()
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
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel_yaw[:, :2]), dim=-1)
        rew_lin_vel = torch.exp(-lin_vel_error / (self.lin_vel_tracking_std**2))

        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel_w[:, 2])
        rew_ang_vel = torch.exp(-ang_vel_error / (self.ang_vel_tracking_std**2))

        rew_lin_vel_z = torch.abs(self.base_lin_vel_yaw[:, 2])
        rew_flat_orientation = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        rew_action_rate = torch.sum(torch.square(self.action_rate), dim=-1)
        rew_torques = torch.sum(torch.square(self.torques), dim=-1)
        rew_joint_vel = torch.sum(torch.square(self.dof_vel), dim=-1)
        joint_limit_excess = torch.clamp(torch.abs(self._joint_pos_normalized) - 1.0, min=0.0)
        rew_joint_limits = torch.sum(joint_limit_excess, dim=-1)
        rew_joint_acc = torch.sum(torch.square(self.dof_acc), dim=-1)
        rew_alive = torch.ones(self.num_envs, device=self.device)
        rew_termination = self.reset_terminated.float()

        rewards = (
            self.cfg.reward_scales["track_lin_vel_xy"] * rew_lin_vel
            + self.cfg.reward_scales["track_ang_vel_z"] * rew_ang_vel
            + self.cfg.reward_scales["lin_vel_z"] * rew_lin_vel_z
            + self.cfg.reward_scales["flat_orientation"] * rew_flat_orientation
            + self.cfg.reward_scales["action_rate"] * rew_action_rate
            + self.cfg.reward_scales["torques"] * rew_torques
            + self.cfg.reward_scales["joint_vel"] * rew_joint_vel
            + self.cfg.reward_scales["joint_pos_limits"] * rew_joint_limits
            + self.cfg.reward_scales["joint_acc"] * rew_joint_acc
            + self.cfg.reward_scales["alive"] * rew_alive
            + self.cfg.reward_scales["termination"] * rew_termination
        )
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_task_tensors()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = (
            (self.base_height < self.cfg.termination_height)
            | (torch.abs(self.base_euler[:, 0]) > self.cfg.max_tilt)
            | (torch.abs(self.base_euler[:, 1]) > self.cfg.max_tilt)
        )
        return fallen, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_tensor = self.robot._ALL_INDICES.to(device=self.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids_list = env_ids_tensor.cpu().tolist()
        super()._reset_idx(env_ids_list)
        self.robot.reset(env_ids_tensor)  # type: ignore[arg-type]

        # joint state noise
        joint_pos = self.nominal_joint_pos[env_ids_tensor].clone()
        joint_pos += sample_uniform(
            -self.cfg.reset_noise["joint_pos"],
            self.cfg.reset_noise["joint_pos"],
            joint_pos.shape,
            str(joint_pos.device),
        )
        joint_vel = sample_uniform(
            -self.cfg.reset_noise["joint_vel"],
            self.cfg.reset_noise["joint_vel"],
            joint_pos.shape,
            str(joint_pos.device),
        )

        # root pose
        root_state = self.nominal_root_state[env_ids_tensor].clone()
        root_state[:, :3] = self.scene.env_origins[env_ids_tensor]
        root_state[:, 2] += self.cfg.init_base_height
        root_state[:, :2] += sample_uniform(
            -self.cfg.reset_noise["pos"], self.cfg.reset_noise["pos"], (len(env_ids_list), 2), str(self.device)
        )

        roll = sample_uniform(
            -self.cfg.reset_noise["roll"], self.cfg.reset_noise["roll"], (len(env_ids_list),), str(self.device)
        )
        pitch = sample_uniform(
            -self.cfg.reset_noise["pitch"], self.cfg.reset_noise["pitch"], (len(env_ids_list),), str(self.device)
        )
        yaw = sample_uniform(
            -self.cfg.reset_noise["yaw"], self.cfg.reset_noise["yaw"], (len(env_ids_list),), str(self.device)
        )
        quat_noise = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root_state[:, 3:7] = math_utils.quat_mul(self.nominal_root_state[env_ids_tensor, 3:7], quat_noise)

        lin_vel = sample_uniform(
            -self.cfg.reset_noise["lin_vel"], self.cfg.reset_noise["lin_vel"], (len(env_ids_list), 3), str(self.device)
        )
        ang_vel = sample_uniform(
            -self.cfg.reset_noise["ang_vel"], self.cfg.reset_noise["ang_vel"], (len(env_ids_list), 3), str(self.device)
        )
        root_state[:, 7:10] = lin_vel
        root_state[:, 10:13] = ang_vel

        # apply state
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids_tensor)  # type: ignore[arg-type]
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids_tensor)  # type: ignore[arg-type]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)  # type: ignore[arg-type]

        # reset buffers
        self.actions[env_ids_tensor] = 0.0
        self.prev_actions[env_ids_tensor] = 0.0
        self.action_rate[env_ids_tensor] = 0.0
        self.command_time_left[env_ids_tensor] = 0.0
        self._resample_commands(env_ids_tensor)
        self._visualize_markers()

    """
    Helpers.
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
        lin_x = sample_uniform(
            self.cfg.command_ranges["lin_vel_x"][0],
            self.cfg.command_ranges["lin_vel_x"][1],
            (num,),
            str(self.device),
        )
        lin_y = sample_uniform(
            self.cfg.command_ranges["lin_vel_y"][0],
            self.cfg.command_ranges["lin_vel_y"][1],
            (num,),
            str(self.device),
        )
        ang_z = sample_uniform(
            self.cfg.command_ranges["ang_vel_z"][0],
            self.cfg.command_ranges["ang_vel_z"][1],
            (num,),
            str(self.device),
        )
        new_cmd = torch.stack((lin_x, lin_y, ang_z), dim=-1)
        if self.cfg.command_smoothing > 0.0:
            alpha = self.cfg.command_smoothing
            self.commands[env_ids] = (1.0 - alpha) * self.commands[env_ids] + alpha * new_cmd
        else:
            self.commands[env_ids] = new_cmd

        self.command_time_left[env_ids] = sample_uniform(
            self.cfg.command_time_range[0], self.cfg.command_time_range[1], (num,), str(self.device)
        )

    def _visualize_markers(self):
        """Render commanded and actual heading markers above each robot."""
        if not hasattr(self, "visualization_markers"):
            return
        marker_locations = self.robot.data.root_pos_w + self._marker_offset
        vel_marker_orientations = self.robot.data.root_quat_w

        cmd_xy = self.commands[:, :2]
        norm = torch.linalg.norm(cmd_xy, dim=-1, keepdim=True)
        fallback = torch.zeros_like(cmd_xy)
        fallback[:, 0] = 1.0
        safe_cmd_xy = torch.where(norm > 1.0e-6, cmd_xy, fallback)

        yaw = torch.atan2(safe_cmd_xy[:, 1], safe_cmd_xy[:, 0])
        up_dirs = self._up_dir.unsqueeze(0).repeat(self.num_envs, 1)
        cmd_marker_orientations = math_utils.quat_from_angle_axis(yaw, up_dirs)

        loc = torch.vstack((marker_locations, marker_locations))
        rots = torch.vstack((vel_marker_orientations, cmd_marker_orientations))
        all_envs = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
        marker_indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, rots, marker_indices=marker_indices)

    def _infer_robot_dof_count(self) -> int:
        """Count articulated DOFs from the first spawned robot prim."""
        first_robot_prim = sim_utils.find_first_matching_prim(self.cfg.robot_cfg.prim_path)
        if first_robot_prim is None:
            raise RuntimeError(f"Failed to locate robot prim matching '{self.cfg.robot_cfg.prim_path}'.")

        count = 0
        stack = [first_robot_prim]
        while stack:
            prim = stack.pop()
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                count += 1
            stack.extend(list(prim.GetChildren()))
        if count == 0:
            raise RuntimeError("Detected zero DOFs for Unitree G1; verify the USD asset and prim path.")
        return count
