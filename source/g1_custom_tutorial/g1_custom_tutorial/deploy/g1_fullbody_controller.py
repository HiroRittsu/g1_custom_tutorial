#!/usr/bin/env python3

"""ROS 2 fullbody controller for the imported G1 locomotion policy."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import torch

import rclpy
from geometry_msgs.msg import Twist
from message_filters import Subscriber, TimeSynchronizer
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32

from g1_custom_tutorial.deploy.g1_policy_spec import (
    G1_ACTION_SCALE,
    G1_ACTUATED_JOINT_NAMES,
    G1_HEIGHT_LIMITS,
    G1_HEIGHT_SCALING,
    build_default_joint_positions,
)


class G1FullbodyController(Node):
    """H1-style ROS 2 controller adapted to the imported 29-DoF G1 policy."""

    def __init__(self):
        super().__init__("g1_fullbody_controller")

        self.declare_parameter("publish_period_ms", 5)
        self.declare_parameter("policy_path", "policy/policy.pt")
        self.declare_parameter("policy_spec_path", "policy/g1_policy_spec.json")
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("height_command_topic", "height_command")
        self.declare_parameter("joint_state_topic", "joint_states")
        self.declare_parameter("imu_topic", "imu")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("joint_command_topic", "joint_command")
        self.declare_parameter("odom_twist_in_body_frame", False)
        self.declare_parameter("use_sim_time", True)

        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        self._cmd_vel = Twist()
        self._height_command = float(G1_HEIGHT_LIMITS["default"])
        self._previous_action = np.zeros(len(G1_ACTUATED_JOINT_NAMES), dtype=np.float32)
        self._action_scale = float(G1_ACTION_SCALE)
        self._policy_counter = 0
        self._decimation = 4
        self._dt = 0.0
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self.action = np.zeros(len(G1_ACTUATED_JOINT_NAMES), dtype=np.float32)
        self._odom_twist_in_body_frame = bool(self.get_parameter("odom_twist_in_body_frame").value)

        self.joint_names = list(G1_ACTUATED_JOINT_NAMES)
        self.default_pos = np.array(
            [build_default_joint_positions(self.joint_names)[name] for name in self.joint_names], dtype=np.float32
        )

        self._load_policy_spec()
        self._load_policy()

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        height_topic = self.get_parameter("height_command_topic").value
        joint_state_topic = self.get_parameter("joint_state_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        joint_command_topic = self.get_parameter("joint_command_topic").value

        self._cmd_vel_subscription = self.create_subscription(Twist, cmd_vel_topic, self._cmd_vel_callback, 10)
        self._height_command_subscription = self.create_subscription(
            Float32, height_topic, self._height_command_callback, 10
        )
        self._joint_publisher = self.create_publisher(JointState, joint_command_topic, qos_profile=sim_qos_profile)

        self._imu_sub_filter = Subscriber(self, Imu, imu_topic, qos_profile=sim_qos_profile)
        self._joint_states_sub_filter = Subscriber(self, JointState, joint_state_topic, qos_profile=sim_qos_profile)
        self._odom_sub_filter = Subscriber(self, Odometry, odom_topic, qos_profile=sim_qos_profile)
        self.sync = TimeSynchronizer([self._joint_states_sub_filter, self._imu_sub_filter, self._odom_sub_filter], 10)
        self.sync.registerCallback(self._tick)

    def _load_policy_spec(self) -> None:
        spec_path = Path(self.get_parameter("policy_spec_path").value)
        if not spec_path.is_file():
            self.get_logger().warning(f"Policy spec not found at {spec_path}. Falling back to built-in defaults.")
            return

        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.joint_names = list(spec.get("joint_names", self.joint_names))
        joint_defaults = spec.get("joint_defaults", {})
        self.default_pos = np.array([joint_defaults.get(name, 0.0) for name in self.joint_names], dtype=np.float32)
        self._action_scale = float(spec.get("action", {}).get("scale", self._action_scale))
        self._decimation = int(spec.get("simulation", {}).get("decimation", self._decimation))

    def _load_policy(self) -> None:
        policy_path = Path(self.get_parameter("policy_path").value)
        with policy_path.open("rb") as handle:
            self.policy = torch.jit.load(io.BytesIO(handle.read()))
        self.policy.eval()

    def _cmd_vel_callback(self, msg: Twist) -> None:
        self._cmd_vel = msg

    def _height_command_callback(self, msg: Float32) -> None:
        clamped = float(np.clip(msg.data, G1_HEIGHT_LIMITS["min"], G1_HEIGHT_LIMITS["max"]))
        self._height_command = clamped

    def _tick(self, joint_state: JointState, imu: Imu, odom: Odometry) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._last_tick_time:
            self._policy_counter = 0
        self._dt = max(now - self._last_tick_time, 0.0)
        self._last_tick_time = now

        self.forward(joint_state, imu, odom)

        joint_command = JointState()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.name = self.joint_names
        action_pos = self.default_pos + self.action * self._action_scale
        joint_command.position = action_pos.tolist()
        joint_command.velocity = np.zeros(len(self.joint_names), dtype=np.float32).tolist()
        joint_command.effort = np.zeros(len(self.joint_names), dtype=np.float32).tolist()
        self._joint_publisher.publish(joint_command)

    def _compute_observation(self, joint_state: JointState, imu: Imu, odom: Odometry) -> np.ndarray:
        quat = np.array([imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z], dtype=np.float64)
        rotation_bi = self._quat_to_rot_matrix(quat).T

        lin_vel = np.array(
            [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z], dtype=np.float32
        )
        if self._odom_twist_in_body_frame:
            lin_vel_b = lin_vel
        else:
            lin_vel_b = np.matmul(rotation_bi, lin_vel.astype(np.float64)).astype(np.float32)

        ang_vel_b = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z], dtype=np.float32)
        gravity_b = np.matmul(rotation_bi, np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)

        obs = np.zeros(100, dtype=np.float32)
        obs[:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = self._height_scaled_command()

        current_joint_pos = np.zeros(len(self.joint_names), dtype=np.float32)
        current_joint_vel = np.zeros(len(self.joint_names), dtype=np.float32)
        name_to_index = {name: idx for idx, name in enumerate(joint_state.name)}
        for i, name in enumerate(self.joint_names):
            idx = name_to_index.get(name)
            if idx is None:
                continue
            current_joint_pos[i] = joint_state.position[idx]
            current_joint_vel[i] = joint_state.velocity[idx]

        obs[12:41] = current_joint_pos - self.default_pos
        obs[41:70] = current_joint_vel
        obs[70:99] = self._previous_action
        obs[99] = self._height_command
        return obs

    def _height_scaled_command(self) -> np.ndarray:
        cmd = np.array(
            [self._cmd_vel.linear.x, self._cmd_vel.linear.y, self._cmd_vel.angular.z], dtype=np.float32
        )
        z_min = float(G1_HEIGHT_SCALING["z_min"])
        z_ref = float(G1_HEIGHT_SCALING["z_ref"])
        denom = max(z_ref - z_min, 1e-6)
        normalized = float(np.clip((self._height_command - z_min) / denom, 0.0, 1.0))
        s_lin = normalized ** float(G1_HEIGHT_SCALING["p_lin"])
        s_ang = normalized ** float(G1_HEIGHT_SCALING["p_ang"])
        cmd[:2] *= s_lin
        cmd[2] *= s_ang
        return cmd

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs_tensor).detach().view(-1).cpu().numpy().astype(np.float32)
        return action

    def forward(self, joint_state: JointState, imu: Imu, odom: Odometry) -> None:
        obs = self._compute_observation(joint_state, imu, odom)
        if self._policy_counter % self._decimation == 0:
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
        self._policy_counter += 1

    @staticmethod
    def _quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )

    def _get_stamp_prefix(self) -> str:
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f"[{now}][{now_ros}]"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = G1FullbodyController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
