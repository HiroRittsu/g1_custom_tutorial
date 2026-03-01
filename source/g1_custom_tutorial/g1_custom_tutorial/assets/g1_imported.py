"""Imported G1 asset configuration used for local USD-based experiments."""

from __future__ import annotations

import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets import G1_MINIMAL_CFG


IMPORTED_G1_USD_PATH = os.environ.get(
    "G1_IMPORTED_USD_PATH",
    "/workspace/imported_g1/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
)


G1_IMPORTED_29DOF_CFG = G1_MINIMAL_CFG.copy()
G1_IMPORTED_29DOF_CFG.spawn.usd_path = IMPORTED_G1_USD_PATH
G1_IMPORTED_29DOF_CFG.init_state.pos = (0.0, 0.0, 0.8)
G1_IMPORTED_29DOF_CFG.init_state.joint_pos = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
}
G1_IMPORTED_29DOF_CFG.init_state.joint_vel = {".*": 0.0}
G1_IMPORTED_29DOF_CFG.actuators = {
    "N7520-14.3": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
        effort_limit_sim=88,
        velocity_limit_sim=32.0,
        stiffness={
            ".*_hip_.*": 100.0,
            "waist_yaw_joint": 200.0,
        },
        damping={
            ".*_hip_.*": 2.0,
            "waist_yaw_joint": 5.0,
        },
        armature=0.01,
    ),
    "N7520-22.5": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
        effort_limit_sim=139,
        velocity_limit_sim=20.0,
        stiffness={
            ".*_hip_roll_.*": 100.0,
            ".*_knee_.*": 150.0,
        },
        damping={
            ".*_hip_roll_.*": 2.0,
            ".*_knee_.*": 4.0,
        },
        armature=0.01,
    ),
    "N5020-16": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_.*",
            ".*_elbow_.*",
            ".*_wrist_roll.*",
            ".*_ankle_.*",
            "waist_roll_joint",
            "waist_pitch_joint",
        ],
        effort_limit_sim=25,
        velocity_limit_sim=37,
        stiffness=40.0,
        damping={
            ".*_shoulder_.*": 1.0,
            ".*_elbow_.*": 1.0,
            ".*_wrist_roll.*": 1.0,
            ".*_ankle_.*": 2.0,
            "waist_.*_joint": 5.0,
        },
        armature=0.01,
    ),
    "W4010-25": ImplicitActuatorCfg(
        joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
        effort_limit_sim=5,
        velocity_limit_sim=22,
        stiffness=40.0,
        damping=1.0,
        armature=0.01,
    ),
}
