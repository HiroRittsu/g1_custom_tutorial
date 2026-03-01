"""Shared deployment metadata for the imported G1 locomotion policy."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from g1_custom_tutorial.assets import G1_IMPORTED_29DOF_CFG


G1_ACTUATED_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

G1_OBSERVATION_TERMS = [
    {"name": "base_lin_vel", "size": 3},
    {"name": "base_ang_vel", "size": 3},
    {"name": "projected_gravity", "size": 3},
    {"name": "velocity_commands", "size": 3},
    {"name": "joint_pos_rel", "size": len(G1_ACTUATED_JOINT_NAMES)},
    {"name": "joint_vel_rel", "size": len(G1_ACTUATED_JOINT_NAMES)},
    {"name": "last_action", "size": len(G1_ACTUATED_JOINT_NAMES)},
    {"name": "height_command", "size": 1},
]

G1_HEIGHT_LIMITS = {"min": 0.24, "max": 0.74, "default": 0.74}
G1_HEIGHT_SCALING = {"z_min": 0.24, "z_ref": 0.74, "p_lin": 1.2, "p_ang": 1.0}
G1_ACTION_SCALE = 0.5
G1_SIM_DT = 0.005
G1_DECIMATION = 4
G1_POLICY_DT = G1_SIM_DT * G1_DECIMATION


def _matches(name: str, pattern: str) -> bool:
    return re.fullmatch(pattern, name) is not None


def _resolve_joint_value(name: str, spec: float | dict[str, float] | None) -> float | None:
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    resolved = None
    for pattern, value in spec.items():
        if _matches(name, pattern):
            resolved = float(value)
    return resolved


def resolve_policy_joint_names_from_env(env) -> list[str]:
    """Extract the policy joint order from the live Isaac Lab environment."""
    term = env.unwrapped.action_manager.get_term("joint_pos")
    joint_names = getattr(term, "joint_names", None) or getattr(term, "_joint_names", None)
    if not joint_names:
        raise RuntimeError("Failed to resolve policy joint order from joint_pos action term.")
    return list(joint_names)


def build_default_joint_positions(joint_names: list[str] | None = None) -> dict[str, float]:
    """Expand regex-based default joint positions into an ordered per-joint map."""
    joint_names = joint_names or list(G1_ACTUATED_JOINT_NAMES)
    defaults = {name: 0.0 for name in joint_names}
    for pattern, value in G1_IMPORTED_29DOF_CFG.init_state.joint_pos.items():
        for name in joint_names:
            if _matches(name, pattern):
                defaults[name] = float(value)
    return defaults


def _build_joint_actuator_map(joint_names: list[str]) -> dict[str, str]:
    joint_to_actuator: dict[str, str] = {}
    for actuator_name, actuator_cfg in G1_IMPORTED_29DOF_CFG.actuators.items():
        for joint_name in joint_names:
            if any(_matches(joint_name, expr) for expr in actuator_cfg.joint_names_expr):
                joint_to_actuator[joint_name] = actuator_name
    missing = [name for name in joint_names if name not in joint_to_actuator]
    if missing:
        raise RuntimeError(f"Missing actuator assignment for joints: {missing}")
    return joint_to_actuator


def build_joint_drive_parameters(joint_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Build per-joint drive parameters for Isaac Sim side reproduction."""
    joint_names = joint_names or list(G1_ACTUATED_JOINT_NAMES)
    defaults = build_default_joint_positions(joint_names)
    actuator_by_joint = _build_joint_actuator_map(joint_names)

    params: list[dict[str, Any]] = []
    for joint_name in joint_names:
        actuator_name = actuator_by_joint[joint_name]
        actuator_cfg = G1_IMPORTED_29DOF_CFG.actuators[actuator_name]
        params.append(
            {
                "name": joint_name,
                "actuator": actuator_name,
                "default_pos": defaults[joint_name],
                "stiffness": _resolve_joint_value(joint_name, actuator_cfg.stiffness),
                "damping": _resolve_joint_value(joint_name, actuator_cfg.damping),
                "effort_limit_sim": getattr(actuator_cfg, "effort_limit_sim", None),
                "velocity_limit_sim": getattr(actuator_cfg, "velocity_limit_sim", None),
                "armature": getattr(actuator_cfg, "armature", None),
            }
        )
    return params


def _with_observation_ranges() -> list[dict[str, Any]]:
    offset = 0
    terms: list[dict[str, Any]] = []
    for term in G1_OBSERVATION_TERMS:
        size = int(term["size"])
        terms.append({"name": term["name"], "size": size, "start": offset, "end": offset + size})
        offset += size
    return terms


def build_policy_manifest(
    joint_names: list[str] | None = None,
    *,
    checkpoint_path: str | None = None,
    policy_path: str | None = None,
    env_cfg: Any | None = None,
    agent_cfg: Any | None = None,
) -> dict[str, Any]:
    """Create a deploy manifest for H1-style ROS2 policy execution."""
    joint_names = joint_names or list(G1_ACTUATED_JOINT_NAMES)
    sim_dt = float(getattr(getattr(env_cfg, "sim", None), "dt", G1_SIM_DT))
    decimation = int(getattr(env_cfg, "decimation", G1_DECIMATION))
    action_scale = float(getattr(getattr(env_cfg, "actions", None), "joint_pos", None).scale) if getattr(getattr(env_cfg, "actions", None), "joint_pos", None) else G1_ACTION_SCALE
    return {
        "policy_format": "torchscript",
        "policy_path": policy_path,
        "checkpoint_path": checkpoint_path,
        "observation_dim": sum(term["size"] for term in G1_OBSERVATION_TERMS),
        "action_dim": len(joint_names),
        "observation_terms": _with_observation_ranges(),
        "joint_names": joint_names,
        "joint_defaults": build_default_joint_positions(joint_names),
        "joint_drive": build_joint_drive_parameters(joint_names),
        "action": {
            "type": "joint_position_target",
            "scale": action_scale,
            "use_default_offset": True,
        },
        "command_topics": {
            "velocity": "cmd_vel",
            "height": "height_command",
            "joint_state": "joint_states",
            "imu": "imu",
            "odom": "odom",
            "joint_command": "joint_command",
        },
        "height_command": {
            "limits": G1_HEIGHT_LIMITS,
            "scaling": G1_HEIGHT_SCALING,
        },
        "simulation": {
            "dt": sim_dt,
            "decimation": decimation,
            "policy_dt": sim_dt * decimation,
            "policy_rate_hz": 1.0 / (sim_dt * decimation),
        },
        "training": {
            "experiment_name": getattr(agent_cfg, "experiment_name", None),
            "empirical_normalization": getattr(agent_cfg, "empirical_normalization", None),
        },
    }


def export_policy_manifest(
    output_dir: str | Path,
    joint_names: list[str] | None = None,
    *,
    checkpoint_path: str | None = None,
    policy_path: str | None = None,
    env_cfg: Any | None = None,
    agent_cfg: Any | None = None,
) -> Path:
    """Write the deploy manifest next to the exported policy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "g1_policy_spec.json"
    manifest = build_policy_manifest(
        joint_names,
        checkpoint_path=checkpoint_path,
        policy_path=policy_path,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
