"""Height/velocity coupling utilities for manager-based MDP.

This module adds a lightweight height command buffer and helpers to
sample height per-env and to produce height-scaled velocity commands
without introducing a new CommandCfg type.
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # only for type hints
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_height_buffers(env: "ManagerBasedRLEnv", center: float = 0.74, width: float = 0.0) -> None:
    """Ensure per-env height buffers exist on the env.

    Buffers:
      - env._height_command: [num_envs, 1]
      - env._height_center: [num_envs]
      - env._height_width:  [num_envs]
    """
    device = getattr(env, "device", "cpu")
    num_envs = getattr(env, "num_envs", 1)
    if not hasattr(env, "_height_command"):
        env._height_command = torch.full((num_envs, 1), float(center), device=device, dtype=torch.float32)  # type: ignore[attr-defined]
    if not hasattr(env, "_height_center"):
        env._height_center = torch.full((num_envs,), float(center), device=device, dtype=torch.float32)  # type: ignore[attr-defined]
    if not hasattr(env, "_height_width"):
        env._height_width = torch.full((num_envs,), float(width), device=device, dtype=torch.float32)  # type: ignore[attr-defined]


def init_height_center_width(
    env: "ManagerBasedRLEnv", env_ids, center: float = 0.74, width: float = 0.0
) -> None:
    """Initialize height buffers with given center/width and sample first commands.

    The second argument is intentionally `env_ids` to comply with EventManager's
    expected signature (env, env_ids, ...). The value is ignored at startup.
    """
    _ensure_height_buffers(env, center=center, width=width)
    env._height_center.fill_(float(center))  # type: ignore[attr-defined]
    env._height_width.fill_(float(width))  # type: ignore[attr-defined]
    resample_height_command(env, None)


def set_height_center_width(
    env: "ManagerBasedRLEnv", center: float | None = None, width: float | None = None
) -> None:
    """Update center/width (broadcast to all envs)."""
    _ensure_height_buffers(env)
    if center is not None:
        env._height_center.fill_(float(center))  # type: ignore[attr-defined]
    if width is not None:
        env._height_width.fill_(float(width))  # type: ignore[attr-defined]


def resample_height_command(env: "ManagerBasedRLEnv", env_ids) -> None:
    """Sample height per env using stored center/width and write to env._height_command.

    If `env_ids` is provided (Tensor or list), only those envs are resampled; otherwise all envs.
    """
    _ensure_height_buffers(env)
    center = env._height_center  # type: ignore[attr-defined]
    width = env._height_width  # type: ignore[attr-defined]
    low = center - width
    high = center + width
    # clamp to non-negative heights and ensure low<=high
    low, high = torch.minimum(low, high), torch.maximum(low, high)
    if env_ids is None or env_ids == slice(None):
        rand = torch.rand_like(center)
        samples = low + (high - low) * rand
        env._height_command[:, 0] = samples  # type: ignore[attr-defined]
    else:
        env_ids_tensor = torch.as_tensor(env_ids, device=center.device)
        rand = torch.rand_like(center[env_ids_tensor])
        samples = low[env_ids_tensor] + (high[env_ids_tensor] - low[env_ids_tensor]) * rand
        env._height_command[env_ids_tensor, 0] = samples  # type: ignore[attr-defined]


def height_command(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return current height command as [num_envs, 1] tensor."""
    _ensure_height_buffers(env)
    return env._height_command  # type: ignore[attr-defined]


def _smooth_scale_from_height(z_cmd: torch.Tensor, z_min: float, z_ref: float, power: float) -> torch.Tensor:
    """Compute smooth scaling s in [0,1] from height command.

    s = clamp(((z_cmd - z_min) / (z_ref - z_min)), 0, 1) ** power
    """
    denom = max(z_ref - z_min, 1e-6)
    s = (z_cmd - z_min) / denom
    s = torch.clamp(s, 0.0, 1.0)
    if power != 1.0:
        s = torch.pow(s, power)
    return s


def height_scaled_velocity_commands(
    env: "ManagerBasedRLEnv",
    vel_cmd_name: str = "base_velocity",
    z_min: float = 0.70,
    z_ref: float = 0.74,
    p_lin: float = 1.2,
    p_ang: float = 1.0,
) -> torch.Tensor:
    """Return velocity command scaled as a function of the height command.

    - Scales linear x/y by s^p_lin and angular z by s^p_ang.
    - Keeps any extra components (e.g., heading) unchanged.
    """
    _ensure_height_buffers(env)
    cmd = env.command_manager.get_command(vel_cmd_name)
    z = env._height_command[:, 0]  # type: ignore[attr-defined]
    s_lin = _smooth_scale_from_height(z, z_min, z_ref, p_lin).unsqueeze(1)
    s_ang = _smooth_scale_from_height(z, z_min, z_ref, p_ang).unsqueeze(1)
    # Defensive: handle command vectors of length >=3 (vx, vy, yaw)
    out = cmd.clone()
    if out.shape[1] >= 2:
        out[:, 0:2] = out[:, 0:2] * s_lin
    if out.shape[1] >= 3:
        out[:, 2:3] = out[:, 2:3] * s_ang
    return out
