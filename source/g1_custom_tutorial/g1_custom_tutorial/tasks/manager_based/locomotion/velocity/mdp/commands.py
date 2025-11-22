"""Manager ベースの MDP における 高さ⇔速度 の連携ユーティリティ。

このモジュールは、軽量な高さコマンド用バッファと、
各環境ごとの高さサンプル、そして高さに応じてスケールした速度コマンドを
生成するヘルパーを提供します（新たな CommandCfg 型を導入せずに実現）。
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # only for type hints
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_height_buffers(env: "ManagerBasedRLEnv", center: float = 0.74, width: float = 0.0) -> None:
    """環境ごとの高さ用バッファが存在することを保証します。

    バッファ:
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
    """高さバッファを与えられた center/width で初期化し、最初のコマンドをサンプルします。

    第2引数 `env_ids` は EventManager の想定シグネチャ（env, env_ids, ...）に合わせるために
    意図的に受け取っています。起動時は値を使用しません。
    """
    _ensure_height_buffers(env, center=center, width=width)
    env._height_center.fill_(float(center))  # type: ignore[attr-defined]
    env._height_width.fill_(float(width))  # type: ignore[attr-defined]
    resample_height_command(env, None)


def set_height_center_width(
    env: "ManagerBasedRLEnv", center: float | None = None, width: float | None = None
) -> None:
    """center/width を更新（全環境にブロードキャスト）。"""
    _ensure_height_buffers(env)
    if center is not None:
        env._height_center.fill_(float(center))  # type: ignore[attr-defined]
    if width is not None:
        env._height_width.fill_(float(width))  # type: ignore[attr-defined]


def resample_height_command(env: "ManagerBasedRLEnv", env_ids) -> None:
    """保持している center/width を用いて各環境の高さをサンプルし、env._height_command に書き込みます。

    `env_ids`（Tensor または list）が渡された場合は、その環境のみ再サンプルします。省略時は全環境です。
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
    """現在の高さコマンドを [num_envs, 1] テンソルとして返します。"""
    _ensure_height_buffers(env)
    return env._height_command  # type: ignore[attr-defined]


def _smooth_scale_from_height(z_cmd: torch.Tensor, z_min: float, z_ref: float, power: float) -> torch.Tensor:
    """高さコマンドから [0,1] の滑らかなスケーリング s を計算します。

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
    """高さコマンドの関数としてスケールした速度コマンドを返します。

    - 並進 x/y は s^p_lin、角速度 z は s^p_ang でスケーリング。
    - 追加成分（例: heading）はそのまま保持します。
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
