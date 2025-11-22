from __future__ import annotations

from typing import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass


@configclass
class PlateMarkerCfg:
    size: tuple[float, float, float] = (0.4, 0.4, 0.01)
    color_actual: tuple[float, float, float] = (1.0, 0.5, 0.1)  # オレンジ
    color_cmd: tuple[float, float, float] = (0.2, 0.5, 1.0)      # 青
    # マーカー用のルートパス（絶対パスにする）
    prim_base: str = "/World/Visuals/HeightMarkers"


def _build_marker_cfg(prim_path: str, size: tuple[float, float, float], color: tuple[float, float, float]):
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "plate": sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


def _get_markers(env, key: str) -> VisualizationMarkers | None:
    return getattr(env, key, None)


def _set_markers(env, key: str, markers: VisualizationMarkers) -> None:
    setattr(env, key, markers)


def spawn_height_plate(env, env_ids: Sequence[int] | None, plate_cfg: PlateMarkerCfg | None = None) -> None:
    """骨盤位置と高さコマンドをプレートで可視化（インスタンス生成）。"""
    cfg = plate_cfg or PlateMarkerCfg()
    asset = env.scene["robot"]
    body_ids, _ = asset.find_bodies("pelvis")
    pelvis_id = int(body_ids[0]) if len(body_ids) > 0 else 0

    pos_pelvis = asset.data.body_pos_w[:, pelvis_id]
    pos_cmd = pos_pelvis.clone()
    pos_cmd[:, 2] = env._height_command[:, 0]  # type: ignore[attr-defined]

    # actual 側マーカー
    if _get_markers(env, "_height_marker_actual") is None:
        cfg_actual = _build_marker_cfg(
            f"{cfg.prim_base}/actual",
            cfg.size,
            cfg.color_actual,
        )
        marker_actual = VisualizationMarkers(cfg_actual)
        _set_markers(env, "_height_marker_actual", marker_actual)

    # command 側マーカー
    if _get_markers(env, "_height_marker_cmd") is None:
        cfg_cmd = _build_marker_cfg(
            f"{cfg.prim_base}/cmd",
            cfg.size,
            cfg.color_cmd,
        )
        marker_cmd = VisualizationMarkers(cfg_cmd)
        _set_markers(env, "_height_marker_cmd", marker_cmd)

    _update_marker_poses(env, pos_pelvis, pos_cmd)


def update_height_plate(env, env_ids: Sequence[int] | None, plate_cfg: PlateMarkerCfg | None = None) -> None:
    """骨盤位置と高さコマンドでプレートの位置を更新。"""
    asset = env.scene["robot"]
    body_ids, _ = asset.find_bodies("pelvis")
    pelvis_id = int(body_ids[0]) if len(body_ids) > 0 else 0

    pos_pelvis = asset.data.body_pos_w[:, pelvis_id]
    pos_cmd = pos_pelvis.clone()
    pos_cmd[:, 2] = env._height_command[:, 0]  # type: ignore[attr-defined]

    # まだ作られていない場合は生成から
    if _get_markers(env, "_height_marker_actual") is None or _get_markers(env, "_height_marker_cmd") is None:
        spawn_height_plate(env, env_ids, plate_cfg)
        return

    _update_marker_poses(env, pos_pelvis, pos_cmd)


def _update_marker_poses(env, pos_pelvis: torch.Tensor, pos_cmd: torch.Tensor) -> None:
    # type: ignore は mypy 対策
    marker_actual: VisualizationMarkers = _get_markers(env, "_height_marker_actual")  # type: ignore[assignment]
    marker_cmd: VisualizationMarkers = _get_markers(env, "_height_marker_cmd")        # type: ignore[assignment]

    # num_envs 個分のマーカーを各 env の骨盤位置 / コマンド高さに配置
    marker_actual.visualize(translations=pos_pelvis)
    marker_cmd.visualize(translations=pos_cmd)
