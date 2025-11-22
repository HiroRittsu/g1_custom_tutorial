"""Debug visuals for Manager-based tasks.

This module provides light-weight USD drawing helpers to visualize commands.
In particular, it renders a per-env velocity arrow whose anchor height follows
the commanded base height (env._height_command) so that users can see both
heading and target height at a glance in Play mode.

Drawing is best-effort and no-op if Kit/USD context is unavailable.
"""

from __future__ import annotations

import math
from typing import Sequence, TYPE_CHECKING

import torch

try:
    # Omniverse USD context
    import omni.usd  # type: ignore
    from pxr import Gf, Sdf, Usd, UsdGeom  # type: ignore

    def _get_stage():
        return omni.usd.get_context().get_stage()
except Exception:  # pragma: no cover - headless fallback
    omni = None  # type: ignore
    Usd = None  # type: ignore
    UsdGeom = None  # type: ignore
    Gf = None  # type: ignore
    Sdf = None  # type: ignore

    def _get_stage():  # type: ignore
        return None

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


ARROW_ROOT_PATH = "/World/Debug/HeightSyncedVelocity"


def _define_prim(stage, path: str, type_name: str = "Xform"):
    if stage.GetPrimAtPath(path):
        return stage.GetPrimAtPath(path)
    return stage.DefinePrim(Sdf.Path(path), type_name)


def _ensure_arrow_for_env(stage, env_index: int):
    """Create (if missing) three arrows for the env: Vel, Heading, Yaw.

    Structure:
      /World/Debug/HeightSyncedVelocity/env_{i}
        ├── Vel (Cone: velocity, points along +Z in local frame)
        ├── Heading (Cone: heading dir)
        └── Yaw (Cone: angular vel magnitude along ±Z)

    We rotate Vel/Heading so the cone lies along +X and then yaw-rotate them in XY.
    """
    root = _define_prim(stage, ARROW_ROOT_PATH, "Xform")
    env_path = f"{ARROW_ROOT_PATH}/env_{env_index}"
    env_xf = _define_prim(stage, env_path, "Xform")
    # helper to define a colored cone
    def _cone(name: str, color):
        path = f"{env_path}/{name}"
        if not stage.GetPrimAtPath(path):
            c = UsdGeom.Cone.Define(stage, Sdf.Path(path))
            # make base geometry larger for visibility
            c.CreateHeightAttr(0.6)
            c.CreateRadiusAttr(0.08)
            c.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
        return stage.GetPrimAtPath(path)

    _cone("Vel", (0.10, 0.80, 0.90))      # cyan
    _cone("Heading", (0.95, 0.60, 0.10))  # orange
    _cone("Yaw", (0.85, 0.10, 0.85))      # magenta
    return env_xf


def spawn_height_velocity_arrows(env: "ManagerBasedRLEnv", env_ids: Sequence[int] | None) -> None:
    """Startup event: ensure debug arrows exist for all envs.

    Safe to call multiple times.
    """
    stage = _get_stage()
    if stage is None:
        return  # headless or USD unavailable
    # cache a torso/anchor body id if available (best-effort)
    try:
        anchor_ids, _ = env.scene["robot"].find_bodies("torso_link")
        if len(anchor_ids) > 0:
            env._height_viz_anchor_body_id = int(anchor_ids[0])  # type: ignore[attr-defined]
    except Exception:
        env._height_viz_anchor_body_id = None  # type: ignore[attr-defined]

    # create for all envs (env_ids ignored at startup)
    for i in range(getattr(env, "num_envs", 0)):
        _ensure_arrow_for_env(stage, i)


def update_height_velocity_arrows(env: "ManagerBasedRLEnv", env_ids: Sequence[int] | None) -> None:
    """Interval/step event: position and orient arrows by (heading, height_cmd).

    - Position XY at robot base position
    - Position Z at env_origin_z + height_command
    - Orientation: yaw from commanded (vx, vy); tilt by +90deg about Y to make cone lie in XY
    - Scale height by commanded speed magnitude (clamped)
    """
    stage = _get_stage()
    if stage is None:
        return
    asset = env.scene["robot"]
    cmd = env.command_manager.get_command("base_velocity")
    z_cmd = getattr(env, "_height_command", torch.zeros(env.num_envs, 1, device=cmd.device))[:, 0]

    # choose envs to update
    if env_ids is None:
        env_ids_tensor = torch.arange(env.num_envs, device=cmd.device)
    else:
        env_ids_tensor = torch.as_tensor(env_ids, device=cmd.device)

    # compute per-env transforms
    # prefer torso/body anchor if available; otherwise use root
    anchor_id = getattr(env, "_height_viz_anchor_body_id", None)
    if anchor_id is None:
        pos_w = asset.data.root_pos_w[env_ids_tensor]
    else:
        pos_w = asset.data.body_pos_w[env_ids_tensor, anchor_id]
    origins = env.scene.env_origins[env_ids_tensor]
    cols = cmd.shape[1]
    vxvy = cmd[env_ids_tensor, :2]
    speed = torch.clamp(torch.norm(vxvy, dim=1), 0.0, 2.0)
    # Heading (if present in command vector)
    if cols >= 4:
        heading = cmd[env_ids_tensor, 3]
    else:
        heading = torch.atan2(vxvy[:, 1], vxvy[:, 0])
    # angular velocity Z (if present)
    omega = cmd[env_ids_tensor, 2] if cols >= 3 else torch.zeros_like(speed)

    for j in range(env_ids_tensor.shape[0]):
        i = int(env_ids_tensor[j].item())
        # target position
        x = float(pos_w[j, 0].item())
        y = float(pos_w[j, 1].item())
        z = float((origins[j, 2] + z_cmd[j]).item())

        # orientation: rotate so local +Z aligns with world +X, then yaw by heading
        rot_y_deg = 90.0
        rot_z_deg = math.degrees(float(heading[j].item())) if speed[j] > 1e-3 else 0.0
        # scale: thicker XY and longer Z (bounded)
        sp = float(speed[j].item())
        length = 0.6 + 0.8 * min(1.0, sp)  # 0.6..1.4
        thickness = 1.5
        scale = Gf.Vec3f(thickness, thickness, max(0.1, length))

        env_xf = _ensure_arrow_for_env(stage, i)
        # Common translate for the group
        UsdGeom.XformCommonAPI(env_xf).SetTranslate(Gf.Vec3d(x, y, z))

        # Velocity arrow
        vel_prim = stage.GetPrimAtPath(f"{ARROW_ROOT_PATH}/env_{i}/Vel")
        vel_api = UsdGeom.XformCommonAPI(vel_prim)
        vel_api.SetRotate((0.0, rot_y_deg, rot_z_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        vel_api.SetScale(scale)

        # Heading arrow (fixed length)
        hdg_prim = stage.GetPrimAtPath(f"{ARROW_ROOT_PATH}/env_{i}/Heading")
        hdg_api = UsdGeom.XformCommonAPI(hdg_prim)
        hdg_len = 1.2
        hdg_api.SetRotate((0.0, rot_y_deg, math.degrees(float(heading[j].item()))), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        hdg_api.SetScale(Gf.Vec3f(1.3, 1.3, hdg_len))

        # Yaw arrow (vertical, along ±Z based on sign of omega)
        yaw_prim = stage.GetPrimAtPath(f"{ARROW_ROOT_PATH}/env_{i}/Yaw")
        yaw_api = UsdGeom.XformCommonAPI(yaw_prim)
        om = abs(float(omega[j].item()))
        yaw_len = 0.4 + 0.8 * min(1.0, om)  # 0.4..1.2
        yaw_up = float(1.0 if omega[j].item() >= 0.0 else -1.0)
        # reset rotation: keep cone along +Z, flip by 180deg for negative
        yaw_api.SetRotate((180.0 if yaw_up < 0.0 else 0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        yaw_api.SetScale(Gf.Vec3f(1.3, 1.3, max(0.1, yaw_len)))
