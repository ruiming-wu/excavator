from __future__ import annotations

import argparse
import math
import os
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from excavator_sim.common import get_paths


@dataclass
class SceneRandomization:
    # Stage convention: x forward, y lateral, z up.
    truck_x_min: float = 0.0
    truck_x_max: float = 2.0
    truck_y_abs_min: float = 2.0
    truck_y_abs_max: float = 3.0
    pile_x_min: float = 2.5
    pile_x_max: float = 3.5
    pile_y_min: float = -1.0
    pile_y_max: float = 1.0
    truck_len: float = 3.0
    truck_wid: float = 1.8
    truck_h: float = 0.8
    truck_thickness: float = 0.04
    stone_count: int = 250
    stone_size_min: float = 0.05
    stone_size_max: float = 0.08
    stone_drop_height_min: float = 0.5
    stone_drop_height_max: float = 0.7
    stone_density: float = 500.0
    stone_spawn_interval_s: float = 1.0 / 30.0
    physics_dt: float = 1.0 / 60.0
    render_every_n_steps: int = 3
    friction_static: float = 2.1
    friction_dynamic: float = 2.0
    friction_restitution: float = 0.02


@dataclass
class StoneSpec:
    sx: float
    sy: float
    sz: float
    z: float


class _RosJointBridge:
    def __init__(self, stage, excavator_prim: str):
        from pxr import Sdf  # type: ignore

        self._Sdf = Sdf
        self._rclpy = None
        self._node = None
        self._joint_msg_type = None
        self._inited_here = False
        self._last_pub_wall_time = -1.0
        self._pub_period = 1.0 / 60.0
        self._last_ready_pub_wall_time = -1.0
        self._ready_pub_period = 1.0 / 10.0
        self._reset_requested = False
        self._ready_state = False
        self._targets: Dict[str, float] = {}
        self._initial_targets: Dict[str, float] = {}
        self._joint_prims: Dict[str, object] = {}
        self._int_msg_type = None
        self._stones_pub = None
        self._stones_in_truck_count = 0
        self._last_stones_pub_wall_time = -1.0
        self._stones_pub_period = 1.0 / 5.0

        # Discover all revolute joints under excavator root.
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if not path.startswith(excavator_prim + "/"):
                continue
            if "RevoluteJoint" not in prim.GetTypeName():
                continue
            name = prim.GetName()
            self._joint_prims[name] = prim
            attr = prim.GetAttribute("drive:angular:physics:targetPosition")
            # PhysX angular drive target is in degrees; keep ROS interface in radians.
            deg = float(attr.Get() or 0.0) if attr else 0.0
            self._targets[name] = math.radians(deg)
        self._initial_targets = dict(self._targets)

        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore
            from sensor_msgs.msg import JointState  # type: ignore
            from std_msgs.msg import Bool  # type: ignore
            from std_msgs.msg import Int32  # type: ignore

            self._rclpy = rclpy
            self._joint_msg_type = JointState
            self._ready_msg_type = Bool
            self._int_msg_type = Int32
            if not rclpy.ok():
                rclpy.init(args=None)
                self._inited_here = True
            self._node = Node("excavator_joint_bridge")
            self._pub = self._node.create_publisher(JointState, "/excavator/joint_states", 50)
            self._ready_pub = self._node.create_publisher(Bool, "/excavator/ready", 10)
            self._stones_pub = self._node.create_publisher(Int32, "/excavator/stones_in_truck", 10)
            self._sub = self._node.create_subscription(JointState, "/excavator/cmd_joint", self._on_cmd_joint, 50)
            self._reset_sub = self._node.create_subscription(Int32, "/excavator/reset", self._on_reset, 10)
            print(
                f"[sim] ROS joint bridge ready: sub=/excavator/cmd_joint,/excavator/reset pub=/excavator/joint_states,/excavator/ready,/excavator/stones_in_truck joints={list(self._joint_prims.keys())}",
                flush=True,
            )
            self._publish_ready()
        except Exception as exc:
            print(f"[sim] ROS joint bridge disabled (rclpy unavailable): {exc}", flush=True)

    def _on_cmd_joint(self, msg):
        names = list(msg.name)
        positions = list(msg.position)
        if not names or not positions:
            return
        for i, name in enumerate(names):
            if i >= len(positions):
                break
            if name in self._joint_prims:
                # ROS command uses radians.
                self._targets[name] = float(positions[i])

    def _on_reset(self, msg):
        data = getattr(msg, "data", 0)
        if int(data) == 1:
            self._reset_requested = True

    def consume_reset_requested(self) -> bool:
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False

    def reset_targets_to_initial(self):
        # Reset commanded joint goals so the excavator does not keep following
        # stale pre-reset commands.
        self._targets = dict(self._initial_targets)
        self._apply_targets()
        # Reset publish time base so joint_states resumes immediately after
        # simulation step counter is reset to 0.
        self._last_pub_wall_time = -1.0

    def _apply_targets(self):
        for name, target in self._targets.items():
            prim = self._joint_prims.get(name)
            if prim is None:
                continue
            attr = prim.GetAttribute("drive:angular:physics:targetPosition")
            if not attr:
                attr = prim.CreateAttribute("drive:angular:physics:targetPosition", self._Sdf.ValueTypeNames.Float)
            # PhysX angular drive target uses degrees.
            attr.Set(float(math.degrees(target)))

    def _publish_joint_states(self, sim_time_s: float):
        if self._node is None:
            return
        msg = self._joint_msg_type()
        # Publish in stable order for consumers.
        names = sorted(self._joint_prims.keys())
        msg.name = names
        msg.position = []
        msg.velocity = []
        for name in names:
            prim = self._joint_prims[name]
            pos_attr = prim.GetAttribute("state:angular:physics:position")
            vel_attr = prim.GetAttribute("state:angular:physics:velocity")
            tgt_attr = prim.GetAttribute("drive:angular:physics:targetPosition")
            # PhysX state angular position/velocity are degrees(/s); publish radians(/s) to ROS.
            pos_deg = float(pos_attr.Get()) if pos_attr and pos_attr.Get() is not None else float(tgt_attr.Get() or 0.0)
            vel_deg = float(vel_attr.Get()) if vel_attr and vel_attr.Get() is not None else 0.0
            pos = math.radians(pos_deg)
            vel = math.radians(vel_deg)
            msg.position.append(pos)
            msg.velocity.append(vel)
        try:
            from rclpy.time import Time  # type: ignore

            msg.header.stamp = Time(seconds=float(sim_time_s)).to_msg()
        except Exception:
            msg.header.stamp = self._node.get_clock().now().to_msg()
        self._pub.publish(msg)

    def _publish_ready(self):
        if self._node is None:
            return
        msg = self._ready_msg_type()
        msg.data = bool(self._ready_state)
        self._ready_pub.publish(msg)

    def set_ready(self, ready: bool):
        ready = bool(ready)
        if self._ready_state != ready:
            self._ready_state = ready
            self._publish_ready()
            if self._node is not None:
                self._node.get_logger().info(f"/excavator/ready -> {self._ready_state}")

    def set_stones_in_truck_count(self, count: int):
        self._stones_in_truck_count = max(0, int(count))

    def _publish_stones_in_truck(self):
        if self._node is None or self._int_msg_type is None or self._stones_pub is None:
            return
        msg = self._int_msg_type()
        msg.data = int(self._stones_in_truck_count)
        self._stones_pub.publish(msg)

    def tick(self, sim_time_s: float):
        if self._rclpy is None or self._node is None:
            return
        now = time.monotonic()
        self._rclpy.spin_once(self._node, timeout_sec=0.0)
        self._apply_targets()
        if self._last_ready_pub_wall_time < 0.0 or (now - self._last_ready_pub_wall_time) >= self._ready_pub_period:
            self._publish_ready()
            self._last_ready_pub_wall_time = now
        if self._last_pub_wall_time < 0.0 or (now - self._last_pub_wall_time) >= self._pub_period:
            self._publish_joint_states(sim_time_s)
            self._last_pub_wall_time = now
        if self._last_stones_pub_wall_time < 0.0 or (now - self._last_stones_pub_wall_time) >= self._stones_pub_period:
            self._publish_stones_in_truck()
            self._last_stones_pub_wall_time = now

    def close(self):
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        if self._rclpy is not None and self._inited_here and self._rclpy.ok():
            try:
                self._rclpy.shutdown()
            except Exception:
                pass


def _import_simulation_app(headless: bool):
    from isaacsim import SimulationApp  # type: ignore

    return SimulationApp({"headless": headless})


def _enable_extensions() -> None:
    import omni.kit.app  # type: ignore

    kit_app = omni.kit.app.get_app()
    if kit_app is None:
        raise RuntimeError("omni.kit.app.get_app() returned None")
    ext_mgr = kit_app.get_extension_manager()

    required_exts = ["isaacsim.ros2.bridge", "isaacsim.asset.importer.urdf"]
    for ext_name in required_exts:
        was_enabled = bool(ext_mgr.is_extension_enabled(ext_name))
        ext_mgr.set_extension_enabled_immediate(ext_name, True)
        now_enabled = bool(ext_mgr.is_extension_enabled(ext_name))
        print(f"[sim] enable request sent: {ext_name} (before={was_enabled}, after={now_enabled})", flush=True)
        if not now_enabled:
            raise RuntimeError(f"failed to enable required extension: {ext_name}")


def _import_urdf_asset(urdf_path: str) -> str | None:
    import omni.kit.commands  # type: ignore

    if not Path(urdf_path).exists():
        return None
    ok, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not ok:
        return None
    import_config.fix_base = True
    import_config.merge_fixed_joints = False
    import_config.make_default_prim = False
    import_config.create_physics_scene = False
    ok, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        # In Isaac Sim 5.1 this expects a USD file path; empty means import to current stage.
        dest_path="",
        get_articulation_root=False,
    )
    if not ok or not imported_path:
        return None
    imported_path = str(imported_path)
    if not imported_path.startswith("/"):
        imported_path = f"/{imported_path}"
    return imported_path


def _add_dynamic_box(
    stage,
    prim_path: str,
    size=(1.0, 1.0, 1.0),
    translation=(0.0, 0.0, 0.5),
    color=(0.5, 0.3, 0.1),
    density: float = 100.0,
):
    from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore
    from omni.physx.scripts import physicsUtils  # type: ignore

    # Let physicsUtils create/populate transform ops to avoid xform-op incompatibility warnings.
    physicsUtils.add_rigid_box(
        stage,
        prim_path,
        size=Gf.Vec3f(*size),
        position=Gf.Vec3f(*translation),
        orientation=Gf.Quatf(1.0),
        color=Gf.Vec3f(*color),
        density=float(density),
    )
    # Set display color after creation.
    cube = UsdGeom.Cube.Get(stage, prim_path)
    if cube:
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    # Explicit mass properties to avoid transient invalid inertia/mass warnings.
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        mass = max(0.01, float(density) * sx * sy * sz)
        ixx = max(1e-6, (1.0 / 12.0) * mass * (sy * sy + sz * sz))
        iyy = max(1e-6, (1.0 / 12.0) * mass * (sx * sx + sz * sz))
        izz = max(1e-6, (1.0 / 12.0) * mass * (sx * sx + sy * sy))
        mass_api.CreateMassAttr().Set(mass)
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(ixx, iyy, izz))


def _add_visual_box(stage, prim_path: str, size=(1.0, 1.0, 1.0), translation=(0.0, 0.0, 0.5), color=(0.5, 0.3, 0.1)):
    from pxr import Gf, UsdGeom  # type: ignore

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*translation))
    xform.SetScale(Gf.Vec3f(size[0], size[1], size[2]))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _add_static_collision_box(stage, prim_path: str, size=(1.0, 1.0, 1.0), translation=(0.0, 0.0, 0.5), color=(0.5, 0.3, 0.1)):
    from pxr import UsdPhysics  # type: ignore

    _add_visual_box(stage, prim_path, size=size, translation=translation, color=color)
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        if not UsdPhysics.CollisionAPI.Get(stage, prim.GetPath()):
            UsdPhysics.CollisionAPI.Apply(prim)


def _create_high_friction_material(stage, cfg: SceneRandomization) -> str:
    from pxr import UsdPhysics, UsdShade  # type: ignore

    mat_path = "/World/Physics_Materials/HighFriction"
    UsdShade.Material.Define(stage, mat_path)
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not mat_prim or not mat_prim.IsValid():
        return mat_path

    mat_api = UsdPhysics.MaterialAPI.Get(stage, mat_prim.GetPath())
    if not mat_api:
        mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
    mat_api.CreateStaticFrictionAttr().Set(float(cfg.friction_static))
    mat_api.CreateDynamicFrictionAttr().Set(float(cfg.friction_dynamic))
    mat_api.CreateRestitutionAttr().Set(float(cfg.friction_restitution))
    return mat_path


def _bind_physics_material(stage, prim_path: str, material_path: str):
    from pxr import UsdShade  # type: ignore

    prim = stage.GetPrimAtPath(prim_path)
    mat_prim = stage.GetPrimAtPath(material_path)
    if not prim or not prim.IsValid() or not mat_prim or not mat_prim.IsValid():
        return

    material = UsdShade.Material(mat_prim)
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    # Bind as physics material purpose.
    binding.Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")


def _add_open_truck_shell(
    stage,
    root_path: str,
    bottom_center_xy: tuple[float, float],
    cfg: SceneRandomization,
    physics_material_path: str | None = None,
):
    x, y = bottom_center_xy
    t = cfg.truck_thickness
    lx, ly, h = cfg.truck_len, cfg.truck_wid, cfg.truck_h
    z0 = 0.0  # bottom level
    _add_static_collision_box(stage, f"{root_path}/floor", size=(lx, ly, t), translation=(x, y, z0 + t * 0.5), color=(0.2, 0.35, 0.7))
    _add_static_collision_box(
        stage,
        f"{root_path}/left_wall",
        size=(lx, t, h),
        translation=(x, y + (ly * 0.5 - t * 0.5), z0 + h * 0.5),
        color=(0.2, 0.35, 0.7),
    )
    _add_static_collision_box(
        stage,
        f"{root_path}/right_wall",
        size=(lx, t, h),
        translation=(x, y - (ly * 0.5 - t * 0.5), z0 + h * 0.5),
        color=(0.2, 0.35, 0.7),
    )
    _add_static_collision_box(
        stage,
        f"{root_path}/back_wall",
        size=(t, ly, h),
        translation=(x - (lx * 0.5 - t * 0.5), y, z0 + h * 0.5),
        color=(0.2, 0.35, 0.7),
    )
    _add_static_collision_box(
        stage,
        f"{root_path}/front_wall",
        size=(t, ly, h),
        translation=(x + (lx * 0.5 - t * 0.5), y, z0 + h * 0.5),
        color=(0.2, 0.35, 0.7),
    )
    if physics_material_path:
        for part in ["floor", "left_wall", "right_wall", "back_wall", "front_wall"]:
            _bind_physics_material(stage, f"{root_path}/{part}", physics_material_path)


def _build_stone_specs(rng: random.Random, cfg: SceneRandomization) -> list[StoneSpec]:
    specs: list[StoneSpec] = []
    while len(specs) < cfg.stone_count:
        size = rng.uniform(cfg.stone_size_min, cfg.stone_size_max)
        specs.append(
            StoneSpec(
                sx=size,
                sy=size * rng.uniform(0.7, 1.3),
                sz=size * rng.uniform(0.7, 1.3),
                z=rng.uniform(cfg.stone_drop_height_min, cfg.stone_drop_height_max),
            )
        )
    return specs


def _setup_stone_pile_root(stage, root_path: str, rng: random.Random, cfg: SceneRandomization):
    from pxr import UsdGeom  # type: ignore

    UsdGeom.Xform.Define(stage, root_path)
    cx = rng.uniform(cfg.pile_x_min, cfg.pile_x_max)
    cy = rng.uniform(cfg.pile_y_min, cfg.pile_y_max)
    return (cx, cy), _build_stone_specs(rng, cfg)


def _spawn_one_stone(
    stage,
    root_path: str,
    center_xy: tuple[float, float],
    index: int,
    spec: StoneSpec,
    stone_density: float,
    physics_material_path: str | None = None,
):
    from pxr import Gf, Sdf, UsdPhysics  # type: ignore

    cx, cy = center_xy
    prim_path = f"{root_path}/stone_{index:03d}"
    _add_dynamic_box(
        stage,
        prim_path,
        size=(spec.sx, spec.sy, spec.sz),
        translation=(cx, cy, spec.z),
        color=(0.42, 0.28, 0.16),
        density=stone_density,
    )
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        # 5.1 API: Get() expects (stage, path), not a Prim.
        rb = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
        if not rb:
            rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        vel_attr = prim.GetAttribute("physics:velocity")
        if not vel_attr:
            vel_attr = prim.CreateAttribute("physics:velocity", Sdf.ValueTypeNames.Vector3f)
        vel_attr.Set(Gf.Vec3f(0.0, 0.0, -0.2))
        if physics_material_path:
            _bind_physics_material(stage, prim_path, physics_material_path)


def _ensure_fallback_stage_light(stage):
    from pxr import Gf, UsdGeom, UsdLux  # type: ignore

    if not stage.GetPrimAtPath("/World/Lights").IsValid():
        UsdGeom.Xform.Define(stage, "/World/Lights")
    if not stage.GetPrimAtPath("/World/Lights/ambient").IsValid():
        ambient = UsdLux.DomeLight.Define(stage, "/World/Lights/ambient")
        ambient.CreateIntensityAttr(1800.0)
        ambient.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    if not stage.GetPrimAtPath("/World/Lights/defaultLight").IsValid():
        default_light = UsdLux.SphereLight.Define(stage, "/World/Lights/defaultLight")
        default_light.CreateIntensityAttr(90000.0)
        default_light.CreateRadiusAttr(1.0)
        default_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.XformCommonAPI(default_light.GetPrim()).SetTranslate((6.5, 0.0, 12.0))


def _hold_articulation_pose(stage, root_prim_path: str):
    from pxr import Sdf, UsdPhysics  # type: ignore

    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return

    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(root_prim_path + "/"):
            continue
        if "RevoluteJoint" not in prim.GetTypeName():
            continue

        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(30000.0)
        drive.CreateDampingAttr(6000.0)
        drive.CreateMaxForceAttr(2.0e6)

        # Keep current authored/default pose unless user sends command.
        target_attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        if not target_attr or not target_attr.HasAuthoredValue():
            prim.CreateAttribute(
                "drive:angular:physics:targetPosition",
                Sdf.ValueTypeNames.Float,
            ).Set(0.0)


def _setup_physics_world():
    from isaacsim.core.api import World  # type: ignore

    return World(stage_units_in_meters=1.0)


def _attach_sensors(excavator_prim: str):
    from isaacsim.sensors.camera import Camera  # type: ignore
    from isaacsim.sensors.rtx import LidarRtx  # type: ignore
    import omni.usd  # type: ignore
    from pxr import UsdGeom  # type: ignore
    
    stage = omni.usd.get_context().get_stage()
    driver_mount_prim = excavator_prim
    bucket_mount_prim = excavator_prim
    if stage:
        # Mount driver camera on house_link and bucket camera on arm_link when available.
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if not p.startswith(excavator_prim + "/"):
                continue
            if prim.GetName() == "house_link":
                driver_mount_prim = p
            elif prim.GetName() == "arm_link":
                bucket_mount_prim = p

    camera_driver = Camera(
        prim_path=f"{driver_mount_prim}/camera_driver",
        # Driver camera: center-forward viewpoint on upper body.
        position=(0.75, 0.0, 1.35),
        frequency=20,
        resolution=(640, 360),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    camera_bucket = Camera(
        prim_path=f"{bucket_mount_prim}/camera_bucket",
        # Bucket camera: mounted on arm_link looking toward bucket area.
        position=(3.7, -0.4, 2.3),
        frequency=20,
        resolution=(640, 360),
        orientation=(0.0, 0.382683, 0.0, -0.923880),
    )
    lidar = LidarRtx(
        prim_path=f"{excavator_prim}/lidar",
        position=(0.75, 0.0, 2.0),
        # Face down: rotate +90 deg around Y (w, x, y, z).
        orientation=(0.7071, 0.0, 0.7071, 0.0),
    )
    camera_driver.initialize()
    camera_bucket.initialize()
    lidar.initialize()
    lidar_prim_path = ""
    if stage:
        # Prefer lidar prim under excavator; fall back to any lidar-like prim.
        under_excavator: list[str] = []
        fallback_any: list[str] = []
        for p in stage.Traverse():
            path = p.GetPath().pathString
            name = p.GetName().lower()
            type_name = p.GetTypeName().lower()
            if "lidar" not in name and "lidar" not in type_name:
                continue
            if path.startswith(excavator_prim + "/"):
                under_excavator.append(path)
            else:
                fallback_any.append(path)
        if under_excavator:
            lidar_prim_path = sorted(under_excavator)[0]
        elif fallback_any:
            lidar_prim_path = sorted(fallback_any)[0]
        print(f"[sim] resolved lidar prim: {lidar_prim_path or '(none)'}", flush=True)
    # Keep aperture ratio consistent with image ratio to silence camera warnings.
    if stage:
        for cam_path in (f"{driver_mount_prim}/camera_driver", f"{bucket_mount_prim}/camera_bucket"):
            cam_prim = stage.GetPrimAtPath(cam_path)
            if cam_prim and cam_prim.IsValid():
                usd_cam = UsdGeom.Camera(cam_prim)
                if usd_cam:
                    # Set focal length to 8mm as requested.
                    usd_cam.GetFocalLengthAttr().Set(8.0)
                    horizontal = float(usd_cam.GetHorizontalApertureAttr().Get() or 2.0955)
                    vertical = horizontal * 360.0 / 640.0
                    usd_cam.GetVerticalApertureAttr().Set(vertical)
        if driver_mount_prim != excavator_prim:
            print(f"[sim] driver camera mount prim: {driver_mount_prim}")
        else:
            print(f"[sim] driver camera mount prim not found, fallback to: {excavator_prim}")
        if bucket_mount_prim != excavator_prim:
            print(f"[sim] bucket camera mount prim: {bucket_mount_prim}")
        else:
            print(f"[sim] bucket camera mount prim not found, fallback to: {excavator_prim}")

    print("[sim] sensor topics (via ROS2 bridge graph):")
    print("[sim]  /excavator/camera_driver/rgb -> sensor_msgs/Image")
    print("[sim]  /excavator/camera_bucket/rgb -> sensor_msgs/Image")
    print("[sim]  /excavator/lidar/points -> sensor_msgs/PointCloud2")
    print("[sim]  /excavator/joint_states -> sensor_msgs/JointState")
    print("[sim]  /excavator/cmd_joint <- sensor_msgs/JointState")
    print("[sim]  /excavator/reset <- std_msgs/Int32 (1 triggers reset)")
    print("[sim]  /excavator/ready -> std_msgs/Bool")
    print("[sim]  /excavator/stones_in_truck -> std_msgs/Int32")

    return {
        "driver_mount_prim": driver_mount_prim,
        "bucket_mount_prim": bucket_mount_prim,
        "camera_driver_prim": f"{driver_mount_prim}/camera_driver",
        "camera_bucket_prim": f"{bucket_mount_prim}/camera_bucket",
        "lidar_prim": lidar_prim_path,
        "width": 640,
        "height": 360,
    }


def _setup_ros2_bridge_graph(excavator_prim: str, sensor_paths: dict):
    try:
        from pxr import Sdf  # type: ignore
        import omni.graph.core as og  # type: ignore
        import omni.usd  # type: ignore

        graph_path = "/World/ROS2BridgeGraph"
        camera_driver_prim = str(sensor_paths.get("camera_driver_prim"))
        camera_bucket_prim = str(sensor_paths.get("camera_bucket_prim"))
        lidar_prim = str(sensor_paths.get("lidar_prim", ""))
        stage = omni.usd.get_context().get_stage()
        lidar_enabled = False
        if stage and lidar_prim:
            lidar_candidate = stage.GetPrimAtPath(lidar_prim)
            lidar_enabled = bool(lidar_candidate and lidar_candidate.IsValid())
        if lidar_prim and not lidar_enabled:
            print(f"[sim] lidar prim invalid, skip lidar ROS bridge: {lidar_prim}", flush=True)
        width = int(sensor_paths.get("width", 640))
        height = int(sensor_paths.get("height", 360))

        create_nodes = [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("CreateRPDriver", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("CreateRPBucket", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("CameraDriver", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CameraBucket", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]
        connect_edges = [
            ("OnPlaybackTick.outputs:tick", "Clock.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "CreateRPDriver.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "CreateRPBucket.inputs:execIn"),
            ("Context.outputs:context", "Clock.inputs:context"),
            ("Context.outputs:context", "CameraDriver.inputs:context"),
            ("Context.outputs:context", "CameraBucket.inputs:context"),
            ("ReadSimTime.outputs:simulationTime", "Clock.inputs:timeStamp"),
            ("CreateRPDriver.outputs:execOut", "CameraDriver.inputs:execIn"),
            ("CreateRPBucket.outputs:execOut", "CameraBucket.inputs:execIn"),
            ("CreateRPDriver.outputs:renderProductPath", "CameraDriver.inputs:renderProductPath"),
            ("CreateRPBucket.outputs:renderProductPath", "CameraBucket.inputs:renderProductPath"),
        ]
        set_values = [
            ("Clock.inputs:topicName", "/clock"),
            ("ReadSimTime.inputs:resetOnStop", False),
            ("CreateRPDriver.inputs:width", width),
            ("CreateRPDriver.inputs:height", height),
            ("CreateRPBucket.inputs:width", width),
            ("CreateRPBucket.inputs:height", height),
            ("CameraDriver.inputs:topicName", "/excavator/camera_driver/rgb"),
            ("CameraDriver.inputs:frameId", "excavator_camera_driver"),
            ("CameraDriver.inputs:type", "rgb"),
            ("CameraDriver.inputs:resetSimulationTimeOnStop", True),
            ("CameraDriver.inputs:frameSkipCount", 2),
            ("CameraBucket.inputs:topicName", "/excavator/camera_bucket/rgb"),
            ("CameraBucket.inputs:frameId", "excavator_camera_bucket"),
            ("CameraBucket.inputs:type", "rgb"),
            ("CameraBucket.inputs:resetSimulationTimeOnStop", True),
            ("CameraBucket.inputs:frameSkipCount", 2),
        ]
        if lidar_enabled:
            create_nodes.extend(
                [
                    ("CreateRPLidar", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("Lidar", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                ]
            )
            connect_edges.extend(
                [
                    ("OnPlaybackTick.outputs:tick", "CreateRPLidar.inputs:execIn"),
                    ("Context.outputs:context", "Lidar.inputs:context"),
                    ("CreateRPLidar.outputs:execOut", "Lidar.inputs:execIn"),
                    ("CreateRPLidar.outputs:renderProductPath", "Lidar.inputs:renderProductPath"),
                ]
            )
            set_values.extend(
                [
                    ("CreateRPLidar.inputs:width", 1),
                    ("CreateRPLidar.inputs:height", 1),
                    ("Lidar.inputs:topicName", "/excavator/lidar/points"),
                    ("Lidar.inputs:frameId", "excavator_lidar"),
                    ("Lidar.inputs:type", "point_cloud"),
                    ("Lidar.inputs:resetSimulationTimeOnStop", True),
                    ("Lidar.inputs:frameSkipCount", 2),
                ]
            )

        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: create_nodes,
                og.Controller.Keys.CONNECT: connect_edges,
                og.Controller.Keys.SET_VALUES: set_values,
            },
        )
        og.Controller.attribute(f"{graph_path}/CreateRPDriver.inputs:cameraPrim").set([Sdf.Path(camera_driver_prim)])
        og.Controller.attribute(f"{graph_path}/CreateRPBucket.inputs:cameraPrim").set([Sdf.Path(camera_bucket_prim)])
        if lidar_enabled:
            og.Controller.attribute(f"{graph_path}/CreateRPLidar.inputs:cameraPrim").set([Sdf.Path(lidar_prim)])
        print(f"[sim] ROS2 bridge graph created at {graph_path}")
    except Exception as exc:
        print(f"[sim] ROS2 graph auto-setup skipped: {exc}")
        print("[sim] Use Isaac Sim Action Graph to bind camera/lidar/joint topics to ROS2 bridge if needed.")


def _randomized_positions(rng: random.Random, cfg: SceneRandomization):
    truck_x = rng.uniform(cfg.truck_x_min, cfg.truck_x_max)
    signed = -1.0 if rng.random() < 0.5 else 1.0
    truck_y = signed * rng.uniform(cfg.truck_y_abs_min, cfg.truck_y_abs_max)
    return (truck_x, truck_y)


def _remove_prim_if_exists(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def _build_randomized_environment(
    stage,
    rng: random.Random,
    cfg: SceneRandomization,
    physics_material_path: str | None = None,
):
    _remove_prim_if_exists(stage, "/World/TruckOrBin")
    _remove_prim_if_exists(stage, "/World/SoilPile")
    truck_bottom_center = _randomized_positions(rng, cfg)
    _add_open_truck_shell(stage, "/World/TruckOrBin", truck_bottom_center, cfg, physics_material_path=physics_material_path)
    pile_center, stone_specs = _setup_stone_pile_root(stage, "/World/SoilPile", rng, cfg)
    return truck_bottom_center, pile_center, stone_specs


def _count_stones_in_truck(
    stage,
    spawned_stones: int,
    truck_bottom_center: tuple[float, float],
    cfg: SceneRandomization,
) -> int:
    from pxr import UsdGeom  # type: ignore

    if spawned_stones <= 0:
        return 0

    cx, cy = truck_bottom_center
    half_x = cfg.truck_len * 0.5 - cfg.truck_thickness
    half_y = cfg.truck_wid * 0.5 - cfg.truck_thickness
    floor_z = cfg.truck_thickness
    top_z = cfg.truck_h
    inside = 0

    for i in range(int(spawned_stones)):
        prim_path = f"/World/SoilPile/stone_{i:03d}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue

        xformable = UsdGeom.Xformable(prim)
        world_m = xformable.ComputeLocalToWorldTransform(0)
        pos = world_m.ExtractTranslation()
        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2])

        if (cx - half_x) <= x <= (cx + half_x) and (cy - half_y) <= y <= (cy + half_y) and floor_z <= z <= top_z:
            inside += 1
    return inside


def run(headless: bool, excavator_asset: str | None, seed: int | None):
    print("[sim] creating SimulationApp", flush=True)
    print(f"[sim] requested asset: {excavator_asset}", flush=True)
    if not excavator_asset or not str(excavator_asset).lower().endswith(".urdf"):
        raise ValueError(f"[sim] only URDF is supported now, got: {excavator_asset}")
    if not Path(excavator_asset).exists():
        raise FileNotFoundError(f"[sim] URDF asset not found: {excavator_asset}")

    simulation_app = _import_simulation_app(headless=headless)
    joint_bridge: Optional[_RosJointBridge] = None
    try:
        print("[sim] enabling extensions", flush=True)
        _enable_extensions()
        print("[sim] setting up world", flush=True)
        world = _setup_physics_world()
        import omni.usd  # type: ignore

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is None after World initialization")
        world.scene.add_default_ground_plane()
        _ensure_fallback_stage_light(stage)
        print("[sim] lighting: fixed stage light enabled", flush=True)

        rng = random.Random(seed)
        cfg = SceneRandomization()
        high_friction_material = _create_high_friction_material(stage, cfg)
        # Bind to default ground plane prims if present.
        _bind_physics_material(stage, "/World/defaultGroundPlane", high_friction_material)
        _bind_physics_material(stage, "/World/defaultGroundPlane/GroundPlane", high_friction_material)

        excavator_prim = "/World/Excavator"
        imported_prim = _import_urdf_asset(excavator_asset)
        if not imported_prim:
            raise RuntimeError(f"[sim] failed to import URDF: {excavator_asset}")
        excavator_prim = imported_prim
        if not excavator_prim.startswith("/World/"):
            try:
                from isaacsim.core.utils.prims import move_prim  # type: ignore

                target_prim = "/World/Excavator"
                if excavator_prim != target_prim:
                    move_prim(excavator_prim, target_prim)
                    excavator_prim = target_prim
            except Exception as exc:
                print(f"[sim] warning: failed to move imported prim to /World: {exc}", flush=True)
        print(f"[sim] URDF imported prim: {excavator_prim}", flush=True)
        _hold_articulation_pose(stage, excavator_prim)
        joint_bridge = _RosJointBridge(stage, excavator_prim)

        truck_bottom_center, pile_center, stone_specs = _build_randomized_environment(
            stage, rng, cfg, physics_material_path=high_friction_material
        )

        sensor_paths = _attach_sensors(excavator_prim)
        _setup_ros2_bridge_graph(excavator_prim, sensor_paths)

        world.reset()
        world.play()
        if joint_bridge is not None:
            joint_bridge.set_ready(False)
        print(f"[sim] randomized truck bottom-center (x, y): {truck_bottom_center}", flush=True)
        print(
            f"[sim] pile bottom range x=[{cfg.pile_x_min}, {cfg.pile_x_max}] y=[{cfg.pile_y_min}, {cfg.pile_y_max}]",
            flush=True,
        )
        print(
            f"[sim] stones: count={len(stone_specs)}, spawn interval={cfg.stone_spawn_interval_s}s",
            flush=True,
        )
        print(
            f"[sim] stones drop point (x, y): ({pile_center[0]}, {pile_center[1]}), z range=[{cfg.stone_drop_height_min}, {cfg.stone_drop_height_max}]",
            flush=True,
        )
        print(f"[sim] entering main loop (headless={headless})", flush=True)

        spawn_interval_steps = max(1, int(round(cfg.stone_spawn_interval_s / cfg.physics_dt)))
        render_every_n_steps = max(1, int(cfg.render_every_n_steps))
        step_count = 0
        stone_idx = 0
        last_stones_count_update_wall_time = -1.0
        stones_count_update_period = 1.0 / 5.0
        print(
            f"[sim] timing: physics={1.0/cfg.physics_dt:.1f}Hz, render={1.0/(cfg.physics_dt*render_every_n_steps):.1f}Hz, joint_states={1.0/joint_bridge._pub_period if joint_bridge else 0:.1f}Hz",
            flush=True,
        )

        if headless:
            # In headless mode, is_running() may become false immediately on some Isaac Sim builds.
            # Keep stepping until interrupted by user (Ctrl+C) or external process termination.
            while True:
                if joint_bridge is not None:
                    joint_bridge.tick(step_count * cfg.physics_dt)
                    if joint_bridge.consume_reset_requested():
                        joint_bridge.set_ready(False)
                        truck_bottom_center, pile_center, stone_specs = _build_randomized_environment(
                            stage, rng, cfg, physics_material_path=high_friction_material
                        )
                        stone_idx = 0
                        joint_bridge.set_stones_in_truck_count(0)
                        last_stones_count_update_wall_time = -1.0
                        step_count = 0
                        world.reset()
                        joint_bridge.reset_targets_to_initial()
                        world.play()
                        print(f"[sim] reset env: truck (x, y)=({truck_bottom_center[0]}, {truck_bottom_center[1]})", flush=True)
                        print(f"[sim] reset env: pile drop (x, y)=({pile_center[0]}, {pile_center[1]})", flush=True)
                        print("[sim] reset env: joint_states publish timer reset", flush=True)
                if stone_idx < len(stone_specs) and (step_count % spawn_interval_steps == 0):
                    _spawn_one_stone(
                        stage,
                        "/World/SoilPile",
                        pile_center,
                        stone_idx,
                        stone_specs[stone_idx],
                        stone_density=cfg.stone_density,
                        physics_material_path=high_friction_material,
                    )
                    stone_idx += 1
                    if stone_idx >= len(stone_specs) and joint_bridge is not None:
                        joint_bridge.set_ready(True)
                render_this_step = (step_count % render_every_n_steps) == 0
                world.step(render=render_this_step)
                step_count += 1
                if joint_bridge is not None:
                    now = time.monotonic()
                    if (
                        last_stones_count_update_wall_time < 0.0
                        or (now - last_stones_count_update_wall_time) >= stones_count_update_period
                    ):
                        stones_in_truck = _count_stones_in_truck(stage, stone_idx, truck_bottom_center, cfg)
                        joint_bridge.set_stones_in_truck_count(stones_in_truck)
                        last_stones_count_update_wall_time = now
        else:
            while simulation_app.is_running():
                if joint_bridge is not None:
                    joint_bridge.tick(step_count * cfg.physics_dt)
                    if joint_bridge.consume_reset_requested():
                        joint_bridge.set_ready(False)
                        truck_bottom_center, pile_center, stone_specs = _build_randomized_environment(
                            stage, rng, cfg, physics_material_path=high_friction_material
                        )
                        stone_idx = 0
                        joint_bridge.set_stones_in_truck_count(0)
                        last_stones_count_update_wall_time = -1.0
                        step_count = 0
                        world.reset()
                        joint_bridge.reset_targets_to_initial()
                        world.play()
                        print(f"[sim] reset env: truck (x, y)=({truck_bottom_center[0]}, {truck_bottom_center[1]})", flush=True)
                        print(f"[sim] reset env: pile drop (x, y)=({pile_center[0]}, {pile_center[1]})", flush=True)
                        print("[sim] reset env: joint_states publish timer reset", flush=True)
                if stone_idx < len(stone_specs) and (step_count % spawn_interval_steps == 0):
                    _spawn_one_stone(
                        stage,
                        "/World/SoilPile",
                        pile_center,
                        stone_idx,
                        stone_specs[stone_idx],
                        stone_density=cfg.stone_density,
                        physics_material_path=high_friction_material,
                    )
                    stone_idx += 1
                    if stone_idx >= len(stone_specs) and joint_bridge is not None:
                        joint_bridge.set_ready(True)
                render_this_step = (step_count % render_every_n_steps) == 0
                world.step(render=render_this_step)
                step_count += 1
                if joint_bridge is not None:
                    now = time.monotonic()
                    if (
                        last_stones_count_update_wall_time < 0.0
                        or (now - last_stones_count_update_wall_time) >= stones_count_update_period
                    ):
                        stones_in_truck = _count_stones_in_truck(stage, stone_idx, truck_bottom_center, cfg)
                        joint_bridge.set_stones_in_truck_count(stones_in_truck)
                        last_stones_count_update_wall_time = now
    except Exception:
        print("[sim] fatal error in run():", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        if joint_bridge is not None:
            joint_bridge.close()
        print("[sim] closing SimulationApp", flush=True)
        simulation_app.close()


def parse_args():
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Run excavator simulation scene")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument(
        "--asset",
        default=os.environ.get("EXCAVATOR_ASSET_PATH", str(paths.assets / "excavator_4dof" / "excavator_4dof.urdf")),
        help="Excavator URDF asset path",
    )
    seed_env = os.environ.get("EXCAVATOR_SEED", "").strip()
    default_seed = int(seed_env) if seed_env else None
    parser.add_argument("--seed", type=int, default=default_seed, help="Random seed; omit for different layout each run")
    return parser.parse_args()


def main():
    args = parse_args()
    run(
        headless=args.headless,
        excavator_asset=args.asset,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
