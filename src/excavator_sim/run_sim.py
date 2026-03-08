from __future__ import annotations

import argparse
import math
import os
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from excavator_sim.common import get_paths


@dataclass
class SceneRandomization:
    # Stage convention: x forward, y lateral, z up.
    truck_x_min: float = -2.0
    truck_x_max: float = 2.0
    truck_y_abs_min: float = 2.0
    truck_y_abs_max: float = 3.0
    pile_x_min: float = 2.5
    pile_x_max: float = 3.5
    pile_y_min: float = -1.0
    pile_y_max: float = 1.0
    truck_len: float = 3.0
    truck_wid: float = 1.8
    truck_h: float = 1.2
    truck_thickness: float = 0.04
    stone_count: int = 200
    stone_size_min: float = 0.05
    stone_size_max: float = 0.15
    stone_drop_height_min: float = 0.5
    stone_drop_height_max: float = 0.7
    stone_spawn_interval_s: float = 0.2
    physics_dt: float = 1.0 / 60.0


@dataclass
class StoneSpec:
    sx: float
    sy: float
    sz: float
    z: float


class _RosJointBridge:
    def __init__(self, stage, excavator_prim: str):
        from pxr import Sdf  # type: ignore

        self._stage = stage
        self._root = excavator_prim
        self._Sdf = Sdf
        self._rclpy = None
        self._node = None
        self._joint_msg_type = None
        self._inited_here = False
        self._last_pub_time = -1.0
        self._pub_period = 1.0 / 30.0
        self._targets: Dict[str, float] = {}
        self._joint_prims: Dict[str, object] = {}

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

        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore
            from sensor_msgs.msg import JointState  # type: ignore

            self._rclpy = rclpy
            self._joint_msg_type = JointState
            if not rclpy.ok():
                rclpy.init(args=None)
                self._inited_here = True
            self._node = Node("excavator_joint_bridge")
            self._pub = self._node.create_publisher(JointState, "/excavator/joint_states", 50)
            self._sub = self._node.create_subscription(JointState, "/excavator/cmd_joint", self._on_cmd_joint, 50)
            print(
                f"[sim] ROS joint bridge ready: sub=/excavator/cmd_joint pub=/excavator/joint_states joints={list(self._joint_prims.keys())}",
                flush=True,
            )
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

    def tick(self, sim_time_s: float):
        if self._rclpy is None or self._node is None:
            return
        self._rclpy.spin_once(self._node, timeout_sec=0.0)
        self._apply_targets()
        if self._last_pub_time < 0.0 or (sim_time_s - self._last_pub_time) >= self._pub_period:
            self._publish_joint_states(sim_time_s)
            self._last_pub_time = sim_time_s

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


def _add_reference(stage, prim_path: str, asset_path: str) -> bool:
    from pxr import Sdf, UsdGeom  # type: ignore

    if not Path(asset_path).exists():
        return False
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(asset_path)
    UsdGeom.Xformable(prim)
    return True


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


def _add_dynamic_box(stage, prim_path: str, size=(1.0, 1.0, 1.0), translation=(0.0, 0.0, 0.5), color=(0.5, 0.3, 0.1)):
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
        density=100.0,
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
        density = 100.0
        mass = max(0.01, density * sx * sy * sz)
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


def _add_open_truck_shell(stage, root_path: str, bottom_center_xy: tuple[float, float], cfg: SceneRandomization):
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


def _build_stone_specs(rng: random.Random, cfg: SceneRandomization) -> list[StoneSpec]:
    specs: list[StoneSpec] = []
    while len(specs) < cfg.stone_count:
        size = rng.uniform(cfg.stone_size_min, cfg.stone_size_max)
        specs.append(
            StoneSpec(
                sx=size,
                sy=size,
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


def _spawn_one_stone(stage, root_path: str, center_xy: tuple[float, float], index: int, spec: StoneSpec):
    from pxr import Gf, Sdf, UsdPhysics  # type: ignore

    cx, cy = center_xy
    prim_path = f"{root_path}/stone_{index:03d}"
    _add_dynamic_box(
        stage,
        prim_path,
        size=(spec.sx, spec.sy, spec.sz),
        translation=(cx, cy, spec.z),
        color=(0.42, 0.28, 0.16),
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


def _setup_lighting(stage):
    from pxr import Gf, UsdGeom, UsdLux  # type: ignore

    UsdGeom.Xform.Define(stage, "/World/Lights")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(1800.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(4500.0)
    key.CreateAngleAttr(0.53)
    UsdGeom.XformCommonAPI(key.GetPrim()).SetRotate((45.0, -30.0, 0.0))


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
        drive.CreateStiffnessAttr(120000.0)
        drive.CreateDampingAttr(18000.0)
        drive.CreateMaxForceAttr(1.0e9)

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


def _attach_sensors(world, excavator_prim: str):
    from isaacsim.sensors.camera import Camera  # type: ignore
    from isaacsim.sensors.rtx import LidarRtx  # type: ignore
    import omni.usd  # type: ignore
    from pxr import UsdGeom  # type: ignore

    camera = Camera(
        prim_path=f"{excavator_prim}/camera_front",
        position=(2.0, 0.0, 2.0),
        frequency=20,
        resolution=(1280, 720),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    lidar = LidarRtx(
        prim_path=f"{excavator_prim}/lidar",
        position=(1.5, 0.0, 2.2),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    camera.initialize()
    lidar.initialize()
    # Keep aperture ratio consistent with image ratio to silence camera warning.
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(f"{excavator_prim}/camera_front") if stage else None
    if cam_prim and cam_prim.IsValid():
        usd_cam = UsdGeom.Camera(cam_prim)
        if usd_cam:
            horizontal = float(usd_cam.GetHorizontalApertureAttr().Get() or 2.0955)
            vertical = horizontal * 720.0 / 1280.0
            usd_cam.GetVerticalApertureAttr().Set(vertical)

    print("[sim] sensor topics (via ROS2 bridge graph):")
    print("[sim]  /excavator/camera_front/rgb -> sensor_msgs/Image")
    print("[sim]  /excavator/lidar/points -> sensor_msgs/PointCloud2")
    print("[sim]  /excavator/joint_states -> sensor_msgs/JointState")


def _setup_ros2_bridge_graph(excavator_prim: str):
    try:
        import omni.graph.core as og  # type: ignore

        graph_path = "/World/ROS2BridgeGraph"
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "Clock.inputs:execIn"),
                    ("Context.outputs:context", "Clock.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime", "Clock.inputs:timeStamp"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("Clock.inputs:topicName", "/clock"),
                    ("ReadSimTime.inputs:resetOnStop", False),
                ],
            },
        )
        print(f"[sim] ROS2 bridge graph created at {graph_path}")
    except Exception as exc:
        print(f"[sim] ROS2 graph auto-setup skipped: {exc}")
        print("[sim] Use Isaac Sim Action Graph to bind camera/lidar/joint topics to ROS2 bridge if needed.")


def _randomized_positions(rng: random.Random, cfg: SceneRandomization):
    truck_x = rng.uniform(cfg.truck_x_min, cfg.truck_x_max)
    signed = -1.0 if rng.random() < 0.5 else 1.0
    truck_y = signed * rng.uniform(cfg.truck_y_abs_min, cfg.truck_y_abs_max)
    return (truck_x, truck_y)


def run(headless: bool, scene_usd: str | None, excavator_asset: str | None, seed: int | None):
    print("[sim] creating SimulationApp", flush=True)
    print(f"[sim] requested asset: {excavator_asset}", flush=True)
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
        if scene_usd:
            print(
                f"[sim] warning: --scene-usd is currently ignored in 5.1 stability mode: {scene_usd}",
                flush=True,
            )
        world.scene.add_default_ground_plane()
        _setup_lighting(stage)

        rng = random.Random(seed)
        cfg = SceneRandomization()
        truck_bottom_center = _randomized_positions(rng, cfg)

        from pxr import UsdGeom  # type: ignore

        excavator_prim = "/World/Excavator"
        excavator_loaded = False
        if excavator_asset:
            if excavator_asset.lower().endswith(".urdf"):
                imported_prim = _import_urdf_asset(excavator_asset)
                excavator_loaded = imported_prim is not None
                if imported_prim:
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
            else:
                excavator_loaded = _add_reference(stage, excavator_prim, excavator_asset)

        if not excavator_loaded:
            # Fallback placeholder body to keep the world valid when asset is absent.
            UsdGeom.Xform.Define(stage, excavator_prim)
            _add_dynamic_box(stage, f"{excavator_prim}/base", size=(2.8, 1.2, 1.0), translation=(0.0, 0.0, 0.5), color=(0.9, 0.8, 0.2))
            print(f"[sim] excavator asset not found: {excavator_asset}", flush=True)

        _add_open_truck_shell(stage, "/World/TruckOrBin", truck_bottom_center, cfg)
        pile_center, stone_specs = _setup_stone_pile_root(stage, "/World/SoilPile", rng, cfg)

        _attach_sensors(world, excavator_prim)
        _setup_ros2_bridge_graph(excavator_prim)

        world.reset()
        world.play()
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
        step_count = 0
        stone_idx = 0

        if headless:
            # In headless mode, is_running() may become false immediately on some Isaac Sim builds.
            # Keep stepping until interrupted by user (Ctrl+C) or external process termination.
            while True:
                if joint_bridge is not None:
                    joint_bridge.tick(step_count * cfg.physics_dt)
                if stone_idx < len(stone_specs) and (step_count % spawn_interval_steps == 0):
                    _spawn_one_stone(stage, "/World/SoilPile", pile_center, stone_idx, stone_specs[stone_idx])
                    stone_idx += 1
                world.step(render=False)
                step_count += 1
        else:
            while simulation_app.is_running():
                if joint_bridge is not None:
                    joint_bridge.tick(step_count * cfg.physics_dt)
                if stone_idx < len(stone_specs) and (step_count % spawn_interval_steps == 0):
                    _spawn_one_stone(stage, "/World/SoilPile", pile_center, stone_idx, stone_specs[stone_idx])
                    stone_idx += 1
                world.step(render=True)
                step_count += 1
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
    parser.add_argument("--scene-usd", default=os.environ.get("EXCAVATOR_SCENE_USD", ""), help="Optional scene USD")
    parser.add_argument(
        "--asset",
        default=os.environ.get("EXCAVATOR_ASSET_PATH", str(paths.assets / "excavator_4dof" / "excavator_4dof.urdf")),
        help="Excavator asset path (.urdf or .usd)",
    )
    seed_env = os.environ.get("EXCAVATOR_SEED", "").strip()
    default_seed = int(seed_env) if seed_env else None
    parser.add_argument("--seed", type=int, default=default_seed, help="Random seed; omit for different layout each run")
    args = parser.parse_args()
    if args.asset and not Path(args.asset).exists():
        fallback = paths.assets / "excavator_4dof" / "excavator_4dof.urdf"
        if fallback.exists():
            print(f"[sim] asset not found, fallback to: {fallback}", flush=True)
            args.asset = str(fallback)
    return args


def main():
    args = parse_args()
    run(
        headless=args.headless,
        scene_usd=args.scene_usd or None,
        excavator_asset=args.asset,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
