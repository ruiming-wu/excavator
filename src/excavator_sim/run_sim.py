from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

from excavator_sim.common import get_paths


@dataclass
class SceneRandomization:
    truck_x_min: float = 4.0
    truck_x_max: float = 7.0
    truck_y_min: float = -2.0
    truck_y_max: float = 2.0
    pile_x_min: float = 1.0
    pile_x_max: float = 3.0
    pile_y_min: float = -1.5
    pile_y_max: float = 1.5


def _import_simulation_app(headless: bool):
    # Isaac Sim 5.x typically uses isaacsim.simulation_app; older releases used omni.isaac.kit.
    try:
        from isaacsim import SimulationApp  # type: ignore

        return SimulationApp({"headless": headless})
    except Exception:
        from omni.isaac.kit import SimulationApp  # type: ignore

        return SimulationApp({"headless": headless})


def _enable_extensions(app) -> None:
    ext_mgr = app.get_extension_manager()
    # ROS2 bridge extension name differs across versions; try both.
    for ext_name in ["isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"]:
        if ext_mgr.get_extension_id(ext_name) != "":
            ext_mgr.set_extension_enabled_immediate(ext_name, True)


def _create_stage(scene_usd: str | None):
    import omni.usd  # type: ignore

    ctx = omni.usd.get_context()
    if scene_usd and Path(scene_usd).exists():
        ctx.open_stage(scene_usd)
    else:
        ctx.new_stage()
    return ctx.get_stage()


def _add_reference(stage, prim_path: str, asset_path: str) -> bool:
    from pxr import Sdf, UsdGeom  # type: ignore

    if not Path(asset_path).exists():
        return False
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(asset_path)
    UsdGeom.Xformable(prim)
    return True


def _add_dynamic_box(stage, prim_path: str, size=(1.0, 1.0, 1.0), translation=(0.0, 0.0, 0.5), color=(0.5, 0.3, 0.1)):
    from pxr import Gf, UsdGeom  # type: ignore
    from omni.physx.scripts import physicsUtils  # type: ignore

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(max(size))
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*translation))
    xform.SetScale(Gf.Vec3f(size[0], size[1], size[2]))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    physicsUtils.add_rigid_box(
        stage,
        prim_path,
        size=Gf.Vec3f(*size),
        position=Gf.Vec3f(*translation),
        orientation=Gf.Quatf(1.0),
        color=Gf.Vec3f(*color),
        density=100.0,
    )


def _setup_physics_world():
    from omni.isaac.core import World  # type: ignore

    return World(stage_units_in_meters=1.0)


def _attach_sensors(world, excavator_prim: str):
    # This keeps the implementation version-tolerant; the canonical ROS2 bridge graph is created in bridge UIs.
    # We instantiate basic Isaac sensors so the prims exist and can be wired to ROS2 topics.
    from omni.isaac.sensor import Camera, LidarRtx  # type: ignore

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
                    ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                    ("Clock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "Clock.inputs:execIn"),
                    ("Context.outputs:context", "Clock.inputs:context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("Clock.inputs:topicName", "/clock"),
                ],
            },
        )
        print(f"[sim] ROS2 bridge graph created at {graph_path}")
    except Exception as exc:
        print(f"[sim] ROS2 graph auto-setup skipped: {exc}")
        print("[sim] Use Isaac Sim Action Graph to bind camera/lidar/joint topics to ROS2 bridge if needed.")


def _randomized_positions(rng: random.Random, cfg: SceneRandomization):
    truck = (
        rng.uniform(cfg.truck_x_min, cfg.truck_x_max),
        rng.uniform(cfg.truck_y_min, cfg.truck_y_max),
        1.2,
    )
    pile = (
        rng.uniform(cfg.pile_x_min, cfg.pile_x_max),
        rng.uniform(cfg.pile_y_min, cfg.pile_y_max),
        0.6,
    )
    return truck, pile


def run(headless: bool, scene_usd: str | None, excavator_asset: str | None, seed: int):
    simulation_app = _import_simulation_app(headless=headless)
    try:
        _enable_extensions(simulation_app)
        stage = _create_stage(scene_usd)

        world = _setup_physics_world()
        world.scene.add_default_ground_plane()

        rng = random.Random(seed)
        truck_pos, pile_pos = _randomized_positions(rng, SceneRandomization())

        from pxr import UsdGeom  # type: ignore

        excavator_prim = "/World/Excavator"
        excavator_loaded = False
        if excavator_asset:
            excavator_loaded = _add_reference(stage, excavator_prim, excavator_asset)

        if not excavator_loaded:
            # Fallback placeholder body to keep the world valid when asset is absent.
            UsdGeom.Xform.Define(stage, excavator_prim)
            _add_dynamic_box(stage, f"{excavator_prim}/base", size=(2.8, 1.2, 1.0), translation=(0.0, 0.0, 0.5), color=(0.9, 0.8, 0.2))
            print(f"[sim] excavator asset not found: {excavator_asset}")

        _add_dynamic_box(stage, "/World/TruckOrBin", size=(3.0, 2.0, 2.4), translation=truck_pos, color=(0.2, 0.35, 0.7))
        _add_dynamic_box(stage, "/World/SoilPile", size=(1.8, 1.8, 1.2), translation=pile_pos, color=(0.45, 0.25, 0.1))

        _attach_sensors(world, excavator_prim)
        _setup_ros2_bridge_graph(excavator_prim)

        world.reset()
        print(f"[sim] randomized truck position: {truck_pos}")
        print(f"[sim] randomized pile position: {pile_pos}")

        while simulation_app.is_running():
            world.step(render=True)
    finally:
        simulation_app.close()


def parse_args():
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Run excavator simulation scene")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--scene-usd", default=os.environ.get("EXCAVATOR_SCENE_USD", ""), help="Optional scene USD")
    parser.add_argument(
        "--asset",
        default=os.environ.get("EXCAVATOR_ASSET_PATH", str(paths.assets / "excavator.usd")),
        help="Excavator USD asset path",
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("EXCAVATOR_SEED", "42")))
    return parser.parse_args()


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
