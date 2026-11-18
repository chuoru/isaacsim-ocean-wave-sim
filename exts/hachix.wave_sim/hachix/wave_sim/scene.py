"""Isaac Sim adapter: the only module that imports omni / pxr."""

from __future__ import annotations

from dataclasses import dataclass, field

import carb.eventdispatcher
import numpy as np
import omni.kit.app
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from .hull import HullMesh
from .hydrodynamics import HydroParameters, Pose, compute_wrench
from .waves import Wavefield, WaveParameters

@dataclass(frozen=True)
class RaftSpec:
    size: tuple[float, float, float] = (2.0, 1.5, 0.3)
    mass: float = 150.0
    start_xy: tuple[float, float] = (0.0, 0.0)
    drop_height: float = 0.5

    @property
    def inertia(self) -> tuple[float, float, float]:
        a, b, c = self.size
        return (self.mass * (b * b + c * c) / 12.0, self.mass * (a * a + c * c) / 12.0, self.mass * (a * a + b * b) / 12.0)


@dataclass(frozen=True)
class OceanSpec:
    extent: float = 40.0
    cells: int = 100


@dataclass(frozen=True)
class WaveSimConfig:
    waves: WaveParameters = field(default_factory=lambda: WaveParameters.vrx_ocean_waves(amplitude=0.5, period=6.0))
    raft: RaftSpec = field(default_factory=RaftSpec)
    ocean: OceanSpec = field(default_factory=OceanSpec)
    hydro: HydroParameters = field(default_factory=HydroParameters)
    physics_hz: float = 60.0
    root: str = "/World/WaveSim"

    @property
    def raft_path(self) -> str:
        return f"{self.root}/Raft"

    @property
    def water_path(self) -> str:
        return f"{self.root}/Water"


def _ocean_grid(spec: OceanSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rest xy [V,2], face vertex counts, face vertex indices for a square quad grid."""
    n = spec.cells + 1
    axis = np.linspace(-spec.extent / 2.0, spec.extent / 2.0, n)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    ij = np.arange(n * n).reshape(n, n)
    quads = np.stack([ij[:-1, :-1], ij[1:, :-1], ij[1:, 1:], ij[:-1, 1:]], axis=-1).reshape(-1, 4)
    return np.column_stack([xx.ravel(), yy.ravel()]), np.full(len(quads), 4), quads.ravel()


def _preview_material(stage: Usd.Stage, path: str, color: tuple[float, float, float], roughness: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


class WaveSimScene:
    """Water surface mesh plus one floating raft, driven by physics-step and app-update callbacks."""

    def __init__(self, config: WaveSimConfig) -> None:
        self.config = config
        self.wavefield = Wavefield.from_parameters(config.waves)
        self.hull = HullMesh.box(config.raft.size)
        self._grid_xy, self._face_counts, self._face_indices = _ocean_grid(config.ocean)
        self._raft: RigidPrim | None = None
        self._water_points: Usd.Attribute | None = None
        self._physics_sub: int | None = None
        self._update_sub = None

    def build(self, stage: Usd.Stage) -> None:
        """Author the scene under `config.root`, replacing any previous copy."""
        cfg = self.config
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.RemovePrim(cfg.root)
        UsdGeom.Xform.Define(stage, cfg.root)

        scene = UsdPhysics.Scene.Define(stage, f"{cfg.root}/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)

        UsdLux.DomeLight.Define(stage, f"{cfg.root}/Sky").CreateIntensityAttr(800.0)
        sun = UsdLux.DistantLight.Define(stage, f"{cfg.root}/Sun")
        sun.CreateIntensityAttr(2500.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 20.0, 0.0))

        water = UsdGeom.Mesh.Define(stage, cfg.water_path)
        water.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(self._face_counts.astype(np.int32)))
        water.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(self._face_indices.astype(np.int32)))
        water.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self._water_points = water.CreatePointsAttr()
        self._write_water(0.0)
        UsdShade.MaterialBindingAPI.Apply(water.GetPrim()).Bind(
            _preview_material(stage, f"{cfg.root}/Looks/Water", (0.02, 0.18, 0.32), 0.15)
        )

        raft = UsdGeom.Mesh.Define(stage, cfg.raft_path)
        raft.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(self.hull.vertices.astype(np.float32)))
        raft.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(self.hull.triangles), 3, dtype=np.int32)))
        raft.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(self.hull.triangles.ravel().astype(np.int32)))
        raft.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        start_z = self.wavefield.height(np.array([cfg.raft.start_xy], dtype=float), 0.0)[0] + cfg.raft.drop_height
        UsdGeom.Xformable(raft).AddTranslateOp().Set(Gf.Vec3d(*cfg.raft.start_xy, start_z))
        UsdShade.MaterialBindingAPI.Apply(raft.GetPrim()).Bind(
            _preview_material(stage, f"{cfg.root}/Looks/Raft", (0.75, 0.45, 0.15), 0.6)
        )
        prim = raft.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(cfg.raft.mass)
        # Wrenches are applied at the body origin, so the center of mass must sit there.
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*cfg.raft.inertia))

        SimulationManager.setup_simulation(dt=1.0 / cfg.physics_hz, device="cpu")

    def attach(self) -> None:
        """Start applying hydrodynamics each physics step and animating the water each app update."""
        self.detach()
        self._raft = RigidPrim(self.config.raft_path)
        self._physics_sub = SimulationManager.register_callback(self._on_physics_step, SimulationEvent.PHYSICS_PRE_STEP)
        self._update_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
            on_event=lambda _: self._write_water(SimulationManager.get_simulation_time()),
            observer_name="hachix.wave_sim.water",
        )

    def detach(self) -> None:
        if self._physics_sub is not None:
            SimulationManager.deregister_callback(self._physics_sub)
        self._physics_sub = None
        self._update_sub = None
        self._raft = None

    def raft_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulated position, orientation (wxyz), linear and angular velocity (rad/s), world frame."""
        position, orientation = self._raft.get_world_poses()
        lin_vel, ang_vel = self._raft.get_velocities()
        return position.numpy()[0], orientation.numpy()[0], lin_vel.numpy()[0], ang_vel.numpy()[0]

    def water_points(self) -> np.ndarray:
        return np.asarray(self._water_points.Get())

    def _write_water(self, t: float) -> None:
        points = self.wavefield.displace(self._grid_xy, t).astype(np.float32)
        self._water_points.Set(Vt.Vec3fArray.FromNumpy(points))

    def _on_physics_step(self, dt: float, context: object) -> None:
        if not self._raft.is_physics_tensor_entity_valid():
            return
        position, orientation, lin_vel, ang_vel = self.raft_state()
        wrench = compute_wrench(
            self.hull,
            Pose(position.astype(float), orientation.astype(float)),
            lin_vel.astype(float),
            ang_vel.astype(float),
            self.wavefield,
            SimulationManager.get_simulation_time(),
            self.config.hydro,
        )
        self._raft.apply_forces_and_torques_at_pos(wrench.force[None], wrench.torque[None])
