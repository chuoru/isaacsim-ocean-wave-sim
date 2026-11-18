"""Isaac Sim adapter: the only plugin module that imports omni / pxr.

`Ocean` animates a water mesh from a wavefield. `Floater` applies hull hydrodynamics to a rigid body that
already exists on the stage. Neither touches the physics scene, lights or stage settings; that is the
caller's scene, as in `examples/raft.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import carb.eventdispatcher
import numpy as np
import omni.kit.app
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from .hull import HullMesh
from .hydrodynamics import HydroParameters, Pose, compute_wrench
from .waves import Wavefield, WaveParameters


@dataclass(frozen=True)
class OceanSpec:
    waves: WaveParameters = field(default_factory=WaveParameters.vrx_ocean_waves)
    extent: float = 40.0
    cells: int = 100
    path: str = "/World/Ocean"
    color: tuple[float, float, float] = (0.02, 0.18, 0.32)


def _ocean_grid(extent: float, cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rest xy [V,2], face vertex counts, face vertex indices for a square quad grid."""
    n = cells + 1
    axis = np.linspace(-extent / 2.0, extent / 2.0, n)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    ij = np.arange(n * n).reshape(n, n)
    quads = np.stack([ij[:-1, :-1], ij[1:, :-1], ij[1:, 1:], ij[:-1, 1:]], axis=-1).reshape(-1, 4)
    return np.column_stack([xx.ravel(), yy.ravel()]), np.full(len(quads), 4), quads.ravel()


def preview_material(stage: Usd.Stage, path: str, color: tuple[float, float, float], roughness: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def hull_from_prim(prim: Usd.Prim) -> HullMesh:
    """Hull for a `UsdGeom.Cube` or `UsdGeom.Mesh`, in the prim's frame with its world scale baked in."""
    scale = np.array(Gf.Transform(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).GetScale())
    if prim.IsA(UsdGeom.Cube):
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return HullMesh.box(tuple(size * scale))
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        hull = HullMesh.from_polygons(
            np.asarray(mesh.GetPointsAttr().Get()), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        )
        return HullMesh(hull.vertices * scale, hull.triangles)
    raise TypeError(f"{prim.GetPath()} is a {prim.GetTypeName()}; pass a Cube or Mesh prim, or an explicit HullMesh")


class Ocean:
    """Water surface mesh animated from a wavefield on every app update."""

    def __init__(self, spec: OceanSpec | None = None) -> None:
        self.spec = spec or OceanSpec()
        self.wavefield = Wavefield.from_parameters(self.spec.waves)
        self._grid_xy, self._face_counts, self._face_indices = _ocean_grid(self.spec.extent, self.spec.cells)
        self._points: Usd.Attribute | None = None
        self._update_sub = None

    def build(self, stage: Usd.Stage) -> "Ocean":
        """Author the water mesh at `spec.path`, replacing any previous copy. The stage must be z-up."""
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise ValueError("isaacsim.ocean.wave_sim needs a z-up stage: call UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)")
        stage.RemovePrim(self.spec.path)
        water = UsdGeom.Mesh.Define(stage, self.spec.path)
        water.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(self._face_counts.astype(np.int32)))
        water.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(self._face_indices.astype(np.int32)))
        water.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self._points = water.CreatePointsAttr()
        self._write(0.0)
        UsdShade.MaterialBindingAPI.Apply(water.GetPrim()).Bind(
            preview_material(stage, f"{self.spec.path}/Looks/Water", self.spec.color, 0.15)
        )
        return self

    def attach(self) -> "Ocean":
        self.detach()
        self._update_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
            on_event=lambda _: self._write(SimulationManager.get_simulation_time()),
            observer_name=f"isaacsim.ocean.wave_sim.ocean:{self.spec.path}",
        )
        return self

    def detach(self) -> None:
        """Stop animating. Also happens on its own once the water prim is gone, e.g. after opening another stage."""
        self._update_sub = None

    def height(self, xy: np.ndarray, t: float | None = None) -> np.ndarray:
        """Water surface z at world xy [M,2], at simulation time `t` (default: now)."""
        return self.wavefield.height(np.asarray(xy, dtype=float), SimulationManager.get_simulation_time() if t is None else t)

    def water_points(self) -> np.ndarray:
        return np.asarray(self._points.Get())

    def _write(self, t: float) -> None:
        if not self._points.IsValid():
            self.detach()
            return
        self._points.Set(Vt.Vec3fArray.FromNumpy(self.wavefield.displace(self._grid_xy, t).astype(np.float32)))


class Floater:
    """Buoyancy, pressure drag and damping from an `Ocean`, applied to one existing rigid body each physics step."""

    def __init__(
        self,
        ocean: Ocean,
        body_path: str,
        hull: HullMesh | None = None,
        hydro: HydroParameters | None = None,
    ) -> None:
        self.ocean = ocean
        self.body_path = body_path
        self.hull = hull
        self.hydro = hydro or HydroParameters()
        self._body: RigidPrim | None = None
        self._physics_sub: int | None = None

    def attach(self, stage: Usd.Stage) -> "Floater":
        """Start applying hydrodynamics. Without an explicit hull, it is read from the body prim now."""
        self.detach()
        if self.hull is None:
            self.hull = hull_from_prim(stage.GetPrimAtPath(self.body_path))
        self._body = RigidPrim(self.body_path)
        self._physics_sub = SimulationManager.register_callback(self._on_physics_step, SimulationEvent.PHYSICS_PRE_STEP)
        return self

    def detach(self) -> None:
        if self._physics_sub is not None:
            SimulationManager.deregister_callback(self._physics_sub)
        self._physics_sub = None
        self._body = None

    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulated position, orientation (wxyz), linear and angular velocity (rad/s), world frame."""
        position, orientation = self._body.get_world_poses()
        lin_vel, ang_vel = self._body.get_velocities()
        return position.numpy()[0], orientation.numpy()[0], lin_vel.numpy()[0], ang_vel.numpy()[0]

    def _on_physics_step(self, dt: float, context: object) -> None:
        if not self._body.valid:
            self.detach()
            return
        if not self._body.is_physics_tensor_entity_valid():
            return
        position, orientation, lin_vel, ang_vel = (a.astype(float) for a in self.state())
        wrench = compute_wrench(
            self.hull,
            Pose(position, orientation),
            lin_vel,
            ang_vel,
            self.ocean.wavefield,
            SimulationManager.get_simulation_time(),
            self.hydro,
        )
        # The torque is about the body origin, so the force must act there too, wherever the center of mass is.
        self._body.apply_forces_and_torques_at_pos(wrench.force[None], wrench.torque[None], positions=position[None])
