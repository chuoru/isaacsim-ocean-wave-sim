"""Example: a box raft floating on Gerstner waves.

Shows the whole recipe for floating your own body: a z-up stage with a physics scene, an `Ocean`, a rigid
body prim with mass and a collider, and a `Floater` that ties the two together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from ..hull import HullMesh
from ..hydrodynamics import HydroParameters
from ..ocean import Floater, Ocean, OceanSpec, preview_material
from ..waves import WaveParameters

ROOT = "/World/WaveSim"


@dataclass(frozen=True)
class RaftSpec:
    size: tuple[float, float, float] = (2.0, 1.5, 0.3)
    mass: float = 150.0
    start_xy: tuple[float, float] = (0.0, 0.0)
    drop_height: float = 0.5
    path: str = f"{ROOT}/Raft"

    @property
    def inertia(self) -> tuple[float, float, float]:
        a, b, c = self.size
        return (self.mass * (b * b + c * c) / 12.0, self.mass * (a * a + c * c) / 12.0, self.mass * (a * a + b * b) / 12.0)


@dataclass(frozen=True)
class RaftExample:
    waves: WaveParameters = field(default_factory=lambda: WaveParameters.vrx_ocean_waves(amplitude=0.5, period=6.0))
    raft: RaftSpec = field(default_factory=RaftSpec)
    hydro: HydroParameters = field(default_factory=HydroParameters)
    physics_hz: float = 60.0


def build_raft_scene(stage: Usd.Stage, example: RaftExample | None = None) -> tuple[Ocean, Floater]:
    """Author the full example under `/World/WaveSim` on `stage` and start it. Press Play to run."""
    example = example or RaftExample()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.RemovePrim(ROOT)
    UsdGeom.Xform.Define(stage, ROOT)

    scene = UsdPhysics.Scene.Define(stage, f"{ROOT}/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    UsdLux.DomeLight.Define(stage, f"{ROOT}/Sky").CreateIntensityAttr(800.0)
    sun = UsdLux.DistantLight.Define(stage, f"{ROOT}/Sun")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 20.0, 0.0))

    ocean = Ocean(OceanSpec(waves=example.waves, path=f"{ROOT}/Water")).build(stage)
    hull = HullMesh.box(example.raft.size)
    _author_raft(stage, example.raft, hull, ocean)
    SimulationManager.setup_simulation(dt=1.0 / example.physics_hz, device="cpu")

    ocean.attach()
    floater = Floater(ocean, example.raft.path, hull=hull, hydro=example.hydro).attach(stage)
    return ocean, floater


def _author_raft(stage: Usd.Stage, spec: RaftSpec, hull: HullMesh, ocean: Ocean) -> None:
    """The visual, the collider and the hydrodynamic hull are one mesh, so what you see is what floats."""
    raft = UsdGeom.Mesh.Define(stage, spec.path)
    raft.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(hull.vertices.astype(np.float32)))
    raft.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(hull.triangles), 3, dtype=np.int32)))
    raft.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(hull.triangles.ravel().astype(np.int32)))
    raft.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    start_z = ocean.height([spec.start_xy], 0.0)[0] + spec.drop_height
    UsdGeom.Xformable(raft).AddTranslateOp().Set(Gf.Vec3d(*spec.start_xy, start_z))
    UsdShade.MaterialBindingAPI.Apply(raft.GetPrim()).Bind(preview_material(stage, f"{ROOT}/Looks/Raft", (0.75, 0.45, 0.15), 0.6))

    prim = raft.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(spec.mass)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*spec.inertia))
