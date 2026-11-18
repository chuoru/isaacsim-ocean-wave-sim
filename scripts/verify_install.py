"""Proof that isaacsim.ocean.wave_sim works as a standalone Isaac Sim extension outside this repo.

Copies the extension folder to a temp directory, loads it only through the extension manager (this
repo is never put on sys.path), then floats the menu raft example and a user-authored scaled Cube.
Exits nonzero on any failed check.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
SOURCE = Path(__file__).resolve().parents[1] / "exts" / "isaacsim.ocean.wave_sim"
install_root = Path(tempfile.mkdtemp(prefix="wave_sim_install_"))
shutil.copytree(SOURCE, install_root / "isaacsim.ocean.wave_sim", ignore=shutil.ignore_patterns("__pycache__"))

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import omni.kit.actions.core  # noqa: E402
import omni.kit.app  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

if SimulationManager.get_active_physics_engine() != "physx":
    SimulationManager.switch_physics_engine("physx")

failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {message}", flush=True)
    if not ok:
        failures.append(message)


def play_for(seconds: float) -> None:
    app.update()
    app_utils.play()
    while SimulationManager.get_simulation_time() < seconds:
        app.update()


manager = omni.kit.app.get_app().get_extension_manager()
manager.add_path(str(install_root))
manager.set_extension_enabled_immediate("isaacsim.ocean.wave_sim", True)
ext_id = manager.get_enabled_extension_id("isaacsim.ocean.wave_sim")
import isaacsim.ocean.wave_sim  # noqa: E402

loaded_from = Path(isaacsim.ocean.wave_sim.__path__[0]).resolve()
check(bool(ext_id), f"install: enabled {ext_id} from {install_root}")
check(install_root.resolve() in loaded_from.parents, f"install: package imported from the copy ({loaded_from})")

from isaacsim.ocean.wave_sim.examples.raft import RaftSpec  # noqa: E402
from isaacsim.ocean.wave_sim.hydrodynamics import WATER_DENSITY  # noqa: E402
from isaacsim.ocean.wave_sim.ocean import Floater, Ocean, OceanSpec  # noqa: E402
from isaacsim.ocean.wave_sim.waves import WaveParameters  # noqa: E402

action = omni.kit.actions.core.get_action_registry().get_action(ext_id, "create_ocean_and_raft")
check(action is not None, "example: Wave Sim > Examples > Ocean + Raft action registered")
stage = stage_utils.create_new_stage()
action.execute()
spec = RaftSpec()
check(stage.GetPrimAtPath(spec.path).HasAPI(UsdPhysics.RigidBodyAPI), "example: menu action authored the raft rigid body")
play_for(3.0)
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402

z = RigidPrim(spec.path).get_world_poses()[0].numpy()[0].astype(float)
app_utils.stop()
surface = Ocean(OceanSpec(waves=WaveParameters.vrx_ocean_waves(amplitude=0.5, period=6.0))).wavefield.height(z[None, :2], 3.0)[0]
check(z[2] - surface > -spec.size[2], f"example: raft afloat after 3 s (z {z[2]:.3f} m, surface {surface:.3f} m)")

size, mass = (1.2, 0.8, 0.4), 100.0
stage = stage_utils.create_new_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
cube = UsdGeom.Cube.Define(stage, "/World/Crate")
cube.CreateSizeAttr(1.0)
UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(3.0, -2.0, 0.6))
UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*size))
UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(mass)
SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")

ocean = Ocean(OceanSpec(waves=WaveParameters(amplitude=0.0), path="/World/Ocean")).build(stage).attach()
floater = Floater(ocean, "/World/Crate").attach(stage)
check(abs(floater.hull.volume / (size[0] * size[1] * size[2]) - 1.0) < 1e-6, f"user body: hull read from scaled Cube, volume {floater.hull.volume:.4f} m^3")
play_for(15.0)
heights = []
while SimulationManager.get_simulation_time() < 20.0:
    app.update()
    heights.append(floater.state()[0][2])
app_utils.stop()
draft = size[2] / 2.0 - sum(heights) / len(heights)
analytic = mass / (WATER_DENSITY * size[0] * size[1])
check(abs(draft / analytic - 1.0) < 0.05, f"user body: crate draft {draft:.5f} m vs analytic {analytic:.5f} m (err {abs(draft / analytic - 1.0):.2%})")

y_up = stage_utils.create_new_stage()
UsdGeom.SetStageUpAxis(y_up, UsdGeom.Tokens.y)
try:
    Ocean().build(y_up)
    refused = False
except ValueError:
    refused = True
check(refused, "boundary: Ocean refuses a y-up stage with a ValueError")

shutil.rmtree(install_root, ignore_errors=True)
print("FAILED: " + "; ".join(failures) if failures else "ALL CHECKS PASSED", flush=True)
exit_code = 1 if failures else 0
app.close(exit_code=exit_code)
sys.exit(exit_code)
