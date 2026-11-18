"""Shared launcher for the standalone scripts. They load the extension the way users do, so its imports must follow `launch`."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXT_FOLDER = Path(__file__).resolve().parents[1] / "exts"
EXT_NAME = "isaacsim.ocean.wave_sim"


def launch(headless: bool):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": headless})
    from isaacsim.core.simulation_manager import SimulationManager

    if SimulationManager.get_active_physics_engine() != "physx":
        SimulationManager.switch_physics_engine("physx")
    print(f"physics engine: {SimulationManager.get_active_physics_engine()}", flush=True)

    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.add_path(str(EXT_FOLDER))
    manager.set_extension_enabled_immediate(EXT_NAME, True)
    if not manager.is_extension_enabled(EXT_NAME):
        sys.exit(f"could not enable {EXT_NAME} from {EXT_FOLDER}")
    return app


@dataclass
class Trace:
    t: np.ndarray
    position: np.ndarray
    orientation: np.ndarray
    surface: np.ndarray
    wall_seconds: float

    @property
    def roll_pitch_deg(self) -> np.ndarray:
        w, x, y, z = self.orientation.T
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        return np.degrees(np.column_stack([roll, pitch]))


def run(app, example, duration: float, on_frame=None) -> Trace:
    """Build the scene on a fresh stage, play for `duration` simulated seconds, sample once per app update."""
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.ocean.wave_sim.examples.raft import build_raft_scene
    from isaacsim.core.simulation_manager import SimulationManager

    stage = stage_utils.create_new_stage()
    ocean, floater = build_raft_scene(stage, example)
    app.update()
    app_utils.play()

    samples = []
    start = time.perf_counter()
    while app_utils.is_playing() and SimulationManager.get_simulation_time() < duration:
        app.update()
        t = SimulationManager.get_simulation_time()
        position, orientation, _, _ = floater.state()
        surface = ocean.height(position[None, :2], t)[0]
        samples.append((t, *position, *orientation, surface))
        if on_frame is not None:
            on_frame(ocean, floater, t)
    wall = time.perf_counter() - start
    app_utils.stop()
    floater.detach()
    ocean.detach()
    data = np.array(samples, dtype=float)
    return Trace(data[:, 0], data[:, 1:4], data[:, 4:8], data[:, 8], wall)
