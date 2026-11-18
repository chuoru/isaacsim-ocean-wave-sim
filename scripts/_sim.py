"""Shared launcher for the standalone scripts. Omni imports must follow SimulationApp creation."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXT_ROOT = Path(__file__).resolve().parents[1] / "exts" / "hachix.wave_sim"
sys.path.insert(0, str(EXT_ROOT))


def launch(headless: bool):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": headless})
    from isaacsim.core.simulation_manager import SimulationManager

    if SimulationManager.get_active_physics_engine() != "physx":
        SimulationManager.switch_physics_engine("physx")
    print(f"physics engine: {SimulationManager.get_active_physics_engine()}", flush=True)
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


def run(app, config, duration: float, on_frame=None) -> Trace:
    """Build the scene on a fresh stage, play for `duration` simulated seconds, sample once per app update."""
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from hachix.wave_sim.scene import WaveSimScene
    from isaacsim.core.simulation_manager import SimulationManager

    stage = stage_utils.create_new_stage()
    scene = WaveSimScene(config)
    scene.build(stage)
    scene.attach()
    app.update()
    app_utils.play()

    samples = []
    start = time.perf_counter()
    while app_utils.is_playing() and SimulationManager.get_simulation_time() < duration:
        app.update()
        t = SimulationManager.get_simulation_time()
        position, orientation, _, _ = scene.raft_state()
        surface = scene.wavefield.height(position[None, :2].astype(float), t)[0]
        samples.append((t, *position, *orientation, surface))
        if on_frame is not None:
            on_frame(scene, t)
    wall = time.perf_counter() - start
    app_utils.stop()
    scene.detach()
    data = np.array(samples, dtype=float)
    return Trace(data[:, 0], data[:, 1:4], data[:, 4:8], data[:, 8], wall)
