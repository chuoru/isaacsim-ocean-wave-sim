"""Time compute_wrench and its parts on the demo raft. Pure numpy, no Isaac."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exts" / "hachix.wave_sim"))
from hachix.wave_sim.hull import HullMesh  # noqa: E402
from hachix.wave_sim.hydrodynamics import HydroParameters, Pose, compute_wrench, submerged_triangles  # noqa: E402
from hachix.wave_sim.waves import Wavefield, WaveParameters  # noqa: E402

REPEATS = 5
CALLS = 2000
DT = 1.0 / 120.0

hull = HullMesh.box((2.0, 1.5, 0.3))
params = HydroParameters()
sea_states = {
    "calm": WaveParameters(amplitude=0.0),
    "vrx 0.5 m / 6 s": WaveParameters.vrx_ocean_waves(amplitude=0.5, period=6.0),
}
print(f"hull: {len(hull.vertices)} vertices, {len(hull.triangles)} triangles; {REPEATS} repeats x {CALLS} calls")


def states(field: Wavefield):
    """Raft riding the surface at 2 m drift, slightly heeled, so every call clips a waterline."""
    rng = np.random.default_rng(0)
    for i in range(CALLS):
        t = i * DT
        xy = np.array([[0.05 * t, 0.0]])
        z = field.height(xy, t)[0] + 0.1
        q = np.array([1.0, *rng.normal(scale=0.05, size=3)])
        yield Pose(np.array([*xy[0], z]), q / np.linalg.norm(q)), rng.normal(scale=0.3, size=3), rng.normal(scale=0.2, size=3), t


for name, wave in sea_states.items():
    field = Wavefield.from_parameters(wave)
    inputs = list(states(field))
    full, height_only, clip_only, wet = [], [], [], []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for pose, v, w, t in inputs:
            wrench = compute_wrench(hull, pose, v, w, field, t, params)
        full.append((time.perf_counter() - start) / CALLS)
        assert np.isfinite(wrench.force).all() and wrench.force[2] > 0.0

        worlds = [hull.vertices @ pose.rotation().T + pose.position for pose, *_ in inputs]
        start = time.perf_counter()
        depths = [field.height(world[:, :2], t) - world[:, 2] for world, (_, _, _, t) in zip(worlds, inputs)]
        height_only.append((time.perf_counter() - start) / CALLS)

        start = time.perf_counter()
        for world, depth in zip(worlds, depths):
            parent, _, _ = submerged_triangles(world, hull.triangles, depth)
        clip_only.append((time.perf_counter() - start) / CALLS)
        wet.append(len(parent))

    ms = lambda xs: f"median {1e3 * np.median(xs):.3f} ms (range {1e3 * min(xs):.3f}-{1e3 * max(xs):.3f})"
    share = lambda xs: f"{np.median(xs) / np.median(full):.0%}"
    print(f"[{name}] compute_wrench {ms(full)}; last call wet sub-triangles {wet[-1]}")
    print(f"[{name}]   height solve {ms(height_only)} = {share(height_only)}; clipping {ms(clip_only)} = {share(clip_only)}")
    print(f"[{name}]   budget at {1 / DT:.0f} Hz physics: {np.median(full) / DT:.1%} of each step")
