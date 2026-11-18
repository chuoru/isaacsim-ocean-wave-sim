"""Headless Isaac Sim proof that the raft floats correctly. Exits nonzero on any failed check."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from _sim import launch, run  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--physics-hz", type=float, default=None)
args = parser.parse_args()
app = launch(headless=True)

from hachix.wave_sim.hydrodynamics import WATER_DENSITY  # noqa: E402
from hachix.wave_sim.scene import WaveSimConfig  # noqa: E402
from hachix.wave_sim.waves import WaveParameters  # noqa: E402

failures: list[str] = []
rate = {} if args.physics_hz is None else {"physics_hz": args.physics_hz}


def check(ok: bool, message: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {message}", flush=True)
    if not ok:
        failures.append(message)


calm = WaveSimConfig(waves=WaveParameters(amplitude=0.0), **rate)
print(f"physics rate: {calm.physics_hz:.0f} Hz", flush=True)
raft = calm.raft
trace = run(app, calm, duration=20.0)
tail = trace.t >= trace.t[-1] - 5.0
draft = raft.size[2] / 2.0 - trace.position[tail, 2]
analytic = raft.mass / (WATER_DENSITY * raft.size[0] * raft.size[1])
drift = np.linalg.norm(trace.position[:, :2] - trace.position[0, :2], axis=1).max()
print(f"calm: {len(trace.t)} samples over {trace.t[-1]:.2f} s sim in {trace.wall_seconds:.2f} s wall", flush=True)
check(np.isfinite(trace.position).all() and np.isfinite(trace.orientation).all(), "calm: no NaN in raft state")
check(
    abs(draft.mean() / analytic - 1.0) < 0.05,
    f"calm: mean draft over last 5 s {draft.mean():.5f} m vs analytic {analytic:.5f} m "
    f"(err {abs(draft.mean() / analytic - 1.0):.2%}, std {draft.std():.2e} m)",
)
check(drift < 0.5, f"calm: max horizontal drift {drift:.4f} m < 0.5 m")

waves = WaveSimConfig(waves=WaveParameters.vrx_ocean_waves(amplitude=0.5, period=6.0), **rate)
frames: list[np.ndarray] = []


def snapshot_water(scene, t):
    if len(frames) < 2 and t > 1.0 + 0.5 * len(frames):
        frames.append(scene.water_points().copy())


trace = run(app, waves, duration=40.0, on_frame=snapshot_water)
settled = trace.t >= 10.0
heave = trace.position[settled, 2]
surface = trace.surface[settled]
corr = np.corrcoef(heave, surface)[0, 1]
freeboard = (trace.position[:, 2] - trace.surface).min()
roll_pitch = np.abs(trace.roll_pitch_deg[settled]).max(axis=0)
print(f"waves: {len(trace.t)} samples over {trace.t[-1]:.2f} s sim in {trace.wall_seconds:.2f} s wall", flush=True)
print(
    f"waves: heave std {heave.std():.3f} m, surface std {surface.std():.3f} m, "
    f"horizontal travel {np.linalg.norm(trace.position[-1, :2] - trace.position[0, :2]):.2f} m",
    flush=True,
)
check(np.isfinite(trace.position).all() and np.isfinite(trace.orientation).all(), "waves: no NaN in raft state")
check(corr > 0.7, f"waves: corr(heave, surface at raft xy) {corr:.3f} > 0.7 over t >= 10 s")
check(freeboard > -raft.size[2], f"waves: center never more than one raft height below surface for the whole run (min {freeboard:.3f} m)")
check(roll_pitch.max() < 30.0, f"waves: max |roll| {roll_pitch[0]:.1f} deg, |pitch| {roll_pitch[1]:.1f} deg < 30 deg")
moved = np.abs(frames[1] - frames[0]).max() if len(frames) == 2 else 0.0
check(moved > 1e-3, f"waves: water mesh points change between frames 0.5 s apart (max |dp| {moved:.3f} m)")

print("FAILED: " + "; ".join(failures) if failures else "ALL CHECKS PASSED", flush=True)
exit_code = 1 if failures else 0
app.close(exit_code=exit_code)
sys.exit(exit_code)
