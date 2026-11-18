"""Standalone demo: a raft floating on Gerstner waves. Prints the raft's z / roll / pitch time series."""

from __future__ import annotations

import argparse
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from _sim import launch, run  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--duration", type=float, default=30.0, help="simulated seconds")
parser.add_argument("--amplitude", type=float, default=0.5, help="VRX mean wave amplitude [m]")
parser.add_argument("--period", type=float, default=6.0, help="VRX mean wave period [s]")
parser.add_argument("--print-every", type=float, default=0.5, help="time-series sample spacing [s]")
parser.add_argument("--screenshot", type=str, default=None, help="save a viewport PNG at the end (GUI only)")
args = parser.parse_args()

app = launch(headless=args.headless)

import omni.usd  # noqa: E402
from isaacsim.ocean.wave_sim.examples.raft import RaftExample  # noqa: E402
from isaacsim.ocean.wave_sim.waves import WaveParameters  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

CAMERA = "/World/DemoCamera"
captured = []

def follow_raft(ocean, floater, t):
    if args.headless:
        return
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(CAMERA):
        UsdGeom.Camera.Define(stage, CAMERA).CreateFocalLengthAttr(18.0)
        UsdGeom.Xformable(stage.GetPrimAtPath(CAMERA)).AddTransformOp()
        from omni.kit.viewport.utility import get_active_viewport

        get_active_viewport().camera_path = CAMERA
    target = Gf.Vec3d(*floater.state()[0].astype(float))
    view = Gf.Matrix4d().SetLookAt(target + Gf.Vec3d(7.0, -7.0, 3.5), target, Gf.Vec3d(0.0, 0.0, 1.0))
    UsdGeom.Xformable(stage.GetPrimAtPath(CAMERA)).GetOrderedXformOps()[0].Set(view.GetInverse())
    if args.screenshot and not captured and t >= args.duration - 1.0:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        captured.append(capture_viewport_to_file(get_active_viewport(), args.screenshot))
        print(f"screenshot requested at t={t:.2f} s", flush=True)


example = RaftExample(waves=WaveParameters.vrx_ocean_waves(amplitude=args.amplitude, period=args.period))
trace = run(app, example, duration=args.duration, on_frame=follow_raft)

print(f"{'t[s]':>7} {'z[m]':>8} {'surface[m]':>10} {'roll[deg]':>9} {'pitch[deg]':>10}")
roll_pitch = trace.roll_pitch_deg
for i in np.searchsorted(trace.t, np.arange(0.0, trace.t[-1], args.print_every)):
    print(f"{trace.t[i]:7.2f} {trace.position[i, 2]:8.3f} {trace.surface[i]:10.3f} {roll_pitch[i, 0]:9.2f} {roll_pitch[i, 1]:10.2f}")
print(f"simulated {trace.t[-1]:.2f} s in {trace.wall_seconds:.2f} s wall ({trace.t[-1] / trace.wall_seconds:.2f}x real time)")

if captured:
    for _ in range(30):
        app.update()
    print(f"screenshot: {args.screenshot} exists={os.path.exists(args.screenshot)}", flush=True)

app.close()
