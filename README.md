# isaacsim.ocean.wave_sim: ocean waves and buoyancy for Isaac Sim

Isaac Sim has no water or buoyancy. `isaacsim.ocean.wave_sim` is an Isaac Sim extension that adds both: a Gerstner wavefield that animates a water mesh, and hull hydrodynamics that float any rigid body on it. The extension ships with one example: a 2.0 x 1.5 x 0.3 m, 150 kg raft.

The wavefield is ported from [wave_sim_vrx](https://github.com/srmainwaring/wave_sim_vrx) and the hydrodynamics follow [asv_wave_sim](https://github.com/srmainwaring/asv_wave_sim). See [Attribution](#attribution) before you redistribute it.

## Requirements

- Isaac Sim 6.2. It's tested on the pip install (`isaacsim[all,extscache]==6.2.0.0`, Python 3.12). It uses `isaacsim.core.experimental.prims` and `isaacsim.core.simulation_manager`, which are not in Isaac Sim 4.x. Other 6.x releases are likely to work but are untested.
- numpy, which Isaac Sim already ships. There are no other dependencies and nothing to compile.

## Install

The extension is the folder `exts/isaacsim.ocean.wave_sim`. It is self-contained, so you can copy that folder anywhere. Then point Isaac Sim at its parent folder in one of these ways:

- **GUI.** Open **Window > Extensions**, then the hamburger menu, then **Settings**. Add the parent folder (for example `/path/to/isaacsim-ocean-wave-sim/exts`) to the extension search paths. Search for `isaacsim.ocean.wave_sim` and enable it.
- **Launch flag.** Run `isaacsim --ext-folder /path/to/isaacsim-ocean-wave-sim/exts --enable isaacsim.ocean.wave_sim`.
- **Standalone Python app.** After you create `SimulationApp`, run this code:

  ```python
  import omni.kit.app

  manager = omni.kit.app.get_app().get_extension_manager()
  manager.add_path("/path/to/isaacsim-ocean-wave-sim/exts")
  manager.set_extension_enabled_immediate("isaacsim.ocean.wave_sim", True)
  ```

Once the extension is enabled, `import isaacsim.ocean.wave_sim...` works from any script, the Script Editor, or another extension. To use it from your own extension, add `"isaacsim.ocean.wave_sim" = {}` to that extension's `[dependencies]`.

The name follows Isaac Sim's own `isaacsim.<area>.<feature>` convention, as in `isaacsim.robot.surface_gripper`. Like NVIDIA's extensions, the folder has no `isaacsim/__init__.py` or `isaacsim/ocean/__init__.py`, and Kit adds `isaacsim/ocean/wave_sim` to the `isaacsim` package when the extension is enabled. So the import works only after the extension is enabled, not from a plain `sys.path` entry. `isaacsim.` is NVIDIA's namespace. If NVIDIA ever ships an extension called `isaacsim.ocean.wave_sim`, the two will clash.

`scripts/verify_install.py` proves the install path. It copies the extension folder to a temp directory and loads it only through the extension manager. It then runs the raft example from the menu action and floats a scaled `Cube` the script authors itself. It exits nonzero if any check fails.

## Float your own body

The plugin has two parts, both in `isaacsim.ocean.wave_sim.ocean`:

- `Ocean(OceanSpec(...))` authors the water mesh and animates it on every app update. `ocean.height(xy, t)` gives the surface z at any world xy.
- `Floater(ocean, body_path, hull=None, hydro=None)` applies buoyancy, pressure drag and damping to an existing rigid body on every physics step.

The plugin does not create a physics scene or lights, and it does not change stage settings or the physics rate. Those belong to your scene. Your body must already have `RigidBodyAPI`, a collider, and a mass. The stage must be z-up in meters. `Ocean.build` raises a `ValueError` on a y-up stage.

```python
import omni.usd
from isaacsim.ocean.wave_sim.ocean import Floater, Ocean, OceanSpec
from isaacsim.ocean.wave_sim.waves import WaveParameters

stage = omni.usd.get_context().get_stage()
ocean = Ocean(OceanSpec(waves=WaveParameters.vrx_ocean_waves(amplitude=0.5, period=6.0))).build(stage).attach()
floater = Floater(ocean, "/World/MyBoat").attach(stage)
# Press Play. Call floater.detach() and ocean.detach() to stop.
```

Without an explicit `hull`, `Floater.attach` reads one from the body prim. It handles a `UsdGeom.Cube` and a `UsdGeom.Mesh` with convex faces, and it bakes in the prim's world scale. The mesh must be closed with outward normals, and its origin should sit at the center of mass. For any other prim, pass a `HullMesh`, for example `HullMesh.box(size)` or `HullMesh.from_polygons(points, counts, indices)`. A finer hull gives smoother forces on a swell. `HullMesh.box` subdivides to 8 cells per meter. A plain 6-face box floats correctly in calm water but responds coarsely to waves.

Each `Floater` costs about 1.4 ms per physics step for the 1104-triangle raft hull. That is the median of 5 runs of 2000 calls from `scripts/bench_wrench.py`. The cost scales with the triangle count.

## The raft example

`isaacsim/ocean/wave_sim/examples/raft.py` is the full recipe. It sets up a z-up stage with a physics scene and lights, an `Ocean`, a raft rigid body whose visual, collider and hull are one mesh, a `Floater`, and a 60 Hz physics rate. You can run it three ways:

- In the GUI, click **Wave Sim > Examples > Ocean + Raft** and press **Play**. The scene is authored under `/World/WaveSim`, and a second click replaces it.
- From Python, call `build_raft_scene(stage, RaftExample(...))`.
- From the command line, use the scripts below.

The scripts use the `isaacsim` conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacsim
export OMNI_KIT_ACCEPT_EULA=YES
cd scripts

python run_raft_demo.py --headless --duration 30 --amplitude 0.5 --period 6   # prints raft z / roll / pitch
python run_raft_demo.py --screenshot /tmp/raft.png                            # GUI, saves a viewport image
python verify_isaac.py                                                        # physics proof, exits nonzero on failure
python verify_install.py                                                      # install proof, exits nonzero on failure
```

On a hybrid-GPU laptop, set `DISPLAY=:0 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` for the GUI. The first GUI launch compiles shaders for several minutes, and the window looks frozen until it finishes.

The numpy core has a pytest suite that needs no Isaac. Run it with `python -m pytest tests -q` from the repo root.

## Layout

- `exts/isaacsim.ocean.wave_sim/config/extension.toml` is the extension manifest.
- `isaacsim/ocean/wave_sim/waves.py` holds `WaveParameters`, `wave_components`, and `Wavefield.displace` / `Wavefield.height`. It is pure numpy.
- `hull.py` holds `HullMesh`, with the `box` and `from_polygons` factories. It is pure numpy.
- `hydrodynamics.py` holds `HydroParameters` and `compute_wrench`. It is pure numpy.
- `ocean.py` holds `Ocean`, `Floater`, and `hull_from_prim`. It is the only plugin module that imports `omni` or `pxr`.
- `examples/raft.py` holds the raft example. `extension.py` adds its menu item.
- `scripts/` holds the demo, the two headless proofs, and the benchmark. `tests/` holds the pytest suite.

## Parameters

`WaveParameters` has the fields and defaults of the VRX `<wave>` SDF element: `number`, `scale`, `angle`, `steepness`, `amplitude`, `period`, `phase`, and `direction`. `WaveParameters.vrx_ocean_waves(**overrides)` starts from the VRX `ocean_waves` model, which has 3 components, scale 2.5, angle 0.3, period 8 s, and direction (1, 1). The raft example uses `amplitude=0.5, period=6.0`. With scale 2.5 the three component amplitudes are 0.2, 0.5, and 1.25 m, so crests reach about 2 m.

`HydroParameters` has the asv_wave_sim `<hydrodynamics>` fields and defaults: `damping_on`, `pressure_drag_on`, `cDampL1/L2/R1/R2`, `cPDrag1/2`, `fPDrag`, `cSDrag1/2`, `fSDrag`, and `vRDrag`. Water density is 1025 kg/m^3 and g is 9.81 m/s^2.

`OceanSpec` sets the waves, the patch size (40 m wide with 100 x 100 cells), the prim path, and the color. The patch is fixed in place, so a body that drifts far leaves the visible water. The physics keeps working out there.

## How it differs from the Gazebo plugins

- **Newton residual.** VRX `ComputeDepthDirectly` drops the steepness `q` from the residual but keeps it in the Jacobian. Whenever `q < 1`, VRX misses the rendered surface. With `amplitude=0.8, period=3, steepness=0.6` the miss reaches 1.49 m. `Wavefield.height` uses `q` in both and inverts `displace` to about 1e-14 m.
- **Shader typo.** The VRX vertex shader writes `P.y -= q*a*dx*s`, where `dy` is meant. `displace` uses `dy`.
- **Clipping.** asv_wave_sim walks triangles with if-chains in CGAL. Here, `submerged_triangles` sorts each triangle's vertices by depth and looks up its wet sub-triangles in a table indexed by the submerged-vertex count. It is vectorized over all triangles.
- **Center of pressure.** This port uses the exact closed form for linear pressure over a triangle. With centroids instead, a level raft at equilibrium got a 0.04 N m yaw torque.
- **Depth inside a triangle** is linear between the vertex depths. That is exact in calm water and an approximation on a curved swell.
- **Not ported:** viscous (ITTC skin friction) drag, the FFT and spectrum wave models, waterline-length output, and added mass.

## Attribution

`waves.py` ports the trochoidal wavefield from [wave_sim_vrx](https://github.com/srmainwaring/wave_sim_vrx) (Copyright 2019 Rhys Mainwaring, Apache License 2.0). The VRX shader carries a UUV Simulator Apache-2.0 notice as well.

`hydrodynamics.py` follows the method of [asv_wave_sim](https://github.com/srmainwaring/asv_wave_sim) (Copyright 2019 Rhys Mainwaring). asv_wave_sim is licensed **GPL-3.0**, not Apache-2.0. This module is a new vectorized numpy implementation of the published method (clipping, hydrostatic pressure, pressure drag, damping), not a translation of its C++. The project owner needs to decide whether that is enough separation before the extension is distributed.
