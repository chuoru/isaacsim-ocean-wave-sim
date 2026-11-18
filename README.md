# robot_on_ship: ocean waves and a floating raft for Isaac Sim 6.2

Isaac Sim has no water or buoyancy. The `hachix.wave_sim` extension adds both: a Gerstner wavefield that animates a water mesh, and hull hydrodynamics that float a rigid body on it. The demo is a 2.0 x 1.5 x 0.3 m, 150 kg raft.

## Layout

- `exts/hachix.wave_sim/hachix/wave_sim/waves.py` holds the wavefield: `WaveParameters`, `wave_components`, and `Wavefield.displace` / `Wavefield.height`. Pure numpy.
- `hull.py` holds `HullMesh`, a closed triangle mesh with a subdivided box factory. Pure numpy.
- `hydrodynamics.py` holds `HydroParameters` and `compute_wrench`, the force and torque about the center of mass. Pure numpy.
- `scene.py` is the Isaac adapter and the only module that imports `omni` or `pxr`. It builds the water mesh, the raft rigid body, and the physics callback.
- `extension.py` adds the **Wave Sim > Create Ocean + Raft** menu item.
- `scripts/run_raft_demo.py` runs the demo standalone. `scripts/verify_isaac.py` is the headless proof. `scripts/bench_wrench.py` times `compute_wrench`.
- `tests/` holds the pytest suite for the numpy core. It needs no Isaac.

## Run it

All commands use the `isaacsim` conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacsim
export OMNI_KIT_ACCEPT_EULA=YES
```

To run the unit tests:

```bash
python -m pytest tests -q
```

To run the demo headless and print the raft's z, roll, and pitch:

```bash
cd scripts
python run_raft_demo.py --headless --duration 30 --amplitude 0.5 --period 6
```

To watch it in the GUI, drop `--headless`. Add `--screenshot /path/out.png` to save a viewport image one second before the end. On a hybrid-GPU laptop, set `DISPLAY=:0 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`. The first GUI launch compiles shaders for several minutes.

To prove the physics headless (exits nonzero on any failure):

```bash
cd scripts
python verify_isaac.py               # 60 Hz physics, the default
python verify_isaac.py --physics-hz 120
```

## Use the extension in the Isaac Sim GUI

1. Add `exts/` to the extension search paths. Use **Window > Extensions > Settings**, or pass `--ext-folder /path/to/robot_on_ship/exts` at launch.
2. Enable `hachix.wave_sim`.
3. Click **Wave Sim > Create Ocean + Raft**. The scene is authored under `/World/WaveSim` on the open stage. A second click replaces it.
4. Press **Play**. The water animates and the raft floats.

From Python, `WaveSimScene(WaveSimConfig(...))` with `.build(stage)` and `.attach()` gives the same scene with your own parameters.

## Parameters

`WaveParameters` has the fields and defaults of the VRX `<wave>` SDF element: `number`, `scale`, `angle`, `steepness`, `amplitude`, `period`, `phase`, and `direction`. `WaveParameters.vrx_ocean_waves(**overrides)` starts from the VRX `ocean_waves` model (3 components, scale 2.5, angle 0.3, period 8 s, direction (1, 1)). The demo default is `amplitude=0.5, period=6.0`. With scale 2.5 the three component amplitudes are 0.2, 0.5, and 1.25 m, so crests reach about 2 m.

`HydroParameters` has the asv_wave_sim `<hydrodynamics>` fields and defaults: `damping_on`, `pressure_drag_on`, `cDampL1/L2/R1/R2`, `cPDrag1/2`, `fPDrag`, `cSDrag1/2`, `fSDrag`, and `vRDrag`. Water density is 1025 kg/m^3 and g is 9.81 m/s^2.

`WaveSimConfig` also sets `RaftSpec` (size, mass, start xy, and drop height above the local surface at t=0), `OceanSpec` (the patch is 40 m wide with 100 x 100 cells), and `physics_hz` (60).

## How it differs from the Gazebo plugins

- **Newton residual.** VRX `ComputeDepthDirectly` drops the steepness `q` from the residual but keeps it in the Jacobian. Whenever `q < 1`, VRX misses the rendered surface; with `amplitude=0.8, period=3, steepness=0.6` the miss reaches 1.49 m. `Wavefield.height` uses `q` in both and inverts `displace` to about 1e-14 m. The Jacobian is a symmetric 2x2 matrix, so it is inverted in closed form.
- **Shader typo.** The VRX vertex shader writes `P.y -= q*a*dx*s`, where `dy` is meant. `displace` uses `dy`.
- **Clipping.** asv_wave_sim walks triangles with if-chains in CGAL. `submerged_triangles` sorts each triangle's vertices by depth and looks up its wet sub-triangles in a table indexed by the submerged-vertex count, vectorized over all triangles.
- **Center of pressure.** asv splits each wet triangle at a horizontal line and combines two partial centers. This port uses the exact closed form for linear pressure over a triangle, `(sum d_i x_i + sum d * sum x) / (4 sum d)`. With centroids instead, a level raft at equilibrium got a 0.04 N m yaw torque from the asymmetric face diagonals.
- **Depth inside a triangle** is linear between the vertex depths. That is exact in calm water and an approximation on a curved swell. The hull is subdivided to 8 cells per meter to keep the error small.
- **Not ported:** viscous (ITTC skin friction) drag, FFT and spectrum wave models, and waterline-length output.

## Attribution

`waves.py` ports the trochoidal wavefield from [wave_sim_vrx](https://github.com/srmainwaring/wave_sim_vrx) (Copyright 2019 Rhys Mainwaring, Apache License 2.0). The VRX shader carries a UUV Simulator Apache-2.0 notice as well.

`hydrodynamics.py` follows the method of [asv_wave_sim](https://github.com/srmainwaring/asv_wave_sim) (Copyright 2019 Rhys Mainwaring). asv_wave_sim is licensed **GPL-3.0**, not Apache-2.0. This module is a new vectorized numpy implementation of the published method (clipping, hydrostatic pressure, pressure drag, damping), not a translation of its C++. The project owner needs to decide whether that is enough separation for how the project is licensed.
