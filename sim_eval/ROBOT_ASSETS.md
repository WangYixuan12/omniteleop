# Vega connector and wrist cameras

The source of mounting geometry is `src/omniteleop/follower/wbik.yaml` and its
`vega_with_robotiq_wrist_cam.urdf`. `robot_model.py` resolves this config without
loading the hardware stack. `OMNITELEOP_SRC`, `OMNITELEOP_WBIK_YAML`, and
`YIXUAN_UTILITIES_ROOT` are supported; use the same overrides for sim and WBC.
The source checkout is the default, replacing the old `/home/yixuan/omniteleop` path.

`SimObsRecorder` attaches RGB sensors to `L_ee` / `R_ee` using the composed
`L_wrist_camera` / `R_wrist_camera` transforms. It converts optical +Z forward,
+Y down to USD -Z forward, +Y up. Reset poses no longer change camera mounting.
Existing image sizing and optional ZED field-of-view processing still apply.

## Build when the Vega asset is missing

`build_vega_assets.py` prepares the current follower body and accessories plus the
articulated 2F-85 subtree from the installed BEHAVIOR asset
`source/franka/franka_robotiq_source.urdf`. It copies/converts meshes into an
isolated generated directory and constructs both hands with twelve mimic joints.
It does not require the historical `omnigibson_build` directory.

```bash
export YIXUAN_UTILITIES_ROOT="$HOME/yixuan_utilities/src/yixuan_utilities"
~/miniconda3/envs/behavior51/bin/python sim_eval/build_vega_assets.py --prepare-only
```

Prepared URDF, meshes, import config and source-hash manifest are under
`sim_eval/generated/vega_robotiq/` (ignored by git). With a compatible Isaac runtime:

```bash
~/miniconda3/envs/behavior51/bin/python sim_eval/build_vega_assets.py --import-prepared
```

The import stage adds the head camera and holonomic base, restores passive-joint
limits and PhysX mimic references, and writes the robot definition into
`omnigibson-robot-assets/models/vega_robotiq/vega_robotiq.yaml`. That definition
references the generated asset in `objects/robot/vega_robotiq/`.

**Current validation status:** the asset is installed under
`omnigibson-robot-assets/objects/robot/vega_robotiq`, with its definition under
`models/vega_robotiq`. Preparation used the real source assets: 71 links, 42 moving
joints before wheel conversion, and 12 mimics. Live validation in `behavior51`
(Python 3.11, Isaac Sim 5.1) passed in an empty scene and the actual `Rs_int`
pick-place scene: nominal joint error below 1e-6 rad, both grippers open/close/reopen
through their full normalized range, all three RGB cameras render non-degenerate
frames, and head/EEF/wrist-camera FK agrees with the follower URDF within 1e-6.
The definition disables only joint-adjacent contacts and measured structural mesh
overlaps that otherwise pin the shoulder and head links at the real nominal posture.

The current gripper mount correction is **+33.6892 mm along each EEF's Z**,
computed from the old/new rigid reference mesh poses. This preserves the original
actuated graft's seating convention; it is not the +234.1892 mm rigid-frame offset.
Wrist optical transforms agree with Pinocchio FK within 6e-16.

## Migrate an existing historical graft

If the original actuated graft is available, `port_vega_wrist_cameras.py` remains
an alternative preparation path. It preserves its linkage and collision settings:

```bash
python sim_eval/port_vega_wrist_cameras.py \
  --source-config /path/to/omnigibson_build/vega_robotiq_source_config.yaml
```

Import the generated `source_config.yaml` with BEHAVIOR's
`omnigibson/examples/robots/import_custom_robot.py`, reapply the original
`configure_robotiq_mimics.py`, and deploy the verified asset with its definition.
See `SUMMARY.md` for the historical import. `VegaOGEnv` rejects assets missing
the updated connector and camera links.
