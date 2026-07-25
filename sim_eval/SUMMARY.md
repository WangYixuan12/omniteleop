# Vega Sim-Eval Pipeline — Working Summary

Simulation-based **eval + scripted data-collection** pipeline in OmniGibson for the Dexmate **Vega**
robot, mirroring the real ManiFlow teleop stack in `omniteleop` so the same policies/checkpoints apply.
Whole-body control **reuses** `omniteleop/.../follower/whole_body_ik.py` **(**`VegaWholeBodyIK`**) +** `wbik.yaml`
**verbatim**; demos come from a scripted expert; SceneDiff env-state uses sim ground-truth object poses.

Reference pipeline: `/home/yixuan/BEHAVIOR-1K/BEHAVIOR-DemoGen` (R1 + CuRobo picks apple→bowl on a desk in
Rs_int). `~/DemoGen` is empty — real content is at that BEHAVIOR-1K path.

---



## Status at a glance


| Phase | What                                                                                | State                                     |
| ----- | ----------------------------------------------------------------------------------- | ----------------------------------------- |
| A     | Tabletop pick-place scaffolding for Vega                                            | ✅ done                                    |
| B     | Replace rigid gripper visuals with two actuated **Robotiq 2F-85** grippers          | ✅ done; Panda model retained for rollback |
| C     | Bowl **beside** apple; real Robotiq grasp (calc grasp pose → solve IK, no teleport) | ✅ seed 0 `SUCCESS=True`                   |
| D     | **Mobile manipulation** — base drives to table via WBC, then manipulates            | ✅ Robotiq seed 0 `SUCCESS=True`           |
| P4    | Multi-seed data collection → ManiFlow zarr                                          | ⏳ not started                             |
| P5    | Policy eval (async rollout, success rate)                                           | ⏳ not started                             |


---



## Architecture — process-split WBC (ZMQ over localhost)

```
behavior env (py3.11, OmniGibson/Isaac, numpy<2)     dexmate venv (pinocchio/pink/daqp, numpy2)
  VegaOGEnv: vega_robotiq + task objects   REQ→   wbc_service.py: VegaWholeBodyIK(wbik.yaml)
   scripted expert -> world L/R/head targets ─────  solve(left,right,head, current_q, dt)
   apply WBC result to controllers each tick ←REP   -> {base_twist, base_pose, torso(3),
   render obs / record / rollout                          l_arm(7), r_arm(7), head(3), success}
```

Only small float arrays cross the socket. `VegaWholeBodyIK` = pinocchio + pink + daqp differential-IK QP.

- WBC service python: `/home/yixuan/miniforge3/envs/dexmate/bin/python`
- Env python: `behavior` conda env.

---



## Run conventions (every OmniGibson run)

- Strip CoppeliaSim from `LD_LIBRARY_PATH` (its old Qt5 breaks pymeshlab):
`CLEAN_LD=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v CoppeliaSim | paste -sd:)`
- `OMNIGIBSON_HEADLESS=1`, `python -u`, from `cd /home/yixuan/omniteleop/sim_eval`.
- Isaac launch ≈ 90 s per run.
- `run_episode.py --task {carry,pickplace,mobilepickplace} --seed N --port P --out ....mp4`
renders head-cam **and** 3rd-person mp4 (`_3rd` suffix).

---



## Key files (`/home/yixuan/omniteleop/sim_eval/`)

- `vega_og_env.py` (`VegaOGEnv`) — the core env + WBC bridge. Loads `vega_robotiq` with
`grasping_direction="upper"`; task-level commands remain `grip_close=-1.0`, `grip_open=+1.0`.
`GRASP_WINDOW=0` → sticky grasp magnetizes on first finger contact. Gripper helpers derive the driven
joints and finger links from the robot definition, and `finger_grasp_point` uses its explicit pad-contact
rays (with an AABB fallback). Other helpers: `is_grasping`, `link_pose`, `obj_pos`, `wbc_tick`,
`drive_base`, `capture_third_person`, `_set_cam_lookat`. `base_x_max` forward-park clamp.
Mobile `wbc_overrides` (current): `head_world_position_cost=[10000,10000,10000] if mobile else [0,0,0]`,
`enable_torso_top_x_anchor=False`, `enable_com_safety=False`, `posture_cost_torso=200 if mobile else 20`.
Base gains (from wbik vr_teleop): kp_xy=1.0, kp_yaw=1.5, max_lin=0.45, max_ang=0.9, deadband 0.02/0.04,
accel 0.4/0.8.
- `base_ctrl.py` — pure-numpy port of `omniteleop/follower/base_closed_loop.py`:
`wrap_pi, limit_twist, pose_error_body, pd_twist, shape_twist`.
- `tasks/pick_place_task.py` (`PickPlaceTask`, tabletop) — base fixed at table. `ROBOT_POS=(0.58,0.40,0.03)`,
`APPLE_XY=(1.20,0.74)`, `BOWL_XY=(1.20,0.56)` (side-by-side, same x). Phases
`[pregrasp45, approach40, settle40, close45, lift35, carry50, place45, release25, done15]`.
The pregrasp point is the midpoint of the current Robotiq pad center and the apple, so both pre-contact
segments move monotonically toward the apple instead of clearing away and doubling back. It targets the
configured pad rays, closes residual pad-position error during settle/close, then calibrates the
closed-gripper/object offset for carry and placement. Apple scale 0.8; bowl `fixed_base`. Success = apple
`Inside`/`OnTop` bowl. `test_pick_place_trajectory.py` locks down the monotonic approach invariant.
- `tasks/mobile_pick_place_task.py` (`MobilePickPlaceTask(PickPlaceTask)`) — `MOBILE=True`,
`ROBOT_POS=(-0.35,0.40,0.03)`, `BASE_X_MAX=0.56`, `ARRIVE_X=0.52`; it reuses the tabletop object layout.
The 10 cm earlier park point compensates for the 2F-85 pad center's longer reach. `expert_reset` captures
base-relative arm rest poses + head frame, builds a
**reachable** nav head target (`_Thead_nav` = nominal head translated to parked base_x). `expert_step`:
navigate (head=`_Thead_nav`, arms ride base) until `base_x≥ARRIVE_X` → `super().expert_reset` →
manipulate (delegate to parent; head TRACKS live head each tick). `_armlog` prints torso/arm joints.
- `run_episode.py` — `mobile=getattr(task,"MOBILE",False)`; `VegaOGEnv(..., lock_base=not mobile, mobile=mobile, ...)`; sets `env.base_x_max` for mobile. Renders head + 3rd-person mp4.
- `wbc_service.py` (dexmate venv) — ZMQ REP wrapping `VegaWholeBodyIK`; `reset(q)` + `solve(...)`.

---



## Robot import (Robotiq 2F-85 actuated graft; current)

Deployed as a separate model at
`/home/yixuan/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/vega_robotiq/`; the working Panda model at
`models/vega/` is unchanged and remains the rollback path.

- Source/build files live in
`/home/yixuan/yixuan_utilities/src/yixuan_utilities/assets/robot/vega-urdf/omnigibson_build/`:
`graft_robotiq_gripper.py`, `vega_robotiq_source_config.yaml`,
`vega_robotiq_robot_definition.yaml`, and `configure_robotiq_mimics.py`.
- The graft replaces each rigid reference gripper visual with the official actuated 2F-85 linkage from
BEHAVIOR-1K, mounted coincident with `L_ee`/`R_ee` (zero translation and rotation) with eef +z as the
approach direction. The original rigid mesh begins at −3.51 mm and the actuated base at −3.60 mm relative
to `L_ee`; zero mount offset reproduces that reference alignment. The former +20 mm spacer left an ~8.5 mm
visible gap and was removed. Per-part collision meshes are retained via `no_decompose_links`.
- The robot has 42 DOFs: 26 body DOFs plus 16 Robotiq revolute joints. Each side exposes two driven outer
knuckles (`left_outer_knuckle_joint`, `right_outer_knuckle_joint`); the remaining six linkage joints per
side are PhysX mimics. `configure_robotiq_mimics.py` restores the passive-joint limits dropped by import,
reauthors clean rigid `rotX` mimic constraints, and verifies all 12 references, gearings, and limits.
- Robot definition: `grasping_direction="upper"`; 0 rad is open and 0.7854 rad is closed. The controller is
inverted at the task boundary so **command −1 = close, +1 = open**, matching the former task convention.
- Assisted-grasp rays use the BEHAVIOR Robotiq pad geometry: `[0.008, 0.010, 0]` and
`[0.008, 0.040, 0]`, from each left inner finger to its opposing right inner finger. These are also the
scripted expert's grasp target.
- Import sequence: graft the URDF/assets, run `import_custom_robot.py --config ...`, run
`configure_robotiq_mimics.py`, then copy the verified object asset and robot definition to
`models/vega_robotiq/`.

Static validation: 63 links, 62 joints, 42 non-fixed DOFs, 18 Robotiq links, 16 Robotiq joints, 12 URDF
mimic tags, zero mount transforms, no unresolved meshes, and no obsolete rigid gripper visuals. Fresh live
articulation confirmed both master joints move from 0 to 0.7854 under close commands. The deployed USD has
zero `localPos` on both mount joints and all 12 mimic joints have the expected references, gearings, axes,
and limits.

The reference implementation and assisted-grasp geometry came from
[BEHAVIOR-1K PR #1890](https://github.com/StanfordVL/BEHAVIOR-1K/pull/1890).

## Previous Franka Panda fallback (preserved)

The prior parallel-jaw graft remains deployed as `models/vega/` and its source remains in
`omnigibson_build/graft_panda_gripper.py`, `vega_panda_source_config.yaml`, and
`vega_panda_robot_definition.yaml`. It is no longer the default sim-eval robot.

## `wbik.yaml` nominal_posture

torso_j1=0.78, torso_j2=1.5708, torso_j3=0.0, head_j1=−0.40, L_arm_j1=0.844, L_arm_j2=0.3, L_arm_j4=−1.556,
L_arm_j5=1.271, R_arm mirror (j1/j2/j5 negated), all others=0.

---



## WBC base-following (how the mobile base moves)

`solve()` returns `base_pose` (world x,y,yaw) + `base_twist` (feed-forward in reference base frame). Follower:
`pd_twist` (rotate feed-forward into measured base frame + PD on SE(2) pose error) → `shape_twist`
(deadband/clamp/slew) → chassis velocity. The base follows the **HEAD** via `head_world_position_cost`
`[x,y,z]`:

- **z = 0 (free)** lets the torso over-extend/collapse to shove the head forward → bad arm pose at grasp.
- **z ON** holds camera height so the torso does **not** collapse.

---



## Robotiq task validation (2026-07-22)

- Tabletop, seed 0 after the zero-gap + direct-approach fixes: `SUCCESS=True`; the run starts directly with
`pregrasp → approach`, grasps at tick 170, and releases the apple in the bowl. Artifacts:
`/tmp/vega_robotiq_gap_path_fix_seed0.mp4` and `/tmp/vega_robotiq_gap_path_fix_seed0_3rd.mp4`.
- Mobile, seed 0: `SUCCESS=True`; the base navigates from x=−0.35 to the Robotiq park point, then completes
the same grasp/carry/place. Final apple `[1.187, 0.537, 0.801]`, bowl `[1.182, 0.541, 0.794]`.
Artifacts: `/tmp/vega_robotiq_mobile_seed0_v3.mp4` and
`/tmp/vega_robotiq_mobile_seed0_v3_3rd.mp4`.
- OmniGibson logs a benign finger-inference warning because the multi-link 2F-85 fingers do not share one
parent. The explicit finger links, master joints, and assisted-grasp points in the robot definition are
loaded and used, so this generic inference limitation does not affect the validated task behavior.



## Mobile fine grasp — resolved (historical Panda `mobile_v10`)

Navigation: base drives −0.35 → 0.66 via WBC head-following + `base_ctrl` shaping, parks at `base_x_max`.
Earlier failure mode: during nav the torso COLLAPSED (parked torso_j1≈0.166/0.45 vs nominal 0.78) → bad
grasp start pose → apple knocked off.

Root cause: `head_world_position_cost` z-axis was free (0). Fix iterations:

- v8: reachable nav head target → ee_err→0.001 but apple knocked.
- v9: `posture_cost_torso=200` → no change.
- **v10:** `head_world_position_cost=[10000,10000,10000]` **(enable z tracking)** — pick-and-place **SUCCESS**
(`SUCCESS=True`). Artifacts: `.../scratchpad/mobile_v10.log`, `mobile_v10.mp4`, `mobile_v10_3rd.mp4`.

Physics/IK mounting of the Panda hand on `L_ee`/`R_ee` was functionally correct. This section is retained as
history; the current Robotiq mobile validation and its earlier park distance are documented above.

---



## Resolved issue — panda gripper visual / “floating wrist cam” (2026-07-22)

The apparent wrist camera was mis-oriented Panda visual geometry; the deployed robot still has exactly one
camera prim (the head camera). The physical graft joints and collision meshes were already correct.

Root cause: the canonical Panda visuals are Z-up DAE, but the bundled Assimp-generated OBJ files have the
coordinate mapping `(x, y, z)_obj = (x, z, -y)_dae`. The graft deliberately used OBJ because the `behavior`
environment has no `pycollada`; importing those OBJ coordinates unchanged put the hand and finger visuals
90 degrees out of plane. Older imports also lacked usable Panda visual prims and exposed the white collision
geometry instead.

Fix: `graft_panda_gripper.py` now rotates OBJ vertices and normals by +90 degrees around X
(`(x, y, z) -> (x, -z, y)`) while staging the assets, retains their MTL materials, and asserts the corrected
hand is Y-long and finger is Z-long. The robot was re-grafted, re-imported, and copied to
`datasets/omnigibson-robot-assets/models/vega/` with the Panda robot definition restored.

Validation:

- All six deployed `L/R_panda_{hand,leftfinger,rightfinger}` links have visual references.
- Deployed finger visual extents are `[0.021003, 0.026429, 0.053767]` XYZ, matching the canonical Franka USD;
before the fix they were `[0.021003, 0.053767, 0.026429]`.
- Fresh seed-0 mobile-v10 episode: `SUCCESS=True`; grasp remains attached through lift/carry/place.
- Artifacts: `scratchpad/mobile_visual_fixed_v10.mp4` and `mobile_visual_fixed_v10_3rd.mp4`.

---



## Later phases (from plan `delegated-soaring-torvalds.md`)

- **P4 collect** — N randomized episodes → ManiFlow zarr (`point_cloud (T,1024,6)`, `state (T,32)`,
`action (T,29)`, `env_state (T,6)`, `episode_ends`). Obs pipeline = head cam depth+rgb → `unproject_head_frame`
→ world crop `(0.20,-0.90,0.60)/(1.70,1.05,1.32)` → FPS 1024. Verify with `vis_episode_processed_wbc.py --zarr`
  - 1-epoch smoke.
- **P5 eval** — load ManiFlow `.ckpt`, async chunk scheduler, apply 29-D actions via WBC, report success rate.



## Pipeline

```bash
(behavior) python sim_eval/collect_demos.py --task mobilepickplace --episodes 2 --out ~/Dexmate/data/sim_mobilepickplace
# --depth-dropout

(dexmate_lerobot) python scripts/port_wbc_mobile_hdf5.py \
  --raw-dir ~/Dexmate/data/sim_mobilepickplace/raw_data \
  --root ~/Dexmate/data/sim_mobilepickplace/processed_wbc \
  --repo-id sim_mobilepickplace \
  --split-csv ~/Dexmate/data/sim_mobilepickplace/raw_data/split.csv \
  --positions-dir ~/Dexmate/data/sim_mobilepickplace/scene_diff/positions \
  --object_nums 2 \
  --overwrite

(dexmate_lerobot) python scripts/vis_episode_processed_wbc.py \
  --dataset_dir ~/Dexmate/data/sim_mobilepickplace/processed_wbc/test/sim_mobilepickplace \
  --episode_index 0
  
```

