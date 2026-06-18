# Vega-1 2D Lidar SLAM (slam_toolbox)

ROS2 Humble + slam_toolbox in Docker, fed by the robot's Zenoh sensors through
a custom bridge. Produces an occupancy-grid map and a continuous
`map -> base_link` pose republished to Zenoh (`dm/vg3cfe65689d-1/slam/pose`)
for the omniteleop stack. Everything runs on the Jetson (vega-1); WiFi loss
only drops visualization.

## Architecture

```
RPLidar (10 Hz)──dexsensor──┐ zenoh (mTLS overlay, robopil config)
chassis steer/drive (100 Hz)┤
                            ▼
            bridge/zenoh_ros_bridge.py          [container, ros:humble]
              ├─ /scan  (LaserScan, 720 uniform bins)
              ├─ /odom + TF odom->base_link  (swerve FK, 50 Hz)
              ├─ static TF base_link->laser  (URDF lidar_mount)
              └─ map->base_link  ──► zenoh slam/pose (JSON, 20 Hz)
                            ▲
            slam_toolbox (mapping or localization) + foxglove :8765
```

Key facts discovered during bring-up (2026-06-11):

- The robot's zenoh runs as an **mTLS peer overlay** (scouting 224.0.0.237:50297),
config + certs in `~/.dexmate/comm/zenoh/robopil/` — default zenoh
multicast sees NOTHING. The container mounts the certs at the same path.
- Lidar topic is `sensors/lidar_2d_front/scan` (note the `/scan` suffix),
~~2070 pts/rev, **non-uniform** angle spacing (binned to 720 uniform bins),
invalid returns encoded as range 0, sensor range_max 12 m,
Jetson-clock timestamps (~~1 ms offset → sensor stamping).
- Firmware topics (`state/chassis/`*, `state/ultrasonic`) have a constant
**+73 ms clock offset** → odometry uses receive-time stamping.
- `state/chassis_imu` is **not published** by this robot — odometry is
wheel-only; slam_toolbox's wide correlation search absorbs yaw drift.
- The repo is an **sshfs mount** the docker daemon can't bind-mount →
`scripts/sync_to_local.sh` rsyncs `SLAM/` to `/home/dexmate/yixuan_slam_runtime`
(mounted as `/slam`). The same applies to the results tree
`/home/dexmate/yixuan/Dexmate/SLAM` (`$SLAM_RESULTS_DIR`, also sshfs): the
container can never write there directly, so host-side script stages copy
results out. Container working files stay in the runtime dir.
- Zenoh peer discovery of the firmware peer takes **~10-20 s** after container
start — wait before judging missing topics.

## Prerequisites

- `dexsensor launch --sensor lidar_2d` running on the Jetson (user process —
keep it in a tmux; the bridge logs a warning if the lidar goes silent >2 s).
- Docker image built once: `cd SLAM/docker && docker compose --profile mapping build`
(tegra kernel quirk: builds use `network: host`, already configured).

## Operating guide — step by step

All commands run on the Jetson (`vega-1`). `<repo>` = `~/yixuan/omniteleop_yifan`.
Two phases: **mapping** (drive the robot around to build a map) then
**localization** (load that map, robot tracks its pose in it). Run them one at a
time — never both containers at once (they would both grab `/scan` and TF).

### Step 0 — prerequisites (every session)

```bash
dexsensor launch --sensor lidar_2d          # data source; keep in its own tmux
cd ~/yixuan/omniteleop_yifan/SLAM
./scripts/sync_to_local.sh                  # run after ANY edit under SLAM/
```

> The container mounts the local runtime dir, not the sshfs repo. Forgetting
> `sync_to_local.sh` is the #1 "my change did nothing" cause.

### Step 1 — start mapping (fresh)

```bash
cd ~/yixuan/omniteleop_yifan/SLAM/docker
docker compose --profile mapping up -d --force-recreate   # always an empty graph
sleep 20                                    # Zenoh peer discovery: 10–20 s
docker logs --tail 15 yixuan-slam-mapping
```

> **Start fresh.** slam_toolbox freezes the map while the robot is stationary,
> so a container left up for hours serves a stale map (seen: 0.43 m phantom
> scan↔map offset that dropped to 3.5 cm after `--force-recreate`).

### Step 2 — live view (Foxglove, optional)

On the workstation: Foxglove Studio → `ws://192.168.0.193:8765` → 3D panel →
subscribe `/map`, `/scan`, `/tf`, `/odom`. Set the `/scan` **Color mode → Flat**
(its `inf` no-return bins otherwise trip a cosmetic black/magenta warning).
WiFi loss only drops this view; mapping keeps running on the robot.
### Step 3 — verify the scan lands on walls (before driving)

```bash
# stage A (container): dump live map + scan + pose
docker exec yixuan-slam-mapping bash -c \
  'source /opt/ros/humble/setup.bash && python3 /slam/scripts/vis_map_snapshot.py --dump'
# stage B (host): render -> $SLAM_RESULTS_DIR/debug/map_snapshot.png
conda run -n yixuan_yifan python \
  /home/dexmate/yixuan_slam_runtime/scripts/vis_map_snapshot.py --render
```

Two stages because rclpy lives only in the container, and only the host can
write the sshfs results tree. **Read it:** black=walls, white=free, gray=unknown
(simply unobserved), red=live scan, blue=robot. Red must sit on black — expect
~3.5 cm on a fresh map. If red is mirrored or tilted off the walls, calibrate
before driving (see [Calibration reference](#calibration-reference-params-in-bridgebridge_paramsyaml)).

### Step 4 — DRIVE THE ROBOT MANUALLY to build the map

The only phase where the robot moves. Teleop slowly around the space (your usual
exo/joycon/VR teleop, or chassis velocity commands):

- **Slow and smooth**, especially turns — odometry is wheel-only (no IMU), so
fast spins drift most. Pause briefly to let scan matching settle.
- **Cover the whole interior** in overlapping passes, not just the perimeter —
point the lidar at every surface you want mapped.
- **Close the loop**: return to where you started; slam_toolbox snaps it shut
and straightens the whole map.
- Re-run Step 3 periodically: walls should stay single-line. A double wall =
drift → drive back over that area or slow down.

### Step 5 — save the map

```bash
cd <repo>/SLAM
./scripts/save_map.sh lab_v1      # -> runtime maps/ + $SLAM_RESULTS_DIR/maps/
```
Writes `lab_v1.{posegraph,data,pgm,yaml}`: `.posegraph`/`.data` reload for
localization; `.pgm`/`.yaml` is the viewable occupancy image.

### Step 6 — export a point cloud (optional)

```bash
# stage A (container): accumulate scans for 30 s
docker exec yixuan-slam-mapping bash -c \
  'source /opt/ros/humble/setup.bash && python3 /slam/scripts/export_pointcloud.py --dump --secs 30'
# stage B (host): -> $SLAM_RESULTS_DIR/pointcloud/lab_v1.ply + preview PNG
conda run -n yixuan_yifan python \
  /home/dexmate/yixuan_slam_runtime/scripts/export_pointcloud.py --convert --name lab_v1
```
Planar cloud only (map cells z=0 + scans z=0.144 m). Driving during the window
enriches coverage; the map cells already hold everything from Step 4. Details:
[Point cloud export](#point-cloud-export-step-6).

### Step 7 — localize against the saved map

```bash
cd <repo>/SLAM/docker
docker compose --profile mapping down
MAP_NAME=lab_v1 docker compose --profile localization up -d
conda run -n yixuan_yifan python <repo>/SLAM/scripts/probe_slam_pose.py
```
slam_toolbox now only tracks pose in the saved map (no new mapping). It assumes
`(0,0,0)` at startup — **place the robot at the map origin** so it locks on.
`probe_slam_pose.py` prints the Zenoh pose stream (`…/slam/pose`, ~20 Hz, low
age) the omniteleop stack consumes; verify with the Step 3 snapshot. Re-map a
changed space → back to Step 1.

## Results location

All results land under `/home/dexmate/yixuan/Dexmate/SLAM` (override with
`SLAM_RESULTS_DIR`): `maps/` (save_map.sh), `debug/` (vis_scan.py,
vis_map_snapshot.py), `pointcloud/` (export_pointcloud.py).

## Point cloud export (Step 6)

slam_toolbox is 2D — no native point cloud. `scripts/export_pointcloud.py`
derives a **planar** cloud: occupied map cells (z=0) + live scans projected into
the map frame and accumulated over a window (z = lidar height 0.144 m). A true
3D cloud is impossible here (2D RPLidar only — vega-1 has no 3D lidar); 3D would
need the head ZED depth (separate pipeline). Commands: operating guide Step 6.

Pose JSON schema: `{timestamp_ns, frame: "map", child_frame: "base_link", pos: [x,y,z], quat_wxyz: [w,x,y,z], yaw}`.

## Calibration reference (params in `bridge/bridge_params.yaml`)

First time on this robot, stationary, ~1 min. Watch `/scan` in Foxglove (Step 2)
or re-run the Step 3 snapshot after each change (then
`sync_to_local.sh && docker compose --profile mapping up -d --force-recreate`).
All **PENDING — needs operator**:

1. **Angle direction** (`scan.angle_sign`, default `+1`): walk to the robot's
   **left**; if you appear on its **right**, the scan is mirrored → set `-1.0`.
2. **Yaw offset** (`scan.laser_yaw`, default `0`): face the robot square at a
   flat wall; the wall should be ⟂ to x. A consistent tilt is the offset (rad).
3. **Drive units** (`odom.drive_state_mode`, default `ms`): clear ~1 m ahead,
   `conda run -n yixuan_yifan python <repo>/SLAM/scripts/odom_sanity.py --test units`
   (drives 0.2 m/s × 3 s; prints `ms` vs `rad` verdict).
4. **Odometry accuracy** (`odom_sanity.py --test watch` while teleoperating):
   1 m forward |x err| <5 %; 0.5 m strafe y sign+magnitude; 360° spin yaw
   closure ≲15° wheel-only (scan matching corrects the residual).

## Verification status (2026-06-11 bring-up, robot stationary)


| Check                                                                    | Result                                               |
| ------------------------------------------------------------------------ | ---------------------------------------------------- |
| swerve FK unit tests (`tests/test_swerve_odometry.py`)                   | 10/10 pass                                           |
| /scan rate / format in container                                         | 10.0 Hz, 720 bins, frame `laser`                     |
| /odom at standstill                                                      | identity pose, zero twist, 50 Hz                     |
| TF tree                                                                  | base_link->laser (0.2975, 0, 0.144); odom->base_link |
| mapping: /map + map->base_link                                           | produced within seconds                              |
| slam/pose over Zenoh (conda consumer)                                    | 20 Hz, ~10 ms age                                    |
| save_map.sh                                                              | .posegraph/.data/.pgm/.yaml written                  |
| localization from saved map                                              | pose 20 Hz, standstill jitter < 1 mm                 |
| container restart (`restart: unless-stopped`)                            | pose stream auto-recovers                            |
| physical calibration (angle_sign, laser_yaw, drive units, odom accuracy) | **PENDING — needs operator**                         |
| real mapping loop + loop closure quality                                 | **PENDING — needs teleop drive**                     |


## Files

```
docker/      Dockerfile (ros:humble + dexcomm==0.4.19), compose.yaml (host net,
             profiles: mapping / localization / tools)
bridge/      zenoh_ros_bridge.py (the bridge node), swerve_odometry.py (pure FK),
             bridge_params.yaml (topics, calibration, stamping)
config/      slam_mapping.yaml, slam_localization.yaml (slam_toolbox params)
launch/      slam_mapping.launch.py, slam_localization.launch.py
scripts/     probe_topics.py, vis_scan.py (step-0 probes, conda env),
             vis_map_snapshot.py (map+scan PNG), export_pointcloud.py (planar
             cloud -> results .ply, two-stage), odom_sanity.py (units/watch),
             probe_slam_pose.py (pose consumer), save_map.sh, sync_to_local.sh,
             check_scan_msg.py, check_scan_tf.py
tests/       test_swerve_odometry.py (pytest, runs in conda env or container)
```

## Troubleshooting

- **No topics in container**: wait 20 s (peer gossip); check certs mount;
`docker compose run --rm tools python3 /slam/scripts/probe_topics.py --skip-discovery`.
- **No lidar**: is `dexsensor launch --sensor lidar_2d` alive? Bridge warns in
`docker logs yixuan-slam-mapping`.
- **map_saver timeout**: already uses `save_map_timeout:=15.0`; slam_toolbox
must have processed ≥1 scan.
- **Edits not taking effect**: you forgot `scripts/sync_to_local.sh` — the
container mounts `/home/dexmate/yixuan_slam_runtime`, not the repo.
- **Foxglove `/scan` magenta/black or no dots**: cosmetic — `inf` no-return
bins trip the color-mapper. Set the `/scan` panel **Color mode → Flat**, point
size 3–5. Confirm the scan is fine with the Step 3 snapshot (red on black);
empty there too → lidar down (`dexsensor`); map empty but scan ok → SLAM
container not up.
- **Large rigid scan↔map offset, map won't change**: stale frozen map from a
long-running container. Restart fresh: `docker compose --profile mapping up -d
--force-recreate` (operating guide Step 1).

