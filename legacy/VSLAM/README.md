## SLAM

### ZED positional tracking → robot base pose

Official doc [https://www.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking](https://www.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking).

Robot base pose in an odom frame from head-ZED VSLAM+IMU tracking fused with torso/head forward kinematics: `T_odom_base = T_odom_track · T_track_cam · T_base_cam(q)⁻¹`, odom = base at the first OK-tracked sample. 

```
tests/test_head_zedx_depth.py --enable-tracking   # owns the camera (one process only);
  └→ sensors/head_camera/pose                     # SDK GEN_3 tracking, quaternion_xyzw + translation (float64-exact)
scripts/zed_base_pose_node.py                     # fusion node
  ├← sensors/head_camera/pose + state/{head,torso}
  └→ state/base_pose_zed                          # + 1 Hz log, CSV/MP4 (XY + z trace) on exit
src/omniteleop/follower/zed_base_pose.py          # the math: VegaHeadCamFK (pinocchio, dexmate_urdf vega_1.urdf,
                                                  #   base → zed_depth_frame = optical frame, no axis remap),
                                                  #   ZedBasePoseEstimator (strict SE(3)), pose_from_quat_trans
                                                  #   (rebuilds exact rotation from the SDK quaternion)
tests/test_zed_base_pose.py                       # offline unit tests (no camera/Zenoh)
```

Run on dexmate machine:

```bash
# robot.head.set_joint_pos([0.0, 0.0, 0.79], wait_time=3.0, exit_on_reach=True)

(yixuan_yifan) LOCAL=/home/dexmate/yixuan/Dexmate_local/SLAM && mkdir -p "$LOCAL/svo" "$LOCAL/logs" "$LOCAL/maps" "$LOCAL/maps/lab_v1_spatial_snapshots" && python /home/dexmate/yixuan/omniteleop_yifan/scripts/zed_area_map_headless.py \
      --resolution SVGA \
      --depth-mode NEURAL \
      --record-svo "$LOCAL/svo/lab_v1.svo2" \
      --tum-file "$LOCAL/logs/lab_v1.tum" \
      2>&1 | tee "$LOCAL/logs/lab_v1.log"
# depth will be estimated offline, so quality does not matter
# --resolution HD1200 
Press Ctrl-C once

(yixuan_yifan) LOCAL=/home/dexmate/yixuan/Dexmate_local/SLAM && python /home/dexmate/yixuan/omniteleop_yifan/scripts/zed_area_map_headless.py \
    --svo-file "$LOCAL/svo/lab_v1.svo2" \
    --depth-mode NEURAL \
    --output-area-file "$LOCAL/maps/lab_v1.area" \
    --tum-file "$LOCAL/logs/lab_v1_offline.tum" \
    --spatial-map-file "$LOCAL/maps/lab_v1_visual_mesh.obj" \
    --spatial-map-snapshot-dir "$LOCAL/maps/lab_v1_spatial_snapshots" \
    --spatial-map-snapshot-period 50 \
    --spatial-map-texture \
    2>&1 | tee "$LOCAL/logs/lab_v1_offline.log"

mkdir -p /home/dexmate/yixuan/Dexmate/SLAM/zed/results
rsync -av \
  /home/dexmate/yixuan/Dexmate_local/SLAM/ \
  /home/dexmate/yixuan/Dexmate/SLAM/zed/results/
------------------------------------------------------------------------------------------------------
(lerobot) python /home/yixuan/omniteleop/scripts/view_mesh.py /home/yixuan/Dexmate/SLAM/zed/results/maps/lab_v1_visual_mesh.obj --traj /home/yixuan/Dexmate/SLAM/zed/results/logs/lab_v1_offline.tum

python /home/yixuan/omniteleop/scripts/view_mesh.py '/home/yixuan/Dexmate/SLAM/zed/results/maps/lab_v1_spatial_snapshots/spatial_map_*.obj' # N/P
------------------------------------------------------------------------------------------------------
# 1.
(yixuan_yifan) python tests/test_head_zedx_depth.py \
    --enable-tracking \
    --area-file /home/dexmate/yixuan/Dexmate/SLAM/zed/results/maps/lab_v1.area \
    --localization-only
# zed.get_position() reads back the pose that grab() already computed on the full frame

# 2. fusion node — Ctrl-C (or --duration N) to stop; rendering runs strictly after capture
(yixuan_yifan) python scripts/zed_base_pose_node.py --save_video /home/dexmate/yixuan/Dexmate/SLAM/test/tmp.mp4

# unit tests
# python -m pytest tests/test_zed_base_pose.py -q

# 3. 
(yixuan_yifan) python scripts/drive_box_record.py \
    --zed-area-file /home/dexmate/yixuan/Dexmate/SLAM/zed/results/maps/lab_v1.area \
    --output /home/dexmate/yixuan/Dexmate/SLAM/test/drive_box.hdf5
# --zed-area-file just for checking
# change --leg-time or revise build_legs
# --closed-loop-source odom/zed

# 4.
python scripts/plot_drive_box.py --input /home/dexmate/yixuan/Dexmate/SLAM/test/drive_box.hdf5

# ffmpeg -y -i /home/yixuan/Dexmate/SLAM/test/turn/drive_box.mp4 -vf "fps=10,scale=880:-1:flags=lanczos,palettegen" /tmp/palette.png
# ffmpeg -y -i /home/yixuan/Dexmate/SLAM/test/turn/drive_box.mp4 -i /tmp/palette.png -lavfi "fps=10,scale=880:-1:flags=lanczos[x];[x][1:v]paletteuse" /home/yixuan/Dexmate/SLAM/test/turn/drive_box.gif
```

> Landmines: ZED SDK poses are denormalized float32 — the publisher sends the SDK quaternion+translation and the node rebuilds an exact SE(3) (`pose_from_quat_trans`), never the raw 4x4 matrix. `state/*` source timestamps run ~100 ms ahead of the local clock (separate clocks), so the node matches joints against the camera publisher's local capture `timestamp_ns` and logs the chosen joint deltas in CSV. The camera sits ~1.46 m from the base, so ZED orientation noise is amplified ~2.5 cm/° into base XY.

