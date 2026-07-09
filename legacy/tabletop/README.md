# Legacy Tabletop Manipulation

These notes are archived from the root README. They describe the pre-WBC tabletop
workflow; the current mobile manipulation pipeline starts in
[../../README.md](../../README.md).

## Tabletop Manip

1. Follow Robot preparation in [../../README.md](../../README.md)

2. Follow Robot preparation in [../../README.md](../../README.md)

3.1 test wrist camera via  ZED SDK (do not run dexsensor):

```python
python /home/dexmate/yixuan/omniteleop/tests/test_wrist_zedm_depth.py --duration 10 --save-dir /home/dexmate/yixuan/Dexmate
```

3.2 check RoI:

```python
python /home/yixuan/omniteleop/tests/crop_and_resize.py
```

1. ssh lambda, conda activate dexmate
2. run
  (dexmate) python src/omniteleop/follower/vr_robot_controller.py # workspace_check = True, joint positions in vr_mode_const, head_mode and left_arm_mode in src/omniteleop/configs/vega_1_f5d6.yaml
3. (dexmate) python src/omniteleop/leader/vr_reader.py --manual_reposition /home/yixuan/Dexmate/data/raw_data/episode_0.hdf5
4. (dexmate) python src/omniteleop/leader/vr_reader.py # start-mode=fixed_pose

> debug /target/eef_used_by_ik/right: IK input
> /action/joint/right_arm: IK output + step clamping

```bash
Follow exactly:
# start_mode in main()
follow_hand mode:
1. Trigger (track head)
2. Trigger (track hand)
3. X
4. Move to new position
5. Y (track head, initialize hand, track hand)
6. A-Manip-B-X
7. Repeat 3-6

fixed_pose mode:
1. Move to new position
2. Y (initialize hand, track head)
3. A-Trigger (track hand)-Manip-B-X
4. Reset
5. Repeat 2-3

python /home/yixuan/omniteleop/scripts/save_rgb_print_fps.py # if you forgot position last time

python /home/dexmate/yixuan/omniteleop_yifan/scripts/admittance_control.py run #calibrate first

before shutdown:
python -m omniteleop.follower.safearm_shutdown # reposition arm so that they fall onto table

visualize:
python /home/yixuan/omniteleop/scripts/vis_episode.py --hdf5 /home/yixuan/Dexmate/data/raw_data/episode_100.hdf5
```

```bash
python /home/yixuan/omniteleop/legacy/tabletop/scripts/rename_raw_data.py # revise train/val/test range
```

```bash
python /home/yixuan/omniteleop/scripts/vis_teleop_curves.py --episode-id 0 # (Optional) (Need to save_debug in vr_reader)
```

## Policy

See [Lerobot README](../../../lerobot_yifan/README.md).
Run infer_dexmate.py before deploy.

## Deploy

Legacy tabletop policy/deploy scripts are archived under `legacy/tabletop/`.
They are kept as references only; the current mobile manipulation pipeline starts
in the root README's WBC section.
