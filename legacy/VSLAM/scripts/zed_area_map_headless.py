#!/usr/bin/env python3
"""Headless ZED VSLAM area-map builder/relocalizer.

This is the official positional-tracking sample's mapping path without the
OpenGL/freeglut viewer, so it works over SSH on a headless Jetson. Stop any other
camera owner first (for example dexsensor head_camera).

Examples:
  python scripts/zed_area_map_headless.py \
      --output-area-file /home/dexmate/yixuan/Dexmate/SLAM/zed/maps/lab_v1.area \
      --tum-file /home/dexmate/yixuan/Dexmate/SLAM/zed/logs/lab_v1.tum

  python scripts/zed_area_map_headless.py \
      --input-area-file /home/dexmate/yixuan/Dexmate/SLAM/zed/maps/lab_v1.area \
      --localization-only
"""

from __future__ import annotations

import argparse
import pathlib
import signal
import sys
import time
from typing import Optional

import pyzed.sl as sl


_STOP = False


def _handle_stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def _enum_name(value) -> str:
    return str(value).split(".")[-1]


def _safe_print(*args, **kwargs) -> None:
    """``print`` that tolerates a dead stdout (e.g. a ``| tee`` killed by Ctrl-C).

    Ctrl-C sends SIGINT to the whole foreground group, so a piped ``tee`` dies
    while this script (which installs its own SIGINT handler) keeps running. The
    area map is only saved *after* the grab loop, so a BrokenPipeError on the
    first post-interrupt print would otherwise abort the export before
    ``save_area_map`` ever runs. Swallow the broken-pipe and keep going.
    """
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except (BrokenPipeError, OSError):
        pass


def _resolution(name: str):
    try:
        return getattr(sl.RESOLUTION, name)
    except AttributeError as exc:
        allowed = ["SVGA", "HD1080", "HD1200", "HD720", "VGA", "HD2K"]
        raise argparse.ArgumentTypeError(
            f"unknown resolution {name!r}; expected one of {allowed}"
        ) from exc


def _tracking_mode(name: str):
    try:
        return getattr(sl.POSITIONAL_TRACKING_MODE, name)
    except AttributeError as exc:
        allowed = ["GEN_1", "GEN_2", "GEN_3"]
        raise argparse.ArgumentTypeError(
            f"unknown tracking mode {name!r}; expected one of {allowed}"
        ) from exc


def _spatial_map_type(name: str):
    key = name.upper()
    aliases = {"FPC": "FUSED_POINT_CLOUD", "POINT_CLOUD": "FUSED_POINT_CLOUD"}
    key = aliases.get(key, key)
    try:
        return getattr(sl.SPATIAL_MAP_TYPE, key)
    except AttributeError as exc:
        allowed = ["MESH", "FUSED_POINT_CLOUD"]
        raise argparse.ArgumentTypeError(
            f"unknown spatial map type {name!r}; expected one of {allowed}"
        ) from exc


def _mapping_resolution(name: str):
    try:
        return getattr(sl.MAPPING_RESOLUTION, name.upper())
    except AttributeError as exc:
        allowed = [n for n in dir(sl.MAPPING_RESOLUTION) if n.isupper()]
        raise argparse.ArgumentTypeError(
            f"unknown spatial mapping resolution {name!r}; expected one of {allowed}"
        ) from exc


def _mapping_range(name: str):
    key = name.upper()
    aliases = {"NEAR": "SHORT", "FAR": "LONG"}
    key = aliases.get(key, key)
    try:
        return getattr(sl.MAPPING_RANGE, key)
    except AttributeError as exc:
        allowed = [n for n in dir(sl.MAPPING_RANGE) if n.isupper()] + ["NEAR", "FAR"]
        raise argparse.ArgumentTypeError(
            f"unknown spatial mapping range {name!r}; expected one of {allowed}"
        ) from exc


def _mesh_filter(name: str):
    key = name.upper()
    if key == "NONE":
        return None
    try:
        return getattr(sl.MESH_FILTER, key)
    except AttributeError as exc:
        allowed = ["NONE"] + [n for n in dir(sl.MESH_FILTER) if n.isupper()]
        raise argparse.ArgumentTypeError(
            f"unknown mesh filter {name!r}; expected one of {allowed}"
        ) from exc


def _make_spatial_map(map_type):
    if map_type == sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD:
        return sl.FusedPointCloud()
    return sl.Mesh()


def _default_spatial_suffix(map_type) -> str:
    return ".ply" if map_type == sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD else ".obj"


def _save_spatial_map(map_obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = map_obj.save(path.as_posix())
    if not ok:
        raise RuntimeError(f"failed to save spatial map: {path}")
    _safe_print(f"[zed-map] spatial map saved: {path}")


def _export_area_map(zed: sl.Camera, output_area_file: pathlib.Path) -> None:
    output_area_file.parent.mkdir(parents=True, exist_ok=True)
    _safe_print(f"[zed-map] saving area map: {output_area_file}")
    status = zed.save_area_map(output_area_file.as_posix())
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"save_area_map failed: {status}")

    while True:
        export_state = zed.get_area_export_state()
        if export_state != sl.AREA_EXPORTING_STATE.RUNNING:
            break
        time.sleep(0.05)

    if export_state != sl.AREA_EXPORTING_STATE.SUCCESS:
        raise RuntimeError(f"area export failed: {export_state}")
    _safe_print(f"[zed-map] area map saved: {output_area_file}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-id", type=int, default=0, help="GMSL camera id/port.")
    p.add_argument("--resolution", type=_resolution, default=None)
    p.add_argument(
        "--rate",
        type=int,
        default=0,
        help="Camera FPS. 0 leaves the SDK default untouched, matching the C++ sample.",
    )
    p.add_argument("--depth-mode", default="NEURAL")
    p.add_argument("--tracking-mode", type=_tracking_mode, default=sl.POSITIONAL_TRACKING_MODE.GEN_3)
    p.add_argument("--input-area-file", type=pathlib.Path)
    p.add_argument("--output-area-file", type=pathlib.Path)
    p.add_argument("--svo-file", type=pathlib.Path, help="Optional SVO/SVO2 input for offline map generation.")
    p.add_argument("--record-svo", type=pathlib.Path, help="Optional SVO2 recording while mapping live.")
    p.add_argument(
        "--svo-real-time",
        action="store_true",
        help="Replay SVO at recorded wall-clock speed. Default is fastest replay, matching the C++ sample.",
    )
    p.add_argument("--tum-file", type=pathlib.Path, help="Write camera trajectory in TUM format.")
    p.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means until Ctrl-C or SVO end.")
    p.add_argument("--log-period", type=float, default=10.0)
    p.add_argument(
        "--max-grab-errors",
        type=int,
        default=3,
        help="Abort after this many consecutive grab errors. Important for failing fast on corrupted SVOs.",
    )
    p.add_argument(
        "--disable-self-calib",
        action="store_true",
        help="Disable ZED self-calibration at open. Default keeps self-calibration enabled, matching C++.",
    )
    p.add_argument(
        "--spatial-map-file",
        type=pathlib.Path,
        help="Optional visualization geometry export (.obj/.ply): enables ZED Spatial Mapping.",
    )
    p.add_argument(
        "--spatial-map-type",
        type=_spatial_map_type,
        default=sl.SPATIAL_MAP_TYPE.MESH,
        help="Visualization geometry type: MESH or FUSED_POINT_CLOUD.",
    )
    p.add_argument(
        "--spatial-map-resolution",
        type=_mapping_resolution,
        default=sl.MAPPING_RESOLUTION.HIGH,
        help="Spatial Mapping resolution preset: LOW, MEDIUM, HIGH.",
    )
    p.add_argument(
        "--spatial-map-range",
        type=_mapping_range,
        default=sl.MAPPING_RANGE.SHORT,
        help="Spatial Mapping range preset: SHORT/NEAR, MEDIUM, LONG/FAR.",
    )
    p.add_argument("--spatial-map-max-memory-mb", type=int, default=2048)
    p.add_argument(
        "--spatial-map-texture",
        action="store_true",
        help="Save mesh texture data and apply it on final export. Mesh only.",
    )
    p.add_argument(
        "--spatial-map-filter",
        type=_mesh_filter,
        default=sl.MESH_FILTER.LOW,
        help="Final mesh filter: NONE, LOW, MEDIUM, HIGH. Mesh only.",
    )
    p.add_argument(
        "--spatial-map-snapshot-dir",
        type=pathlib.Path,
        help="Optional directory for periodic async spatial-map snapshots.",
    )
    p.add_argument(
        "--spatial-map-snapshot-period",
        type=float,
        default=10.0,
        help="Seconds between snapshot requests. 0 disables snapshots.",
    )
    p.add_argument("--localization-only", action="store_true", help="Use an input map for localization without updating it.")
    p.add_argument("--2d-ground-mode", action="store_true", help="Constrain tracking to a 2D ground plane.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.svo_file and args.record_svo:
        print("[zed-map] --svo-file and --record-svo are mutually exclusive", file=sys.stderr)
        return 2
    if args.localization_only and not args.input_area_file:
        print("[zed-map] --localization-only requires --input-area-file", file=sys.stderr)
        return 2
    if args.rate < 0:
        print("[zed-map] --rate must be >= 0", file=sys.stderr)
        return 2
    if args.max_grab_errors < 1:
        print("[zed-map] --max-grab-errors must be >= 1", file=sys.stderr)
        return 2
    if args.spatial_map_max_memory_mb <= 0:
        print("[zed-map] --spatial-map-max-memory-mb must be > 0", file=sys.stderr)
        return 2
    if args.spatial_map_type == sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD:
        if args.spatial_map_texture:
            print("[zed-map] --spatial-map-texture is only valid for mesh maps", file=sys.stderr)
            return 2
        if args.spatial_map_filter is not None:
            print("[zed-map] --spatial-map-filter is only valid for mesh maps", file=sys.stderr)
            return 2
    if args.spatial_map_snapshot_dir and args.spatial_map_snapshot_period <= 0:
        print("[zed-map] --spatial-map-snapshot-dir requires --spatial-map-snapshot-period > 0", file=sys.stderr)
        return 2
    spatial_mapping_requested = args.spatial_map_file is not None or args.spatial_map_snapshot_dir is not None

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    zed = sl.Camera()

    init = sl.InitParameters()
    init.coordinate_units = sl.UNIT.METER
    # Keep the official mapping sample convention for .area creation/relocalization.
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    if args.resolution is not None:
        init.camera_resolution = args.resolution
    if args.rate > 0:
        init.camera_fps = args.rate
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.camera_disable_self_calib = args.disable_self_calib
    if args.svo_file:
        init.set_from_svo_file(args.svo_file.as_posix())
        init.svo_real_time_mode = args.svo_real_time
    else:
        init.set_from_camera_id(args.camera_id)

    print(
        f"[zed-map] open camera resolution={_enum_name(args.resolution) if args.resolution else 'SDK_DEFAULT'} "
        f"fps={args.rate if args.rate > 0 else 'SDK_DEFAULT'} depth={args.depth_mode} "
        f"self_calib={'disabled' if args.disable_self_calib else 'enabled'} "
        f"svo_real_time={args.svo_real_time} "
        f"source={args.svo_file if args.svo_file else f'camera_id:{args.camera_id}'}",
        flush=True,
    )
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"[zed-map] failed to open camera: {status}", file=sys.stderr)
        zed.close()
        return 1

    tracking = sl.PositionalTrackingParameters()
    if args.input_area_file:
        tracking.area_file_path = args.input_area_file.as_posix()
    tracking.set_floor_as_origin = False
    tracking.set_gravity_as_origin = True
    tracking.enable_area_memory = True
    tracking.enable_pose_smoothing = False
    tracking.enable_imu_fusion = True
    tracking.set_as_static = False
    tracking.depth_min_range = -1
    tracking.enable_2d_ground_mode = args.__dict__["2d_ground_mode"]
    tracking.enable_localization_only = args.localization_only
    tracking.mode = args.tracking_mode

    status = zed.enable_positional_tracking(tracking)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"[zed-map] failed to enable positional tracking: {status}", file=sys.stderr)
        zed.close()
        return 1

    print(
        "[zed-map] tracking enabled: "
        f"mode={_enum_name(args.tracking_mode)} "
        f"area_memory={tracking.enable_area_memory} "
        f"imu_fusion={tracking.enable_imu_fusion} "
        f"localization_only={tracking.enable_localization_only} "
        f"input_area={args.input_area_file}",
        flush=True,
    )

    spatial_map = None
    spatial_map_snapshot = None
    spatial_snapshot_pending = False
    spatial_snapshot_index = 0
    next_spatial_snapshot_ts: Optional[float] = None
    if spatial_mapping_requested:
        spatial_params = sl.SpatialMappingParameters(
            resolution=args.spatial_map_resolution,
            mapping_range=args.spatial_map_range,
            max_memory_usage=args.spatial_map_max_memory_mb,
            save_texture=args.spatial_map_texture,
            use_chunk_only=False,
            reverse_vertex_order=False,
            map_type=args.spatial_map_type,
        )
        status = zed.enable_spatial_mapping(spatial_params)
        if status > sl.ERROR_CODE.SUCCESS:
            print(f"[zed-map] failed to enable spatial mapping: {status}", file=sys.stderr)
            zed.disable_positional_tracking()
            zed.close()
            return 1
        spatial_map = _make_spatial_map(args.spatial_map_type)
        spatial_map_snapshot = _make_spatial_map(args.spatial_map_type)
        _safe_print(
            "[zed-map] spatial mapping enabled: "
            f"type={_enum_name(args.spatial_map_type)} "
            f"resolution={_enum_name(args.spatial_map_resolution)} "
            f"range={_enum_name(args.spatial_map_range)} "
            f"max_memory={args.spatial_map_max_memory_mb}MB "
            f"texture={args.spatial_map_texture} "
            f"final={args.spatial_map_file} snapshots={args.spatial_map_snapshot_dir}"
        )

    if args.record_svo:
        args.record_svo.parent.mkdir(parents=True, exist_ok=True)
        rec_status = zed.enable_recording(
            sl.RecordingParameters(args.record_svo.as_posix(), sl.SVO_COMPRESSION_MODE.H265)
        )
        if rec_status > sl.ERROR_CODE.SUCCESS:
            print(f"[zed-map] failed to enable SVO recording: {rec_status}", file=sys.stderr)
            zed.disable_positional_tracking()
            zed.close()
            return 1
        print(f"[zed-map] recording SVO: {args.record_svo}", flush=True)

    tum = None
    if args.tum_file:
        args.tum_file.parent.mkdir(parents=True, exist_ok=True)
        tum = args.tum_file.open("w", encoding="utf-8", buffering=1)
        print(f"[zed-map] writing TUM trajectory: {args.tum_file}", flush=True)

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 30
    pose = sl.Pose()
    frames = 0
    t0 = time.monotonic()
    last_log = 0.0
    last_sm: Optional[str] = None
    last_odom: Optional[str] = None
    consecutive_grab_errors = 0
    failed = False

    try:
        while not _STOP:
            if args.duration > 0 and time.monotonic() - t0 >= args.duration:
                break
            grab_status = zed.grab(runtime)
            if grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print("[zed-map] reached end of SVO", flush=True)
                break
            if grab_status > sl.ERROR_CODE.SUCCESS:
                consecutive_grab_errors += 1
                print(
                    f"[zed-map] grab failed ({consecutive_grab_errors}/"
                    f"{args.max_grab_errors}): {grab_status}",
                    flush=True,
                )
                if consecutive_grab_errors >= args.max_grab_errors:
                    failed = True
                    print(
                        "[zed-map] aborting after repeated grab errors; "
                        "not exporting an area map",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                time.sleep(0.02)
                continue
            consecutive_grab_errors = 0

            track_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
            frames += 1

            q = pose.get_orientation().get()
            tr = pose.get_translation().get()
            ts = pose.timestamp.get_nanoseconds() / 1e9
            if tum is not None and track_state == sl.POSITIONAL_TRACKING_STATE.OK:
                tum.write(
                    f"{ts:.9f} {tr[0]:.9f} {tr[1]:.9f} {tr[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
                )

            now = time.monotonic()
            if spatial_map_snapshot is not None and args.spatial_map_snapshot_dir is not None:
                if next_spatial_snapshot_ts is None:
                    next_spatial_snapshot_ts = ts
                if not spatial_snapshot_pending and ts >= next_spatial_snapshot_ts:
                    zed.request_spatial_map_async()
                    spatial_snapshot_pending = True
                    next_spatial_snapshot_ts = ts + args.spatial_map_snapshot_period
                if spatial_snapshot_pending and zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_spatial_map_async(spatial_map_snapshot)
                    suffix = (
                        args.spatial_map_file.suffix
                        if args.spatial_map_file
                        else _default_spatial_suffix(args.spatial_map_type)
                    )
                    snapshot_path = (
                        args.spatial_map_snapshot_dir
                        / f"spatial_map_{spatial_snapshot_index:04d}{suffix}"
                    )
                    _save_spatial_map(spatial_map_snapshot, snapshot_path)
                    spatial_snapshot_index += 1
                    spatial_snapshot_pending = False

            pts = zed.get_positional_tracking_status()
            sm = _enum_name(pts.spatial_memory_status)
            odom = _enum_name(pts.odometry_status)
            fusion = _enum_name(pts.tracking_fusion_status)
            spatial_state = (
                _enum_name(zed.get_spatial_mapping_state())
                if spatial_mapping_requested
                else "OFF"
            )
            should_log = (
                now - last_log >= args.log_period
                or sm != last_sm
                or odom != last_odom
            )
            if should_log:
                print(
                    f"[zed-map] t={now - t0:7.1f}s frame={frames:6d} "
                    f"ts={ts:.6f} state={_enum_name(track_state)} "
                    f"odom={odom} spatial_memory={sm} spatial_map={spatial_state} fusion={fusion} "
                    f"conf={pose.pose_confidence:3d} "
                    f"xyz=({tr[0]:+.3f},{tr[1]:+.3f},{tr[2]:+.3f}) "
                    f"q=({q[0]:+.5f},{q[1]:+.5f},{q[2]:+.5f},{q[3]:+.5f})",
                    flush=True,
                )
                last_sm, last_odom = sm, odom
                last_log = now
    finally:
        if tum is not None:
            tum.close()
        if args.record_svo:
            zed.disable_recording()

    if failed:
        if spatial_mapping_requested:
            zed.disable_spatial_mapping()
        zed.disable_positional_tracking()
        zed.close()
        return 1

    if spatial_map is not None and args.spatial_map_file is not None:
        _safe_print(f"[zed-map] extracting spatial map for visualization: {args.spatial_map_file}")
        status = zed.extract_whole_spatial_map(spatial_map)
        if status > sl.ERROR_CODE.SUCCESS:
            if spatial_mapping_requested:
                zed.disable_spatial_mapping()
            zed.disable_positional_tracking()
            zed.close()
            raise RuntimeError(f"extract_whole_spatial_map failed: {status}")
        if args.spatial_map_type == sl.SPATIAL_MAP_TYPE.MESH and isinstance(spatial_map, sl.Mesh):
            if args.spatial_map_filter is not None:
                filter_params = sl.MeshFilterParameters()
                filter_params.set(args.spatial_map_filter)
                spatial_map.filter(filter_params, True)
            if args.spatial_map_texture:
                spatial_map.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)
        _save_spatial_map(spatial_map, args.spatial_map_file)

    if args.output_area_file:
        _export_area_map(zed, args.output_area_file)

    if spatial_mapping_requested:
        zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    _safe_print(f"[zed-map] done, frames={frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
