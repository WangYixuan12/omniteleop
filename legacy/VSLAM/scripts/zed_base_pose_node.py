#!/usr/bin/env python3
"""Robot base pose from head-ZED positional tracking + FK — fusion node.

Subscribes to the camera pose published by the head publisher
(``tests/test_head_zedx_depth.py --enable-tracking``) and to the robot joint
states, fuses them through ``omniteleop.follower.zed_base_pose`` and publishes
the base pose in the odom frame (= base frame at the first OK-tracked sample):

  sensors/<id>/pose  (quat+trans, JSON)──┐
  state/torso        (JointStateCodec)   ├─→ T_odom_base ─→ state/base_pose_zed (JSON)
  state/head         (JointStateCodec) ──┘                  + 1 Hz log, optional CSV/video

Pose precision: the camera pose is built from the publisher's
quaternion+translation fields, re-normalized in float64 to an exactly
orthonormal rotation (only float32 quantization ~1e-7 remains). The SDK's raw
4x4 matrix is not used — it is denormalized float32 (det(R)≈1.002 observed),
and the quaternion path was verified equivalent to it on hardware (max
0.00001° / 0.0000 mm over a 180° drive) at 2.4x lower cost. During capture
nothing heavy runs in the loop (per-sample cost ≪ 1 ms at 30 Hz); video
rendering happens strictly after the capture ends and the Zenoh node shuts down.

The fusion keeps short head/torso state histories and pairs each camera pose
with the joint samples nearest to the pose publisher's local capture timestamp
(``timestamp_ns``). It deliberately does not match on ``state/*`` source
timestamps: they were observed ~100 ms ahead of local receive time, so those
driver clocks are not comparable to the camera publisher's clock. If the pose
publisher timestamp is not close to this node's local receive time, the node
falls back to receive-time matching.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import tyro
from dexcomm import Node
from dexcomm.codecs import JointStateCodec, JsonDataCodec
from loguru import logger

sys.path.insert(0, (pathlib.Path(__file__).resolve().parent.parent / "src").as_posix())

from omniteleop.follower.zed_base_pose import (  # noqa: E402
    VegaHeadCamFK,
    ZedBasePoseEstimator,
    pose_from_quat_trans,
    se3_to_xy_yaw,
)


@dataclasses.dataclass
class Args:
    sensor_id: str = "head_camera"
    """Camera whose pose stream to consume: sensors/<sensor_id>/pose."""

    namespace: str = ""
    """Zenoh namespace passed to dexcomm.Node (matches the publisher)."""

    publish_topic: str = "state/base_pose_zed"
    """Topic for the fused base pose (JsonDataCodec)."""

    urdf_path: Optional[str] = None
    """Vega URDF override; default resolves vega_1.urdf from dexmate_urdf."""

    max_state_age_s: float = 0.5
    """Reject fusion when the freshest joint state (local receive time) is older
    than this — prevents silently using stale kinematics."""

    max_state_match_s: float = 0.08
    """Reject fusion when the nearest locally received joint sample is farther
    than this from the camera-pose match time. This prevents fast head motion
    from leaking through a mismatched pose/joint pair."""

    max_pose_clock_skew_s: float = 1.0
    """Use the pose publisher's timestamp for matching only when it is within
    this many seconds of local receive time; otherwise fall back to receive time."""

    min_confidence: int = 0
    """Skip poses below this SDK pose_confidence (0..100). 0 = accept all."""

    duration: float = 0.0
    """If > 0, stop after this many seconds (0 = run forever)."""

    save_csv: Optional[str] = None
    """Append per-sample rows (times, joints, match deltas, x y z rpy, T 16) here."""

    save_video: Optional[str] = None
    """Write an animated base trajectory (.mp4 via ffmpeg or .gif via Pillow)
    on exit: growing XY path + heading arrow and z-vs-time height trace."""

    video_fps: int = 10
    """Playback frame rate of the animation."""

    log_period_s: float = 1.0
    """Wall period between base-pose log lines."""


def _configure_zenoh_like_dexcontrol() -> Optional[pathlib.Path]:
    """Mirror dexcontrol's default ZENOH_CONFIG discovery (see head publisher)."""
    existing = os.getenv("ZENOH_CONFIG")
    if existing:
        return pathlib.Path(existing).expanduser()
    base_dir = pathlib.Path("~/.dexmate/comm/zenoh").expanduser()
    for pattern in ("**/*.dzcfg", "**/zenoh*config*.json5"):
        matches = sorted(p for p in base_dir.glob(pattern) if p.is_file())
        if matches:
            os.environ["ZENOH_CONFIG"] = matches[0].as_posix()
            return matches[0]
    return None


def _pose_msg_to_se3(msg: dict) -> np.ndarray:
    """Build the camera pose T_world_cam (4,4) from one sensors/<id>/pose message.

    Quaternion-only: the SDK's quaternion is re-normalized in float64 to an
    exactly orthonormal rotation (``pose_from_quat_trans``). The SDK's raw 4x4
    matrix is no longer used — it is denormalized float32, and the quaternion
    path was verified equivalent to it (max 0.00001° / 0.0000 mm over a 180°
    drive) and is 2.4x cheaper. A publisher too old to send the quaternion
    fields is a hard error here, by design.
    """
    missing = [
        k for k in ("quaternion_xyzw", "translation", "state", "confidence",
                    "sequence", "timestamp_ns") if k not in msg
    ]
    if missing:
        raise ValueError(
            f"pose message missing keys {missing}; got keys {sorted(msg)}. "
            "If only quaternion_xyzw/translation are missing, the head publisher "
            "predates the quaternion fields — restart it with --enable-tracking."
        )
    return pose_from_quat_trans(msg["quaternion_xyzw"], msg["translation"], what="pose msg")


def _joint_pos(msg: dict, what: str) -> np.ndarray:
    if "pos" not in msg:
        raise ValueError(f"{what} state message has no 'pos' field; keys: {sorted(msg)}")
    pos = np.asarray(msg["pos"], dtype=np.float64)
    if pos.shape != (3,):
        raise ValueError(f"{what} 'pos' must have 3 entries, got shape {pos.shape}")
    return pos


class _MsgHistory:
    """Thread-safe fixed-size message history keyed by local receive time."""

    def __init__(self, maxlen: int = 300) -> None:
        self._items: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, msg: dict) -> None:
        item = dict(msg)
        item.setdefault("receive_time_ns", time.time_ns())
        item["receive_time_ns"] = int(item["receive_time_ns"])
        with self._lock:
            self._items.append(item)

    def nearest(self, target_receive_time_ns: int) -> tuple[Optional[dict], float]:
        with self._lock:
            items = list(self._items)
        if not items:
            return None, math.inf
        msg = min(
            items,
            key=lambda item: abs(int(item["receive_time_ns"]) - target_receive_time_ns),
        )
        dt_s = (int(msg["receive_time_ns"]) - target_receive_time_ns) / 1e9
        return msg, dt_s


def _se3_to_roll_pitch_yaw(T: np.ndarray) -> tuple[float, float, float]:
    R = T[:3, :3]
    pitch = math.atan2(-float(R[2, 0]), math.hypot(float(R[2, 1]), float(R[2, 2])))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return roll, pitch, yaw


def main() -> None:
    args = tyro.cli(Args)
    if args.save_video:
        # fail at startup, not after a long capture
        suffix = pathlib.Path(args.save_video).suffix.lower()
        if suffix not in (".mp4", ".gif"):
            raise ValueError(f"--save-video must end in .mp4 or .gif, got {suffix!r}")
        if args.video_fps <= 0:
            raise ValueError(f"--video-fps ({args.video_fps}) must be > 0")

    zenoh_config = _configure_zenoh_like_dexcontrol()
    logger.info(f"Zenoh config: {zenoh_config if zenoh_config else '<dexcomm default>'}")

    fk = VegaHeadCamFK(urdf_path=args.urdf_path)
    estimator = ZedBasePoseEstimator(fk)
    logger.info(f"FK model: {fk.urdf_path} → frame {fk.camera_frame!r}")

    node = Node(name="zed_base_pose_node", namespace=args.namespace)
    pose_sub = node.create_subscriber(
        f"sensors/{args.sensor_id}/pose", decoder=JsonDataCodec.decode
    )
    head_history = _MsgHistory()
    torso_history = _MsgHistory()
    _head_sub = node.create_subscriber(
        "state/head", callback=head_history.add, decoder=JointStateCodec.decode
    )
    _torso_sub = node.create_subscriber(
        "state/torso", callback=torso_history.add, decoder=JointStateCodec.decode
    )
    base_pub = node.create_publisher(args.publish_topic, encoder=JsonDataCodec.encode)
    logger.info(
        f"sensors/{args.sensor_id}/pose + state/{{head,torso}} → {args.publish_topic!r}"
    )

    csv_file = None
    if args.save_csv:
        csv_path = pathlib.Path(args.save_csv).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w")
        csv_file.write(
            "t_wall_s,seq,state,confidence,x,y,z,roll,pitch,yaw,"
            "pose_match_time_ns,pose_match_time_source,pose_receive_time_ns,"
            "head_receive_time_ns,torso_receive_time_ns,"
            "head_match_dt_ms,torso_match_dt_ms,"
            "head_j1,head_j2,head_j3,torso_j1,torso_j2,torso_j3,"
            + ",".join(f"T{i}{j}" for i in range(4) for j in range(4))
            + "\n"
        )

    history: list[tuple[float, float, float, float, float]] = []  # t, x, y, z, yaw

    last_seq = None
    fused = skipped_state = skipped_stale = skipped_conf = 0
    t_start = time.perf_counter()
    t_last_log = 0.0
    last_state_logged = None
    last_pose_source_meta = None

    try:
        while True:
            now = time.perf_counter()
            if args.duration > 0 and now - t_start >= args.duration:
                break

            msg = pose_sub.get_latest()
            if msg is None or msg.get("sequence") == last_seq:
                time.sleep(0.005)
                continue
            T_world_cam = _pose_msg_to_se3(msg)
            last_seq = msg["sequence"]

            pose_source_meta = (
                msg.get("coordinate_system"),
                msg.get("area_file_path"),
                bool(msg.get("localization_only", False)),
                bool(msg.get("area_memory", False)),
            )
            if pose_source_meta != last_pose_source_meta:
                logger.info(
                    "Pose source: "
                    f"coordinate_system={pose_source_meta[0]} "
                    f"area_file={pose_source_meta[1]} "
                    f"localization_only={pose_source_meta[2]} "
                    f"area_memory={pose_source_meta[3]}"
                )
                last_pose_source_meta = pose_source_meta

            if msg["state"] != last_state_logged:
                logger.info(
                    f"Tracking state (publisher): {msg['state']} "
                    f"confidence={msg['confidence']}"
                )
                last_state_logged = msg["state"]

            if msg["state"] != "OK":
                skipped_state += 1
                continue
            if int(msg["confidence"]) < args.min_confidence:
                skipped_conf += 1
                continue

            pose_receive_time_ns = int(msg.get("receive_time_ns", time.time_ns()))
            pose_timestamp_ns = int(msg.get("timestamp_ns", pose_receive_time_ns))
            pose_clock_skew_s = abs(pose_receive_time_ns - pose_timestamp_ns) / 1e9
            if pose_clock_skew_s <= args.max_pose_clock_skew_s:
                pose_match_time_ns = pose_timestamp_ns
                pose_match_time_source = "timestamp_ns"
            else:
                pose_match_time_ns = pose_receive_time_ns
                pose_match_time_source = "receive_time_ns"
            head_msg, head_match_dt_s = head_history.nearest(pose_match_time_ns)
            torso_msg, torso_match_dt_s = torso_history.nearest(pose_match_time_ns)
            if head_msg is None or torso_msg is None:
                skipped_stale += 1
                if (skipped_stale % 30) == 1:
                    logger.warning(
                        f"No joint states yet (head={head_msg is not None}, "
                        f"torso={torso_msg is not None}) — is the robot stack up?"
                    )
                continue
            now_ns = time.time_ns()
            age_head_s = (now_ns - int(head_msg["receive_time_ns"])) / 1e9
            age_torso_s = (now_ns - int(torso_msg["receive_time_ns"])) / 1e9
            if max(age_head_s, age_torso_s) > args.max_state_age_s:
                skipped_stale += 1
                if (skipped_stale % 30) == 1:
                    logger.warning(
                        f"Stale joint states (head {age_head_s:.2f}s, torso "
                        f"{age_torso_s:.2f}s > {args.max_state_age_s}s) — skipping"
                    )
                continue
            if max(abs(head_match_dt_s), abs(torso_match_dt_s)) > args.max_state_match_s:
                skipped_stale += 1
                if (skipped_stale % 30) == 1:
                    logger.warning(
                        f"No close joint sample for pose seq={msg['sequence']} "
                        f"(head {head_match_dt_s * 1000:+.1f} ms, torso "
                        f"{torso_match_dt_s * 1000:+.1f} ms > "
                        f"{args.max_state_match_s * 1000:.0f} ms, "
                        f"match={pose_match_time_source}) — skipping"
                    )
                continue

            head_pos = _joint_pos(head_msg, "head")
            torso_pos = _joint_pos(torso_msg, "torso")

            first = not estimator.anchored
            T_odom_base = estimator.update(T_world_cam, torso_pos, head_pos)
            fused += 1
            if first:
                logger.info(
                    "Anchored odom frame := base @ first OK sample "
                    f"(seq={last_seq}, confidence={msg['confidence']})"
                )

            x, y, yaw = se3_to_xy_yaw(T_odom_base)
            # Raw camera path in the SAME odom frame, FK NOT applied (anchor is set
            # by update() above, so odom_T_track is never None here). Overlays the
            # base curve; their gap is exactly the head->base FK lever arm.
            cam_x, cam_y, cam_yaw = se3_to_xy_yaw(estimator.odom_T_track @ T_world_cam)
            roll, pitch, _yaw_rpy = _se3_to_roll_pitch_yaw(T_odom_base)
            z = float(T_odom_base[2, 3])
            t_wall = now - t_start

            base_pub.publish(
                {
                    "T_odom_base": T_odom_base.reshape(-1).tolist(),
                    "x": x,
                    "y": y,
                    "z": z,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "cam_x": cam_x,
                    "cam_y": cam_y,
                    "cam_yaw": cam_yaw,
                    "state": msg["state"],
                    "confidence": int(msg["confidence"]),
                    "timestamp_ns": int(msg["timestamp_ns"]),
                    "pose_match_time_ns": pose_match_time_ns,
                    "pose_match_time_source": pose_match_time_source,
                    "pose_receive_time_ns": pose_receive_time_ns,
                    "head_receive_time_ns": int(head_msg["receive_time_ns"]),
                    "torso_receive_time_ns": int(torso_msg["receive_time_ns"]),
                    "head_match_dt_ms": head_match_dt_s * 1000.0,
                    "torso_match_dt_ms": torso_match_dt_s * 1000.0,
                    "head_pos": head_pos.tolist(),
                    "torso_pos": torso_pos.tolist(),
                    "zed_timestamp_ns": int(msg.get("zed_timestamp_ns", 0)),
                    "raw_pose_coordinate_system": msg.get("coordinate_system"),
                    "area_file_path": msg.get("area_file_path"),
                    "localization_only": bool(msg.get("localization_only", False)),
                    "area_memory": bool(msg.get("area_memory", False)),
                    "sequence": int(last_seq),
                }
            )
            if csv_file is not None:
                csv_file.write(
                    f"{t_wall:.4f},{last_seq},{msg['state']},{msg['confidence']},"
                    f"{x:.6f},{y:.6f},{z:.6f},{roll:.6f},{pitch:.6f},{yaw:.6f},"
                    f"{pose_match_time_ns},"
                    f"{pose_match_time_source},"
                    f"{pose_receive_time_ns},"
                    f"{int(head_msg['receive_time_ns'])},"
                    f"{int(torso_msg['receive_time_ns'])},"
                    f"{head_match_dt_s * 1000.0:.3f},"
                    f"{torso_match_dt_s * 1000.0:.3f},"
                    + ",".join(f"{v:.9f}" for v in head_pos)
                    + ","
                    + ",".join(f"{v:.9f}" for v in torso_pos)
                    + ","
                    + ",".join(f"{v:.9f}" for v in T_odom_base.reshape(-1))
                    + "\n"
                )
            if args.save_video:
                history.append((t_wall, x, y, z, yaw))

            if now - t_last_log >= args.log_period_s:
                logger.info(
                    f"base in odom: x={x:+.3f} y={y:+.3f} z={z:+.3f} m "
                    f"pitch={math.degrees(pitch):+.1f}° "
                    f"yaw={math.degrees(yaw):+.1f}° "
                    f"match={pose_match_time_source} "
                    f"joint dt H/T={head_match_dt_s * 1000:+.0f}/"
                    f"{torso_match_dt_s * 1000:+.0f} ms "
                    f"(conf={msg['confidence']}, "
                    f"fused={fused}, skipped state/conf/stale="
                    f"{skipped_state}/{skipped_conf}/{skipped_stale})"
                )
                t_last_log = now

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        # capture is over: persist the CSV and drop off the network FIRST, so
        # the (potentially slow) matplotlib/ffmpeg rendering below runs with no
        # subscriber threads or robot-facing state left.
        if csv_file is not None:
            csv_file.close()
            logger.info(f"CSV written: {args.save_csv} ({fused} rows)")
        node.shutdown()
        logger.info(
            f"Done: fused={fused}, skipped non-OK={skipped_state}, "
            f"low-confidence={skipped_conf}, stale/missing joints={skipped_stale}"
        )
        if args.save_video:
            if len(history) >= 2:
                _write_video(args.save_video, history, args.video_fps)
            else:
                logger.warning(
                    f"--save-video skipped: only {len(history)} fused samples"
                )


def _pick_mp4_codec() -> str:
    """libx264 when the ffmpeg build has it; else software mpeg4 (the Jetson's
    /usr/local ffmpeg ships no libx264 and its h264_v4l2m2m wrapper fails to
    configure, so matplotlib's default 'h264' dies there)."""
    import subprocess

    from matplotlib import rcParams

    out = subprocess.run(
        [rcParams["animation.ffmpeg_path"], "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return "libx264" if " libx264 " in out else "mpeg4"


def _write_video(
    path: str,
    history: list[tuple[float, float, float, float, float]],
    fps: int,
) -> None:
    """Animate the base trajectory: XY path + heading arrow and z over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    arr = np.asarray(history, dtype=np.float64)
    t, x, y, z, yaw = arr.T

    n_frames = int(np.ceil((t[-1] - t[0]) * fps)) + 1
    if n_frames > 3000:
        raise ValueError(
            f"{n_frames} frames for {t[-1] - t[0]:.0f}s at fps={fps}, "
            "lower --video-fps (cap: 3000 frames)"
        )

    out = pathlib.Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp4":
        if not animation.FFMpegWriter.isAvailable():
            raise RuntimeError("ffmpeg not found — use a .gif output or install ffmpeg")
        writer = animation.FFMpegWriter(fps=fps, codec=_pick_mp4_codec(), bitrate=2000)
    elif out.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError(f"video path must end in .mp4 or .gif, got {out.suffix!r}")

    # fixed, equal-aspect limits over the whole run; floor the span so a parked
    # robot doesn't get a microscopic axis range
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = max((x.max() - x.min()) / 2, (y.max() - y.min()) / 2) * 1.15 + 1e-9
    half = max(half, 0.25)
    arrow_len = 0.18 * half
    z_min, z_max = float(z.min()), float(z.max())
    z_pad = max((z_max - z_min) * 0.15, 0.02)
    t_right = float(t[-1]) if t[-1] > t[0] else float(t[0] + 1.0 / fps)

    fig, (ax_xy, ax_z) = plt.subplots(
        1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )
    ax_xy.set_xlim(cx - half, cx + half)
    ax_xy.set_ylim(cy - half, cy + half)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("base XY in odom")
    ax_xy.plot(x, y, color="0.85", lw=0.8, zorder=1)  # full path for context
    xy_trail, = ax_xy.plot([], [], color="tab:blue", lw=1.6, zorder=2)
    ax_xy.scatter([x[0]], [y[0]], c="g", s=40, label="start", zorder=3)
    xy_dot = ax_xy.scatter([], [], c="r", s=50, zorder=4, label="base")
    arrow = ax_xy.quiver(
        [x[0]], [y[0]], [arrow_len * np.cos(yaw[0])], [arrow_len * np.sin(yaw[0])],
        angles="xy", scale_units="xy", scale=1.0, color="r", width=0.008, zorder=5,
    )
    ax_xy.legend(loc="upper right")

    ax_z.set_xlim(float(t[0]), t_right)
    ax_z.set_ylim(z_min - z_pad, z_max + z_pad)
    ax_z.grid(True, alpha=0.3)
    ax_z.set_xlabel("t [s]")
    ax_z.set_ylabel("z [m]")
    ax_z.set_title("base z in odom")
    ax_z.plot(t, z, color="0.85", lw=0.8, zorder=1)
    z_trail, = ax_z.plot([], [], color="tab:orange", lw=1.6, zorder=2)
    z_dot = ax_z.scatter([], [], c="r", s=45, zorder=3)
    z_cursor = ax_z.axvline(float(t[0]), color="0.35", lw=0.8, alpha=0.6, zorder=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    frame_times = t[0] + np.arange(n_frames) / fps
    logger.info(
        f"Rendering {n_frames} frames ({n_frames / fps:.1f}s video of "
        f"{t[-1] - t[0]:.1f}s trajectory) → {out}"
    )
    with writer.saving(fig, out.as_posix(), dpi=110):
        for f, tf in enumerate(frame_times):
            i = int(np.searchsorted(t, tf, side="right") - 1)
            i = max(i, 0)
            xy_trail.set_data(x[: i + 1], y[: i + 1])
            xy_dot.set_offsets([[x[i], y[i]]])
            arrow.set_offsets([[x[i], y[i]]])
            arrow.set_UVC([arrow_len * np.cos(yaw[i])], [arrow_len * np.sin(yaw[i])])
            z_trail.set_data(t[: i + 1], z[: i + 1])
            z_dot.set_offsets([[t[i], z[i]]])
            z_cursor.set_xdata([t[i], t[i]])
            fig.suptitle(
                f"base pose in odom   t={t[i]:6.2f}s   "
                f"x={x[i]:+.3f} y={y[i]:+.3f} z={z[i]:+.3f} m   "
                f"dz={z[i] - z[0]:+.3f} m   yaw={np.degrees(yaw[i]):+.1f}deg"
            )
            writer.grab_frame()
            if n_frames > 100 and f % (n_frames // 5) == 0 and f > 0:
                logger.info(f"  …{100 * f // n_frames}%")
    plt.close(fig)
    logger.info(f"Video written: {out}")


if __name__ == "__main__":
    main()
