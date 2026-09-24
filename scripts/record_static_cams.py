#!/usr/bin/env python3
"""Free-running recorder for the two STATIC RealSense D455s, to run alongside a
``wbc_vr_robot.py --record`` teleop take.

Why this exists
---------------
``calib/packing_data/scripts/record_pw_episode.py`` is a *burst* recorder: it
buffers a fixed ``--frames`` count in RAM and dumps once. That is right for a
30-frame calibration still, and unusable for a minutes-long demo. This script
streams to disk instead, at a fixed rate, for as long as you let it run, and
writes the SAME HDF5 layout -- so ``realsense_depth_to_fs_disp.py``,
``processing.fs_stereo.run_fs`` and ``processing.pack.pack_dataset`` consume the
result with no changes.

The teleop/static split across two machines
-------------------------------------------
The D455s hang off the workstation that has them on USB; ``wbc_vr_robot.py`` runs
on the robot workstation. Those are different hosts, so the two recordings only
line up if their clocks do.

Both ends of that are handled here:

  * librealsense ``global_time`` is REQUIRED (the script refuses to start
    without it). With it, ``frame.get_timestamp()`` is milliseconds on THIS
    host's epoch clock, already corrected for USB transport delay -- not a
    device-uptime counter. That lands in ``capture_meta/camera_<c>_host_ns``.
  * ``--teleop-host`` runs an NTP-style offset probe against the teleop host
    over ssh, before and after the take, and stores both as ROOT attrs
    ``teleop_clock_{pre,post}_{offset,rtt,measured_at}_ns`` (file-level, NOT on
    ``capture_meta``). Two samples, so clock DRIFT across a long session is
    visible rather than assumed away.

Offline, a static frame and a teleop frame are on a common clock via::

    static_capture_on_teleop_clock = camera_<c>_host_ns + teleop_minus_local_ns
    head_capture_on_teleop_clock   = obs/images/head_frame_ns
                                     - meta/camera_ntp/head/offset_ns

then nearest-neighbour match -- implemented by ``scripts/align_static_teleop.py``,
which interpolates ``teleop_minus_local_ns`` between the pre/post probes.

The quantisation floor is half the RECORD period, not the sensor period: at the
15 Hz default that is +-33.3 ms, and at 30 Hz +-16.7 ms. It dominates everything
else -- an NTP-synced host pair sits 2-5 ms apart, and measured drift across a
16-minute take was 1.5 ms.

Extrinsic seed
--------------
``--seed-from`` copies ``camera_<c>_extrinsics`` VERBATIM from a reference
episode (normally the calibration take whose ``config.json`` you solved), so
every episode in a session carries the SAME seed and one solved calibration
delta applies to all of them. Re-fitting the table plane per episode is what
made the earlier calibration drift between takes; this script will not do it.
``--seed identity`` is the only alternative, and then the seed is honestly the
identity rather than a plausible-looking guess.

Prefer ``--seed-from``. Episodes from the burst recorder DO carry a usable
``observations/images/camera_<c>_extrinsics`` (``extrinsic_init=table``), so they
are valid references. ``--seed identity`` makes camera_0's OPTICAL frame the
world, which leaves the table tilted and the bbox non-axis-aligned, and it means
a ``config.json`` delta solved against a table seed does NOT carry over -- the
delta is stored relative to the seed. Mixing the two seed conventions across a
calibration take and its demo take is the failure this flag invites.

Usage
-----
Run ON the host the cameras are plugged into, writing to LOCAL disk::

    python scripts/record_static_cams.py \
        --out-dir /home/yifan/static_takes/packing \
        --seed-from .../recorded/calib_static/episode_1.hdf5 \
        --teleop-host lambda --rate 15

Ctrl-C ends the take and finalises the file. Needs ``pyrealsense2`` + ``h5py``
(here: the ``calibration`` conda env), NOT the teleop env.

Session mode (one launch per data-collection session)
------------------------------------------------------
``--follow-teleop-dir`` keeps the cameras open for the whole session and cuts one
static file per teleop episode, so the recorder is NOT relaunched between takes::

    python scripts/record_static_cams.py \
        --out-dir /data/Dexmate/static \
        --follow-teleop-dir /data/Dexmate/data \
        --seed-from .../calib_static/episode_1.hdf5 \
        --teleop-host local

It watches the teleop follower's ``--save-dir`` (the follower must run with
``--streaming-recorder``, which creates ``episode_<N>.hdf5.partial`` at frame zero
and renames it to ``episode_<N>.hdf5`` on a clean stop):

  * a fresh ``episode_<N>.hdf5.partial`` opens ``<out-dir>/episode_<N>.hdf5.partial``
    -- SAME id as the teleop take, so the two files pair by name;
  * ``episode_<N>.hdf5`` appearing finalises it as ``episode_<N>.hdf5``
    (``teleop_disposition = complete``);
  * a ``.partial`` whose mtime stops advancing for ``--stall-s`` is a
    quarantined/aborted teleop take: the static file STAYS ``.hdf5.partial``
    (``aborted``), mirroring the teleop convention that a ``.partial`` is not
    training input;
  * a ``.partial`` that vanishes without a final file is a zero-row discard on the
    teleop side; the static file is deleted (nothing to pair).

Frames grabbed between takes are dropped. ``--teleop-host local`` is for the
same-host layout (teleop follower and D455s on one machine): the clock offset is
recorded as exactly 0 so ``align_static_teleop.py`` works unchanged.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import deque
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise SystemExit(
        f"{exc}\nrun this with a Python that has pyrealsense2, e.g.\n"
        "  /home/yifan/miniconda3/envs/calibration/bin/python"
    )

# Native D455 848x480 for both IR and colour. Depth+IR must share a frame, so they
# stay the same size; colour matches so the stored streams share one geometry.
# This is NOT the 1280x720 / 1280x800 layout of record_pw_episode.py.
from omniteleop.common import realsense_capture_meta as rsmeta
from omniteleop.common.realsense_frameset_guard import FramesetGuard

IR_W, IR_H = 848, 480
COLOR_W, COLOR_H = 848, 480
FPS = 30

SETTLE_S = 4.0          # D455 auto-exposure / auto-white-balance convergence
MAX_SATURATED = 0.20    # refuse a blown-out colour channel rather than record it
QUEUE_DEPTH = 8         # per-camera grabber -> writer backlog before we abort
POLL_SLEEP_S = 0.002    # grabber idle sleep between poll_for_frames()

# Scaled from the 1280x720 IR / 1280x800 colour lzf benchmark (colour 1.93 MB,
# left_ir 0.68, right_ir 0.65, depth 0.54) by pixel count at 848x480.
_MB_PER_FRAME = {"color": 0.77, "left_ir": 0.30, "right_ir": 0.29, "depth": 0.24}
# The ASIC depth is NOT worth its 16% of the file when FoundationStereo runs on the
# IR pair anyway. Measured on 08-20 episode 0, same three frames, same calibration:
# table depth noise 2.24 mm rms (ASIC) vs 1.20 mm (Fast-FoundationStereo), 318k vs
# 493k surviving workspace points, and cross-camera nearest-neighbour distance
# 95.5 mm vs 65.4 mm. Hence "fs" is the default; "full" only adds a stream that is
# strictly worse than what is derived from the two already being stored.
_STREAM_SETS = {
    "fs": ("left_ir", "right_ir", "color"),             # FoundationStereo path (default)
    "full": ("left_ir", "right_ir", "color", "depth"),  # + ASIC depth, for comparison
    "rs": ("color", "depth"),                           # RealSense-depth path only
}


# --------------------------------------------------------------------------- #
# clock probe
# --------------------------------------------------------------------------- #
def probe_clock_offset(host: str, samples: int = 7) -> dict:
    """NTP-style (remote - local) clock offset in ns, measured over ssh.

    One remote timestamp per round trip, so the estimate assumes a symmetric
    path; the reported rtt is what tells you how much to trust it. The first
    call pays ssh connection setup, so we take the LOWEST-rtt half like
    ``wbc_vr_robot._camera_clock_calibration_from_samples`` does.

    ``host == "local"`` means the teleop follower runs on THIS machine: the offset
    is exactly 0 by construction (same clock) and no ssh is attempted.
    """
    if host == "local":
        return {"host": "local", "offset_ns": 0, "rtt_ns": 0, "samples": 0,
                "measured_at_ns": time.time_ns()}
    ctl = f"/tmp/.rs_clock_probe_{os.getpid()}"
    base = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "-o", "ControlMaster=auto", "-o", f"ControlPath={ctl}",
        "-o", "ControlPersist=30", host, "date +%s.%N",
    ]
    pairs: list[tuple[int, int]] = []
    last_err = ""
    for _ in range(samples):
        t0 = time.time_ns()
        try:
            out = subprocess.run(base, capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            last_err = "ssh timed out"
            continue
        t3 = time.time_ns()
        if out.returncode != 0:
            last_err = (out.stderr or "ssh failed").strip().splitlines()[-1][:200]
            continue
        try:
            t1 = int(round(float(out.stdout.strip()) * 1e9))
        except ValueError:
            last_err = f"unparseable remote time {out.stdout.strip()!r}"
            continue
        pairs.append((t1 - (t0 + t3) // 2, t3 - t0))
    subprocess.run(["ssh", "-o", f"ControlPath={ctl}", "-O", "exit", host],
                   capture_output=True)
    if not pairs:
        raise RuntimeError(f"clock probe against {host!r} produced no samples: {last_err}")
    pairs.sort(key=lambda p: p[1])
    keep = pairs[: max(1, len(pairs) // 2)]
    return {
        "host": host,
        "offset_ns": int(round(float(np.mean([p[0] for p in keep])))),
        "rtt_ns": int(round(float(np.mean([p[1] for p in keep])))),
        "samples": len(pairs),
        "measured_at_ns": time.time_ns(),
    }


# --------------------------------------------------------------------------- #
# colour control
# --------------------------------------------------------------------------- #
_COLOR_OPTS = {"exposure": rs.option.exposure, "gain": rs.option.gain,
               "white_balance": rs.option.white_balance}


LOCK_GAIN = 16.0          # --exposure auto-lock: fixed colour gain (low = less noise)
LOCK_TARGET = 110.0       # --exposure auto-lock: target frame mean, 0..255
LOCK_ITERS = 6
LOCK_SETTLE_FRAMES = 8    # frames for a new exposure to take effect on the D455


def lock_exposure(pipe, color_sensor, rs_id, target, gain, max_exposure):
    """Software auto-exposure that CONVERGES ONCE and then stays fixed.

    The D455 auto-exposure cannot be read back on this host (no UVC frame
    metadata), so instead: fix the gain, then iterate the manual exposure with
    e <- e * target / mean until the colour frame mean sits at ``target``. Returns
    the final exposure. Raises if the scene cannot be brought to target within
    the exposure range without clipping.
    """
    rng = color_sensor.get_option_range(rs.option.exposure)
    lo, hi = max(rng.min, 1.0), min(rng.max, max_exposure)
    color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
    color_sensor.set_option(rs.option.gain, float(gain))
    exp = float(np.clip(100.0, lo, hi))
    mean = None
    for _ in range(LOCK_ITERS):
        color_sensor.set_option(rs.option.exposure, exp)
        for _ in range(LOCK_SETTLE_FRAMES):
            fr = pipe.wait_for_frames().get_color_frame()
        img = np.asanyarray(fr.get_data())
        mean = float(img.mean())
        if mean < 1.0:
            raise ValueError(f"{rs_id}: colour frame is black at exposure {exp:g}")
        if abs(mean - target) / target < 0.03:
            break
        exp = float(np.clip(exp * target / mean, lo, hi))
    sat = float((img >= 250).mean())
    if abs(mean - target) / target > 0.15:
        raise ValueError(
            f"{rs_id}: auto-lock could not reach mean {target:g} (got {mean:.0f} at "
            f"exposure {exp:g}, range [{lo:g}, {hi:g}]) -- scene too dark or too bright "
            f"for gain {gain:g}")
    if sat > MAX_SATURATED:
        raise ValueError(f"{rs_id}: auto-lock exposure {exp:g} leaves {sat * 100:.1f}% saturated")
    return exp, mean


def parse_color_control(exposure, gain, white_balance, n_cams=2):
    """--exposure/--gain/--white-balance -> {name: None | "auto-lock" | [v_cam0, v_cam1]}.

    Each flag is ``auto``, one value applied to every camera, or a comma pair in
    camera_0,camera_1 order. ``--exposure auto-lock`` converges a software
    auto-exposure once at start-up and then freezes it (per camera), at a fixed
    gain (``--gain`` value, default LOCK_GAIN). Manual exposure REQUIRES manual
    gain: leaving gain to whatever the device last held is the silent
    inheritance this script otherwise guards against.
    """
    out = {}
    for name, raw in (("exposure", exposure), ("gain", gain), ("white_balance", white_balance)):
        if raw is None or str(raw).strip().lower() == "auto":
            out[name] = None
            continue
        if name == "exposure" and str(raw).strip().lower() == "auto-lock":
            out[name] = "auto-lock"
            continue
        parts = [p.strip() for p in str(raw).split(",")]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            raise ValueError(f"--{name.replace('_', '-')}: expected 'auto', V or V0,V1; got {raw!r}")
        if len(vals) == 1:
            vals = vals * n_cams
        if len(vals) != n_cams:
            raise ValueError(f"--{name.replace('_', '-')}: need 1 or {n_cams} values, got {len(vals)}")
        out[name] = vals
    if out["exposure"] == "auto-lock":
        if out["gain"] is None:
            out["gain"] = [LOCK_GAIN] * n_cams
    elif (out["exposure"] is None) != (out["gain"] is None):
        raise ValueError("--exposure and --gain must both be manual or both be auto")
    return out


def apply_color_control(color_sensor, rs_id, cam_idx, ctl, pipe=None, max_exposure=None):
    """Set auto / auto-lock / manual exposure-gain-white-balance and read back.

    Options persist on the device across processes, so auto is set EXPLICITLY
    when requested rather than assumed. Returns the effective settings.
    """
    lock = ctl["exposure"] == "auto-lock"
    manual_exp = ctl["exposure"] is not None
    manual_wb = ctl["white_balance"] is not None
    color_sensor.set_option(rs.option.enable_auto_exposure, 0.0 if manual_exp else 1.0)
    color_sensor.set_option(rs.option.enable_auto_white_balance, 0.0 if manual_wb else 1.0)
    eff = {"auto_exposure": not manual_exp, "auto_white_balance": not manual_wb,
           "exposure_mode": "auto-lock" if lock else ("manual" if manual_exp else "auto")}
    if lock:
        if pipe is None or max_exposure is None:
            raise ValueError("auto-lock needs the pipeline and the exposure ceiling")
        gain = float(ctl["gain"][cam_idx])
        exp, mean = lock_exposure(pipe, color_sensor, rs_id, LOCK_TARGET, gain, max_exposure)
        eff.update(exposure=exp, gain=gain, lock_mean=mean)
    for name, opt in _COLOR_OPTS.items():
        if ctl[name] is None or (lock and name in ("exposure", "gain")):
            continue
        want = float(ctl[name][cam_idx])
        rng = color_sensor.get_option_range(opt)
        if not rng.min <= want <= rng.max:
            raise ValueError(f"{rs_id}: {name} {want:g} outside [{rng.min:g}, {rng.max:g}]")
        color_sensor.set_option(opt, want)
        got = float(color_sensor.get_option(opt))
        if abs(got - want) > max(rng.step, 1e-6):
            raise ValueError(f"{rs_id}: {name} set to {want:g} but device reports {got:g}")
        eff[name] = got
    return eff


def describe_color_control(eff):
    if eff["auto_exposure"] and eff["auto_white_balance"]:
        return "auto exposure, auto white balance"
    parts = []
    if eff["auto_exposure"]:
        parts.append("auto exposure")
    elif eff.get("exposure_mode") == "auto-lock":
        parts.append(f"locked exposure {eff['exposure']:g} gain {eff['gain']:g} "
                     f"(mean {eff['lock_mean']:.0f})")
    else:
        parts.append(f"exposure {eff['exposure']:g} gain {eff['gain']:g}")
    parts.append("auto WB" if eff["auto_white_balance"] else f"WB {eff['white_balance']:g}K")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# grabber
# --------------------------------------------------------------------------- #
def _grabber(rs_id, cam_idx, args, streams, barrier, info_q, frame_q, stop_evt, t0_box):
    """One process per camera: open it, decimate to --rate, push frames to the writer.

    Emits exactly one record per schedule tick, always carrying the freshest
    frameset held at that instant (poll, don't wait -- ``wait_for_frames`` would
    hand us a frame up to 33 ms stale and blow the head-camera sync budget).
    A tick without a new complete frameset waits; missed ticks and SDK frame
    numbers remain observable in the metadata.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # parent owns Ctrl-C; we stop on stop_evt
    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(rs_id)
        # Every stream is ENABLED regardless of --streams: K_ir, the distortion
        # coefficients and the stereo baseline are read off the IR stream profiles,
        # and the four-stream configuration is the one verified to start together on
        # both units. --streams selects what gets STORED, which is where the disk
        # cost actually is.
        cfg.enable_stream(rs.stream.infrared, 1, IR_W, IR_H, rs.format.y8, FPS)
        cfg.enable_stream(rs.stream.infrared, 2, IR_W, IR_H, rs.format.y8, FPS)
        cfg.enable_stream(rs.stream.depth, IR_W, IR_H, rs.format.z16, FPS)
        # RGB8, not BGR8: the packing pipeline treats camera_<c>_color as RGB.
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, FPS)

        profile = pipe.start(cfg)
        dev = profile.get_device()

        depth_sensor = dev.first_depth_sensor()
        if not depth_sensor.supports(rs.option.emitter_enabled):
            raise ValueError(f"{rs_id}: device does not expose emitter_enabled")
        depth_sensor.set_option(rs.option.emitter_enabled, float(args.emitter))
        if depth_sensor.supports(rs.option.laser_power):
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            depth_sensor.set_option(rs.option.laser_power,
                                    rng.max if args.emitter else rng.min)
        # Scan/manual tools leave device options behind; explicitly restore IR AE.
        rsmeta.set_option_checked(depth_sensor, rs.option.enable_auto_exposure, 1)
        depth_scale = float(depth_sensor.get_depth_scale())

        # global_time is what makes get_timestamp() a HOST epoch clock instead of
        # a device counter. Without it nothing here can be matched to the teleop
        # take, so this is fatal rather than a warning.
        for sensor in dev.query_sensors():
            if not sensor.supports(rs.option.global_time_enabled):
                continue
            sensor.set_option(rs.option.global_time_enabled, 1.0)
            if sensor.get_option(rs.option.global_time_enabled) != 1.0:
                raise ValueError(
                    f"{rs_id}: global_time_enabled would not stay on for "
                    f"{sensor.get_info(rs.camera_info.name)!r}; frame timestamps "
                    f"would be device-uptime and could not be synced to teleop")

        # Sensor options persist on the device across processes: a manual exposure
        # left behind by any earlier tool would be inherited silently.
        color_sensor = dev.first_color_sensor()
        color_ctl = apply_color_control(color_sensor, rs_id, cam_idx, args.color_control,
                                        pipe=pipe, max_exposure=1e6 / (100.0 * FPS))

        col = profile.get_stream(rs.stream.color).as_video_stream_profile()
        serial_profile = getattr(args, "camera_profiles", {}).get(rs_id, {})
        rsmeta.apply_serial_profile(dev, rs, serial_profile)
        if serial_profile:
            color_ctl["serial_profile_json"] = json.dumps(serial_profile, sort_keys=True)
            if "color_exposure_option" in serial_profile:
                color_ctl.update(auto_exposure=False, exposure_mode="manual",
                                 exposure=serial_profile["color_exposure_option"], gain=serial_profile["color_gain"])
            if "white_balance" in serial_profile:
                color_ctl.update(auto_white_balance=False, white_balance=serial_profile["white_balance"])

        ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        k_ir, k_rgb = ir1.get_intrinsics(), col.get_intrinsics()
        if (k_ir.width, k_ir.height) != (IR_W, IR_H):
            raise ValueError(f"{rs_id}: IR negotiated {k_ir.width}x{k_ir.height}")
        if (k_rgb.width, k_rgb.height) != (COLOR_W, COLOR_H):
            raise ValueError(f"{rs_id}: colour negotiated {k_rgb.width}x{k_rgb.height}")

        K_ir = np.array([[k_ir.fx, 0, k_ir.ppx],
                         [0, k_ir.fy, k_ir.ppy], [0, 0, 1]], np.float32)
        K_rgb = np.array([[k_rgb.fx, 0, k_rgb.ppx],
                          [0, k_rgb.fy, k_rgb.ppy], [0, 0, 1]], np.float32)

        baseline = float(np.linalg.norm(
            np.asarray(ir1.get_extrinsics_to(ir2).translation, float)))
        if not 0.05 < baseline < 0.15:
            raise ValueError(f"{rs_id}: implausible stereo baseline {baseline} m")

        ex = col.get_extrinsics_to(ir1)  # librealsense stores rotation column-major
        T_rgb_to_ir = np.eye(4, dtype=np.float32)
        T_rgb_to_ir[:3, :3] = np.asarray(ex.rotation, float).reshape(3, 3).T
        T_rgb_to_ir[:3, 3] = np.asarray(ex.translation, float)

        deadline = time.perf_counter() + SETTLE_S
        probe = None
        while time.perf_counter() < deadline:
            probe = pipe.wait_for_frames()
        sample = np.asanyarray(probe.get_color_frame().get_data())
        quality_roi = tuple(getattr(args, "quality_roi_values", (0., 0., 1., 1.)))
        saturated = float(rsmeta.color_statistics(sample, quality_roi)[6])
        if saturated > MAX_SATURATED:
            raise ValueError(
                f"{rs_id}: colour is {saturated * 100:.1f}% saturated after settling "
                f"-- exposure is wrong (check scene lighting, and that no other tool "
                f"left manual exposure on the device)")

        info_q.put({
            "rs_id": rs_id, "photometric_telemetry": True, "quality_roi": quality_roi, "frameset_guard": True,
            "sensor_settings_json": rsmeta.sensor_settings(dev, rs),
            "ir_distortion_model": str(k_ir.model),
            "rgb_distortion_model": str(k_rgb.model), "cam": cam_idx, "ok": True,
            "K_ir": K_ir, "K_rgb": K_rgb, "T_rgb_to_ir": T_rgb_to_ir,
            "baseline": baseline, "depth_scale": depth_scale,
            "ir_coeffs": list(k_ir.coeffs), "rgb_coeffs": list(k_rgb.coeffs),
            "saturated": saturated, "color_control": color_ctl,
        })
    except Exception as exc:
        info_q.put({"rs_id": rs_id, "cam": cam_idx, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"})
        return

    # The parent is the extra barrier party: it sets t0_box before joining, so the
    # shared tick schedule is guaranteed published by the time we read it here.
    # It aborts the barrier if the OTHER camera failed to open, which lands us in
    # BrokenBarrierError -- release the device instead of hanging on a peer that
    # is never going to arrive.
    try:
        barrier.wait()
    except Exception as exc:
        pipe.stop()
        frame_q.put({"cam": cam_idx, "error": f"start aborted: {type(exc).__name__}"})
        return
    t0_ns = int(t0_box.value)
    if t0_ns <= 0:
        pipe.stop()
        frame_q.put({"cam": cam_idx, "error": "tick anchor was never published"})
        return
    period_ns = int(round(1e9 / args.rate))

    latest = None
    option_readback = None
    frameset_guard = FramesetGuard(streams)
    rejected_since = None
    next_option_readback = 0.0
    tick = 0
    try:
        while not stop_evt.is_set():
            fs = pipe.poll_for_frames()
            while fs:                    # drain: only the newest frameset matters
                latest = fs
                fs = pipe.poll_for_frames()

            now = time.time_ns()
            due = t0_ns + tick * period_ns
            if now < due or latest is None:
                time.sleep(POLL_SLEEP_S)
                continue
            # Behind schedule by more than a period (writer stall, USB hiccup):
            # skip the missed ticks outright instead of emitting a burst of
            # backdated frames that never existed.
            if now - due >= period_ns:
                tick += int((now - due) // period_ns)
                due = t0_ns + tick * period_ns

            f1, f2 = latest.get_infrared_frame(1), latest.get_infrared_frame(2)
            fc, fd = latest.get_color_frame(), latest.get_depth_frame()
            if not f1 or not f2 or not fc or not fd:
                raise ValueError(f"{rs_id}: incomplete frameset at tick {tick}")

            src = {"left_ir": f1, "right_ir": f2, "color": fc, "depth": fd}
            rec = {
                "cam": cam_idx, "tick": tick,
                # Colour is the reference stream: it is the one both --streams sets
                # store, so the recorded stamp means the same thing either way.
                "host_ns": int(round(fc.get_timestamp() * 1e6)),
                "tick_ns": due,
                "fnums": (int(f1.get_frame_number()), int(f2.get_frame_number()),
                          int(fc.get_frame_number()), int(fd.get_frame_number())),
            }
            rec.update(rsmeta.frame_metadata([f1, f2, fc, fd], rs))
            rejected = frameset_guard.inspect(rec["fnums"], rec["stream_timestamp_ms"])
            if rejected is not None:
                latest = None
                if rejected_since is None:
                    rejected_since = time.monotonic()
                if frameset_guard.rejected <= 5 or frameset_guard.rejected % 30 == 0:
                    print(f"[{rs_id}] dropped invalid frameset #{frameset_guard.rejected}: "
                          f"{rejected}; tick={tick}, frame_numbers={rec['fnums']}", flush=True)
                if time.monotonic() - rejected_since > 1.0:
                    raise ValueError(f"{rs_id}: no coherent advancing frameset for 1 second: {rejected}")
                continue
            rejected_since = None
            rec["framesets_rejected_total"] = np.int64(frameset_guard.rejected)
            if time.monotonic() >= next_option_readback:
                option_readback = rsmeta.option_snapshot([color_sensor, depth_sensor], rs)
                next_option_readback = time.monotonic() + 1.0
            rec.update(option_readback)
            rec["color_quality"] = rsmeta.color_statistics(np.asanyarray(fc.get_data()), quality_roi)
            for s in streams:
                rec[s] = np.array(src[s].get_data(), copy=True)

            try:
                frame_q.put(rec, timeout=2.0)
            except queue.Full:
                raise ValueError(
                    f"{rs_id}: writer fell {QUEUE_DEPTH} frames behind -- disk cannot "
                    f"keep up with {args.rate} Hz. Lower --rate or write to a faster "
                    f"local disk; --streams cannot go below 'fs' without losing the IR "
                    f"pair FoundationStereo needs.")
            latest = None
            tick += 1
    except Exception as exc:
        frame_q.put({"cam": cam_idx, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        pipe.stop()
        frame_q.put({"cam": cam_idx, "done": True})


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
class EpisodeWriter:
    """Appends paired ticks into one HDF5 in the record_pw_episode.py layout."""

    def __init__(self, path, cams, streams, info, seeds, compression):
        self.path = path
        self.cams = cams
        self.guard_cams = {c for c in cams if info[c].get("frameset_guard", False)}
        self.telemetry_cams = {c for c in cams if info[c].get("photometric_telemetry", False)}
        self.streams = streams
        self.info = info
        self.seeds = seeds
        self.comp = compression
        self.n = 0
        self.f = h5py.File(path, "w")
        self.f.attrs["cam_ids"] = [f"camera_{c}" for c in cams]
        self.g = self.f.create_group("observations/images")
        self.rg = (self.f.require_group("realsense_depth")
                   if "depth" in streams else None)
        self.mg = self.f.require_group("capture_meta")
        self.ds = {}
        shapes = {
            "left_ir": ((IR_H, IR_W), np.uint8, self.g, "camera_{c}_left_ir"),
            "right_ir": ((IR_H, IR_W), np.uint8, self.g, "camera_{c}_right_ir"),
            "color": ((COLOR_H, COLOR_W, 3), np.uint8, self.g, "camera_{c}_color"),
            "depth": ((IR_H, IR_W), np.uint16, self.rg, "camera_{c}"),
        }
        for c in cams:
            for s in streams:
                shp, dt, grp, name = shapes[s]
                self.ds[(c, s)] = grp.create_dataset(
                    name.format(c=c), shape=(0,) + shp, maxshape=(None,) + shp,
                    dtype=dt, chunks=(1,) + shp, compression=self.comp)
            if "depth" in streams:
                self.rg[f"camera_{c}"].attrs["depth_scale"] = info[c]["depth_scale"]
                self.rg[f"camera_{c}"].attrs["frame"] = "left_ir"
            for name, shp, dt in (("host_ns", (), np.int64),
                                  ("tick_ns", (), np.int64),
                                  ("timestamp_ms", (), np.float64),
                                  ("frame_numbers", (4,), np.int64)):
                self.ds[(c, name)] = self.mg.create_dataset(
                    f"camera_{c}_{name}", shape=(0,) + shp, maxshape=(None,) + shp,
                    dtype=dt, chunks=(256,) + shp)
            for name, shape, dtype in rsmeta.specs(4):
                self.ds[(c, name)] = self.mg.create_dataset(
                    f"camera_{c}_{name}", shape=(0,) + shape, maxshape=(None,) + shape,
                    dtype=dtype, chunks=(256,) + shape)
            if c in self.guard_cams:
                self.ds[(c, "framesets_rejected_total")] = self.mg.create_dataset(
                    f"camera_{c}_framesets_rejected_total", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(256,))
                self.mg.attrs[f"camera_{c}_frameset_guard"] = 'stored streams advance; IR pair <=2 ms; RGB/IR <=20 ms; counter cumulative since process start'
            if c in self.telemetry_cams:
                for name, shape, dtype in rsmeta.telemetry_specs():
                    self.ds[(c, name)] = self.mg.create_dataset(
                        f"camera_{c}_{name}", shape=(0,) + shape, maxshape=(None,) + shape,
                        dtype=dtype, chunks=(256,) + shape)
                self.mg.attrs[f"camera_{c}_sensor_option_fields"] = json.dumps(rsmeta.OPTION_FIELDS)
                self.mg.attrs[f"camera_{c}_sensor_option_order"] = 'color,depth'
                self.mg.attrs[f"camera_{c}_sensor_option_semantics"] = '1 Hz SDK option readback with query bounds; NOT per-frame actual exposure'
                self.mg.attrs[f"camera_{c}_color_quality_fields"] = json.dumps(rsmeta.COLOR_FIELDS)
                self.mg.attrs[f"camera_{c}_quality_roi"] = info[c]["quality_roi"]
            rsmeta.describe(self.mg, f"camera_{c}", ["left_ir", "right_ir", "color", "depth"])
            self.mg.attrs[f"camera_{c}_serial"] = info[c]["rs_id"]
            for key in ("sensor_settings_json", "ir_distortion_model", "rgb_distortion_model"):
                self.mg.attrs[f"camera_{c}_{key}"] = info[c][key]
            self.mg.attrs[f"camera_{c}_ir_coeffs"] = info[c]["ir_coeffs"]
            self.mg.attrs[f"camera_{c}_rgb_coeffs"] = info[c]["rgb_coeffs"]

    def append(self, recs):
        """recs: {cam -> record} for ONE tick, every camera present."""
        i = self.n
        for c, r in recs.items():
            for s in self.streams:
                d = self.ds[(c, s)]
                d.resize(i + 1, axis=0)
                d[i] = r[s]
            for name, val in (("host_ns", r["host_ns"]),
                              ("tick_ns", r["tick_ns"]),
                              ("timestamp_ms", r["host_ns"] / 1e6),
                              ("frame_numbers", np.asarray(r["fnums"], np.int64))):
                d = self.ds[(c, name)]
                d.resize(i + 1, axis=0)
                d[i] = val
            for name, _, _ in rsmeta.specs(4):
                d = self.ds[(c, name)]
                d.resize(i + 1, axis=0)
                d[i] = r[name]
            if c in self.guard_cams:
                d = self.ds[(c, "framesets_rejected_total")]
                d.resize(i + 1, axis=0)
                d[i] = r["framesets_rejected_total"]
            if c in self.telemetry_cams:
                for name, _, _ in rsmeta.telemetry_specs():
                    d = self.ds[(c, name)]
                    d.resize(i + 1, axis=0)
                    d[i] = r[name]
        self.n += 1

    def close(self, extra_attrs):
        """Write the per-frame constants (now that T is known) and close.

        Returns the frame count; 0 means nothing was recorded and the file was
        removed. It must not RAISE on that path -- close() runs from a finally
        block, where an exception here would mask the real failure.
        """
        T = self.n
        if T == 0:
            self.f.close()
            Path(self.path).unlink(missing_ok=True)
            return 0
        rep = lambda a: np.repeat(np.asarray(a)[None], T, axis=0)  # noqa: E731
        for c in self.cams:
            nfo = self.info[c]
            self.g.create_dataset(f"camera_{c}_intrinsics", data=rep(nfo["K_ir"]))
            self.g.create_dataset(f"camera_{c}_rgb_intrinsics", data=rep(nfo["K_rgb"]))
            self.g.create_dataset(f"camera_{c}_rgb_to_ir_extrinsic",
                                  data=rep(nfo["T_rgb_to_ir"]))
            self.g.create_dataset(f"camera_{c}_baseline",
                                  data=np.full(T, nfo["baseline"], np.float32))
            self.g.create_dataset(f"camera_{c}_extrinsics", data=rep(self.seeds[c]))
        for k, v in extra_attrs.items():
            self.f.attrs[k] = v
        self.f.attrs["num_frames"] = T
        self.f.close()
        return T


# --------------------------------------------------------------------------- #
def load_seeds(path, cams, serials):
    """camera_<c>_extrinsics[0] from a reference episode, with a serial check.

    Reusing a solved dataset's seed is the whole point of --seed-from: the
    calibration delta in config.json is expressed RELATIVE to this matrix, so a
    different seed silently invalidates it.
    """
    with h5py.File(path, "r") as f:
        seeds = {}
        for c in cams:
            key = f"observations/images/camera_{c}_extrinsics"
            if key not in f:
                raise ValueError(f"{path}: missing {key}")
            E = np.asarray(f[key][0], np.float32)
            if E.shape != (4, 4):
                raise ValueError(f"{path}: {key}[0] is {E.shape}, expected (4, 4)")
            spread = float(np.abs(np.asarray(f[key]) - E).max())
            if spread > 1e-6:
                raise ValueError(
                    f"{path}: {key} varies across frames by {spread:.2e} -- it is not "
                    f"a single static seed, refusing to copy it")
            seeds[c] = E
            ref = f["capture_meta"].attrs.get(f"camera_{c}_serial")
            if ref is not None and str(ref) != str(serials[c]):
                raise ValueError(
                    f"{path}: camera_{c} was serial {ref}, this rig has {serials[c]}. "
                    f"The seed belongs to a different physical camera; re-solve the "
                    f"calibration or pass the matching reference episode.")
    return seeds


def next_episode(out_dir: Path) -> int:
    n = 0
    while (out_dir / f"episode_{n}.hdf5").exists():
        n += 1
    return n


_TELEOP_PARTIAL_RE = re.compile(r"^episode_(\d+)\.hdf5\.partial$")


def newest_teleop_partial(teleop_dir: Path) -> tuple[int, Path, float] | None:
    """The most recently modified ``episode_<N>.hdf5.partial`` in the teleop save dir.

    Returns ``(episode_id, path, mtime_age_s)`` or ``None``. The streaming recorder
    appends rows at the record rate, so the mtime of a LIVE take advances continuously;
    a stale mtime is an aborted (quarantined) take that will never be renamed.
    """
    best = None
    for p in teleop_dir.iterdir():
        m = _TELEOP_PARTIAL_RE.match(p.name)
        if m is None:
            continue
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:   # renamed/unlinked between listing and stat
            continue
        if best is None or mtime > best[2]:
            best = (int(m.group(1)), p, mtime)
    if best is None:
        return None
    return best[0], best[1], time.time() - best[2]


def report_take(path: Path, cams, rate: float, teleop_host, clock_pre, clock_post) -> None:
    """Print the per-camera repeat/jitter/skew summary for one finished file."""
    with h5py.File(path, "r") as f:
        for c in cams:
            fn = np.asarray(f[f"capture_meta/camera_{c}_frame_numbers"])[:, 2]
            reps = int((np.diff(fn) == 0).sum())
            hs = np.asarray(f[f"capture_meta/camera_{c}_host_ns"])
            jitter = np.abs(hs - np.asarray(f[f"capture_meta/camera_{c}_tick_ns"])) / 1e6
            print(f"  camera_{c}: {reps} repeated frame(s), capture-vs-tick offset "
                  f"median {np.median(jitter):.1f} ms, p95 {np.percentile(jitter, 95):.1f} ms")
        h0 = np.asarray(f["capture_meta/camera_0_host_ns"], np.int64)
        h1 = np.asarray(f["capture_meta/camera_1_host_ns"], np.int64)
        d = np.abs(h0 - h1) / 1e6
        print(f"  camera_0 vs camera_1 capture skew: median {np.median(d):.1f} ms, "
              f"max {d.max():.1f} ms")
    if clock_pre and clock_post:
        drift = (clock_post["offset_ns"] - clock_pre["offset_ns"]) / 1e6
        print(f"  {teleop_host} clock drift over the take: {drift:+.2f} ms")


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="episode_<N>.hdf5 is written here; use LOCAL disk. Required "
                         "unless --follow-teleop-dir is given, where it defaults to "
                         "<follow-teleop-dir>_static")
    ap.add_argument("--episode", type=int, default=None,
                    help="default: next free index in --out-dir")
    ap.add_argument("--rate", type=float, default=15.0,
                    help="recording rate in Hz (default 15). Keep this above the teleop "
                         "record rate to tighten alignment -- the match gap is bounded "
                         "by half THIS period alone (15 Hz -> +-33 ms), while equal-rate "
                         "independent streams can alias and reuse frames")
    ap.add_argument("--streams", choices=sorted(_STREAM_SETS), default="fs",
                    help="fs = IR pair + colour, FoundationStereo derives depth from the "
                         "IR pair and beats the ASIC (~16%% smaller, ~16%% less writer "
                         "throughput); full = fs + ASIC depth; rs = colour + ASIC depth only")
    ap.add_argument("--emitter", type=int, default=1, choices=[0, 1],
                    help="1 = IR projector on (better depth on bare tabletops)")
    ap.add_argument("--exposure", default="auto",
                    help="colour exposure: 'auto' (device AE, drifts as the arm moves); "
                         "'auto-lock' (converge once at start-up to a frame mean of "
                         f"{LOCK_TARGET:g} at fixed gain, then FREEZE -- per camera); one "
                         "value for both cameras; or V0,V1 in camera_0,camera_1 order "
                         "(D455 units of 100 us). Manual values need --gain too.")
    ap.add_argument("--gain", default="auto",
                    help="colour gain 0..128, same forms as --exposure; low gain + "
                         f"longer exposure is less noisy (auto-lock default {LOCK_GAIN:g})")
    ap.add_argument("--white-balance", default="auto",
                    help="colour temperature in K (2800..6500), same forms as "
                         "--exposure; tune per camera so a shared white surface "
                         "matches between views")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after this many seconds (default: until Ctrl-C)")
    ap.add_argument("--seed-from", type=Path, default=None,
                    help="episode_<N>.hdf5 to copy camera_<c>_extrinsics from")
    ap.add_argument("--seed", choices=["identity"], default=None,
                    help="use the identity seed instead of --seed-from")
    ap.add_argument("--teleop-host", default=None,
                    help="ssh host of the machine running wbc_vr_robot.py; its clock "
                         "offset is probed before and after the take. 'local' = the "
                         "follower runs on THIS host (offset recorded as exactly 0)")
    ap.add_argument("--follow-teleop-dir", type=Path, default=None,
                    help="session mode: keep the cameras running and cut one static file "
                         "per teleop episode by watching this teleop --save-dir for "
                         "episode_<N>.hdf5.partial (streaming recorder). Files are named "
                         "with the SAME episode id. Requires --teleop-host.")
    ap.add_argument("--stall-s", type=float, default=3.0,
                    help="session mode: a teleop .partial whose mtime has not advanced for "
                         "this long is treated as an aborted take and closed (default 3)")
    ap.add_argument("--backfill-s", type=float, default=0.5,
                    help="session mode: on take start, also write the frames captured "
                         "during the last N seconds, so static coverage begins BEFORE the "
                         "teleop's frame zero (the .partial only appears at frame zero)")
    ap.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"],
                    help="lzf is the real-time default; gzip is ~35%% smaller but will "
                         "not keep up above ~5 Hz")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="refuse to start with less than this much free space")
    ap.add_argument("--serials", default=None,
                    help="comma-separated serials of the TWO static D455s. Required when "
                         "more than two RealSenses are attached (e.g. the handheld scan "
                         "camera is still plugged in); camera_<c> is still assigned by "
                         "ascending serial")
    ap.add_argument("--camera-profile", type=Path, help="JSON keyed by selected D455 serial; exposure/gain in SDK option units")
    ap.add_argument("--allow-auto-color", action="store_true", help="intentional tuning only: allow automatic RGB controls on the mobile static pair instead of requiring/selecting its measured profile")
    ap.add_argument("--quality-roi", default="0,0,1,1", help="normalized task ROI x0,y0,x1,y1 for photometric statistics")
    args = ap.parse_args()
    if args.out_dir is None:
        if args.follow_teleop_dir is None:
            ap.error("--out-dir is required without --follow-teleop-dir")
        d = args.follow_teleop_dir
        args.out_dir = d.with_name(d.name + "_static")
    return args


MOBILE_STATIC_SERIALS = frozenset(('239222302971', '244622300362'))
MOBILE_PROFILE = Path(__file__).resolve().parents[1] / 'configs/camera_profiles/static_mobile_20260920.json'


def resolve_mobile_profile(args, serials, profile_path=MOBILE_PROFILE):
    """Prevent default 'auto' from undoing the measured mobile-camera settings.

    Other camera pairs retain their existing behavior. Intentional automatic-color
    experiments on this pair require --allow-auto-color and remain visible in QC.
    """
    if frozenset(serials) != MOBILE_STATIC_SERIALS or args.allow_auto_color:
        return
    if args.camera_profile is None:
        manual = all(isinstance(args.color_control[k], list)
                     for k in ('exposure', 'gain', 'white_balance'))
        defaults = all(args.color_control[k] is None
                       for k in ('exposure', 'gain', 'white_balance'))
        if manual:
            return
        if not defaults:
            raise ValueError('Mobile static capture needs manual RGB exposure/gain AND white balance; '
                             'use --camera-profile, or --allow-auto-color for an intentional experiment')
        if not profile_path.is_file():
            raise ValueError(f'Measured mobile camera profile missing: {profile_path}; '
                             'pass --camera-profile explicitly; refusing silent auto exposure')
        args.camera_profile = profile_path
        args.camera_profiles = json.loads(profile_path.read_text())
        print(f'profile   automatically selected measured mobile profile: {profile_path}', flush=True)
    if not isinstance(args.camera_profiles, dict):
        raise ValueError('Mobile camera profile must be an object keyed by serial')
    required = {'color_exposure_option', 'color_gain', 'white_balance'}
    for serial in serials:
        settings = args.camera_profiles.get(serial)
        if not isinstance(settings, dict) or required - settings.keys():
            raise ValueError(f'{serial}: mobile profile must specify RGB exposure, gain and white balance; '
                             'refusing partial profile with an automatic-color fallback')


def main():
    args = parse_args()
    args.camera_profiles = json.loads(args.camera_profile.read_text()) if args.camera_profile else {}
    if not isinstance(args.camera_profiles, dict):
        raise ValueError('Camera profile must be an object keyed by serial')
    args.quality_roi_values = tuple(float(v) for v in args.quality_roi.split(','))
    rsmeta.color_statistics(np.zeros((COLOR_H, COLOR_W, 3), np.uint8), args.quality_roi_values)
    if args.rate <= 0 or args.rate > FPS:
        raise ValueError(f"--rate must be in (0, {FPS}]; got {args.rate}")
    if (args.seed_from is None) == (args.seed is None):
        raise ValueError(
            "pass exactly one of --seed-from <episode.hdf5> or --seed identity. "
            "There is no default: a wrong extrinsic seed silently invalidates the "
            "solved calibration delta in config.json.")

    streams = _STREAM_SETS[args.streams]
    args.color_control = parse_color_control(args.exposure, args.gain, args.white_balance)
    max_exp = 1e6 / (100.0 * FPS)
    if isinstance(args.color_control["exposure"], list) \
            and max(args.color_control["exposure"]) > max_exp:
        raise ValueError(f"--exposure above {max_exp:g} cannot sustain {FPS} fps")
    attached = sorted(d.get_info(rs.camera_info.serial_number) for d in rs.context().devices)
    if args.serials is not None:
        serials = sorted(x.strip() for x in args.serials.split(",") if x.strip())
        if len(serials) != 2 or len(set(serials)) != 2:
            raise ValueError(f"--serials must name exactly 2 distinct cameras, got {serials}")
        missing = [x for x in serials if x not in attached]
        if missing:
            raise ValueError(f"--serials {missing} not attached; attached: {attached}")
    else:
        serials = attached
        if len(serials) != 2:
            raise ValueError(
                f"expected exactly 2 RealSense devices, found {serials}; pass --serials "
                "<static_a>,<static_b> to pick the two static ones")
    resolve_mobile_profile(args, serials)
    if set(args.camera_profiles) - set(serials):
        raise ValueError(f"Profile contains unselected serials: {set(args.camera_profiles) - set(serials)}")
    cams = list(range(len(serials)))          # camera_<c> by ascending serial
    serial_of = {c: serials[c] for c in cams}

    follow = args.follow_teleop_dir is not None
    if follow:
        if args.episode is not None:
            raise ValueError("--episode is meaningless with --follow-teleop-dir: the static "
                             "file takes the teleop episode's id")
        if args.teleop_host is None:
            raise ValueError("--follow-teleop-dir needs --teleop-host (use 'local' when the "
                             "teleop follower runs on this machine)")
        args.follow_teleop_dir.mkdir(parents=True, exist_ok=True)
        if not np.isfinite(args.stall_s) or args.stall_s <= 0.0:
            raise ValueError(f"--stall-s must be finite and > 0, got {args.stall_s}")
        if not np.isfinite(args.backfill_s) or args.backfill_s < 0.0:
            raise ValueError(f"--backfill-s must be finite and >= 0, got {args.backfill_s}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = None
    if not follow:
        ep = next_episode(args.out_dir) if args.episode is None else args.episode
        out_path = args.out_dir / f"episode_{ep}.hdf5"
        if out_path.exists():
            raise ValueError(f"{out_path} already exists -- refusing to overwrite")

    mb_s = sum(_MB_PER_FRAME[s] for s in streams) * len(cams) * args.rate
    free_gb = shutil.disk_usage(args.out_dir).free / 1e9
    print(f"cameras   camera_0 = {serials[0]}   camera_1 = {serials[1]}  (ascending serial)")
    if follow:
        print(f"output    {args.out_dir}/episode_<N>.hdf5, N following "
              f"{args.follow_teleop_dir}/episode_<N>.hdf5.partial")
    else:
        print(f"output    {out_path}")
    print(f"streams   {', '.join(streams)}  {IR_W}x{IR_H} IR / {COLOR_W}x{COLOR_H} colour  "
          f"@ {args.rate:g} Hz, {args.compression}")
    print(f"rate      ~{mb_s:.0f} MB/s  =  ~{mb_s * 60 / 1000:.1f} GB/min")
    print(f"free      {free_gb:.1f} GB  ->  ~{free_gb * 1000 / mb_s / 60:.0f} min of recording")
    if free_gb < args.min_free_gb:
        raise ValueError(
            f"only {free_gb:.1f} GB free on {args.out_dir} (--min-free-gb "
            f"{args.min_free_gb}). Free space or choose another --out-dir.")
    # An sshfs/NFS target cannot absorb this and the grabbers will abort mid-take.
    dev = os.stat(args.out_dir).st_dev
    with open("/proc/mounts") as fh:
        for line in fh:
            src, mnt, fstype = line.split()[:3]
            try:
                if os.stat(mnt).st_dev == dev and fstype.startswith(("fuse", "nfs", "cifs")):
                    print(f"\n!! {args.out_dir} is on a {fstype} mount ({src}).")
                    print("!! Record to local disk and copy afterwards -- a network "
                          "filesystem will not absorb this write rate.\n")
                    break
            except OSError:
                continue

    seeds = ({c: np.eye(4, dtype=np.float32) for c in cams} if args.seed == "identity"
             else load_seeds(args.seed_from, cams, serial_of))
    if args.seed_from:
        print(f"seed      copied from {args.seed_from}")
    else:
        print("seed      IDENTITY -- solve extrinsics from scratch for this dataset")

    clock_pre = None
    if args.teleop_host and not follow:
        clock_pre = probe_clock_offset(args.teleop_host)
        print(f"clock     {args.teleop_host} - local = "
              f"{clock_pre['offset_ns'] / 1e6:+.2f} ms "
              f"(rtt {clock_pre['rtt_ns'] / 1e6:.1f} ms, {clock_pre['samples']} samples)")

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(cams) + 1)      # +1 = this process, see t0_box below
    info_q, frame_q = ctx.Queue(), ctx.Queue(maxsize=QUEUE_DEPTH * len(cams))
    stop_evt = ctx.Event()
    t0_box = ctx.Value("q", 0)

    procs = [ctx.Process(target=_grabber,
                         args=(serial_of[c], c, args, streams, barrier, info_q,
                               frame_q, stop_evt, t0_box), daemon=True)
             for c in cams]
    for p in procs:
        p.start()

    info = {}
    print(f"\nsettling colour ({SETTLE_S:.0f}s)...")
    try:
        for _ in cams:
            r = info_q.get(timeout=SETTLE_S + 90)
            if not r["ok"]:
                raise ValueError(f"camera {r['rs_id']} failed to start: {r['error']}")
            info[r["cam"]] = r
    except Exception:
        stop_evt.set()
        barrier.abort()      # release any camera that DID open and is waiting
        for p in procs:
            p.join(timeout=10)
        raise
    for c in cams:
        print(f"  camera_{c} ({info[c]['rs_id']}): baseline "
              f"{info[c]['baseline'] * 1000:.1f} mm, colour "
              f"{info[c]['saturated'] * 100:.1f}% saturated, "
              f"{describe_color_control(info[c]['color_control'])}")

    # Anchor the shared tick schedule only once both cameras are past settling, so
    # tick 0 is a real frame on both and the two streams stay index-aligned.
    # Published BEFORE joining the barrier the grabbers are blocked on, so they
    # cannot read the anchor before it exists.
    t0_box.value = time.time_ns() + int(0.5e9)
    barrier.wait()

    compression = None if args.compression == "none" else args.compression
    base_attrs = {
        "emitter_enabled": args.emitter,
        "extrinsic_init": "identity" if args.seed == "identity" else "seed_from",
        "extrinsic_seed_source": str(args.seed_from or "identity"),
        "record_rate_hz": float(args.rate),
        "streams": list(streams),
        "capture_host": os.uname().nodename,
        "timestamp_domain": "global_time (host epoch, ms)",
    }
    for c in cams:
        for k, v in info[c]["color_control"].items():
            base_attrs[f"camera_{c}_color_{k}"] = v

    def _clock_attrs(pre, post) -> dict:
        out = {}
        for tag, probe in (("pre", pre), ("post", post)):
            if probe:
                out[f"teleop_clock_{tag}_host"] = probe["host"]
                out[f"teleop_clock_{tag}_offset_ns"] = np.int64(probe["offset_ns"])
                out[f"teleop_clock_{tag}_rtt_ns"] = np.int64(probe["rtt_ns"])
                out[f"teleop_clock_{tag}_measured_at_ns"] = np.int64(probe["measured_at_ns"])
        return out

    def _post_probe():
        if not args.teleop_host:
            return None
        try:
            return probe_clock_offset(args.teleop_host)
        except Exception as exc:
            print(f"!! post-take clock probe failed: {exc}")
            return None

    # --- session (follow) mode state: one EpisodeWriter per teleop take -------------
    writer = None
    writer_ep = None
    writer_pre = None
    pending: dict[int, dict] = {}
    # Paired ticks captured between takes, newest last; sized to hold --backfill-s.
    recent: deque = deque(maxlen=int(np.ceil(args.backfill_s * args.rate)) + 2)
    finished, errors = set(), []
    interrupted = False
    takes_written = 0
    last_poll = 0.0

    def _start_take(ep: int) -> None:
        nonlocal writer, writer_ep, writer_pre
        final = args.out_dir / f"episode_{ep}.hdf5"
        partial = args.out_dir / f"episode_{ep}.hdf5.partial"
        if final.exists():
            raise ValueError(
                f"{final} already exists: the teleop dir restarted its numbering at "
                f"episode_{ep} while static files for that id are already here. Point "
                f"--out-dir at an empty session directory."
            )
        if partial.exists():
            aside = partial.with_name(f"{partial.name}.{time.time_ns()}")
            partial.rename(aside)
            print(f"!! stale {partial.name} moved aside to {aside.name}")
        writer_pre = probe_clock_offset(args.teleop_host)
        writer = EpisodeWriter(partial, cams, streams, info, seeds, compression)
        writer_ep = ep
        # The teleop .partial only appears when its frame ZERO lands, and this watcher sees
        # it up to a poll later: without backfill the static file started ~0.25 s after the
        # teleop's first row (episode 0: rows 0-2 had no static frame). Replay the paired
        # ticks captured during the last --backfill-s from the ring buffer so the static
        # coverage starts BEFORE frame zero; extra leading frames are harmless to the
        # nearest-neighbour alignment.
        cutoff_ns = time.time_ns() - int(args.backfill_s * 1e9)
        replayed = 0
        for tick_recs in list(recent):
            if min(r["host_ns"] for r in tick_recs.values()) >= cutoff_ns:
                writer.append(tick_recs)
                replayed += 1
        recent.clear()
        print(f"\nRECORDING teleop episode_{ep} -> {partial}   ({time.strftime('%H:%M:%S')}, "
              f"backfilled {replayed} tick(s) from the last {args.backfill_s:g}s)", flush=True)

    def _finish_take(disposition: str) -> None:
        nonlocal writer, writer_ep, writer_pre, takes_written
        ep = writer_ep
        partial = args.out_dir / f"episode_{ep}.hdf5.partial"
        final = args.out_dir / f"episode_{ep}.hdf5"
        post = _post_probe()
        attrs = dict(base_attrs, teleop_episode_id=np.int64(ep),
                     teleop_disposition=disposition,
                     teleop_dir=str(args.follow_teleop_dir))
        attrs.update(_clock_attrs(writer_pre, post))
        T = writer.close(attrs)
        writer, writer_ep = None, None
        if T == 0:
            print(f"  episode_{ep}: no static frames landed inside the take; nothing written")
            return
        if disposition == "complete":
            os.rename(partial, final)        # atomic, same directory
            kept = final
        elif disposition == "discarded":
            partial.unlink()                 # teleop kept no rows: nothing to pair with
            print(f"  episode_{ep}: teleop take discarded (zero rows); static file removed")
            return
        else:
            kept = partial                   # aborted/interrupted: stays .partial like teleop
        takes_written += 1
        print(f"  wrote {kept}  ({T} frames, {kept.stat().st_size / 1e9:.2f} GB, "
              f"{T / args.rate:.1f}s, teleop {disposition})")
        report_take(kept, cams, args.rate, args.teleop_host, writer_pre, post)

    def _poll_teleop() -> None:
        """Open/close the per-take writer from the teleop save dir's .partial state."""
        if writer is None:
            hit = newest_teleop_partial(args.follow_teleop_dir)
            if hit is None:
                return
            ep, _path, age = hit
            if age <= args.stall_s:        # advancing mtime = a live streaming take
                _start_take(ep)
            return
        final = args.follow_teleop_dir / f"episode_{writer_ep}.hdf5"
        partial = args.follow_teleop_dir / f"episode_{writer_ep}.hdf5.partial"
        if final.exists():
            _finish_take("complete")
        elif not partial.exists():
            _finish_take("discarded")
        elif time.time() - partial.stat().st_mtime > args.stall_s:
            _finish_take("aborted")

    if not follow:
        writer = EpisodeWriter(out_path, cams, streams, info, seeds, compression)
        writer_pre = clock_pre

    def _sigint(_sig, _frm):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            print("\nstopping...")
            stop_evt.set()
    signal.signal(signal.SIGINT, _sigint)

    if follow:
        print(f"\nSESSION: waiting for {args.follow_teleop_dir}/episode_<N>.hdf5.partial "
              f"(Ctrl-C to end the session)")
    else:
        print(f"\nRECORDING -> {out_path}   (Ctrl-C to stop)")
    t_start = time.time()
    t_report = t_start
    dropped = 0
    try:
        while len(finished) < len(cams):
            if args.duration and not stop_evt.is_set() \
                    and time.time() - t_start >= args.duration:
                stop_evt.set()
            if follow and not stop_evt.is_set() and time.monotonic() - last_poll >= 0.05:
                last_poll = time.monotonic()
                _poll_teleop()
            try:
                rec = frame_q.get(timeout=0.2 if follow else 1.0)
            except queue.Empty:
                if stop_evt.is_set() and not any(p.is_alive() for p in procs):
                    break
                continue
            if "error" in rec:
                errors.append(f"camera_{rec['cam']}: {rec['error']}")
                stop_evt.set()
                continue
            if rec.get("done"):
                finished.add(rec["cam"])
                continue
            slot = pending.setdefault(rec["tick"], {})
            slot[rec["cam"]] = rec
            if len(slot) == len(cams):
                paired = pending.pop(rec["tick"])
                if writer is not None:
                    writer.append(paired)
                elif follow:
                    recent.append(paired)   # bounded ring buffer for take-start backfill
            # Ticks whose peer never arrives must not pile up while idle between takes.
            if len(pending) > 4 * QUEUE_DEPTH:
                for old in sorted(pending)[:-QUEUE_DEPTH]:
                    pending.pop(old)
            now = time.time()
            if writer is not None and now - t_report >= 5.0:
                t_report = now
                sz = Path(writer.path).stat().st_size / 1e9
                print(f"  {writer.n:6d} frames  {now - t_start:6.1f}s  {sz:5.2f} GB",
                      flush=True)
    finally:
        stop_evt.set()
        for p in procs:
            p.join(timeout=15)
        # Unpaired trailing ticks (one camera stopped a frame earlier than the
        # other) are dropped rather than written half-filled.
        dropped = len(pending)
        if writer is not None:
            if follow:
                _finish_take("interrupted" if not errors else "static_camera_error")
            else:
                post = _post_probe()
                attrs = dict(base_attrs)
                attrs.update(_clock_attrs(writer_pre, post))
                T = writer.close(attrs)
                writer = None

    if errors:
        raise ValueError("recording aborted:\n  " + "\n  ".join(errors))
    if follow:
        print(f"\nsession over: {takes_written} static take(s) written to {args.out_dir}")
        return 0
    if T == 0:
        raise ValueError("no frames were recorded -- nothing written")

    print(f"\nwrote {out_path}  ({T} frames, {out_path.stat().st_size / 1e9:.2f} GB, "
          f"{T / args.rate:.1f}s)")
    if dropped:
        print(f"  {dropped} unpaired trailing tick(s) dropped")
    report_take(out_path, cams, args.rate, args.teleop_host, clock_pre, post)
    return 0


if __name__ == "__main__":
    sys.exit(main())
