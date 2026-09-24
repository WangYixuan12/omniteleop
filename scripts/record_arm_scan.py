#!/usr/bin/env python3
"""Handheld D455 walk-around scan of the Vega robot, frozen at several arm poses.

Same capture end as ``record_scan.py`` -- FoundationStereo turns each stored IR pair
into metric depth, RGB-D odometry chains the frames, a TSDF fuses them -- but aimed at
the ROBOT instead of the room, and with everything needed to *name* the resulting
surface recorded alongside it.

Why the extra channels
----------------------
A scan gives geometry and appearance. Rendering the arm from the head camera needs
three more things: per-link segmentation, a scan->link registration, and the joint
model. ``fit_robot_to_scan.py`` supplies the first two in one step -- it poses the URDF
at the scan-time joint vector and solves the single rigid transform left over -- but
only if that joint vector is known. Typing it by hand is how it is done today
(``--torso 0.78 ... --left-arm 0.844 ...``); this script records it instead, per frame,
with the component publish timestamps, so the fit is reproducible and the "was the
robot actually still?" question is answerable from the file rather than from memory.

Head ``left_rgb`` (+ depth) is recorded for the same reason. Once the scan is registered
to the URDF, the head frames recorded in the SAME pose are an independent check of the
whole render path: project the registered, per-link-labelled cloud into the head camera
and compare against the photograph. That also measures the ``zed_depth_frame`` (URDF)
vs ``head_left_rgb`` (optical) offset, which is currently assumed rather than measured.

Why several poses
-----------------
Four separate reasons, all of which need the poses to be *different*, not just many:

  1. Coverage. A link occluded by the forearm in one pose is exposed in another.
  2. Registration. Each pose is an independent solve of the same seven numbers; if
     they disagree, either a joint readback or the URDF is wrong, and the disagreement
     says which link.
  3. The open right-arm question. ``docs/right_arm_staleness/HANDOVER.md`` closes with
     "the right arm's FK-vs-depth agreement is worse than the left (median 84% vs
     99.3% within 30 mm) ... it needs a purpose-recorded calibration sweep, not this
     investigation." A frozen multi-pose scan IS that sweep: a constant joint-zero
     error shows up as a link displacement that grows down the chain. That finding is
     the OLD arm's; on this robot every arm joint publishes at the full record rate
     (~10 Hz, measured over episode_9), so the ~1 Hz ``R_arm_j1``--``j4`` staleness
     that muddied the original investigation is no longer in the picture at all.
  4. The UNDERSIDE. A walk-around scan photographs whatever faces the room; the faces
     that point at the floor are never seen, no matter how many laps you walk. Teleop
     never rolls the wrist -- over all 16 episodes of that teleop set the
     forearm roll ``*_arm_j5`` spans ~1.1 rad of its 6.1 rad range and never changes
     sign -- so the gripper is palm-down in every frame the robot has ever recorded,
     and the bottom of the forearm, wrist and gripper is simply missing from the atlas.
     Only the ROBOT can fix that, by turning the part over between takes.

Poses that vary ONE joint at a time are the most identifiable for (3); poses that
change the occlusion pattern are the best for (1); (4) needs each pose paired with a
rolled twin.

Rolling the wrist to scan the bottom -- why 120 degrees, not 180
----------------------------------------------------------------
``j5`` is the forearm roll, so links ``l5`` (forearm), ``l6``, ``l7`` and ``robotiq``
all turn with it. It is the only joint that can show the camera their underside.

A half turn is NOT needed and is not worth its cost. The scan camera is handheld and
walks around the robot, so a face only has to stop pointing at the FLOOR to be
photographable -- once its normal is roughly horizontal it is seen head-on from a side
view. Measured on the URDF by sampling link surface normals (a normal counts as
scannable at world ``z > -0.30``), the distal links go from 77.2% of their surface ever
facing the camera to:

    roll      90 deg -> 94.3%      120 deg -> 98.7%      135 deg -> 99.8%
    j5 +- pi with j6 -> -j6, j7 -> -j7 -> 100.0%

``120 deg`` is the knee of that curve. It brings the worst-case bottom face to
``normal.Z = -0.05`` -- horizontal, i.e. straight at a camera held beside the robot --
for two thirds of the wrist travel of a half turn, and it parks ``j5`` between -0.99
and +1.10 rad, NEARER its neutral than the recorded teleop posture itself sits. The
half turn buys the last 1.3 points of coverage for 50% more wrist rotation and parks
j5 at -1.79/+1.97 rad instead. Nothing in the URDF forbids the half turn -- the limit
is +-3.071 -- but the wrist-camera cable run has never been asked to take one, and
coverage is not what is short. Pass a bigger roll to the generator if it ever is.

Sign is per arm because the recorded posture is not symmetric about zero: LEFT rolls
``j5 - 2.0944``, RIGHT rolls ``j5 + 2.0944``.

Only the exact half turn keeps the gripper still, via the wrist's second solution
(``j5 -+ pi, j6 -> -j6, j7 -> -j7``, which moves the gripper origin <= 1 mm). There is
no partial-angle version of that identity, so a 120 deg roll also swings the gripper by
up to 0.35 m. That is checked, not ignored -- see below -- and it helps coverage rather
than hurting it, at the cost of the rolled take no longer being a pixel-matched twin of
its base take.

The upper arm (``l3``/``l4``) is left alone: rolling ``j3`` to turn it over buys only
~7 points, because the rest of its blind spot is the face pressed against the torso,
which no joint angle exposes.

Where scan_poses_dex.json comes from
------------------------------------
``scan_poses_ep9.json`` was drawn from ONE episode. ``scan_poses_dex.json`` replaces it
with the posture the robot actually holds across ALL 16 episodes of the teleop set --
36 699 frames, recorded at ``/data/Dexmate/data`` and since moved to
``/data/Dexmate/wm_data`` -- summarised as k-medoids of the 14-D two-arm joint vector,
each snapped back to the real frame nearest its centre (the label carries the episode
and frame it came from, so any pose can be looked up in the recording it describes).
Every one is a posture the robot has physically held, so reachability and clearance are
facts, not predictions. Torso ``[0.7805, 1.5724, 0.0002]`` and head ``[-0.25, 0, 0]``
are constant in every episode and are pinned in every entry, so the scan is taken in
the data-collection posture rather than near it.

FOUR medoids, not more, and specifically medoids 0, 2, 3 and 6. Coverage saturates
almost immediately, because the teleop postures are all variations of "both arms
forward": these four reach 98.8% of the distal surface (98.3% of the gripper) once
rolled, where all eight reach 98.9%. The rest cost half the scan and buy a tenth of a
point.

Medoid 1 is EXCLUDED, and not for coverage. It is the pose where the two hands come
closest -- ``L_robotiq_finger_neg`` to ``R_robotiq_finger_pos`` is 64 mm, the tightest
pair anywhere on the robot -- and rolling the wrists swings the fingers straight
through each other (-4.3 mm at 120 deg). No roll angle between 60 and 150 deg clears
it in either direction, so the pose has no rolled twin and is dropped. If a later
dataset is re-clustered, RE-RUN THE CHECKER: "the hands are already nearly touching"
is a property of the posture, not of the roll.

Each medoid is followed immediately by its rolled twin, so consecutive takes differ
only by the wrist roll -- the shortest move between takes, and the pair that makes the
underside comparison obvious.

The d6 PAIR IS DELIBERATELY FIRST, NOT LAST, and the run now finishes on ``d3roll``.
``d6roll_ep9f600`` brings the two hands close enough that the swept-volume ``tip_*``
fingertip spheres overlap by 20 mm (``L_robotiq_tip_end_pos`` to
``R_robotiq_tip_end_neg``). The real fingertip geometry clears by 70 mm and the checker
below calls the pose safe -- it drops ``tip_*`` and measures the mesh -- but
``wbc_vr_robot.py``'s homing guard uses the RAW sphere metric and vetoes any waypoint
under its 20 mm ``safe_dist`` with no comparison to where it started. So a run that
ENDS on d6roll leaves the robot unable to home at all: every group aborts on its first
substep with ``self-collision -2.0cm``, whichever one you try. Passing THROUGH the pose
is fine. Keep each (medoid, roll) pair adjacent, and keep d6 off the end.

Verify any edit to this file with the real checker, which poses the URDF the solver
actually uses and includes the GRIPPER geometry:

    python scripts/diagnostics/check_scan_collisions.py \
        --poses scripts/scan_poses_dex.json --measured <measured.json>

It reports ``VegaWholeBodyIK.self_collision_distance()`` -- the same metric the
whole-body QP's SelfCollisionBarrier uses -- at every waypoint AND along the
straight-line interpolation ``interp_to`` walks between them. As written this file
clears at 0.0594 m everywhere, ~6x the 0.01 m floor and ~3x the 0.02 m safe_dist; the
only tighter number in the report is the measured start pose, which is wherever the
robot happens to be standing. Do NOT vet these poses with the bare Dexmate sphere
model (``vega_1_collision_spheres.collision.urdf``): it is built for
``vega_no_effector.urdf`` and has no finger spheres at all, so it cannot see the one
interference that actually bites here.

Budget: 8 takes at ``--scan-time 30`` is ~31 GB and ~9 min of recording.

Where the time actually goes, if you want it shorter still: ``--scan-time`` IS the
take (75 of every ~82 s). ``--settle 3`` + ``--precheck 2`` is under 7% of it, so
trimming the settle is not the lever it looks like -- and it is load-bearing, because
``interp_to`` does not converge per substep: it steps 0.01 rad while ``exit_on_reach``
uses a 0.05 rad tolerance, so every substep returns immediately and the arm arrives
carrying the ramp's accumulated lag. ``--settle`` is where that lag is absorbed and
``--precheck`` is what proves it did (it samples the joints and SKIPS the pose if any
moved more than ``--freeze-tol``, so a rushed settle costs a retry, never a bad file).
The levers that matter are the pose count and ``--scan-time``.

R_arm_j1 works on THIS robot
---------------------------
On the old arm the right shoulder's first motor could not be commanded, so every
right-arm target issued here had element 0 overwritten with the measured angle. That
workaround is RETIRED: on the robot that recorded ``/data/Dexmate/data/episode_9.hdf5``
R_arm_j1 was commanded across 1.75 rad and tracked it with a 0.026 rad median error, so
``DEAD_JOINTS`` is empty and right-arm targets are now sent whole. The whole right arm
can therefore be swept, j1 included -- which is why the ``scan_poses_dex.json`` /
``scan_poses_ep9*.json`` files command both arms where the old
``scan_poses_books*.json`` commanded only the left.

CHECK ``DEAD_JOINTS`` before running this against a different arm.

The robot must be FROZEN while you scan
---------------------------------------
The odometry is photometric against a static scene. A robot that moves mid-take does not
produce a worse atlas, it produces a wrong trajectory. So each pose is recorded into its
OWN file: the robot is commanded (or hand-posed), allowed to settle, verified still, and
only then does recording start. Nothing that moves ever lands in a scan file. The freeze
is verified again afterwards from the recorded readback, and a pose that failed is named
in the summary and in the file's ``frozen`` attribute.

Writes the ``record_scan.py`` / ``record_pw_episode.py`` ``camera_0`` layout so
``fuse_scan.py`` reads each pose file unchanged -- ``camera_0_extrinsics`` deliberately
absent, for the same reason as there: a handheld scan has no known pose until the fusion
solves one.

usage (on lambda -- the only host on the robot's 192.168.50.0/24 link -- with the
handheld D455 on a USB3 cable and the head publisher up):

    # once, if pyrealsense2 is not in the venv yet:
    ~/omniteleop/.venv/bin/pip install pyrealsense2

    # what pose names the robot itself offers
    ~/omniteleop/.venv/bin/python scripts/record_arm_scan.py --list-poses

    # hand-posed / teleop-posed: prompts before each take, commands nothing
    ~/omniteleop/.venv/bin/python scripts/record_arm_scan.py \
        --out-dir /data/scans/arm_09_08 --poses scripts/scan_poses_dex.json --scan-time 75

    # let the script drive the arms through the list (SAFETY: clear the workspace)
    ~/omniteleop/.venv/bin/python scripts/record_arm_scan.py \
        --out-dir /data/scans/arm_09_08 --poses scripts/scan_poses_dex.json --execute

``--poses`` is a JSON list; each entry names any subset of components (see
``scan_poses.example.json``, five poses lifted verbatim from recorded teleop frames,
so every one is known reachable and known collision-free):

    [{"label": "home"},
     {"label": "left_L",   "left_arm": "L_shape", "head": [-0.2, 0.0, 0.0]},
     {"label": "left_up",  "left_arm": [0.064, 0.3, 0.0, -2.756, 1.271, 0.0, 0.0]},
     {"label": "right_j2", "right_arm": {"delta": [0, 0.35, 0, 0, 0, 0, 0]}},
     {"label": "right_j4", "right_arm": {"delta": [0, 0, 0, 0.35, 0, 0, 0]}}]

A component value is a full joint vector, one of the robot's own ``pose_pool`` names
(``--list-poses``), or ``{"delta": [...]}`` measured from where the joint is NOW.
Prefer the delta form for the right arm: a named vendor pose assumes all seven joints
reach their listed values, and ``R_arm_j1`` cannot, so the pose that actually results
is a hybrid whose clearance nobody has checked. One-joint deltas are also the most
informative poses for the kinematic question above.

Omitted components are left where they are. ``{"label": "home"}`` alone records the
robot exactly as it stands. With ``--execute`` the entries are commanded; without it
they are printed as instructions and you pose the robot yourself.

Ctrl-C ends the current pose early and moves to the next; twice within 2 s aborts the
run (the files already written stay valid). Budget ~4.9 GB/min while a pose is
recording. Walk slowly, stay outside ~0.5 m, and keep the whole robot in view.
"""

from __future__ import annotations

import argparse
import os
import json
import queue
import shutil
import signal
import subprocess
import tempfile
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pyrealsense2 as rs

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, (_REPO / "src").as_posix())

from omniteleop.common.head_camera import (  # noqa: E402
    HEAD_CROP_TBLR,
    HEAD_RESIZE_HW,
    ZED_K,
)

from omniteleop.common.capture_integrity import (freeze_ranges, state_snapshot,
                                                ProgressGuard, validate_stereo_stamps)
from omniteleop.common import realsense_capture_meta as rsmeta

IR_W, IR_H = 1280, 720
COLOR_W, COLOR_H = 1280, 800
FPS = 30
SETTLE_S = 2.0
QUEUE_DEPTH = 64        # ~2 s of backlog before we call the disk too slow
MB_PER_FRAME = 2.7      # measured with lzf on this scene (IR pair + colour)

# HW-WORKAROUND(right-j1-dead): RETIRED -- this robot's right shoulder is healthy.
# On the old arm R_arm_j1 could not be commanded, so element 0 of every right-arm target
# was overwritten with the measured angle. On the robot that recorded
# /data/Dexmate/data/episode_9.hdf5 that motor works: j1 was commanded across 1.75 rad
# and tracked it with a 0.026 rad median error (obs-vs-action corr 0.983), and its
# readback updates at the full ~10 Hz instead of the old ~1 Hz. Leaving it listed would
# silently pin the right shoulder to wherever it already is -- which is precisely the
# sweep these scans exist to measure. Keep names URDF-spelled: they are compared against
# dexcontrol's get_joint_name(). Put "R_arm_j1" back to run against the old arm.
DEAD_JOINTS: tuple[str, ...] = ()

# Every joint group worth recording, in the order they are concatenated into
# robot/joint_pos. Names come from the robot, not from here, so the vector stays
# self-describing if a component's joint list ever changes.
STATE_COMPONENTS = ("torso", "head", "left_arm", "right_arm", "left_hand", "right_hand")
COMMANDABLE = ("torso", "head", "left_arm", "right_arm")

# Every robot camera that can be recorded alongside the handheld D455, as
# (HDF5 group == record kind, robot.sensors attribute). The group name IS the kind, so
# ``head/left_rgb`` sits exactly where it always did and the wrists simply add
# ``left_wrist/`` and ``right_wrist/`` beside it. Wrist frames are what make the
# wrist-camera -> head-camera extrinsic solvable from a scan take: both cameras see the
# same static board from the same frozen pose, so each pose is one direct measurement.
CAMERAS = (("head", "head_camera"),
           ("left_wrist", "left_wrist_zedm"),
           ("right_wrist", "right_wrist_zedm"))
CAM_KINDS = tuple(k for k, _ in CAMERAS)
CAM_ATTR = dict(CAMERAS)

# Kinds read STRAIGHT OFF ZENOH instead of through robot.sensors.
#
# The wrists are stereo ZED Minis published by tests/test_wrist_zedm_depth.py on
# sensors/<side>_wrist_zedm/{left_rgb,right_rgb} -- the same ids wbc_vr_robot.py and the
# leader HUD consume (they inject a ZedXCameraConfig for them). A stock dexcontrol Robot
# only knows the built-in *_wrist_camera sensors, a MONOCULAR ZED X One model with a
# single "rgb" stream, which can never receive this publisher. Subscribing directly keeps
# the stereo pair (what makes wrist frames metric) and needs no sensor injection.
DIRECT_ZENOH = ("left_wrist", "right_wrist")


def _bright_rb(pipe, *, skip: int = 3, frames: int = 6, thr: float = 170.0):
    """Mean R/B of near-white pixels over a few delivered colour frames.

    Returns (r_over_b, n_pixels); r_over_b is None when the view holds too few
    near-white pixels for a grey-world estimate to mean anything.
    """
    for _ in range(skip):                       # a WB change takes effect over a frame or two
        pipe.wait_for_frames()
    acc = []
    for _ in range(frames):
        colour = pipe.wait_for_frames().get_color_frame()
        if not colour:
            continue
        px = np.asanyarray(colour.get_data()).reshape(-1, 3).astype(np.float32)
        px = px[px.mean(1) > thr]
        if len(px):
            acc.append(px)
    if not acc:
        return None, 0
    px = np.concatenate(acc)
    if len(px) < 20000:
        return None, len(px)
    m = px.mean(0)
    return float(m[0] / max(m[2], 1e-6)), len(px)


def open_d455(serial: str, *, exposure_us: float | None = None,
              gain: float | None = None, white_balance: float | None = None,
              profile_settings: dict | None = None):
    """Bring one D455 up, re-opening it once if the first frame never arrives.

    2026-09-19: camera_2 (244622300362) started cleanly and then delivered nothing --
    no kernel USB event on its port, nothing else holding the node, and a 10 s deadline
    failed too, so a longer timeout would not have saved it. Closing the pipeline and
    opening it again cleared it, and the device has streamed 30 fps ever since. Only
    RuntimeError -- librealsense's frame timeout -- is retried; every ValueError raised
    below is deterministic and would just fail again.
    """
    for attempt in (0, 1):
        pipe = rs.pipeline()
        try:
            return _open_d455(pipe, serial, exposure_us=exposure_us, gain=gain,
                              white_balance=white_balance, profile_settings=profile_settings)
        except BaseException as exc:
            try:
                pipe.stop()             # raises when it never started: nothing to undo
            except RuntimeError:
                pass
            if attempt or not isinstance(exc, RuntimeError):
                raise
            print(f"[{serial}] bring-up failed -- {exc}; re-opening once", flush=True)
            time.sleep(2.0)


def _open_d455(pipe, serial: str, *, exposure_us: float | None = None,
               gain: float | None = None, white_balance: float | None = None,
               profile_settings: dict | None = None):
    """Start one D455 on the scan profile and return (pipeline, cam_meta, depth_scale).

    Identical bring-up for the handheld and the fixed overhead camera, so camera_1 is
    described by exactly the same intrinsics/baseline/timestamp conventions as
    camera_0 and the fuser needs no special case for it.
    """
    settings = {} if profile_settings is None else profile_settings
    if not isinstance(settings, dict):
        raise ValueError("Camera profile must be an object")
    allowed = {"color_exposure_option", "color_gain", "white_balance",
               "ir_exposure_option", "ir_gain", "laser_power"}
    if set(settings) - allowed:
        raise ValueError(f"Unknown camera profile keys: {set(settings) - allowed}")
    if ("color_exposure_option" in settings) != ("color_gain" in settings):
        raise ValueError("Color manual profile requires both exposure and gain")
    exposure_us = settings.get("color_exposure_option", exposure_us)
    gain = settings.get("color_gain", gain)
    white_balance = settings.get("white_balance", white_balance)
    if exposure_us is None or gain is None or white_balance is None:
        # 2026-09-23: with colour AE on, the D455 reports neither the exposure nor the
        # gain it chose (option readback and frame metadata alike), and turning AE off
        # restores the stored manual values. The old "meter, then lock" therefore locked
        # the factory 166/64/4600 K, which clipped half of a room view.
        raise ValueError(f"{serial}: colour exposure, gain and white balance must be explicit "
                         f"(--camera-profile, or --color-exposure-option/--gain/--white-balance); "
                         f"the D455 cannot meter them")
    devs = {d.get_info(rs.camera_info.serial_number): d
            for d in rs.context().query_devices()}
    if serial not in devs:
        raise ValueError(f"{serial} is not connected; present: {sorted(devs)}")
    usb = devs[serial].get_info(rs.camera_info.usb_type_descriptor)
    if not usb.startswith("3"):
        raise ValueError(f"{serial} is on USB {usb}; the IR pair + colour at 30 fps "
                         f"needs USB 3 -- use the blue-tab cable/port")
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.infrared, 1, IR_W, IR_H, rs.format.y8, FPS)
    cfg.enable_stream(rs.stream.infrared, 2, IR_W, IR_H, rs.format.y8, FPS)
    # ASIC depth is enabled but never stored: it feeds the live TOO CLOSE warning.
    cfg.enable_stream(rs.stream.depth, IR_W, IR_H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, FPS)
    profile = pipe.start(cfg)
    dev = profile.get_device()

    ds = dev.first_depth_sensor()
    if not ds.supports(rs.option.emitter_enabled):
        raise ValueError(f"{serial}: device does not expose emitter_enabled")
    depth_scale = float(ds.get_depth_scale())
    # The projected dots are the only stereo texture the robot's flat painted shells
    # have. Multiple projectors may alter stereo quality; verify the combined setup
    # against a single-emitter capture if depth artifacts persist.
    ds.set_option(rs.option.emitter_enabled, 1.0)
    rsmeta.set_option_checked(ds, rs.option.laser_power,
                              settings.get("laser_power", ds.get_option_range(rs.option.laser_power).max))
    if ("ir_exposure_option" in settings) != ("ir_gain" in settings):
        raise ValueError("IR manual profile requires both exposure and gain")
    rsmeta.set_option_checked(ds, rs.option.enable_auto_exposure,
                              0 if "ir_exposure_option" in settings else 1)
    if "ir_exposure_option" in settings:
        rsmeta.set_option_checked(ds, rs.option.exposure, settings["ir_exposure_option"])
        rsmeta.set_option_checked(ds, rs.option.gain, settings["ir_gain"])
    for sensor in dev.query_sensors():
        if not sensor.supports(rs.option.global_time_enabled):
            continue
        sensor.set_option(rs.option.global_time_enabled, 1.0)
        if sensor.get_option(rs.option.global_time_enabled) != 1.0:
            raise ValueError(f"{serial}: global_time_enabled would not stay on")
    cs = dev.first_color_sensor()
    # Lock per camera for the scan: auto exposure/WB would change a surface's appearance
    # across viewpoints. A fixed setting can also clip or underexpose part of the orbit:
    # inspect coverage and a neutral patch before adopting the values.
    rng = cs.get_option_range(rs.option.exposure)
    if not np.isfinite(exposure_us) or not rng.min <= exposure_us <= rng.max:
        raise ValueError("Requested color exposure outside sensor range")
    cs.set_option(rs.option.enable_auto_exposure, 0)
    cs.set_option(rs.option.enable_auto_white_balance, 0)
    rsmeta.set_option_checked(cs, rs.option.exposure, exposure_us)
    rsmeta.set_option_checked(cs, rs.option.gain, gain)
    rsmeta.set_option_checked(cs, rs.option.white_balance, white_balance)
    rb, n_px = _bright_rb(pipe)
    got = (float(cs.get_option(rs.option.exposure)), float(cs.get_option(rs.option.gain)),
           float(cs.get_option(rs.option.white_balance)))
    if cs.get_option(rs.option.enable_auto_exposure) != 0:
        raise ValueError(f"{serial}: auto-exposure would not stay off")
    if cs.get_option(rs.option.enable_auto_white_balance) != 0:
        raise ValueError(f"{serial}: auto-white-balance would not stay off")
    print(f"[{serial}] colour LOCKED exposure_option={got[0]:.0f} gain={got[1]:.0f} wb={got[2]:.0f}K; "
          f"white renders at R/B {'n/a' if rb is None else f'{rb:.3f}'} over {n_px:,} px",
          flush=True)

    col = profile.get_stream(rs.stream.color).as_video_stream_profile()
    ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    k_ir, k_rgb = ir1.get_intrinsics(), col.get_intrinsics()
    if (k_ir.width, k_ir.height) != (IR_W, IR_H):
        raise ValueError(f"{serial}: IR negotiated {k_ir.width}x{k_ir.height}")
    if (k_rgb.width, k_rgb.height) != (COLOR_W, COLOR_H):
        raise ValueError(f"{serial}: colour negotiated {k_rgb.width}x{k_rgb.height}")
    baseline = float(np.linalg.norm(
        np.asarray(ir1.get_extrinsics_to(ir2).translation, float)))
    if not 0.05 < baseline < 0.15:
        raise ValueError(f"{serial}: implausible stereo baseline {baseline} m")
    ex = col.get_extrinsics_to(ir1)          # librealsense stores rotation column-major
    T_rgb_to_ir = np.eye(4, dtype=np.float32)
    T_rgb_to_ir[:3, :3] = np.asarray(ex.rotation, float).reshape(3, 3).T
    T_rgb_to_ir[:3, 3] = np.asarray(ex.translation, float)
    meta = {
        "serial": serial,
        "K_ir": np.array([[k_ir.fx, 0, k_ir.ppx], [0, k_ir.fy, k_ir.ppy], [0, 0, 1]],
                         np.float32),
        "K_rgb": np.array([[k_rgb.fx, 0, k_rgb.ppx], [0, k_rgb.fy, k_rgb.ppy],
                           [0, 0, 1]], np.float32),
        "T_rgb_to_ir": T_rgb_to_ir,
        "baseline": baseline,
        "ir_coeffs": list(k_ir.coeffs),
        "rgb_coeffs": list(k_rgb.coeffs),
        "sensor_settings_json": rsmeta.sensor_settings(dev, rs),
        "ir_distortion_model": str(k_ir.model),
        "rgb_distortion_model": str(k_rgb.model),
        "color_exposure_option": got[0],
        "color_gain": got[1],
        "color_white_balance": got[2],
        "color_auto_exposure": 0,
        "color_auto_white_balance": 0,
        "color_white_rb": -1.0 if rb is None else rb,
    }
    return pipe, meta, depth_scale


def pick_device(serial: str | None) -> str:
    devs = {d.get_info(rs.camera_info.serial_number): d for d in rs.context().query_devices()}
    if not devs:
        raise ValueError("no RealSense device connected")
    if serial is not None:
        if serial not in devs:
            raise ValueError(f"serial {serial} not connected; present: {sorted(devs)}")
        chosen = serial
    elif len(devs) == 1:
        chosen = next(iter(devs))
    else:
        raise ValueError(
            f"{len(devs)} devices connected ({sorted(devs)}); unplug the rig camera "
            f"you are NOT holding, or pass --serial")
    usb = devs[chosen].get_info(rs.camera_info.usb_type_descriptor)
    if not usb.startswith("3"):
        raise ValueError(
            f"{chosen} is on USB {usb}; the IR pair + colour at 30 fps needs USB 3 -- "
            f"use the blue-tab cable/port")
    return chosen


class ZenohCam:
    """Latest-frame view of one camera's zenoh topics, shaped like dexcontrol's get_obs.

    Exposes the same ``get_obs(obs_keys, include_timestamp)`` contract the rest of this
    script already uses, so the sampler does not care which cameras come through
    dexcontrol and which come off the wire.
    """

    def __init__(self, base: str, streams: list[str], namespace: str):
        from dexcomm import subscribe                                   # noqa: PLC0415
        from dexcomm.codecs import RGBImageCodec                        # noqa: PLC0415
        self.base, self.namespace = base, namespace
        self._latest: dict[str, tuple[np.ndarray, int] | None] = {s: None for s in streams}
        self._lock = threading.Lock()
        self._subs = []
        for stream in streams:
            def _cb(msg, _s=stream):
                # RGBImageCodec.decode gives Message.data = {"data": HxWx3 uint8,
                # "timestamp_ns", "height", "width", "color_format", "receive_time_ns"}.
                # Prefer the payload's own stamp: it is the capture time, while the
                # Message stamp is when this host saw it.
                payload = getattr(msg, "data", msg)
                if isinstance(payload, dict):
                    arr = payload.get("data")
                    ts = int(payload.get("timestamp_ns")
                             or getattr(msg, "timestamp_ns", -1) or -1)
                else:
                    arr, ts = payload, int(getattr(msg, "timestamp_ns", -1) or -1)
                if arr is None:
                    return
                with self._lock:
                    self._latest[_s] = (np.array(arr, copy=True, order="C"), ts)
            self._subs.append(
                subscribe(f"{namespace}/{base}/{stream}", _cb,
                          decoder=RGBImageCodec.decode))

    def get_obs(self, obs_keys: list[str], include_timestamp: bool = False) -> dict:
        with self._lock:
            snap = {k: self._latest.get(k) for k in obs_keys}
        if any(v is None for v in snap.values()):
            return {}
        return {k: {"data": v[0], "timestamp_ns": v[1]} for k, v in snap.items()}

    def get_camera_info(self, force_refresh: bool = False):
        from dexcomm import call_service                                # noqa: PLC0415
        from dexcomm.codecs import JsonDataCodec                        # noqa: PLC0415
        return call_service(f"{self.namespace}/{self.base}/info", request={}, timeout=2.0,
                            request_encoder=JsonDataCodec.encode,
                            response_decoder=JsonDataCodec.decode)

    def close(self):
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                pass


class ScanWriter(threading.Thread):
    """Appends camera framesets, robot-state samples and head frames to one HDF5.

    Three record kinds on three independent time axes, because they arrive at three
    different rates and pairing them by index would be a lie. They share one clock:
    every record carries a host-epoch ``host_ns`` written by the sampler, and the
    RealSense frames carry the librealsense ``global_time`` stamp, which is already
    host epoch (``global_time_enabled`` is forced on below).
    """

    BASE_KINDS = ("cam", "state") + CAM_KINDS

    def __init__(self, path: Path, q: queue.Queue, compression: str,
                 joint_names: list[str], vel_names: list[str], comp_names: list[str],
                 cam_shapes: dict[str, dict[str, tuple[tuple[int, ...], np.dtype]]],
                 has_chassis: bool, static_n: int = 0):
        super().__init__(daemon=True)
        self.path, self.q, self.error = path, q, None
        self._drained = False
        self.static_n = static_n
        # camera_0 is the handheld one; every fixed camera adds a camera_<i> and its own
        # record kind, so each keeps an independent time axis.
        self.KINDS = self.BASE_KINDS + tuple(f"cam{i}" for i in range(1, static_n + 1))
        self.n = {k: 0 for k in self.KINDS}
        self.partial_path = Path(str(path) + ".partial")
        self.f = h5py.File(self.partial_path, "x")
        self.f.attrs["capture_usable"] = False
        comp = None if compression == "none" else compression
        gi = self.f.create_group("observations/images")
        gm = self.f.create_group("capture_meta")
        gr = self.f.create_group("robot")
        gcam = {kind: self.f.create_group(kind) for kind in cam_shapes}

        spec: dict[str, tuple[str, tuple[int, ...], np.dtype, h5py.Group]] = {
            "camera_0_left_ir": ("cam", (IR_H, IR_W), np.uint8, gi),
            "camera_0_right_ir": ("cam", (IR_H, IR_W), np.uint8, gi),
            "camera_0_color": ("cam", (COLOR_H, COLOR_W, 3), np.uint8, gi),
            "camera_0_intrinsics": ("cam", (3, 3), np.float32, gi),
            "camera_0_rgb_intrinsics": ("cam", (3, 3), np.float32, gi),
            "camera_0_rgb_to_ir_extrinsic": ("cam", (4, 4), np.float32, gi),
            "camera_0_baseline": ("cam", (), np.float32, gi),
            "camera_0_host_ns": ("cam", (), np.int64, gm),
            "camera_0_timestamp_ms": ("cam", (), np.float64, gm),
            "camera_0_frame_numbers": ("cam", (3,), np.int64, gm),
            "host_ns": ("state", (), np.int64, gr),
            "joint_pos": ("state", (len(joint_names),), np.float64, gr),
            "comp_ts_ns": ("state", (len(comp_names),), np.int64, gr),
        }
        for i in range(1, static_n + 1):
            # A FIXED D455. Its own time axis, because it free-runs against camera_0 and
            # pairing them by row index would be a lie; join on camera_*_host_ns, which
            # is host epoch for all of them.
            for suffix, shp, dt, grp in (
                    ("left_ir", (IR_H, IR_W), np.uint8, gi),
                    ("right_ir", (IR_H, IR_W), np.uint8, gi),
                    ("color", (COLOR_H, COLOR_W, 3), np.uint8, gi),
                    ("intrinsics", (3, 3), np.float32, gi),
                    ("rgb_intrinsics", (3, 3), np.float32, gi),
                    ("rgb_to_ir_extrinsic", (4, 4), np.float32, gi),
                    ("baseline", (), np.float32, gi),
                    ("host_ns", (), np.int64, gm),
                    ("timestamp_ms", (), np.float64, gm),
                    ("frame_numbers", (3,), np.int64, gm)):
                spec[f"camera_{i}_{suffix}"] = (f"cam{i}", shp, dt, grp)
        for i in range(static_n + 1):
            kind = "cam" if i == 0 else f"cam{i}"
            for suffix, shape, dtype in rsmeta.specs(3):
                spec[f"camera_{i}_{suffix}"] = (kind, shape, dtype, gm)
            rsmeta.describe(gm, f"camera_{i}", ["left_ir", "right_ir", "color"])
        if vel_names:
            spec["joint_vel"] = ("state", (len(vel_names),), np.float64, gr)
        if has_chassis:
            for name in ("chassis_steer", "chassis_wheel_vel", "chassis_wheel_pos"):
                spec[name] = ("state", (2,), np.float64, gr)
        # Spec keys are "<kind>::<dataset>" so that two cameras publishing the same
        # stream name (all three publish "left_rgb") cannot overwrite each other. Only
        # the part after "::" becomes the dataset name inside the group.
        for kind, shapes in cam_shapes.items():
            g = gcam[kind]
            spec[f"{kind}::host_ns"] = (kind, (), np.int64, g)
            for stream, (shape, dtype) in shapes.items():
                spec[f"{kind}::{stream}"] = (kind, shape, dtype, g)
                spec[f"{kind}::{stream}_ts_ns"] = (kind, (), np.int64, g)

        self.kind_of: dict[str, str] = {}
        self.ds: dict[str, h5py.Dataset] = {}
        for name, (kind, shp, dt, grp) in spec.items():
            key = name.split("::", 1)[1] if "::" in name else name
            self.ds[name] = grp.create_dataset(
                key, shape=(0,) + shp, maxshape=(None,) + shp, dtype=dt,
                chunks=(1,) + shp if shp else (4096,),
                compression=comp if shp and len(shp) >= 2 else None)
            self.kind_of[name] = kind

        sdt = h5py.string_dtype()
        gr.create_dataset("joint_names", data=np.array(joint_names, dtype=object), dtype=sdt)
        gr.create_dataset("comp_names", data=np.array(comp_names, dtype=object), dtype=sdt)
        if vel_names:
            gr.create_dataset("vel_joint_names",
                              data=np.array(vel_names, dtype=object), dtype=sdt)

    def run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    return
                kind, rec = item
                i = self.n[kind]
                for name, val in rec.items():
                    d = self.ds[name]
                    d.resize(i + 1, axis=0)
                    d[i] = val
                self.n[kind] += 1
        except Exception as exc:                      # surfaced by the main loop
            self.error = f"{type(exc).__name__}: {exc}"

    def finish(self):
        """Drain the queue and stop the thread, so the datasets can be read safely.

        h5py is not thread-safe, and the freeze verdict is computed from the joint
        rows that were just written -- so the writer has to be off the file before
        anything reads it back.
        """
        if self.is_alive() or not self._drained:
            if self.is_alive():
                self.q.put(None, timeout=5.0)
                self.join(timeout=30.0)
            if self.is_alive():
                raise RuntimeError("Scan writer did not drain; file is incomplete")
            if self.error:
                raise RuntimeError(f"Scan writer failed: {self.error}")
            self._drained = True

    def close(self, attrs: dict):
        self.finish()
        self.f.attrs["num_frames"] = self.n["cam"]
        for i in range(1, self.static_n + 1):
            self.f.attrs[f"num_camera_{i}_frames"] = self.n[f"cam{i}"]
        self.f.attrs["num_state_samples"] = self.n["state"]
        for kind in CAM_KINDS:
            self.f.attrs[f"num_{kind}_frames"] = self.n[kind]
        for k, v in attrs.items():
            self.f.attrs[k] = v
        self.f.close()
        if attrs.get("capture_usable", False) and not attrs.get("aborted", False):
            if self.path.exists():
                raise FileExistsError(self.path)
            self.partial_path.rename(self.path)


_WARNED_NO_STAMP: set[str] = set()


class RobotLink:
    """Validated access to the robot's joint readback and its head camera.

    Every shape this class hands out is checked against the robot's own joint-name
    lists at construction and again on every sample, so a component that silently
    changes width -- or a head publisher running a different crop than
    ``omniteleop.common.head_camera`` describes -- stops the run instead of writing a
    plausible-looking file.
    """

    def __init__(self, robot, cam_streams: dict[str, list[str]], startup_timeout: float):
        self.robot = robot
        self._state_progress = ProgressGuard()
        self._last_camera_stamp = {}
        self.comp: dict[str, tuple[object, list[str]]] = {}
        for name in STATE_COMPONENTS:
            c = getattr(robot, name, None)
            if c is None:
                continue
            names = list(c.get_joint_name())
            if not names:
                raise ValueError(f"{name}: get_joint_name() returned nothing")
            self.comp[name] = (c, names)
        if not self.comp:
            raise ValueError(
                "robot exposes none of "
                f"{STATE_COMPONENTS} -- is this the right ROBOT_NAME?")
        for need in ("left_arm", "right_arm", "head"):
            if need not in self.comp:
                raise ValueError(f"robot has no {need}; an arm scan needs it")

        self.comp_names = list(self.comp)
        self.joint_names: list[str] = []
        self.slice: dict[str, slice] = {}
        for name, (_, names) in self.comp.items():
            self.slice[name] = slice(len(self.joint_names), len(self.joint_names) + len(names))
            self.joint_names += names
        dup = {n for n in self.joint_names if self.joint_names.count(n) > 1}
        if dup:
            raise ValueError(f"duplicate joint names across components: {sorted(dup)}")

        self._wait(lambda: self._try_pos(), "robot joint readback", startup_timeout)
        for name, (c, names) in self.comp.items():
            q = np.asarray(c.get_joint_pos(), dtype=np.float64).ravel()
            if q.shape != (len(names),):
                raise ValueError(
                    f"{name}: get_joint_pos() is {q.shape}, but get_joint_name() lists "
                    f"{len(names)} joints {names}")

        # Velocity is a bonus channel: it makes "is it still?" a direct measurement
        # instead of a finite difference. Components that do not publish it are
        # recorded as not publishing it -- never as zeros.
        self.vel_comp: list[str] = []
        self.vel_names: list[str] = []
        for name, (c, names) in self.comp.items():
            try:
                v = np.asarray(c.get_joint_vel(), dtype=np.float64).ravel()
            except Exception:
                continue
            if v.shape != (len(names),):
                continue
            self.vel_comp.append(name)
            self.vel_names += names

        self.chassis = getattr(robot, "chassis", None)
        if self.chassis is not None:
            for prop in ("steering_angle", "wheel_velocity", "wheel_encoder_pos"):
                a = np.asarray(getattr(self.chassis, prop), dtype=np.float64).ravel()
                if a.shape != (2,):
                    raise ValueError(f"chassis.{prop} is {a.shape}, expected (2,)")

        self.dead = [j for j in DEAD_JOINTS if j in self.joint_names]
        missing = sorted(set(DEAD_JOINTS) - set(self.dead))
        if missing:
            raise ValueError(
                f"dead joints {missing} are not among this robot's joints "
                f"{self.joint_names} -- fix DEAD_JOINTS before recording")

        self.cam_streams = {k: list(v) for k, v in cam_streams.items() if v}
        self.cams: dict[str, object] = {}
        self.cam_shapes: dict[str, dict[str, tuple[tuple[int, ...], np.dtype]]] = {}
        self.cam_info: dict[str, str] = {}
        self.head_K = None
        for kind in CAM_KINDS:                       # probe in a fixed, printable order
            if kind in self.cam_streams:
                self._probe_cam(kind, self.cam_streams[kind], startup_timeout)

    # -- startup helpers -------------------------------------------------------
    @staticmethod
    def _wait(getter, what: str, timeout: float):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            v = getter()
            if v is not None:
                return v
            time.sleep(0.05)
        raise ValueError(
            f"no {what} within {timeout:.0f}s -- is the robot powered and is this host "
            f"on its zenoh link? (ROBOT_NAME + ~/.dexmate/comm/zenoh/*)")

    def _try_pos(self):
        try:
            for _, (c, names) in self.comp.items():
                if np.asarray(c.get_joint_pos()).size != len(names):
                    return None
        except Exception:
            return None
        return True

    def _probe_cam(self, kind: str, streams: list[str], timeout: float):
        attr = CAM_ATTR[kind]
        if kind in DIRECT_ZENOH:
            ns = os.environ.get("ROBOT_NAME", "dm/vg7ae4b55f3e-1")
            cam = ZenohCam(f"sensors/{attr}", streams, ns)
        else:
            cam = getattr(self.robot.sensors, attr, None)
        if cam is None:
            raise ValueError(
                f"robot.sensors has no {attr} -- start the {kind} publisher "
                f"(scripts/diagnostics/start_robot_cams_tmux.sh) or drop {kind} from "
                f"--head-streams/--wrist-streams")
        self.cams[kind] = cam
        obs = self._wait(lambda: self._try_cam(cam, streams),
                         f"{kind} camera frames", timeout)
        self.cam_shapes[kind] = {s: (a.shape, a.dtype) for s, a in obs.items()}
        if kind == "head":
            # ZED_K describes one specific crop+resize of the head view. The wrists have
            # no such constant, so this check is head-only by construction.
            shapes = self.cam_shapes[kind]
            rgb_hw = shapes["left_rgb"][0][:2] if "left_rgb" in shapes else None
            if rgb_hw is not None:
                if tuple(rgb_hw) != tuple(HEAD_RESIZE_HW):
                    raise ValueError(
                        f"head left_rgb is {rgb_hw} but omniteleop.common.head_camera "
                        f"says HEAD_RESIZE_HW={HEAD_RESIZE_HW}; ZED_K describes that "
                        f"view, not this one. The head frames would be recorded with an "
                        f"intrinsic that does not belong to them -- start the publisher "
                        f"whose --crop/--resize match, or fix the constants.")
                self.head_K = np.asarray(ZED_K, np.float64)
        info = cam.get_camera_info(force_refresh=True)
        if not isinstance(info, dict) or "stereo" not in info:
            raise ValueError(f"{kind}: effective stereo calibration unavailable")
        if kind == "head" and "left_rgb" in streams:
            live_K = np.asarray(info["stereo"]["left_K"], dtype=float)
            if live_K.shape != (3, 3) or not np.isfinite(live_K).all() or min(live_K[0, 0], live_K[1, 1]) <= 0:
                raise ValueError("head: invalid live intrinsics")
            self.head_K = live_K
        self.cam_info[kind] = json.dumps(info)

    def _try_cam(self, cam, streams: list[str]):
        try:
            obs = cam.get_obs(obs_keys=list(streams), include_timestamp=True)
        except Exception:
            return None
        out = {}
        for k in streams:
            v = obs.get(k)
            if v is None:
                return None
            arr = v["data"] if isinstance(v, dict) and "data" in v else v
            if arr is None:
                return None
            out[k] = np.ascontiguousarray(arr)
        return out

    @property
    def head_streams(self) -> list[str]:
        return self.cam_streams.get("head", [])

    @property
    def head_shapes(self) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        return self.cam_shapes.get("head", {})

    @property
    def head_info(self) -> str | None:
        return self.cam_info.get("head")

    # -- per sample ------------------------------------------------------------
    def sample_state(self) -> dict:
        pos = np.empty(len(self.joint_names), np.float64)
        ts = np.empty(len(self.comp_names), np.int64)
        snapshots = {}
        for i, name in enumerate(self.comp_names):
            c, names = self.comp[name]
            q, stamp, snapshot = state_snapshot(c, len(names))
            self._state_progress.check(name, stamp)
            snapshots[name] = snapshot
            pos[self.slice[name]] = q
            ts[i] = stamp
        rec = {"host_ns": time.time_ns(), "joint_pos": pos, "comp_ts_ns": ts}
        if self.vel_comp:
            vel = np.empty(len(self.vel_names), np.float64)
            at = 0
            for name in self.vel_comp:
                c, names = self.comp[name]
                v = np.asarray(snapshots[name]["vel"], dtype=np.float64)
                if v.shape != (len(names),) or not np.isfinite(v).all():
                    raise ValueError(f"{name}: get_joint_vel() became {v.shape}, "
                                     f"expected {(len(names),)}")
                vel[at:at + len(names)] = v
                at += len(names)
            rec["joint_vel"] = vel
        if self.chassis is not None:
            for key, prop in (("chassis_steer", "steering_angle"),
                              ("chassis_wheel_vel", "wheel_velocity"),
                              ("chassis_wheel_pos", "wheel_encoder_pos")):
                a = np.asarray(getattr(self.chassis, prop), dtype=np.float64).ravel()
                if a.shape != (2,) or not np.isfinite(a).all():
                    raise ValueError(f"chassis.{prop} became invalid {a.shape}, expected finite (2,)")
                rec[key] = a
        return rec

    def sample_cam(self, kind: str) -> dict | None:
        """One frameset from camera ``kind``, or None if any stream had nothing new.

        A partial sample is dropped rather than padded: a camera's time axis is only
        useful if every row is a real, simultaneous observation. Each camera keeps its
        own axis -- the head runs at a different rate from the wrists, and pairing them
        by row index would be a lie. Join them on ``host_ns`` (or the per-stream
        ``*_ts_ns`` where the publisher supplies one).
        """
        streams = self.cam_streams[kind]
        try:
            obs = self.cams[kind].get_obs(obs_keys=list(streams),
                                          include_timestamp=True)
        except Exception:
            return None
        rec = {f"{kind}::host_ns": time.time_ns()}
        for k in streams:
            v = obs.get(k)
            if v is None:
                return None
            arr = v["data"] if isinstance(v, dict) and "data" in v else v
            # dexcontrol's zed_camera.get_obs docstring advertises "timestamp", but the
            # Zenoh payload it passes through verbatim carries "timestamp_ns". Reading
            # the documented name yielded None -> -1 for EVERY head frame, and a -1
            # column silently matches to sample 0, so consumers aligned head frames
            # against the wrong joint samples. Prefer the real key; keep the documented
            # one as a fallback in case a publisher emits it.
            stamp = None
            if isinstance(v, dict):
                stamp = v.get("timestamp_ns")
                if stamp is None:
                    stamp = v.get("timestamp")
            if arr is None:
                return None
            arr = np.array(arr, copy=True, order="C")
            want_shape, want_dtype = self.cam_shapes[kind][k]
            if arr.shape != want_shape or arr.dtype != want_dtype:
                raise ValueError(
                    f"{kind} {k} became {arr.shape}/{arr.dtype}, expected "
                    f"{want_shape}/{want_dtype} -- the publisher changed mid-take")
            rec[f"{kind}::{k}"] = arr
            if stamp is None and f"{kind}/{k}" not in _WARNED_NO_STAMP:
                _WARNED_NO_STAMP.add(f"{kind}/{k}")
                print(f"    WARNING: {kind} {k} carries no timestamp; {k}_ts_ns will be "
                      f"missing; this frame set is rejected.", flush=True)
            rec[f"{kind}::{k}_ts_ns"] = int(stamp) if stamp is not None else -1
        stamps = [rec[f"{kind}::{k}_ts_ns"] for k in streams]
        if not validate_stereo_stamps(stamps, self._last_camera_stamp.get(kind)):
            return None
        self._last_camera_stamp[kind] = stamps[0]
        return rec

    def joint_dict(self, pos: np.ndarray) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.joint_names, pos)}

    # -- commanding ------------------------------------------------------------
    def resolve_target(self, comp: str, spec) -> np.ndarray:
        """One pose entry -> the absolute joint vector that will actually be sent.

        Three forms: a ``pose_pool`` name, an absolute vector, or ``{"delta": [...]}``
        relative to the CURRENT measured angles. The delta form is the safe way to
        move an arm that carries a dead joint: a named vendor pose assumes all seven
        joints reach their listed values, and one of the right arm's cannot.
        """
        c, names = self.comp[comp]
        cur = np.asarray(c.get_joint_pos(), np.float64).ravel()
        if cur.shape != (len(names),):
            raise ValueError(f"{comp}: readback {cur.shape}, expected {(len(names),)}")
        if isinstance(spec, dict):
            d = np.asarray(spec["delta"], np.float64).ravel()
            if d.shape != (len(names),):
                raise ValueError(f"pose entry for {comp} has a {d.size}-value delta, "
                                 f"expected {len(names)} {names}")
            q, kind = cur + d, "delta"
        elif isinstance(spec, str):
            q, kind = np.asarray(c.get_predefined_pose(spec), np.float64).ravel(), \
                f"pose_pool[{spec}]"
        else:
            q, kind = np.asarray(spec, np.float64).ravel(), "absolute"
        if q.shape != (len(names),):
            raise ValueError(
                f"pose entry for {comp} has {q.size} values, expected {len(names)} "
                f"{names}")
        substituted = False
        for i, n in enumerate(names):
            if n in self.dead:
                if abs(cur[i] - q[i]) > 1e-9:
                    print(f"    {n} is DEAD: target {q[i]:+.4f} -> measured "
                          f"{cur[i]:+.4f} rad ({np.degrees(cur[i]):+.2f} deg)")
                    substituted = True
                q[i] = cur[i]
        if substituted and kind.startswith("pose_pool"):
            print(f"    WARNING: {comp} {kind} with a dead joint pinned elsewhere is "
                  f"NOT the vendor pose any more -- its clearance is unverified. "
                  f"Prefer a small {{\"delta\": [...]}} entry for this arm.")
        lim = c.joint_pos_limit
        if lim is not None:
            lim = np.asarray(lim, np.float64)
            if lim.shape != (len(names), 2):
                raise ValueError(f"{comp}: joint_pos_limit is {lim.shape}, "
                                 f"expected {(len(names), 2)}")
            out = np.clip(q, lim[:, 0], lim[:, 1])
            for i, n in enumerate(names):
                if abs(out[i] - q[i]) > 1e-9:
                    print(f"    {n} target {q[i]:+.4f} clipped to limit {out[i]:+.4f}")
            q = out
        return q


def load_poses(path: Path | None) -> list[dict]:
    if path is None:
        return [{"label": "as_is"}]
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty JSON list of pose objects")
    out = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ValueError(f"{path}[{i}]: expected an object, got {type(e).__name__}")
        unknown = set(e) - {"label"} - set(COMMANDABLE)
        if unknown:
            raise ValueError(f"{path}[{i}]: unknown keys {sorted(unknown)}; "
                             f"allowed: 'label' + {list(COMMANDABLE)}")
        for k in COMMANDABLE:
            if k not in e:
                continue
            v = e[k]
            if isinstance(v, dict):
                if set(v) != {"delta"} or not isinstance(v["delta"], list):
                    raise ValueError(f"{path}[{i}].{k}: an object value must be exactly "
                                     f'{{"delta": [ ... ]}}, got {v}')
                v = v["delta"]
            elif isinstance(v, str):
                continue
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                raise ValueError(f"{path}[{i}].{k}: expected a list of numbers, a "
                                 f'pose_pool name, or {{"delta": [...]}}, got {e[k]!r}')
        label = str(e.get("label", f"pose_{i}"))
        if not label or any(ch in label for ch in "/\\ \t"):
            raise ValueError(f"{path}[{i}]: label {label!r} must be a non-empty "
                             f"filename-safe string")
        out.append({"label": label, **{k: e[k] for k in COMMANDABLE if k in e}})
    labels = [e["label"] for e in out]
    dup = {l for l in labels if labels.count(l) > 1}
    if dup:
        raise ValueError(f"{path}: duplicate pose labels {sorted(dup)}")
    return out


def freeze_report(pos: np.ndarray, names: list[str], tol: float):
    """Peak-to-peak per joint over a recorded segment, and whoever exceeded ``tol``.

    Peak-to-peak, not variance: a joint that is merely republishing a stale value
    (``R_arm_j1``--``j4`` publish at ~1 Hz) reads as perfectly still, which is the
    correct answer for a joint that is not moving, while a joint that actually drifted
    shows the full excursion no matter how it was sampled.
    """
    return freeze_ranges(pos, names, tol)


POSE_REACH_TOL = 0.05      # rad; worst-joint miss that is worth warning about
HOME_STEP = 0.01           # rad per substep, matching wbc_robot_home.ARM_HOME_STEP
HOME_WAIT = 0.1            # s per substep, matching wbc_robot_home._interp_joint


def interp_to(link, resolved: dict, step: float, wait_time: float, abort, stop) -> None:
    """Glide each component to its target in small joint-space steps.

    This is ``omniteleop.wbc_robot_home._interp_joint``'s pattern, which is what
    actually moves this robot. Two things make it the right call here rather than
    ``robot.move_to_joint_pos``:

      * The 0.5 motion plugin stalls on the head -- it raises ``settling stalled: head
        stopped converging (0.542 rad residual) after 2000ms with noprogress`` while
        the head never leaves where it drooped to. Component-level ``set_joint_pos``
        moves it to ~1e-4 rad of target every time.
      * Stepping along the straight joint-space line is EXACTLY the path
        ``check_scan_collisions.py`` vets between waypoints. The motion plugin's
        internal trajectory is not, and never was the thing that got checked.

    ``wait_time`` is deprecated at the component level (removal in dexcontrol 0.6.0) but
    is what ``home_to_nominal`` still uses; revisit when that call site does.
    """
    for name, target in resolved.items():
        comp = link.comp[name][0]
        cur = np.asarray(comp.get_joint_pos(), np.float64).ravel()
        wp = np.asarray(target, np.float64).ravel()
        n = max(1, int(np.ceil(np.max(np.abs(wp - cur)) / step)))
        for i in range(n):
            if abort.is_set() or stop.is_set():
                return
            q = cur + (wp - cur) * (i + 1) / n
            comp.set_joint_pos(q.tolist(), wait_time=wait_time, exit_on_reach=True)




COLLISION_CHECKER = Path(__file__).resolve().parent / "diagnostics/check_scan_collisions.py"
CHECK_GROUPS = ("torso", "head", "left_arm", "right_arm")


def gate_collision_check(link, args, resolved=None) -> None:
    """Refuse --execute unless the ACTUAL commanded path is self-collision free.

    ``interp_to`` glides one component at a time, so the trajectory is an axis-aligned
    staircase whose corners a simultaneous interpolation never visits. The checker is
    run here in its ``sequential`` mode, from the state the robot is in RIGHT NOW --
    not from a historical pose or a previously observed clearance -- and a nonzero exit
    blocks the run. ``--skip-collision-check`` overrides, loudly.
    """
    if not COLLISION_CHECKER.is_file():
        raise ValueError(f"collision checker missing: {COLLISION_CHECKER}")
    measured = {}
    for comp in CHECK_GROUPS:
        if comp not in link.comp:
            raise ValueError(f"cannot gate --execute: robot link has no {comp!r}")
        c, names = link.comp[comp]
        q = np.asarray(c.get_joint_pos(), np.float64).ravel()
        if q.shape != (len(names),) or not np.all(np.isfinite(q)):
            raise ValueError(f"cannot gate --execute: {comp} readback {q!r} is not a "
                             f"finite {len(names)}-vector")
        measured[comp] = q.tolist()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(measured, fh)
        measured_path = fh.name
    pose_path = None
    if resolved is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump([{"label": "next_resolved_pose", **{k: v.tolist() for k, v in resolved.items()}}], fh)
            pose_path = fh.name
    cmd = [sys.executable, str(COLLISION_CHECKER), "--poses", pose_path or str(args.poses),
           "--measured", measured_path, "--path", "sequential"]
    if args.collision_config:
        cmd += ["--config", args.collision_config]
    if args.collision_strict:
        cmd += ["--strict"]
    print("\n" + "=" * 78)
    print("checking the SEQUENTIAL commanded path for self-collision from the pose the")
    print("robot is in now (--skip-collision-check overrides):")
    print("  " + " ".join(cmd), flush=True)
    try:
        rc = subprocess.run(cmd, cwd=str(COLLISION_CHECKER.parents[2])).returncode
    finally:
        os.unlink(measured_path)
        if pose_path is not None:
            os.unlink(pose_path)
    if rc != 0:
        raise ValueError(
            f"collision check failed (exit {rc}); refusing to --execute. Fix the poses "
            f"or the starting state, or pass --skip-collision-check if you have vetted "
            f"the motion another way.")
    for comp, before in measured.items():
        current, _, _ = state_snapshot(link.comp[comp][0], len(before))
        if np.max(np.abs(current - np.asarray(before))) > .01:
            raise ValueError(f"{comp} changed during collision checking; refresh the plan")
    print("collision check passed; proceeding to move the robot.")
    print("=" * 78, flush=True)

def record_pose(pipe, link, args, out_path: Path, pose: dict, serial: str,
                depth_scale: float, cam_meta: dict, stop: threading.Event,
                abort: threading.Event, static: list | None = None) -> dict:
    """Record one frozen pose into ``out_path``. Returns its manifest row."""
    q: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
    writer = ScanWriter(out_path, q, args.compression, link.joint_names,
                        link.vel_names, link.comp_names, link.cam_shapes,
                        link.chassis is not None, len(static or []))
    writer.start()

    state_stop = threading.Event()
    misses = {}
    for kind in link.cam_streams:
        misses[kind] = misses[f"{kind}_try"] = 0

    def sampler():
        period = 1.0 / args.state_hz
        every = (max(1, int(round(args.state_hz / args.cam_hz)))
                 if link.cam_streams else 0)
        k = 0
        last_camera_ok = {kind: time.monotonic() for kind in link.cam_streams}
        while not state_stop.is_set():
            tick = time.perf_counter()
            try:
                q.put(("state", link.sample_state()), timeout=2.0)
                if every and k % every == 0:
                    # All cameras are sampled on the same tick, so a head frame and a
                    # wrist frame from one tick are as simultaneous as this loop can
                    # make them -- which is what the wrist->head solve needs.
                    for kind in link.cam_streams:
                        misses[f"{kind}_try"] += 1
                        rec = link.sample_cam(kind)
                        if rec is None:
                            misses[kind] += 1
                            if time.monotonic() - last_camera_ok[kind] > max(1.0, 3.0 / args.cam_hz):
                                raise ValueError(f"{kind}: no fresh paired frames within camera liveness budget")
                        else:
                            last_camera_ok[kind] = time.monotonic()
                            q.put((kind, rec), timeout=2.0)
            except Exception as exc:
                writer.error = writer.error or f"sampler: {type(exc).__name__}: {exc}"
                return
            k += 1
            sleep = period - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)

    sampler_t = threading.Thread(target=sampler, daemon=True)
    sampler_t.start()

    # Drop whatever librealsense buffered while the robot was moving and the prompts
    # were up. A single stale frame from a moving scene is enough to send the
    # photometric odometry off, and it would be indistinguishable from bad walking.
    for _ in range(64):
        if not pipe.poll_for_frames():
            break
    for _ in range(3):
        pipe.wait_for_frames()

    print(f"\nRECORDING '{pose['label']}' -> {out_path.name}   "
          f"({args.scan_time:.0f}s, or Ctrl-C to end this pose)")
    print("walk slowly; watch SHARPNESS -- it dips when you turn too fast\n")
    t0 = time.time()
    next_static = t0
    n_frames, last_report, last_n = 0, t0, 0
    try:
        while not stop.is_set() and not abort.is_set():
            if time.time() - t0 >= args.scan_time:
                break
            fs = pipe.wait_for_frames()
            f1, f2, fc = fs.get_infrared_frame(1), fs.get_infrared_frame(2), fs.get_color_frame()
            fd = fs.get_depth_frame()
            if not f1 or not f2 or not fc or not fd:
                raise ValueError(f"incomplete frameset at frame {n_frames}")
            ir = np.array(f1.get_data(), copy=True)
            rec = {
                "camera_0_left_ir": ir,
                "camera_0_right_ir": np.array(f2.get_data(), copy=True),
                "camera_0_color": np.array(fc.get_data(), copy=True),
                "camera_0_intrinsics": cam_meta["K_ir"],
                "camera_0_rgb_intrinsics": cam_meta["K_rgb"],
                "camera_0_rgb_to_ir_extrinsic": cam_meta["T_rgb_to_ir"],
                "camera_0_baseline": cam_meta["baseline"],
                "camera_0_host_ns": int(round(fc.get_timestamp() * 1e6)),
                "camera_0_timestamp_ms": float(fc.get_timestamp()),
                "camera_0_frame_numbers": np.array(
                    [f1.get_frame_number(), f2.get_frame_number(), fc.get_frame_number()],
                    np.int64),
            }
            rec.update({f"camera_0_{key}": value for key, value in
                        rsmeta.frame_metadata([f1, f2, fc], rs).items()})
            if static and time.time() >= next_static:
                next_static = time.time() + 1.0 / args.static_hz
                for i, (spipe, smeta) in enumerate(static, start=1):
                    sfs = spipe.poll_for_frames()
                    if not sfs:
                        continue
                    s1 = sfs.get_infrared_frame(1)
                    s2 = sfs.get_infrared_frame(2)
                    sc = sfs.get_color_frame()
                    if not (s1 and s2 and sc):
                        continue
                    q.put((f"cam{i}", {
                        **{f"camera_{i}_{key}": value for key, value in
                           rsmeta.frame_metadata([s1, s2, sc], rs).items()},
                        f"camera_{i}_left_ir": np.array(s1.get_data(), copy=True),
                        f"camera_{i}_right_ir": np.array(s2.get_data(), copy=True),
                        f"camera_{i}_color": np.array(sc.get_data(), copy=True),
                        f"camera_{i}_intrinsics": smeta["K_ir"],
                        f"camera_{i}_rgb_intrinsics": smeta["K_rgb"],
                        f"camera_{i}_rgb_to_ir_extrinsic": smeta["T_rgb_to_ir"],
                        f"camera_{i}_baseline": smeta["baseline"],
                        f"camera_{i}_host_ns": int(round(sc.get_timestamp() * 1e6)),
                        f"camera_{i}_timestamp_ms": float(sc.get_timestamp()),
                        f"camera_{i}_frame_numbers": np.array(
                            [s1.get_frame_number(), s2.get_frame_number(),
                             sc.get_frame_number()], np.int64),
                    }), timeout=2.0)
            try:
                q.put(("cam", rec), timeout=2.0)
            except queue.Full:
                raise ValueError(
                    f"writer fell {QUEUE_DEPTH} frames behind -- disk cannot keep up. "
                    f"Write to the local NVMe, not a USB drive or network mount.")
            if writer.error:
                raise ValueError(f"writer failed: {writer.error}")
            n_frames += 1

            now = time.time()
            if now - last_report >= 1.0:
                sharp = float(np.abs(np.diff(
                    ir[180:540:4, 320:960:4].astype(np.int16), axis=1)).mean())
                z = np.asanyarray(fd.get_data())[::8, ::8]
                z = z[z > 0]
                zmed = float(np.median(z)) * depth_scale if z.size else float("nan")
                near = "  << TOO CLOSE, step back" if zmed < args.near_warn else ""
                fps = (n_frames - last_n) / (now - last_report)
                print(f"  {n_frames:6d} frames  {now - t0:6.1f}s  {fps:5.1f} fps  "
                      f"queue {q.qsize():2d}/{QUEUE_DEPTH}  SHARPNESS {sharp:5.2f}  "
                      f"DEPTH {zmed:4.2f} m{near}",
                      flush=True)
                last_report, last_n = now, n_frames
    except BaseException:
        state_stop.set()
        sampler_t.join(timeout=5.0)
        try:
            writer.close({"purpose": "handheld_robot_arm_scan", "aborted": True,
                          "pose_label": pose["label"]})
        except Exception:
            pass
        raise
    state_stop.set()
    sampler_t.join(timeout=5.0)
    writer.finish()

    n_state = writer.n["state"]
    n_cam = {kind: writer.n[kind] for kind in link.cam_streams}
    n_head = n_cam.get("head", 0)
    if n_state == 0:
        writer.close({"purpose": "handheld_robot_arm_scan", "aborted": True,
                      "pose_label": pose["label"]})
        raise ValueError(f"{out_path.name}: no robot-state samples were recorded")
    pos = np.asarray(writer.ds["joint_pos"][:n_state], np.float64)
    ptp, bad = freeze_report(pos, link.joint_names, args.freeze_tol)
    frozen = not bad
    q_med = np.median(pos, axis=0)

    base_ptp = None
    if link.chassis is not None:
        enc = np.asarray(writer.ds["chassis_wheel_pos"][:n_state], np.float64)
        base_ptp = float((enc.max(0) - enc.min(0)).max())

    usable = frozen and n_frames >= 2 and all(n >= 2 for n in n_cam.values())
    usable = usable and all(writer.n[f"cam{i}"] >= 2 for i in range(1, len(static or []) + 1))
    usable = usable and (base_ptp is None or (np.isfinite(base_ptp) and base_ptp <= args.freeze_tol))
    writer.close({
        "capture_integrity_version": 1,
        "capture_usable": bool(usable),
        "cam_ids": ["camera_0"] + [f"camera_{i}" for i in range(1, len(static or []) + 1)],
        "serial": serial,
        **{k: v for i, (_p, sm) in enumerate(static or [], start=1) for k, v in (
            (f"camera_{i}_serial", sm["serial"]),
            (f"camera_{i}_role", "fixed, does not move within or between poses"),
            (f"camera_{i}_ir_coeffs", sm["ir_coeffs"]),
            (f"camera_{i}_rgb_coeffs", sm["rgb_coeffs"]),
            (f"camera_{i}_exposure_option", sm["color_exposure_option"]),
            (f"camera_{i}_gain", sm["color_gain"]),
            (f"camera_{i}_white_balance", sm["color_white_balance"]),
            (f"camera_{i}_white_rb", sm["color_white_rb"]),
            (f"camera_{i}_sensor_settings_json", sm["sensor_settings_json"]),
            (f"camera_{i}_ir_distortion_model", sm["ir_distortion_model"]),
            (f"camera_{i}_rgb_distortion_model", sm["rgb_distortion_model"]),
            (f"camera_{i}_hz", args.static_hz))},
        "purpose": "handheld_robot_arm_scan",
        "pose_label": pose["label"],
        "pose_index": pose["_index"],
        "scan_note": ("camera_0_extrinsics is absent ON PURPOSE: the per-frame pose "
                      "is unsolved. The scan fuser estimates the trajectory with "
                      "RGB-D odometry over FoundationStereo depth; fit_robot_to_scan.py "
                      "then registers the URDF posed at robot/joint_pos into it."),
        "fps": FPS,
        "emitter_enabled": 1,
        "streams": ["left_ir", "right_ir", "color"],
        "compression": args.compression,
        "timestamp_domain": "global_time (host epoch, ms)",
        "ir_coeffs": cam_meta["ir_coeffs"],
        "rgb_coeffs": cam_meta["rgb_coeffs"],
        **{k: cam_meta[k] for k in ("color_exposure_option", "sensor_settings_json", "ir_distortion_model", "rgb_distortion_model", "color_gain",
                                    "color_white_balance", "color_auto_exposure",
                                    "color_auto_white_balance", "color_white_rb")},
        "robot_frozen": frozen,
        "freeze_tol_rad": args.freeze_tol,
        "joint_pos_median": q_med,
        "joint_ptp": ptp,
        "dead_joints": list(link.dead),
        "head_intrinsic_source": ("live publisher stereo.left_K"
                                  if link.head_K is not None else "none"),
        **({"head_K": link.head_K} if link.head_K is not None else {}),
        **({"head_crop_tblr": np.asarray(HEAD_CROP_TBLR, np.int64),
            "head_resize_hw": np.asarray(HEAD_RESIZE_HW, np.int64)}
           if link.head_K is not None else {}),
        **{f"{kind}_camera_info": info for kind, info in link.cam_info.items()},
        **{f"{kind}_streams": ",".join(st)
           for kind, st in link.cam_streams.items()},
    })
    if writer.error:
        raise ValueError(f"writer failed: {writer.error}")

    dur = time.time() - t0
    actual_path = out_path if out_path.exists() else writer.partial_path
    size_gb = actual_path.stat().st_size / 1e9
    print(f"\nwrote {actual_path.name}  ({n_frames} frames, {size_gb:.2f} GB, {dur:.1f}s, "
          f"{n_frames / max(dur, 1e-9):.1f} fps)")
    for i, (_p, sm) in enumerate(static or [], start=1):
        print(f"  camera_{i}     {writer.n[f'cam{i}']} framesets @ ~{args.static_hz:g} Hz"
              f"  ({sm['serial']})")
    print(f"  robot state {n_state} samples @ ~{args.state_hz:g} Hz")
    for kind in link.cam_streams:
        print(f"  {kind:12s} {n_cam[kind]}/{misses[f'{kind}_try']} framesets"
              + (f"  ({misses[kind]} dropped)" if misses[kind] else ""))
    if frozen:
        print(f"  FROZEN OK: worst joint moved {np.degrees(ptp.max()):.4f} deg "
              f"(tol {np.degrees(args.freeze_tol):.4f} deg)")
    else:
        print("  " + "!" * 70)
        print(f"  ROBOT MOVED during '{pose['label']}' -- this scan's joint vector is "
              f"not one pose:")
        for n, p in bad[:8]:
            print(f"    {n:14s} peak-to-peak {np.degrees(p):8.4f} deg")
        print("  The atlas will still fuse, but fit_robot_to_scan.py has no single "
              "true pose to fit.")
        print("  Re-record this pose, or drop it.")
        print("  " + "!" * 70)
    if base_ptp is not None and base_ptp > args.freeze_tol:
        print(f"  WARNING: chassis wheel encoders moved {base_ptp:.4f} "
              f"(rad or m) -- the base was not parked")

    return {
        "index": pose["_index"],
        "label": pose["label"],
        "file": actual_path.name,
        "frames": n_frames,
        "state_samples": int(n_state),
        "head_frames": int(n_head),
        **{f"{kind}_frames": int(v) for kind, v in n_cam.items() if kind != "head"},
        "seconds": round(dur, 2),
        "gigabytes": round(size_gb, 3),
        "frozen": bool(frozen),
        "usable": bool(usable),
        "worst_joint_ptp_deg": float(np.degrees(ptp.max())),
        "worst_joint": link.joint_names[int(np.argmax(ptp))],
        "joints": {n: round(float(v), 6) for n, v in zip(link.joint_names, q_med)},
    }


def fit_command(row: dict, take_dir: Path) -> str:
    """The fit_robot_to_scan.py invocation this pose is now ready for."""
    j = row["joints"]
    need = ([f"torso_j{i + 1}" for i in range(3)] + [f"head_j{i + 1}" for i in range(3)]
            + [f"L_arm_j{i + 1}" for i in range(7)] + [f"R_arm_j{i + 1}" for i in range(7)])
    missing = [n for n in need if n not in j]
    if missing:
        return (f"# fit_robot_to_scan.py needs {missing}, which this robot did not "
                f"report -- fill them in by hand")

    def vec(prefix, n):
        return " ".join(f"{j[f'{prefix}{i + 1}']:.6f}" for i in range(n))

    return (f"python fit_robot_to_scan.py \\\n"
            f"    --atlas {take_dir}/fused_{row['label']} \\\n"
            f"    --torso {vec('torso_j', 3)} \\\n"
            f"    --head {vec('head_j', 3)} \\\n"
            f"    --left-arm {vec('L_arm_j', 7)} \\\n"
            f"    --right-arm {vec('R_arm_j', 7)} \\\n"
            f"    --out-dir {take_dir}/fit_{row['label']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="take directories are created under here (required unless "
                         "--list-poses)")
    ap.add_argument("--poses", type=Path, default=None,
                    help="JSON list of pose objects; default records one 'as_is' take")
    ap.add_argument("--execute", action="store_true",
                    help="COMMAND the robot to each pose. Without this the poses are "
                         "printed and you pose the robot yourself between takes.")
    ap.add_argument("--list-poses", action="store_true",
                    help="print the robot's own pose_pool names per component and exit")
    ap.add_argument("--static-serial", default=None,
                    help="comma-separated serials of D455s that stay FIXED, each seeing "
                         "the robot from one unmoving viewpoint at every pose. They "
                         "become camera_1, camera_2, ... in the order given, each on "
                         "its own time axis; join to camera_0 on "
                         "capture_meta/camera_*_host_ns")
    ap.add_argument("--static-hz", type=float, default=2.0,
                    help="camera_1 sampling rate. The robot is frozen and the camera "
                         "does not move, so full rate would only store the same view "
                         "hundreds of times; a few Hz still gives frames to average")
    ap.add_argument("--serial", default=None,
                    help="D455 to use; required when more than one is connected")
    ap.add_argument("--scan-time", type=float, default=75.0,
                    help="seconds of orbit per pose")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="dwell after a move before the freeze pre-check")
    ap.add_argument("--move-time", type=float, default=6.0,
                    help="unused since dexcontrol 0.5; --execute paces by --home-step")
    ap.add_argument("--max-vel", type=float, default=0.35,
                    help="unused since dexcontrol 0.5; see --home-step/--home-wait")
    ap.add_argument("--home-step", type=float, default=HOME_STEP,
                    help="--execute: joint-space substep of the glide (rad). Smaller is "
                         "slower and smoother; matches wbc_robot_home ARM_HOME_STEP")
    ap.add_argument("--home-wait", type=float, default=HOME_WAIT,
                    help="--execute: per-substep wait (s); exit_on_reach usually ends "
                         "each substep sooner")
    ap.add_argument("--freeze-tol", type=float, default=0.002,
                    help="max peak-to-peak per joint (rad) for a take to count as "
                         "frozen; 0.002 rad = 0.115 deg")
    ap.add_argument("--precheck", type=float, default=2.0,
                    help="seconds of stillness required BEFORE recording starts")
    ap.add_argument("--state-hz", type=float, default=20.0,
                    help="robot joint sampling rate")
    ap.add_argument("--cam-hz", type=float, default=2.0,
                    help="robot-camera sampling rate, shared by the head and both "
                         "wrists (the scene is frozen, so this is for redundancy and "
                         "exposure variety, not motion)")
    ap.add_argument("--head-streams", default="left_rgb,depth",
                    help="comma-separated head streams to record")
    ap.add_argument("--no-head", action="store_true",
                    help="do not record head-camera frames")
    ap.add_argument("--wrist-streams", default="left_rgb,right_rgb",
                    help="comma-separated streams to record from BOTH wrist cameras; "
                         "these are what make the wrist->head extrinsic solvable")
    ap.add_argument("--no-wrist", action="store_true",
                    help="do not record wrist-camera frames")
    ap.add_argument("--near-warn", type=float, default=0.5,
                    help="live TOO CLOSE warning threshold (m); D455 stereo dies inside "
                         "~0.5 m")
    ap.add_argument("--skip-collision-check", action="store_true",
                    help="run --execute WITHOUT vetting the sequential commanded path "
                         "for self-collision. Only if you have vetted the motion "
                         "another way.")
    ap.add_argument("--collision-config", default=None,
                    help="wbik yaml for the collision check (default: the checker's own).")
    ap.add_argument("--collision-strict", action="store_true",
                    help="also block --execute when the path is merely below safe_dist.")
    ap.add_argument("--color-exposure-option", type=float, default=None,
                    help="colour exposure (raw SDK option units) for every D455 the profile does "
                         "not cover. Same value is used for every pose, so takes stay comparable.")
    ap.add_argument("--camera-profile", type=Path,
                    help="JSON keyed by D455 serial; values in SDK option units. Every D455 "
                         "needs colour exposure, gain and white balance from here or the flags: "
                         "the D455 cannot meter them.")
    ap.add_argument("--gain", type=float, default=None,
                    help="colour gain for every D455 the profile does not cover.")
    ap.add_argument("--white-balance", type=float, default=None,
                    help="white balance (Kelvin) for every D455 the profile does not cover; "
                         "validate it on a neutral patch. Bright scene pixels are not a neutral "
                         "reference.")
    ap.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--startup-timeout", type=float, default=15.0)
    args = ap.parse_args()
    profiles = json.loads(args.camera_profile.read_text()) if args.camera_profile else {}
    if not isinstance(profiles, dict):
        raise ValueError("Camera profile must be a JSON object keyed by serial")

    if args.precheck < 0:
        raise ValueError(f"--precheck must be >= 0, got {args.precheck}")
    for name, v in (("scan-time", args.scan_time), ("state-hz", args.state_hz),
                    ("cam-hz", args.cam_hz), ("freeze-tol", args.freeze_tol),
                    ("home-step", args.home_step), ("home-wait", args.home_wait),
                    ("static-hz", args.static_hz)):
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"--{name} must be finite and > 0, got {v}")
    if args.cam_hz > args.state_hz:
        raise ValueError(f"--cam-hz {args.cam_hz} exceeds --state-hz {args.state_hz}; "
                         f"camera frames are sampled on the state tick")
    head_streams = [] if args.no_head else [s for s in args.head_streams.split(",") if s]
    wrist_streams = [] if args.no_wrist else [s for s in args.wrist_streams.split(",") if s]
    cam_streams: dict[str, list[str]] = {}
    if head_streams:
        cam_streams["head"] = head_streams
    if wrist_streams:
        cam_streams["left_wrist"] = list(wrist_streams)
        cam_streams["right_wrist"] = list(wrist_streams)

    from dexcontrol.robot import Robot                                  # noqa: E402

    print("connecting to the robot ...", flush=True)
    if cam_streams:
        # dexcontrol ships these sensors disabled, so a bare Robot() leaves
        # robot.sensors without them. Enable them before constructing the robot,
        # the same way wbc_vr_robot.py does for --record.
        from dexcontrol.core.config import get_robot_config              # noqa: PLC0415
        configs = get_robot_config()
        # Only the dexcontrol-backed cameras need enabling in the robot config; the
        # DIRECT_ZENOH ones are subscribed to by hand and would just add a subscriber
        # for a stream name that is never published.
        via_dexcontrol = [k for k in cam_streams if k not in DIRECT_ZENOH]
        missing = sorted({CAM_ATTR[k] for k in via_dexcontrol} - set(configs.sensors))
        if missing:
            raise SystemExit(
                f"record_arm_scan was asked to record {missing} but the robot config "
                f"has no such sensor -- pass --no-head and/or --no-wrist to scan "
                f"without them."
            )
        for kind in via_dexcontrol:
            configs.sensors[CAM_ATTR[kind]].enabled = True
        robot = Robot(configs=configs)
    else:
        robot = Robot()
        # dexcontrol's idle monitor pauses every subscriber not read for 5 s (arms, chassis,
    # hands...). The pause re-configures the process-wide Zenoh session and stalls camera
    # delivery ~300 ms once, ~6 s after Robot(). Keep everything subscribed instead
    # (measured 2026-09-02 in record_head_sweep_video.py: 340 ms -> 37 ms worst latency).
    robot.set_subscription_policy("always_on")
    try:
        link = RobotLink(robot, cam_streams, args.startup_timeout)

        if args.list_poses:
            for name in COMMANDABLE:
                if name not in link.comp:
                    continue
                c, names = link.comp[name]
                pool = getattr(c, "_pose_pool", None)
                print(f"{name:10s} joints {names}")
                if pool:
                    for pname, pv in pool.items():
                        print(f"    {pname:20s} {np.round(np.asarray(pv, float), 4).tolist()}")
                else:
                    print("    (no pose_pool)")
            return 0

        if args.out_dir is None:
            raise ValueError("--out-dir is required (unless --list-poses)")
        poses = load_poses(args.poses)
        for i, p in enumerate(poses):
            p["_index"] = i
        if args.execute and not any(set(p) & set(COMMANDABLE) for p in poses):
            raise ValueError("--execute was passed but no pose entry names a component "
                             "to command; drop --execute or fill in --poses")

        q0 = link.sample_state()["joint_pos"]
        jd = link.joint_dict(q0)
        print(f"robot     {len(link.joint_names)} joints over {link.comp_names}")
        print(f"dead      {link.dead} = "
              + ", ".join(f"{n} {jd[n]:+.4f} rad ({np.degrees(jd[n]):+.2f} deg)"
                          for n in link.dead)
              + "  (readback trusted, never commanded)")
        if link.vel_names:
            print(f"velocity  published by {link.vel_comp}")
        else:
            print("velocity  NOT published by any component -- freeze is judged from "
                  "position peak-to-peak alone")
        for kind, shapes in link.cam_shapes.items():
            note = ""
            if kind == "head":
                note = (f"   K from ZED_K (fx {link.head_K[0, 0]:.2f}, "
                        f"cx {link.head_K[0, 2]:.2f})" if link.head_K is not None
                        else "   no left_rgb -> no head intrinsic recorded")
            print(f"{kind:10s}" + ", ".join(f"{k}{v[0]} {v[1]}"
                                            for k, v in shapes.items()) + note)
        if not link.cam_shapes:
            print("cameras   none recorded (--no-head and --no-wrist)")

        serial = pick_device(args.serial)
        selected_serials = {serial, *[x.strip() for x in (args.static_serial or "").split(",") if x.strip()]}
        if set(profiles) - selected_serials:
            raise ValueError(f"Profile contains unselected serials: {set(profiles) - selected_serials}")
        n = 0
        while (args.out_dir / f"take_{n}").exists():
            n += 1
        take_dir = args.out_dir / f"take_{n}"
        args.out_dir.mkdir(parents=True, exist_ok=True)

        free_gb = shutil.disk_usage(args.out_dir).free / 1e9
        mb_s = MB_PER_FRAME * FPS
        n_static = len([x for x in (args.static_serial or "").split(",") if x.strip()])
        mb_s += MB_PER_FRAME * args.static_hz * n_static
        budget_gb = mb_s * args.scan_time * len(poses) / 1000
        print(f"device    {serial} (USB 3)")
        print(f"output    {take_dir}")
        print(f"plan      {len(poses)} pose(s) x {args.scan_time:.0f}s "
              f"~= {budget_gb:.1f} GB of {free_gb:.0f} GB free"
              + ("   [--execute: THE ROBOT WILL MOVE]" if args.execute else
                 "   [poses are printed; you move the robot]"))
        if free_gb < args.min_free_gb or free_gb < budget_gb * 1.2:
            raise ValueError(f"only {free_gb:.1f} GB free for a {budget_gb:.1f} GB plan "
                             f"(--min-free-gb {args.min_free_gb})")

        # Until every camera is streaming nothing is committed: a failed bring-up
        # used to exit with the cameras opened before it still running, and with an
        # empty take_N already claiming the next number on disk.
        pipe = None
        static: list = []
        try:
            pipe, cam_meta, depth_scale = open_d455(
                serial, exposure_us=args.color_exposure_option, gain=args.gain,
                white_balance=args.white_balance, profile_settings=profiles.get(serial, {}))
            baseline = cam_meta["baseline"]

            static_sns = [x.strip() for x in (args.static_serial or "").split(",") if x.strip()]
            if len(set(static_sns)) != len(static_sns):
                raise ValueError(f"--static-serial repeats a serial: {static_sns}")
            if serial in static_sns:
                raise ValueError(
                    f"--static-serial lists {serial}, which is the handheld camera "
                    f"(--serial). A camera cannot be both.")
            present = {d.get_info(rs.camera_info.serial_number)
                       for d in rs.context().query_devices()}
            missing = [x for x in static_sns if x not in present]
            if missing:
                raise ValueError(f"--static-serial {missing} not connected; "
                                 f"present: {sorted(present)}")
            for i, sn in enumerate(static_sns, start=1):
                spipe, smeta, _ = open_d455(
                    sn, exposure_us=args.color_exposure_option, gain=args.gain,
                    white_balance=args.white_balance, profile_settings=profiles.get(sn, {}))
                static.append((spipe, smeta))
                print(f"camera_{i}  {sn} FIXED, baseline "
                      f"{smeta['baseline'] * 1000:.1f} mm, sampled at {args.static_hz:g} Hz")
        except BaseException:
            if pipe is not None:
                pipe.stop()
            for spipe, _sm in static:
                spipe.stop()
            raise

        take_dir.mkdir(parents=True)

        print(f"baseline  {baseline * 1000:.1f} mm   settling auto-exposure "
              f"({SETTLE_S:g}s)...")
        deadline = time.perf_counter() + SETTLE_S
        while time.perf_counter() < deadline:
            pipe.wait_for_frames()
            for spipe, _sm in static:
                spipe.poll_for_frames()

        stop, abort = threading.Event(), threading.Event()
        last_sigint = [0.0]

        def on_sigint(*_):
            now = time.time()
            if now - last_sigint[0] < 2.0:
                abort.set()
                print("\n  aborting the run (second Ctrl-C)", flush=True)
            else:
                stop.set()
                print("\n  ending this pose (Ctrl-C again within 2 s to abort the run)",
                      flush=True)
            last_sigint[0] = now

        signal.signal(signal.SIGINT, on_sigint)

        if args.execute and not args.skip_collision_check:
            gate_collision_check(link, args)

        rows: list[dict] = []
        try:
            for pose in poses:
                if abort.is_set():
                    break
                label = pose["label"]
                targets = {k: pose[k] for k in COMMANDABLE if k in pose}
                print("\n" + "=" * 78)
                print(f"POSE {pose['_index']}/{len(poses) - 1}  '{label}'"
                      + (f"   sets {sorted(targets)}" if targets else
                         "   (records the robot as it stands)"))
                if targets:
                    resolved = {k: link.resolve_target(k, v) for k, v in targets.items()}
                    for k, v in resolved.items():
                        print(f"    {k:10s} " + " ".join(f"{x:+.4f}" for x in v))
                    if args.execute:
                        print(f"    gliding to the pose in {args.home_step:g} rad steps "
                              f"(Ctrl-C stops) ...", flush=True)
                        if not args.skip_collision_check:
                            gate_collision_check(link, args, resolved)
                        interp_to(link, resolved, args.home_step, args.home_wait,
                                  abort, stop)
                        # The freeze check proves the robot was STILL, not that it
                        # ARRIVED. A stalled move is frozen at the wrong pose.
                        reached = {k: np.asarray(link.comp[k][0].get_joint_pos(),
                                                 np.float64).ravel()
                                   for k in resolved}
                        worst, where = 0.0, ""
                        for k, v in resolved.items():
                            e = float(np.abs(reached[k] - v).max())
                            if e > worst:
                                worst, where = e, k
                        if worst > POSE_REACH_TOL:
                            print(f"    WARNING: {where} is {worst:.4f} rad from its "
                                  f"target after the move -- this take will be frozen "
                                  f"at a DIFFERENT pose than the file names. The "
                                  f"recorded joint vector is still the truth.",
                                  flush=True)
                    else:
                        input("    put the robot in this pose, then press Enter ")
                elif not args.execute:
                    input("    press Enter when the robot is posed and still ")

                print(f"    settling {args.settle:g}s ...", flush=True)
                # NOT time.sleep(args.settle): PEP 475 makes time.sleep() resume after a
                # signal handler that does not raise, so on_sigint's Event was set but a
                # long --settle kept sleeping and Ctrl-C looked dead. Poll the events so
                # the settle stays interruptible.
                t_settle = time.perf_counter() + args.settle
                while time.perf_counter() < t_settle:
                    if abort.is_set() or stop.is_set():
                        print("    settle interrupted", flush=True)
                        break
                    time.sleep(0.05)
                if abort.is_set():
                    break
                if stop.is_set():
                    stop.clear()
                    print("    skipping this pose (Ctrl-C during settle)", flush=True)
                    rows.append({"index": pose["_index"], "label": label, "file": None,
                                 "frozen": False, "skipped": "Ctrl-C during settle"})
                    continue

                # Pre-check: refuse to spend a minute of scanning on a pose that is
                # still drifting. Cheap here, unrecoverable afterwards.
                pre = []
                t_end = time.perf_counter() + args.precheck
                while time.perf_counter() < t_end:
                    pre.append(link.sample_state()["joint_pos"])
                    time.sleep(1.0 / args.state_hz)
                bad = []
                if pre:
                    _, bad = freeze_report(np.asarray(pre, np.float64),
                                           link.joint_names, args.freeze_tol)
                elif args.precheck > 0:
                    raise ValueError(
                        f"--precheck {args.precheck}s produced no samples at "
                        f"--state-hz {args.state_hz}")
                if bad:
                    print(f"    NOT STILL after {args.settle:g}s: "
                          + ", ".join(f"{n} {np.degrees(p):.3f} deg" for n, p in bad[:4]))
                    print("    skipping this pose -- fix it and re-run, or raise "
                          "--settle / --freeze-tol")
                    rows.append({"index": pose["_index"], "label": label, "file": None,
                                 "frozen": False, "skipped": "not still at precheck"})
                    continue

                stop.clear()
                out_path = take_dir / f"pose_{pose['_index']:02d}_{label}.hdf5"
                rows.append(record_pose(pipe, link, args, out_path, pose, serial,
                                        depth_scale, cam_meta, stop, abort,
                                        static))
        finally:
            pipe.stop()
            for spipe, _sm in static:
                spipe.stop()

        manifest = {
            "take_dir": str(take_dir),
            "serial": serial,
            "created_unix_ns": time.time_ns(),
            "joint_names": link.joint_names,
            "components": link.comp_names,
            "vel_components": link.vel_comp,
            "dead_joints": list(link.dead),
            "dead_joint_measured_rad": {n: float(jd[n]) for n in link.dead},
            "cam_streams": {k: list(v) for k, v in link.cam_streams.items()},
            "head_streams": head_streams,
            "head_intrinsic": (link.head_K.tolist() if link.head_K is not None else None),
            "head_crop_tblr": list(HEAD_CROP_TBLR),
            "head_resize_hw": list(HEAD_RESIZE_HW),
            "freeze_tol_rad": args.freeze_tol,
            "executed": bool(args.execute),
            "scan_time_s": args.scan_time,
            "poses": rows,
        }
        (take_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        good = [r for r in rows if r.get("usable") and r.get("file")]
        print("\n" + "=" * 78)
        print(f"take {take_dir}   {len(good)}/{len(poses)} pose(s) usable")
        for r in rows:
            if not r.get("file"):
                print(f"  [skip] {r['label']:16s} {r.get('skipped', '')}")
                continue
            flag = "ok  " if r["frozen"] else "MOVED"
            print(f"  [{flag}] {r['label']:16s} {r['frames']:5d} frames  "
                  f"{r['gigabytes']:5.2f} GB  worst {r['worst_joint']} "
                  f"{r['worst_joint_ptp_deg']:.4f} deg")
        if good:
            r = good[0]
            print("\nnext, per pose (a robot at ~0.6-1.2 m wants a finer voxel and "
                  "every frame):")
            print(f"  python fuse_scan.py --scan {take_dir}/{r['file']} \\\n"
                  f"      --out-dir {take_dir}/fused_{r['label']} \\\n"
                  f"      --every 1 --voxel 0.002 --min-depth 0.35 --max-depth 1.2")
            print(fit_command(r, take_dir))
        print("=" * 78)
        return 0 if good and len(good) == len(poses) else 1
    finally:
        try:
            robot.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
