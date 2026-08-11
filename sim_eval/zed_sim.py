"""Optional ZED-camera realism for sim: make the head/wrist images match what the robot publishes.

**Everything here is opt-in and OFF by default** (`ZedSimOptions`), so the baseline pipeline
renders exactly as it did before. Turn switches on individually to close a specific sim/real gap.

The real observation is not "whatever the camera renders" -- it is the output of a specific
pipeline (`tests/test_head_zedx_depth.py`): a ZED X Mini captures SVGA, the SDK computes NEURAL
depth, then BOTH streams are cropped to `HEAD_CROP_TBLR` and resized to `HEAD_RESIZE_HW`
robot-side (RGB `INTER_AREA`, depth `INTER_NEAREST`) before anything downstream sees them.

Every number below was MEASURED on the 41 real takes in ``~/Dexmate/data/box2cloth/raw_data``:

* **`match_zed_fov` -- intrinsics were the big gap.** The real head view is 102.5 deg x 75.8 deg
  with a 1.2 pixel aspect (`ZED_K`: fx 128.36, fy 154.04) and an off-centre principal point
  (165.05, 127.55). OmniGibson's default vega camera renders 63.4 deg x 49.7 deg with square
  pixels and a centred principal point -- barely 40% of the real horizontal field. The 1.2 aspect
  is NOT a weird sensor: the raw SVGA frame has square pixels (`HEAD_BASE_K`, fx == fy == 385.09)
  and the anisotropy comes entirely from the 960x600 -> 320x240 resize.
  **Cost of enabling it:** the wider field puts FEWER pixels on the workspace, so the ManiFlow
  workspace crop has to grow or the cloud starves below the 1024-point FPS budget. The crop then
  has to span the EEF TARGETS, not just the objects (the grasped one is carried back toward the
  robot): measured `--crop-min 0.35 -0.52 0.55 --crop-max 1.65 0.46 1.04` for mobilepickplace,
  against `0.80 -0.30 0.60 .. 1.90 0.70 1.15` for the default narrow field.

* **`axial_noise_mm_per_m2` -- leave it at 0.** Local plane-fit roughness on the real depth is
  0.49 mm median (0.63 mm with the flatness filter fully open, so it is not a selection
  artifact) -- against a 0.289 mm floor from the uint16-millimetre storage quantisation alone.
  NEURAL depth is smoother than the format it is stored in, so injecting noise would move sim
  AWAY from real. The knob exists only because a cheaper `depth_mode` (ULTRA/PERFORMANCE) would
  genuinely be noisy.

* **`depth_dropout` -- what IS different.** Real depth is 1.06-1.22% invalid (zero) against sim's
  0.08%, and it is structured, not salt-and-pepper: 13% of invalid pixels sit on a depth
  discontinuity, the rest cluster on surfaces the stereo matcher struggles with. The two
  mechanisms modelled here -- occlusion boundaries and grazing incidence -- are geometric, so
  they reproduce the measured spatial bias (real bottom-16-rows 2.00% invalid vs top-16-rows
  0.17%: the tabletop seen edge-on) without painting in a hand-tuned mask. Fitted result:
  1.12% invalid, 12% of it on edges. Sim's own "ray escaped the scene" pixels are excluded from
  that budget -- in Rs_int those are the window panes, where a real ZED also returns nothing.

* **`wrist_zedm_fov`.** The real wrist is a ZED Mini at HD720 published with NO crop and an
  anisotropic resize straight to 320x240 -- so it is horizontally SQUASHED (16:9 -> 4:3), as
  `tests/test_wrist_zedm_depth.py`'s own `resize_h` docstring warns. Rendering 16:9 and squashing
  it the same way reproduces that. The focal length is the ZED Mini HD720 spec FOV (90 deg H),
  not a measured calibration: the wrist stream is a policy image input only, never unprojected.

Frame/convention facts live in `obs_pipeline.py`; this module only shapes pixels.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import pathlib

import cv2
import numpy as np

# The crop/resize geometry and ZED_K must stay in lock-step with the real publisher, so load them
# from the real module rather than copying the numbers. `omniteleop.common.__init__` pulls loguru
# (absent from the behavior env), so bypass the package and load the leaf file directly -- it
# imports nothing but numpy.
_HEAD_CAMERA_PY = pathlib.Path("/home/yixuan/omniteleop/src/omniteleop/common/head_camera.py")
_spec = importlib.util.spec_from_file_location("_ot_head_camera", _HEAD_CAMERA_PY)
_hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc)

HEAD_BASE_K = _hc.HEAD_BASE_K              # raw SVGA intrinsic, square pixels
HEAD_CROP_TBLR = _hc.HEAD_CROP_TBLR        # crop applied to the raw SVGA frame
HEAD_RESIZE_HW = _hc.HEAD_RESIZE_HW        # (H, W) published to the policy
ZED_K = _hc.ZED_K                          # intrinsic of the published view
crop_resize_intrinsics = _hc.crop_resize_intrinsics

# Raw capture geometry of the head ZED X Mini (SVGA), the frame HEAD_CROP_TBLR indexes into.
SVGA_HW = (600, 960)

# An OmniGibson camera's principal point is always the image centre, but the ZED's is at
# (495.14, 318.89) -- (+15.1, +18.9) px off centre. So render a canvas whose CENTRE is the ZED's
# principal point, then take the top-left SVGA window out of it: the crop shifts the principal
# point back to exactly where the ZED has it, with no aperture-offset trickery. Rounding the
# canvas to whole pixels costs 0.14 px at SVGA (0.045 px after the resize).
ZED_RENDER_HW = (2 * int(round(float(HEAD_BASE_K[1, 2]))),
                 2 * int(round(float(HEAD_BASE_K[0, 2]))))       # (638, 990)
ZED_WRIST_RENDER_HW = (720, 1280)                                # ZED Mini HD720
WRIST_HFOV_DEG = 90.0

# What the pipeline rendered before any of this -- also what it renders with the switches off.
DEFAULT_RENDER_HW = (240, 320)
DEFAULT_WRIST_RENDER_HW = (240, 320)

# --- depth-sensor behaviour, all measured on box2cloth --------------------------------------
DEPTH_MIN_M = 0.1        # test_head_zedx_depth.depth_min (SDK depth_minimum_distance)
DEPTH_MAX_M = 8.0        # test_head_zedx_depth.depth_max

# Head-camera NEAR CLIP, which is NOT the same thing as the ZED's depth_minimum_distance.
# Vega's own chest yoke (`arm_center`) sits ~0.106 m in front of and below `zed_depth_frame`, so
# at a 0.10 m near plane it STRADDLES the clip and the renderer fills the lower-left corner with
# an unshaded pure-black rectangle (measured 3041 px, RGB <= 8, depth pinned at exactly 0.100).
# It grows with the gaze tilt (271 px at head_j3 -22.6 deg -> 2919 px at -44.5 deg), so it is a
# clipping artifact, not the chest: the real robot sees a LIT chest there, which sim cannot
# reproduce at this near plane anyway.
#
# Measured clip sweep at the head-down pose (head_j3 -47.4 deg), largest black blob / min depth
# anywhere in frame:  0.10 -> 3041 px / 0.100 m | 0.12 -> 2041 px / 0.120 m |
#                     0.15 -> 72 px / 0.390 m   | 0.20 and 0.30 -> identical to 0.15.
# 0.15 m culls the chest cleanly and costs NOTHING: there is no geometry at all between 0.15 and
# 0.390 m, so no real observation is lost and the renders at 0.15/0.20/0.30 are indistinguishable.
NEAR_CLIP_M = 0.15
EDGE_STEP_M = 0.05       # depth step counted as an occlusion boundary (the 5 cm probe that found
                         # 13% of real invalid pixels sitting on one)
EDGE_P = 0.10            # ...but only this fraction of boundary pixels actually drop: NEURAL
                         # fills most occlusion shadows, and a deterministic band came out 17x
                         # too wide (2.45% of the frame vs the ~0.14% real edge share).
GRAZE_P = 0.55           # dropout probability as a surface turns fully edge-on
GRAZE_EXP = 25.0         # how sharply it switches on; high -> only near-tangent surfaces drop
BLOB_PX = 6              # dropout correlation length at SVGA (~2 px in the published view)

# Measured on box2cloth: total 1.06% invalid, of which ~13% sits on a depth discontinuity.
REAL_INVALID_FRACTION = 0.0106
REAL_INVALID_ON_EDGES = 0.13


@dataclasses.dataclass
class ZedSimOptions:
    """Which pieces of ZED realism to apply. All off = the pre-existing sim behaviour."""

    match_zed_fov: bool = False
    """Render the ZED's raw SVGA geometry and apply the real publisher's crop+resize, so the
    recorded intrinsic is ZED_K (102.5 x 75.8 deg, pixel aspect 1.2) instead of OmniGibson's
    63.4 x 49.7 deg. Needs a WIDER ManiFlow workspace crop -- see the module docstring."""

    depth_dropout: bool = True
    """Invalidate depth at occlusion boundaries and on grazing surfaces, matching the real
    sensor's 1.06% invalid rate (sim is otherwise 0.08%)."""

    wrist_zedm_fov: bool = False
    """Render the wrist at ZED Mini HD720 geometry and squash 16:9 -> 4:3 like the publisher."""

    axial_noise_mm_per_m2: float = 0.0
    """sigma = c * z^2 additive depth noise. Measured ~0.2 mm/m^2 on the real sensor, i.e. BELOW
    the 0.289 mm storage-quantisation floor -- leave at 0 unless modelling a noisier depth_mode."""

    @staticmethod
    def add_cli(parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group(
            "ZED camera realism (depth dropout on by default; geometry unchanged)")
        g.add_argument("--match-zed-fov", action=argparse.BooleanOptionalAction, default=False,
                       help="render the real ZED field of view + crop/resize chain (recorded "
                            "intrinsic becomes ZED_K). Needs a wider zarr crop.")
        g.add_argument("--depth-dropout", action=argparse.BooleanOptionalAction, default=True,
                       help="model the real sensor's ~1.1%% invalid-depth pixels (edges + "
                            "grazing incidence)")
        g.add_argument("--wrist-zedm-fov", action=argparse.BooleanOptionalAction, default=False,
                       help="render the wrist at ZED Mini HD720 and squash 16:9 -> 4:3")
        g.add_argument("--axial-noise", type=float, default=0.0, metavar="MM_PER_M2",
                       help="additive depth noise sigma = c*z^2. Real is below the storage "
                            "quantisation floor, so 0 is the faithful value (default 0)")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ZedSimOptions":
        return cls(match_zed_fov=bool(getattr(args, "match_zed_fov", False)),
                   depth_dropout=bool(getattr(args, "depth_dropout", True)),
                   wrist_zedm_fov=bool(getattr(args, "wrist_zedm_fov", False)),
                   axial_noise_mm_per_m2=float(getattr(args, "axial_noise", 0.0)))

    def summary(self) -> str:
        on = [n for n, v in (("match_zed_fov", self.match_zed_fov),
                             ("depth_dropout", self.depth_dropout),
                             ("wrist_zedm_fov", self.wrist_zedm_fov),
                             ("axial_noise", self.axial_noise_mm_per_m2 > 0)) if v]
        return ", ".join(on) if on else "none (OmniGibson defaults)"


def head_render_hw(opts: ZedSimOptions) -> tuple[int, int]:
    """Render size the head VisionSensor must be CONSTRUCTED with (see configure_head_camera)."""
    return ZED_RENDER_HW if opts.match_zed_fov else DEFAULT_RENDER_HW


def wrist_render_hw(opts: ZedSimOptions) -> tuple[int, int]:
    return ZED_WRIST_RENDER_HW if opts.wrist_zedm_fov else DEFAULT_WRIST_RENDER_HW


def configure_head_camera(sensor, opts: ZedSimOptions) -> np.ndarray:
    """Set the head camera's field of view; return the achieved render intrinsic.

    Only the aperture is touched -- the resolution must already be `head_render_hw(opts)`,
    because a VisionSensor's render product is sized at load time (its `image_width` setter is a
    no-op headless, where there is no viewport). `VegaOGEnv` therefore constructs it that way.
    """
    if opts.match_zed_fov:
        want = head_render_hw(opts)
        got = (sensor.image_height, sensor.image_width)
        if got != want:
            raise ValueError(
                f"head camera renders {got[1]}x{got[0]}, expected {want[1]}x{want[0]}. The render "
                f"product is sized at load time, so pass obs_hw=zed_sim.head_render_hw(opts) to "
                "VegaOGEnv."
            )
        # fx = focal_length * width / horizontal_aperture  (USD pinhole)
        sensor.horizontal_aperture = (
            float(sensor.focal_length) * want[1] / float(HEAD_BASE_K[0, 0]))
    return _np(sensor.intrinsic_matrix).reshape(3, 3).astype(np.float64)


def configure_wrist_camera(sensor, opts: ZedSimOptions) -> None:
    """Set the wrist camera to the ZED Mini's HD720 field of view (no-op unless enabled)."""
    if not opts.wrist_zedm_fov:
        return
    w = sensor.image_width
    fx = (w / 2.0) / np.tan(np.radians(WRIST_HFOV_DEG) / 2.0)
    sensor.horizontal_aperture = float(sensor.focal_length) * w / fx


def head_intrinsic(render_k: np.ndarray, opts: ZedSimOptions) -> np.ndarray:
    """Intrinsic of the PUBLISHED head view, given the raw render's K.

    With `match_zed_fov` the SVGA window is cut from the render's top-left, which leaves the
    principal point where it is, so the SVGA intrinsic IS the render intrinsic; the published one
    then follows the real crop+resize exactly. Otherwise the render is already the published view.
    """
    k = np.asarray(render_k, dtype=np.float64)
    if not opts.match_zed_fov:
        return k.astype(np.float32)
    return crop_resize_intrinsics(k, HEAD_CROP_TBLR, HEAD_RESIZE_HW).astype(np.float32)


def head_frame(rgb_render: np.ndarray, depth_render_m: np.ndarray, rng: np.random.Generator,
               render_k: np.ndarray, opts: ZedSimOptions) -> tuple[np.ndarray, np.ndarray]:
    """Raw render -> the published `(head_left_rgb uint8, head_depth uint16 mm)` pair.

    Order matches the robot: the SDK's invalid pixels exist at full resolution and are only then
    cropped/resized (`INTER_NEAREST`, so a dropout survives as a whole output pixel rather than
    being averaged away), which is why the artifacts are applied before the resize.
    """
    rgb = np.asarray(rgb_render)[..., :3]
    depth = np.asarray(depth_render_m, dtype=np.float64)
    k = np.asarray(render_k, dtype=np.float64)

    if opts.match_zed_fov:                       # cut the raw SVGA window out of the canvas
        svga_h, svga_w = SVGA_HW
        rgb = rgb[:svga_h, :svga_w]
        depth = depth[:svga_h, :svga_w]

    if opts.depth_dropout or opts.axial_noise_mm_per_m2 > 0.0:
        depth = _zed_depth_artifacts(depth, rng, k, opts)

    if opts.match_zed_fov:
        rgb = _crop_resize(np.ascontiguousarray(rgb), HEAD_CROP_TBLR, HEAD_RESIZE_HW,
                           cv2.INTER_AREA)
        depth = _crop_resize(depth, HEAD_CROP_TBLR, HEAD_RESIZE_HW, cv2.INTER_NEAREST)
    return np.ascontiguousarray(rgb).astype(np.uint8), _to_uint16_mm(depth)


def wrist_frame(rgb_render: np.ndarray, opts: ZedSimOptions) -> np.ndarray:
    """Raw render -> the published wrist image (16:9 squashed to 4:3 when enabled)."""
    rgb = np.ascontiguousarray(np.asarray(rgb_render)[..., :3])
    if not opts.wrist_zedm_fov:
        return rgb.astype(np.uint8)
    h, w = DEFAULT_WRIST_RENDER_HW
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.uint8)


# --- internals -------------------------------------------------------------------------------

def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _crop_resize(img, crop_tblr, out_hw, interpolation):
    """`img[top:bottom, left:right]` then resize to (h, w) -- the real publisher's helper."""
    top, bottom, left, right = crop_tblr
    cropped = img[top:bottom, left:right]
    h, w = out_hw
    return cv2.resize(cropped, (w, h), interpolation=interpolation)


def _to_uint16_mm(depth_m):
    """Metres -> uint16 millimetres, invalid -> 0 (wbc_vr_robot._grab_head_images)."""
    mm = np.where(np.isfinite(depth_m), np.asarray(depth_m, dtype=np.float64) * 1000.0, 0.0)
    return np.clip(mm, 0.0, 65535.0).astype(np.uint16)


def _incidence_cos(depth_m, k):
    """|cos| between each pixel's viewing ray and its local surface normal.

    1.0 face-on, 0.0 edge-on. Computed from the depth map itself (the same information the
    stereo matcher has), so it needs no scene knowledge.
    """
    h, w = depth_m.shape
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = (u - k[0, 2]) / k[0, 0] * depth_m
    y = (v - k[1, 2]) / k[1, 1] * depth_m
    pts = np.stack([x, y, depth_m], axis=-1)
    du = np.gradient(pts, axis=1)
    dv = np.gradient(pts, axis=0)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    r = pts / np.maximum(np.linalg.norm(pts, axis=-1, keepdims=True), 1e-12)
    return np.abs(np.sum(n * r, axis=-1))


def _blob_field(rng, shape, blob_px):
    """Uniform[0,1) noise correlated over ~`blob_px`, so dropouts cluster like the real ones.

    Nearest-neighbour upsampling of a low-res uniform field keeps the marginal EXACTLY uniform,
    so a threshold at p still drops p of the pixels -- a smoothed field would not.
    """
    h, w = shape
    small = rng.random((max(1, h // blob_px), max(1, w // blob_px)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _zed_depth_artifacts(depth_m, rng, k, opts):
    """Apply the measured ZED depth behaviour; invalid pixels come back as NaN."""
    z = np.asarray(depth_m, dtype=np.float64).copy()
    finite = np.isfinite(z)
    zf = np.where(finite, z, 0.0)

    if opts.axial_noise_mm_per_m2 > 0.0:
        z = z + rng.normal(0.0, opts.axial_noise_mm_per_m2 * 1e-3 * zf ** 2)
    if not opts.depth_dropout:
        return z

    bad = ~finite | (zf < DEPTH_MIN_M) | (zf > DEPTH_MAX_M)

    # Occlusion boundaries. The dropout lands on the FAR side of a depth step -- that is the
    # surface hidden from the other eye, so it has no correspondence to match against. Only
    # EDGE_P of those pixels actually drop; NEURAL recovers the rest.
    edge = np.zeros_like(bad)
    for axis in (0, 1):
        step = np.abs(np.diff(zf, axis=axis)) > EDGE_STEP_M
        far_second = np.diff(zf, axis=axis) > 0.0        # the deeper of the two pixels
        lo = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        hi = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        edge[hi] |= step & far_second
        edge[lo] |= step & ~far_second
    bad |= edge & (_blob_field(rng, zf.shape, BLOB_PX) < EDGE_P)

    # Grazing incidence: a surface turning edge-on gives the matcher almost no parallax to work
    # with, which is what makes the real tabletop drop out along its far edge (measured: real
    # bottom-16-rows 2.00% invalid vs top-16-rows 0.17%).
    p = GRAZE_P * (1.0 - _incidence_cos(zf, k)) ** GRAZE_EXP
    bad |= _blob_field(rng, zf.shape, BLOB_PX) < p

    z[bad] = np.nan
    return z
