"""Offline temporal alignment for a recorded VR teleop episode.

Two outputs, from the leader-side + follower-side HDF5s of one episode:

1. Diagnostics — joins the two files by ``vr_publish_ns`` (lambda clock,
   carried in both), applies the dexmate→lambda offset from
   ``/meta/clock/query_results``, and emits a per-channel lag CSV + histogram
   PNG so the latency budget for each data path is visible at a glance.

2. Aligned episode — resamples every channel onto one shared timeline and
   writes ``raw_data/episode_<N>.hdf5`` in the flat "old" training format
   (the format ``leader/vr_reader_old.py`` recorded directly: the new leader
   HDF5 minus the ``timestamps/`` and ``meta/`` subtrees). Anchor = camera
   capture time; obs/joint comes from the denser follower stream; action is
   re-timed to its follower ``t_apply_ns``; extrinsic is recomputed by FK on
   the aligned joints. See ``_build_aligned_episode`` for the full per-channel
   policy.

Usage::

    python scripts/align_timestamps.py --episode_id 2
    python scripts/align_timestamps.py

By default this processes all episodes under ``/home/yixuan/Dexmate/data``:
``leader/episode_<N>.hdf5`` joined with ``follower/episode_<N>_follower.hdf5``.

Explicit ``--leader``/``--follower`` path mode is still available for ad-hoc files.

Output (per episode):
  * <root_dir>/raw_data/episode_<N>.hdf5   aligned episode, old training format
  * <leader_basename>_lag.csv              per-frame lag table (ms)
  * <leader_basename>_lag.png              per-channel lag histograms

Clock convention (see /home/yixuan/.claude/.../project_dexcontrol_ntp_query.md):
  offset = dexmate − lambda → lambda_ns = dexmate_ns − offset_ns
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Optional

import h5py
import numpy as np


_DEXMATE_CHANNELS = {
    "obs/joint/right_arm",
    "obs/joint/left_arm",
    "obs/joint/head",
    "obs/joint/torso",
    "obs/gripper/right",
    "obs/gripper/left",
    "obs/images/left_rgb",
    "obs/images/right_rgb",
    "obs/images/depth",
}


def _load_ts_tree(h5: h5py.File, prefix: str = "timestamps") -> dict[str, np.ndarray]:
    """Flatten /timestamps/* into a {dotted/path: int64 array} dict."""
    out: dict[str, np.ndarray] = {}
    if prefix not in h5:
        return out

    def walk(group: h5py.Group, dotted: str) -> None:
        for k, v in group.items():
            sub = f"{dotted}/{k}" if dotted else k
            if isinstance(v, h5py.Group):
                walk(v, sub)
            else:
                out[sub] = np.asarray(v[...]).astype(np.int64)

    walk(h5[prefix], "")
    return out


def _read_clock_offset_ns(h5: h5py.File) -> int:
    """Read offset_ns from /meta/clock/query_results; 0 if absent or empty."""
    path = "meta/clock/query_results"
    if path not in h5:
        return 0
    arr = h5[path][...]
    if arr.size == 0:
        return 0
    # Columns: [query_lambda_ns, offset_ns, rtt_ns]; use the first sample.
    return int(arr[0, 1])


def _read_source_map(h5: h5py.File) -> dict[str, str]:
    path = "meta/clock/source_per_channel"
    if path not in h5:
        return {}
    raw = h5[path][()]
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    return json.loads(bytes(raw).decode("utf-8"))


def _to_lambda_ns(dexmate_ns: np.ndarray, offset_ns: int) -> np.ndarray:
    """Convert dexmate-clock ns array to lambda-clock ns, preserving -1 sentinel."""
    mask_unknown = dexmate_ns < 0
    out = dexmate_ns - offset_ns
    out[mask_unknown] = dexmate_ns[mask_unknown]
    return out


def _episode_id_from_leader_path(leader_path: pathlib.Path) -> int:
    m = re.match(r"episode_(\d+)\.hdf5$", leader_path.name)
    if not m:
        raise SystemExit(
            f"Cannot infer episode id from leader filename {leader_path.name!r}."
        )
    return int(m.group(1))


def _leader_path_for_episode(root_dir: pathlib.Path, episode_id: int) -> pathlib.Path:
    name = f"episode_{episode_id}.hdf5"
    # `leader/` first: `raw_data/` now holds this script's *output* (the aligned
    # episode), so it must never be picked up as an input leader file.
    for dirname in ("leader", "raw_data"):
        path = root_dir / dirname / name
        if path.exists():
            return path
    return root_dir / "leader" / name


def _follower_path_for_episode(root_dir: pathlib.Path, episode_id: int) -> pathlib.Path:
    return root_dir / "follower" / f"episode_{episode_id}_follower.hdf5"


def _resolve_follower_path(
    leader_path: pathlib.Path,
    override: Optional[str],
    root_dir: pathlib.Path,
) -> pathlib.Path:
    if override:
        return pathlib.Path(override)
    return _follower_path_for_episode(root_dir, _episode_id_from_leader_path(leader_path))


def _discover_episode_ids(root_dir: pathlib.Path) -> list[int]:
    ids: list[int] = []
    for dirname in ("raw_data", "leader"):
        for path in (root_dir / dirname).glob("episode_*.hdf5"):
            m = re.match(r"episode_(\d+)\.hdf5$", path.name)
            if m:
                ids.append(int(m.group(1)))
    return sorted(set(ids))


def _match_to_first_follower(
    leader_pub_ns: np.ndarray, follower_pub_ns: np.ndarray
) -> np.ndarray:
    """For each leader publish_ns, index of the FIRST follower row with the same
    vr_publish_ns; -1 if no match. Both arrays must be already int64 ns.
    """
    out = np.full(leader_pub_ns.shape, -1, dtype=np.int64)
    # Build a hash: publish_ns → first row index in follower
    seen: dict[int, int] = {}
    for i, ts in enumerate(follower_pub_ns):
        ts_int = int(ts)
        if ts_int not in seen:
            seen[ts_int] = i
    for j, ts in enumerate(leader_pub_ns):
        idx = seen.get(int(ts))
        if idx is not None:
            out[j] = idx
    return out


# ── Aligned-episode export ─────────────────────────────────────────────────────
#
# Resamples a timestamp-instrumented leader episode (+ its follower file) into
# the flat "old" training format that `vr_reader_old.py` used to record
# directly — the new leader HDF5 minus the `timestamps/` and `meta/` subtrees,
# with every channel resampled onto one shared timeline.
#
# Anchor:      camera capture time (`timestamps/obs/images/left_rgb`, dexmate
#              clock). Images are kept verbatim; everything else is interpolated
#              onto this timeline. Output row count == leader frame count.
# obs/joint/*: taken from the follower file (denser stream), linearly
#              interpolated on its per-group `obs/joint_ts/*`.
# obs/gripper: from the leader file, linearly interpolated on
#              `timestamps/obs/gripper/*`. `left` is all-NaN when FC03 never
#              replied (no valid timestamps to interpolate against).
# action/*:    from the leader file, but re-timed — each action's timeline is
#              its follower `t_apply_ns` (when it actually took effect on the
#              robot), matched via `vr_publish_ns`, then linearly interpolated
#              onto the camera timeline.
# extrinsic:   recomputed by FK on the *aligned* torso+head joints, so the
#              camera pose is consistent with the camera frame it pairs with.

_RAW_DATA_DIRNAME = "raw_data"

# Exact leaf set of the vr_reader_old.py recording format (the export target).
_OLD_FORMAT_KEYS = {
    "timestamp_ns",
    "action/joint/left_arm", "action/joint/right_arm", "action/joint/head",
    "action/joint/chassis_vx", "action/joint/chassis_vy", "action/joint/chassis_wz",
    "action/gripper/left", "action/gripper/right",
    "obs/joint/left_arm", "obs/joint/right_arm", "obs/joint/head", "obs/joint/torso",
    "obs/gripper/left", "obs/gripper/right",
    "obs/images/left_rgb", "obs/images/right_rgb", "obs/images/depth",
    "obs/images/intrinsic", "obs/images/extrinsic",
}


def _sorted_dedup(xp: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort ``(xp, fp)`` by ``xp`` ascending; on exact ``xp`` ties keep the last.

    ``np.interp`` requires a strictly increasing sample timeline; gripper-status
    and joint-state timestamps can arrive slightly out of order or repeat, so
    every source channel is run through this before interpolation. ``fp`` may be
    1-D ``(M,)`` or 2-D ``(M, D)``.
    """
    order = np.argsort(xp, kind="stable")
    xp, fp = xp[order], fp[order]
    keep = np.ones(xp.shape[0], dtype=bool)
    keep[:-1] = xp[1:] != xp[:-1]
    return xp[keep], fp[keep]


def _interp_to(t_query: np.ndarray, t_src: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``v_src`` (sampled at ``t_src``) onto ``t_query``.

    ``t_src``/``t_query`` are int64 ns in the *same* clock. ``t_src`` entries
    ``<= 0`` are sentinels and dropped. Out-of-range query points are clamped to
    the endpoint value (``np.interp`` default). ``v_src`` may be 1-D or 2-D;
    returns float64 (caller casts). Raises if < 2 valid source samples.
    """
    t_src = np.asarray(t_src, dtype=np.int64)
    v_src = np.asarray(v_src)
    valid = t_src > 0
    if int(valid.sum()) < 2:
        raise ValueError(
            f"channel has {int(valid.sum())} validly-timestamped samples; "
            "need >= 2 to interpolate"
        )
    t_src, v_src = _sorted_dedup(t_src[valid], v_src[valid])
    # Rebase to a local origin before float64 — raw epoch-ns (~1.8e18) exceeds
    # float64 integer precision (~9e15); a 23-s-span offset does not.
    t_query = np.asarray(t_query, dtype=np.int64)
    t0 = min(int(t_src[0]), int(t_query.min()))
    ts = (t_src - t0).astype(np.float64)
    tq = (t_query - t0).astype(np.float64)
    if v_src.ndim == 1:
        return np.interp(tq, ts, v_src.astype(np.float64))
    out = np.empty((tq.shape[0], v_src.shape[1]), dtype=np.float64)
    for d in range(v_src.shape[1]):
        out[:, d] = np.interp(tq, ts, v_src[:, d].astype(np.float64))
    return out


def _clamp_count(name: str, t_query: np.ndarray, t_src: np.ndarray) -> None:
    """Print how many query points fall outside the source timeline (clamped)."""
    valid = t_src[t_src > 0]
    if valid.size == 0:
        return
    lo = int((t_query < valid.min()).sum())
    hi = int((t_query > valid.max()).sum())
    if lo or hi:
        print(
            f"  [clamp] {name}: {lo} row(s) before source start, {hi} after end "
            "— held at the endpoint sample"
        )


class _CamExtrinsicFK:
    """FK the head-camera link (``zed_depth_frame``) from torso + head joints.

    KinHelper is imported lazily so the lag-diagnostics path needs no robot
    model — only the aligned-episode export pulls this in.
    """

    _ROBOT_NAME = "vega_no_effector"
    _CAM_LINK = "zed_depth_frame"
    _TORSO_JOINTS = ("torso_j1", "torso_j2", "torso_j3")
    _HEAD_JOINTS = ("head_j1", "head_j2", "head_j3")

    def __init__(self) -> None:
        try:
            from yixuan_utilities.kinematics_helper import KinHelper
        except ImportError as exc:
            raise SystemExit(
                "align_timestamps.py needs `yixuan_utilities` to recompute "
                "obs/images/extrinsic — run it in the `dexmate` (or "
                "`dexmate_lerobot`) conda env."
            ) from exc
        self._kin = KinHelper(self._ROBOT_NAME)
        active = self._kin.sapien_robot.get_active_joints()
        self._jidx = {j.name: i for i, j in enumerate(active)}
        self._dof = self._kin.sapien_robot.dof
        if self._CAM_LINK not in self._kin.link_name_to_idx:
            raise ValueError(
                f"link {self._CAM_LINK!r} not in robot model {self._ROBOT_NAME!r}"
            )
        self._eef_idx = self._kin.link_name_to_idx[self._CAM_LINK]

    def fk_batch(self, torso: np.ndarray, head: np.ndarray) -> np.ndarray:
        """``torso`` (N, 3) + ``head`` (N, 3) joints -> ``world_t_cam`` (N, 4, 4)."""
        if torso.ndim != 2 or torso.shape[1] != 3 or head.shape != torso.shape:
            raise ValueError(
                f"expected torso/head both (N, 3); got {torso.shape} / {head.shape}"
            )
        n = torso.shape[0]
        out = np.zeros((n, 4, 4), dtype=np.float32)
        qpos = np.zeros(self._dof, dtype=np.float64)
        for t in range(n):
            for i, name in enumerate(self._TORSO_JOINTS):
                qpos[self._jidx[name]] = torso[t, i]
            for i, name in enumerate(self._HEAD_JOINTS):
                qpos[self._jidx[name]] = head[t, i]
            out[t] = self._kin.compute_fk_from_link_idx(qpos, [self._eef_idx])[0]
        return out


def _build_aligned_episode(
    leader_path: pathlib.Path, follower_path: pathlib.Path
) -> tuple[dict, int]:
    """Resample one leader episode (+ follower) into the flat old-format dict.

    All interpolation runs in the dexmate clock (the camera/joint/apply clock).
    The output ``timestamp_ns`` is converted back to the lambda clock so the
    file stays a drop-in for the vr_reader_old.py format.
    """
    with h5py.File(leader_path, "r") as lh, h5py.File(follower_path, "r") as fh:
        source_map = _read_source_map(lh)
        for ch in (
            "obs/joint/left_arm", "obs/joint/right_arm", "obs/joint/head",
            "obs/joint/torso", "obs/gripper/right", "obs/images/left_rgb",
        ):
            src = source_map.get(ch, "dexmate")
            if src != "dexmate":
                raise ValueError(
                    f"channel {ch!r} has source clock {src!r}; this exporter "
                    "interpolates in the dexmate clock and expects 'dexmate'"
                )

        # Episode-start NTP query (last row) — temporally closest to the data.
        qr = lh["meta/clock/query_results"][...]
        offset_ns = int(qr[-1, 1]) if qr.size else 0
        if not qr.size:
            print("  [warn] leader file has no NTP query; treating dexmate == lambda")

        # ── Reference timeline: camera (left_rgb) capture time, dexmate clock ──
        t_ref = lh["timestamps/obs/images/left_rgb"][...].astype(np.int64)
        n = int(t_ref.shape[0])
        if (t_ref <= 0).any():
            raise ValueError(
                f"{int((t_ref <= 0).sum())} camera reference timestamps are "
                "<= 0 sentinels — cannot anchor on camera capture time"
            )
        dt = np.diff(t_ref)
        if (dt < 0).any():
            raise ValueError(
                f"camera reference timestamps are not non-decreasing "
                f"({int((dt < 0).sum())} inversion(s))"
            )
        n_dup = int((dt == 0).sum())
        if n_dup:
            print(
                f"  [info] {n_dup} repeated camera timestamp(s) "
                "(camera re-served a frame) — kept as-is"
            )

        # ── obs/joint/* ← follower (denser stream) ─────────────────────────────
        obs_joint: dict[str, np.ndarray] = {}
        for grp in ("left_arm", "right_arm", "head", "torso"):
            t_src = fh[f"obs/joint_ts/{grp}"][...].astype(np.int64)
            _clamp_count(f"obs/joint/{grp}", t_ref, t_src)
            obs_joint[grp] = _interp_to(
                t_ref, t_src, fh[f"obs/joint/{grp}"][...]
            ).astype(np.float32)

        # ── obs/gripper/* ← leader (left is dead when FC03 never replied) ──────
        gr_ts = lh["timestamps/obs/gripper/right"][...].astype(np.int64)
        _clamp_count("obs/gripper/right", t_ref, gr_ts)
        obs_grip_right = _interp_to(
            t_ref, gr_ts, lh["obs/gripper/right"][...]
        ).astype(np.float32)
        gl_ts = lh["timestamps/obs/gripper/left"][...].astype(np.int64)
        if int((gl_ts > 0).sum()) < 2:
            print(
                "  [warn] obs/gripper/left has < 2 valid timestamps "
                "(FC03 never replied this session) — output is all-NaN"
            )
            obs_grip_left = np.full(n, np.nan, dtype=np.float32)
        else:
            _clamp_count("obs/gripper/left", t_ref, gl_ts)
            obs_grip_left = _interp_to(
                t_ref, gl_ts, lh["obs/gripper/left"][...]
            ).astype(np.float32)

        # ── action/* ← leader, re-timed publish_ns → t_apply_ns (follower) ─────
        lead_pub = lh["timestamps/action/publish_ns"][...].astype(np.int64)
        match_idx = _match_to_first_follower(
            lead_pub, fh["vr_publish_ns"][...].astype(np.int64)
        )
        matched = match_idx >= 0
        print(
            f"  action re-timing: {int(matched.sum())}/{n} leader actions "
            "matched to a follower t_apply_ns"
        )
        if int(matched.sum()) < 2:
            raise ValueError(
                "fewer than 2 leader actions matched a follower t_apply_ns — "
                "cannot re-time action onto the apply timeline"
            )
        t_apply = fh["t_apply_ns"][...].astype(np.int64)[match_idx[matched]]
        _clamp_count("action/*", t_ref, t_apply)

        def _retime(name: str) -> np.ndarray:
            return _interp_to(t_ref, t_apply, lh[name][...][matched]).astype(np.float32)

        action = {
            "joint": {
                "left_arm": _retime("action/joint/left_arm"),
                "right_arm": _retime("action/joint/right_arm"),
                "head": _retime("action/joint/head"),
                "chassis_vx": _retime("action/joint/chassis_vx"),
                "chassis_vy": _retime("action/joint/chassis_vy"),
                "chassis_wz": _retime("action/joint/chassis_wz"),
            },
            "gripper": {
                "left": _retime("action/gripper/left"),
                "right": _retime("action/gripper/right"),
            },
        }

        # ── obs/images/* : verbatim from leader (the camera is the anchor) ─────
        images = {
            "left_rgb": lh["obs/images/left_rgb"][...],
            "right_rgb": lh["obs/images/right_rgb"][...],
            "depth": lh["obs/images/depth"][...],
            "intrinsic": lh["obs/images/intrinsic"][...].astype(np.float32),
        }

    # ── obs/images/extrinsic : recompute from the aligned (camera-time) joints ─
    print("  recomputing obs/images/extrinsic via FK on aligned torso+head ...")
    images["extrinsic"] = _CamExtrinsicFK().fk_batch(
        obs_joint["torso"], obs_joint["head"]
    )

    data = {
        "timestamp_ns": (t_ref - offset_ns).astype(np.int64),  # dexmate → lambda
        "action": action,
        "obs": {
            "joint": obs_joint,
            "gripper": {"left": obs_grip_left, "right": obs_grip_right},
            "images": images,
        },
    }
    return data, n


def _flatten(d: dict, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _verify_aligned(data: dict, n: int, leader_path: pathlib.Path) -> None:
    """Validate the aligned dict against the old format and print a summary.

    The post-alignment error is ~0 by construction (every channel is
    interpolated onto the camera clock), so instead we print the *pre*-alignment
    staleness of each leader channel vs the camera — i.e. what the resampling
    actually corrected.
    """
    flat = _flatten(data)
    keys = set(flat)
    if keys != _OLD_FORMAT_KEYS:
        raise ValueError(
            "aligned dict does not match the vr_reader_old.py leaf set:\n"
            f"  missing: {sorted(_OLD_FORMAT_KEYS - keys)}\n"
            f"  extra:   {sorted(keys - _OLD_FORMAT_KEYS)}"
        )
    for k, v in flat.items():
        if v.shape[0] != n:
            raise ValueError(f"{k}: first dim {v.shape[0]} != n_rows {n}")
    for k, want in (
        ("obs/images/left_rgb", "uint8"), ("obs/images/right_rgb", "uint8"),
        ("obs/images/depth", "uint16"), ("timestamp_ns", "int64"),
    ):
        if flat[k].dtype != np.dtype(want):
            raise ValueError(f"{k}: dtype {flat[k].dtype} != {want}")

    print(f"  aligned episode: {n} rows, {len(flat)} leaves")
    for k in sorted(flat):
        v = flat[k]
        nan = int(np.isnan(v).sum()) if np.issubdtype(v.dtype, np.floating) else 0
        tag = f"  NaN={nan}" if nan else ""
        print(f"    {k:30s} {str(v.shape):22s} {v.dtype}{tag}")

    with h5py.File(leader_path, "r") as lh:
        cam = lh["timestamps/obs/images/left_rgb"][...].astype(np.int64)
        qr = lh["meta/clock/query_results"][...]
        offset_ns = int(qr[-1, 1]) if qr.size else 0
        print("  pre-alignment staleness vs camera clock (ms; leader timestamps):")
        for ch in (
            "obs/joint/left_arm", "obs/joint/right_arm", "obs/joint/head",
            "obs/joint/torso", "obs/gripper/right",
        ):
            cts = lh[f"timestamps/{ch}"][...].astype(np.int64)
            m = cts > 0
            if m.any():
                d = (cts[m] - cam[m]) / 1e6
                print(
                    f"    {ch:24s} mean={d.mean():+8.1f}  "
                    f"min={d.min():+8.1f}  max={d.max():+8.1f}"
                )
        pub = lh["timestamps/action/publish_ns"][...].astype(np.int64)
        d = ((pub + offset_ns) - cam) / 1e6
        print(
            f"    {'action/publish':24s} mean={d.mean():+8.1f}  "
            f"min={d.min():+8.1f}  max={d.max():+8.1f}"
        )
    print(
        "  (post-alignment every channel sits on the camera clock => ~0 by "
        "construction)"
    )


def _write_h5(data: dict, path: pathlib.Path) -> None:
    """Write a nested dict-of-ndarrays to ``path`` (overwrites any existing)."""

    def recurse(group: h5py.Group, sub: dict) -> None:
        for k, v in sub.items():
            if isinstance(v, dict):
                recurse(group.require_group(k), v)
            elif isinstance(v, np.ndarray):
                group.create_dataset(k, data=v)
            else:
                raise ValueError(f"cannot write {type(v).__name__} at {k!r}")

    with h5py.File(path, "w") as h5:
        recurse(h5, data)


def _export_aligned(
    leader_path: pathlib.Path,
    follower_path: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    """Build, verify and write the aligned old-format episode HDF5."""
    print(f"\nAligned export → {out_path}")
    if not leader_path.exists():
        raise SystemExit(f"Leader file not found: {leader_path}")
    if not follower_path.exists():
        raise SystemExit(f"Follower file not found: {follower_path}")
    data, n = _build_aligned_episode(leader_path, follower_path)
    _verify_aligned(data, n, leader_path)
    if out_path.exists():
        print(f"  (overwriting existing {out_path.name})")
    _write_h5(data, out_path)
    print(f"  wrote {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


def _align_one(
    leader_path: pathlib.Path,
    follower_path: pathlib.Path,
    out_prefix: pathlib.Path,
) -> None:
    if not leader_path.exists():
        raise SystemExit(f"Leader file not found: {leader_path}")
    if not follower_path.exists():
        raise SystemExit(f"Follower file not found: {follower_path}")

    print(f"\nLeader:   {leader_path}")
    print(f"Follower: {follower_path}")

    with h5py.File(leader_path, "r") as lh, h5py.File(follower_path, "r") as fh:
        ts_tree = _load_ts_tree(lh)
        offset_ns = _read_clock_offset_ns(lh)
        source_map = _read_source_map(lh)

        if "action/publish_ns" not in ts_tree:
            raise SystemExit(
                "Leader file has no /timestamps/action/publish_ns; was it recorded "
                "with the new schema?"
            )
        leader_publish_ns = ts_tree["action/publish_ns"]

        if "vr_publish_ns" not in fh:
            raise SystemExit("Follower file has no /vr_publish_ns; schema mismatch?")
        follower_publish_ns = fh["vr_publish_ns"][...].astype(np.int64)
        follower_apply_ns = fh["t_apply_ns"][...].astype(np.int64)  # dexmate clock

        # Pair leader → follower by exact vr_publish_ns match.
        match_idx = _match_to_first_follower(leader_publish_ns, follower_publish_ns)
        n_matched = int((match_idx >= 0).sum())
        print(
            f"Matched {n_matched}/{len(leader_publish_ns)} leader frames to follower ticks."
        )

        # Build per-channel lag table in ms, in lambda clock.
        lags_ms: dict[str, np.ndarray] = {}
        for channel, raw in ts_tree.items():
            if channel == "action/publish_ns":
                continue
            arr = raw.astype(np.int64)
            # Apply offset if the channel originated on dexmate.
            if source_map.get(channel) == "dexmate":
                arr = _to_lambda_ns(arr, offset_ns)
            # Negative sentinel = fallback (image arrival ts) or missing (-1).
            mask = arr > 0
            lag = np.where(mask, (leader_publish_ns - arr) / 1e6, np.nan)
            lags_ms[channel] = lag

        # Follower apply lag (dexmate → lambda first).
        follower_apply_ns_lambda = _to_lambda_ns(follower_apply_ns, offset_ns)
        leader_to_apply_ms = np.full(leader_publish_ns.shape, np.nan, dtype=np.float64)
        valid = match_idx >= 0
        if valid.any():
            leader_to_apply_ms[valid] = (
                follower_apply_ns_lambda[match_idx[valid]] - leader_publish_ns[valid]
            ) / 1e6
        lags_ms["action/apply (follower)"] = leader_to_apply_ms

    # ── Write CSV ──────────────────────────────────────────────────────────────
    import csv

    csv_path = out_prefix.with_suffix(".lag.csv")
    cols = ["frame_idx", "leader_publish_ns"] + sorted(lags_ms.keys())
    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(cols)
        for i in range(len(leader_publish_ns)):
            row = [i, int(leader_publish_ns[i])]
            for c in cols[2:]:
                v = lags_ms[c][i]
                row.append("" if (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}")
            w.writerow(row)
    print(f"Wrote {csv_path}")

    # ── Plot histograms ────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping histograms.")
        return

    keys = sorted(lags_ms.keys())
    n = len(keys)
    cols_grid = 3
    rows_grid = (n + cols_grid - 1) // cols_grid
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(4.5 * cols_grid, 2.8 * rows_grid))
    axes = np.atleast_2d(axes).reshape(rows_grid, cols_grid)

    print("\nPer-channel lag (ms, leader publish − channel timestamp, lambda clock):")
    for i, key in enumerate(keys):
        r, c = divmod(i, cols_grid)
        ax = axes[r][c]
        data = lags_ms[key]
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            ax.set_title(f"{key}\n(no data)")
            ax.axis("off")
            continue
        ax.hist(finite, bins=50, color="steelblue", edgecolor="none")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("ms")
        p50, p95 = float(np.percentile(finite, 50)), float(np.percentile(finite, 95))
        ax.axvline(p50, color="orange", linewidth=1, label=f"p50 {p50:.1f}")
        ax.axvline(p95, color="red", linewidth=1, label=f"p95 {p95:.1f}")
        ax.legend(fontsize=7, loc="upper right")
        print(
            f"  {key:<40s}  n={finite.size:5d}  "
            f"mean={float(finite.mean()):+7.2f}  p50={p50:+7.2f}  p95={p95:+7.2f}  "
            f"max={float(finite.max()):+7.2f}"
        )
    for j in range(n, rows_grid * cols_grid):
        r, c = divmod(j, cols_grid)
        axes[r][c].axis("off")
    fig.suptitle(
        f"Per-channel lag — {leader_path.name}  (offset_ns={offset_ns}, "
        f"n_matched={n_matched}/{len(leader_publish_ns)})",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = out_prefix.with_suffix(".lag.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"\nWrote {png_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--episode_id",
        default="all",
        help="Episode id to process, or 'all' to process every leader/raw_data episode.",
    )
    p.add_argument(
        "--root_dir",
        default="/home/yixuan/Dexmate/data",
        type=pathlib.Path,
        help="Dexmate data root containing raw_data/ or leader/, plus follower/.",
    )
    p.add_argument("--leader", default=None, type=pathlib.Path)
    p.add_argument("--follower", default=None, type=str)
    p.add_argument("--out-prefix", default=None, type=str)
    args = p.parse_args()

    root_dir = args.root_dir

    if args.leader is not None:
        leader_path = args.leader
        follower_path = _resolve_follower_path(leader_path, args.follower, root_dir)
        out_prefix = (
            pathlib.Path(args.out_prefix)
            if args.out_prefix is not None
            else leader_path.with_suffix("")
        )
        _align_one(leader_path, follower_path, out_prefix)
        episode_id = _episode_id_from_leader_path(leader_path)
        aligned_path = root_dir / _RAW_DATA_DIRNAME / f"episode_{episode_id}.hdf5"
        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        _export_aligned(leader_path, follower_path, aligned_path)
        return

    if args.follower is not None:
        raise SystemExit("--follower requires --leader; use --episode_id for root_dir mode.")

    if args.episode_id == "all":
        episode_ids = _discover_episode_ids(root_dir)
        if not episode_ids:
            raise SystemExit(f"No leader/raw_data episode_*.hdf5 files found under {root_dir}")
    else:
        try:
            episode_ids = [int(args.episode_id)]
        except ValueError as exc:
            raise SystemExit("--episode_id must be an integer or 'all'.") from exc

    out_prefix_arg = pathlib.Path(args.out_prefix) if args.out_prefix is not None else None
    multiple = len(episode_ids) > 1
    for episode_id in episode_ids:
        leader_path = _leader_path_for_episode(root_dir, episode_id)
        follower_path = _follower_path_for_episode(root_dir, episode_id)
        if multiple and not follower_path.exists():
            print(f"\nSkipping episode {episode_id}: follower file not found: {follower_path}")
            continue
        if out_prefix_arg is None:
            out_prefix = leader_path.with_suffix("")
        elif multiple:
            out_prefix_arg.mkdir(parents=True, exist_ok=True)
            out_prefix = out_prefix_arg / f"episode_{episode_id}"
        else:
            out_prefix = out_prefix_arg
        _align_one(leader_path, follower_path, out_prefix)
        aligned_path = root_dir / _RAW_DATA_DIRNAME / f"episode_{episode_id}.hdf5"
        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        _export_aligned(leader_path, follower_path, aligned_path)


if __name__ == "__main__":
    main()
