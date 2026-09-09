#!/usr/bin/env python3
r"""Visualize one joystick take as the POLICY sees it: cameras + state/action.

Top row: the three robot cameras the recorder stores (``obs/images/head_left_rgb``,
``left_wrist_rgb``, ``right_wrist_rgb``). Second row: the two static D455 colour streams
from the matching ``record_static_cams.py`` file, time-matched to every teleop row on
the shared clock exactly like ``scripts/align_static_teleop.py`` (nearest capture within
``--max-gap-ms``; a tile is blanked when no static frame is close enough), and a panel of
time-series with a cursor. Bottom: the live numeric values.

The vectors are not re-derived here: every row goes through the porter's own
``compute_frame_state_action`` (``scripts/port_wbc_mobile_hdf5.py``), so what is drawn is
bit-for-bit the 32-D ``observation.state`` (achieved base-frame EEF/head FK + grippers +
world base pose) and the 32-D joystick action (current-base EEF/head targets, binarized
grippers, body-frame chassis intent) that ``port_wbc_mobile_hdf5.py`` would write.

    python scripts/vis_policy_episode.py \
        --teleop_dir /data/Dexmate/data \
        --static_dir /data/Dexmate/static_09_05 \
        --episode 0 \
        --out /data/Dexmate/viz/episode_0_policy.mp4 [--preview-png ...]

    python scripts/vis_policy_episode.py \
        --teleop_dir /data/Dexmate/data \
        --static_dir /data/Dexmate/static_09_05 \
        --episode 4 \
        --interactive

Run in the dexmate2 env (pinocchio for the FK). ``--static_dir`` may be omitted for a
robot-cameras-only visualization. ``--episode N`` loads ``episode_N.hdf5``, or the
quarantined ``episode_N.hdf5.partial`` if that is all that exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from omniteleop.wbc_policy_format import (
    JOYSTICK_ACTION_AXES,
    STATE_AXES,
    WBCPolicyFK,
)

_SCRIPTS = Path(__file__).resolve().parent


def _episode_path(directory: Path, episode: int) -> Path:
    """``episode_<N>.hdf5``, or the quarantined ``.hdf5.partial`` if that is all that exists."""
    complete = directory / f"episode_{episode}.hdf5"
    if complete.is_file():
        return complete
    partial = directory / f"episode_{episode}.hdf5.partial"
    if partial.is_file():
        return partial
    raise FileNotFoundError(f"episode_{episode}.hdf5[.partial] not found in {directory}")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_vis", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TILE_W, TILE_H = 600, 375  # every camera tile; head/statics are 16:10, wrists 16:9
TEXT_H = 300
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox ``img`` into a (h, w, 3) tile without changing its aspect ratio."""
    out = np.zeros((h, w, 3), np.uint8)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    rw, rh = max(1, round(iw * s)), max(1, round(ih * s))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    out[y0 : y0 + rh, x0 : x0 + rw] = resized
    return out


def _label(tile: np.ndarray, text: str, color=(255, 255, 255)) -> None:
    cv2.rectangle(tile, (0, 0), (min(tile.shape[1], 12 + 11 * len(text)), 26), (0, 0, 0), -1)
    cv2.putText(tile, text, (6, 19), FONT, 0.6, color, 1, cv2.LINE_AA)


def _blank(text: str) -> np.ndarray:
    tile = np.full((TILE_H, TILE_W, 3), 40, np.uint8)
    cv2.putText(tile, text, (20, TILE_H // 2), FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return tile


def build_policy_vectors(teleop: Path, state_frame: str) -> tuple[np.ndarray, np.ndarray]:
    """(T,32) observation.state and (T,32) joystick action, via the porter's own code."""
    porter = _load_script("port_wbc_mobile_hdf5")
    fk = WBCPolicyFK()
    with h5py.File(teleop, "r") as raw:
        schema = raw["meta/policy_action_schema"][()].decode()
        n = int(raw["timestamp_ns"].shape[0])
        states, actions = [], []
        for t in range(n):
            action_eef, action_head = porter.load_action_targets(raw, t)
            s, a = porter.compute_frame_state_action(
                raw,
                t,
                fk,
                action_eef,
                action_head,
                state_frame=state_frame,
                policy_action_schema=schema,
            )
            states.append(s)
            actions.append(a)
    states, actions = np.stack(states), np.stack(actions)
    if states.shape[1] != len(STATE_AXES) or actions.shape[1] != len(JOYSTICK_ACTION_AXES):
        raise RuntimeError(f"unexpected vector widths {states.shape} / {actions.shape}")
    return states, actions


# Plot rows: (title, [(label, source, dim)], y-limits or None). "S" = state, "A" = action.
def _plot_rows():
    sd = {a: i for i, a in enumerate(STATE_AXES)}
    ad = {a: i for i, a in enumerate(JOYSTICK_ACTION_AXES)}
    return [
        (
            "L EEF pos [m]  (solid=state, dashed=action)",
            [(k, "S", sd[f"left_t{k}"], "A", ad[f"left_t{k}"]) for k in "xyz"],
        ),
        ("R EEF pos [m]", [(k, "S", sd[f"right_t{k}"], "A", ad[f"right_t{k}"]) for k in "xyz"]),
        (
            "grippers  (state achieved / action binary)",
            [
                ("L", "S", sd["left_gripper"], "A", ad["left_gripper"]),
                ("R", "S", sd["right_gripper"], "A", ad["right_gripper"]),
            ],
        ),
        ("head pos [m]", [(k, "S", sd[f"head_t{k}"], "A", ad[f"head_t{k}"]) for k in "xyz"]),
        (
            "base pose (state only): x,y [m], yaw [rad]",
            [
                ("x", "S", sd["base_x"], None, None),
                ("y", "S", sd["base_y"], None, None),
                ("yaw", "S", sd["base_yaw"], None, None),
            ],
        ),
        (
            "chassis intent (action only) [m/s, rad/s]",
            [
                ("vx", None, None, "A", ad["chassis_intent_vx_body"]),
                ("vy", None, None, "A", ad["chassis_intent_vy_body"]),
                ("wz", None, None, "A", ad["chassis_intent_wz_body"]),
            ],
        ),
    ]


def render_plot_panel(
    t: np.ndarray, states: np.ndarray, actions: np.ndarray, w: int, h: int
) -> tuple[np.ndarray, callable]:
    """Pre-render the time-series once; return the RGB panel and a t->pixel-x mapper."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = _plot_rows()
    dpi = 100
    fig, axes = plt.subplots(len(rows), 1, figsize=(w / dpi, h / dpi), dpi=dpi, sharex=True)
    colors = ["C0", "C1", "C2"]
    for ax, (title, series) in zip(axes, rows, strict=True):
        for ci, (lab, _sk, si, _ak, ai) in enumerate(series):
            c = colors[ci % 3]
            if si is not None:
                ax.plot(t, states[:, si], c, lw=0.9, label=lab)
            if ai is not None:
                ax.plot(
                    t,
                    actions[:, ai],
                    c,
                    lw=0.9,
                    ls="--",
                    label=None if si is not None else lab,
                )
        ax.set_title(title, fontsize=7, loc="left", pad=2)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right", ncol=3, frameon=False)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t [s]", fontsize=7)
    axes[0].set_xlim(float(t[0]), float(t[-1]))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.07, hspace=0.55)
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    x0_px, x1_px = (
        axes[0].transData.transform((float(t[0]), 0))[0],
        axes[0].transData.transform((float(t[-1]), 0))[0],
    )
    plt.close(fig)
    if panel.shape[:2] != (h, w):
        panel = cv2.resize(panel, (w, h), interpolation=cv2.INTER_AREA)
    sx = w / (fig.get_size_inches()[0] * dpi)

    def x_of(sec: float) -> int:
        frac = (sec - float(t[0])) / max(1e-9, float(t[-1]) - float(t[0]))
        return round((x0_px + frac * (x1_px - x0_px)) * sx)

    return panel, x_of


def render_text(
    state: np.ndarray,
    action: np.ndarray,
    matched: tuple[int, int],
    gaps_ms: tuple[float, float],
    w: int,
) -> np.ndarray:
    tile = np.zeros((TEXT_H, w, 3), np.uint8)
    f = lambda v: " ".join(f"{x:+.3f}" for x in v)
    sd = {a: i for i, a in enumerate(STATE_AXES)}
    ad = {a: i for i, a in enumerate(JOYSTICK_ACTION_AXES)}
    blocks = [
        (
            "observation.state (32)",
            [
                f"L eef  t {f(state[0:3])}  r6 {f(state[3:9])}  "
                f"grip {state[sd['left_gripper']]:.3f}",
                f"R eef  t {f(state[10:13])}  r6 {f(state[13:19])}  "
                f"grip {state[sd['right_gripper']]:.3f}",
                f"head   t {f(state[20:23])}  r6 {f(state[23:29])}",
                f"base   x {state[29]:+.3f}  y {state[30]:+.3f}  yaw {state[31]:+.3f}",
            ],
        ),
        (
            "action (32, joystick v1, current-base frame)",
            [
                f"L eef  t {f(action[0:3])}  r6 {f(action[3:9])}  "
                f"grip {action[ad['left_gripper']]:.0f}",
                f"R eef  t {f(action[10:13])}  r6 {f(action[13:19])}  "
                f"grip {action[ad['right_gripper']]:.0f}",
                f"head   t {f(action[20:23])}  r6 {f(action[23:29])}",
                f"chassis intent  vx {action[29]:+.3f}  vy {action[30]:+.3f}  wz {action[31]:+.3f}"
                f"    | static match: cam0 #{matched[0]} ({gaps_ms[0]:.0f} ms)  "
                f"cam1 #{matched[1]} ({gaps_ms[1]:.0f} ms)",
            ],
        ),
    ]
    y = 22
    for title, lines in blocks:
        cv2.putText(tile, title, (10, y), FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        y += 24
        for line in lines:
            cv2.putText(tile, line, (10, y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
        y += 8
    return tile


_RERUN_PLOT_SLUGS = (
    "left_eef_position",
    "right_eef_position",
    "grippers",
    "head_position",
    "base_pose",
    "chassis_intent",
)
_RERUN_STATE_COLORS = ((31, 119, 180), (44, 160, 44), (214, 39, 40))
_RERUN_ACTION_COLORS = ((23, 190, 207), (152, 223, 138), (255, 152, 150))


def _format_rerun_values(
    source_row: int,
    row_count: int,
    sec: float,
    state: np.ndarray,
    action: np.ndarray,
    matched: tuple[int, int],
    gaps_ms: tuple[float, float],
) -> str:
    """Format the numeric values shown beside the interactive Rerun views."""
    f = lambda v: " ".join(f"{x:+.3f}" for x in v)
    sd = {a: i for i, a in enumerate(STATE_AXES)}
    ad = {a: i for i, a in enumerate(JOYSTICK_ACTION_AXES)}

    static_matches = []
    for cam, (idx, gap) in enumerate(zip(matched, gaps_ms, strict=True)):
        if idx < 0:
            static_matches.append(f"cam{cam}: no match")
        else:
            static_matches.append(f"cam{cam}: #{idx} ({gap:.0f} ms)")

    return "\n".join(
        [
            f"# Policy frame {source_row}/{row_count - 1}  ·  t={sec:.3f} s",
            "",
            "## observation.state (32)",
            "```text",
            f"L eef  t {f(state[0:3])}  r6 {f(state[3:9])}  grip {state[sd['left_gripper']]:.3f}",
            f"R eef  t {f(state[10:13])}  r6 {f(state[13:19])}  "
            f"grip {state[sd['right_gripper']]:.3f}",
            f"head   t {f(state[20:23])}  r6 {f(state[23:29])}",
            f"base   x {state[29]:+.3f}  y {state[30]:+.3f}  yaw {state[31]:+.3f}",
            "```",
            "",
            "## action (32, joystick v1, current-base frame)",
            "```text",
            f"L eef  t {f(action[0:3])}  r6 {f(action[3:9])}  "
            f"grip {action[ad['left_gripper']]:.0f}",
            f"R eef  t {f(action[10:13])}  r6 {f(action[13:19])}  "
            f"grip {action[ad['right_gripper']]:.0f}",
            f"head   t {f(action[20:23])}  r6 {f(action[23:29])}",
            f"chassis intent  vx {action[29]:+.3f}  vy {action[30]:+.3f}  wz {action[31]:+.3f}",
            "```",
            "",
            "Static match: " + "  ·  ".join(static_matches),
        ]
    )


def _visualize_rerun(
    teleop_file: h5py.File,
    static_file: h5py.File | None,
    static_idx: dict[int, np.ndarray] | None,
    static_gap: dict[int, np.ndarray] | None,
    t: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    stride: int,
    max_frames: int | None,
    start_row: int,
    playback_fps: float,
    jpeg_quality: int,
) -> int:
    """Log the episode into a spawned Rerun viewer and return the frame count."""
    try:
        import rerun as rr  # noqa: PLC0415
        import rerun.blueprint as rrb  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "--interactive requires rerun-sdk; install the project dependencies first"
        ) from exc

    camera_views = [
        rrb.Spatial2DView(origin="/camera/head_left_rgb", name="head_left_rgb"),
        rrb.Spatial2DView(origin="/camera/left_wrist_rgb", name="left_wrist_rgb"),
        rrb.Spatial2DView(origin="/camera/right_wrist_rgb", name="right_wrist_rgb"),
    ]
    static_views = []
    if static_file is not None:
        static_views = [
            rrb.Spatial2DView(origin=f"/camera/static_{cam}", name=f"static camera_{cam}")
            for cam in (0, 1)
        ]
    n = states.shape[0]
    lookback = max(400, min(n, 5000))
    lookahead = min(80, max(1, n // 50 + 5))
    plot_views = [
        rrb.TimeSeriesView(
            origin=f"/plot/{slug}",
            name=title.replace("solid=state, dashed=action", "state / action"),
            time_ranges=[
                rrb.VisibleTimeRange(
                    "frame",
                    start=rrb.TimeRangeBoundary.cursor_relative(seq=-lookback),
                    end=rrb.TimeRangeBoundary.cursor_relative(seq=lookahead),
                )
            ],
        )
        for slug, (title, _) in zip(_RERUN_PLOT_SLUGS, _plot_rows(), strict=True)
    ]
    image_grid = rrb.Grid(contents=[*camera_views, *static_views], grid_columns=3)
    lower = rrb.Horizontal(
        rrb.Grid(contents=plot_views, grid_columns=2, name="state and action"),
        rrb.TextDocumentView(origin="/frame_values", name="numeric values"),
        column_shares=[2, 1],
    )
    blueprint = rrb.Blueprint(
        rrb.Vertical(image_grid, lower, row_shares=[3, 2]),
        rrb.TimePanel(timeline="frame", fps=playback_fps),
        collapse_panels=True,
    )

    rr.init("vis_policy_episode", spawn=True)
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    rows = _plot_rows()
    for slug, (_, series) in zip(_RERUN_PLOT_SLUGS, rows, strict=True):
        colors = []
        names = []
        for color_i, (label, _state_key, state_i, _action_key, action_i) in enumerate(series):
            if state_i is not None:
                colors.append(_RERUN_STATE_COLORS[color_i % 3])
                names.append(f"state {label}")
            if action_i is not None:
                colors.append(_RERUN_ACTION_COLORS[color_i % 3])
                names.append(f"action {label}")
        rr.log(
            f"plot/{slug}",
            rr.SeriesLines(colors=colors, names=names, widths=[2.0] * len(names)),
            static=True,
        )

    print(
        f"Rerun viewer launched; encoding camera frames as JPEG quality {jpeg_quality} ...",
        flush=True,
    )
    written = 0
    for i in range(start_row, n, stride):
        # Use a dense sequence so --stride still plays at the requested effective FPS.
        rr.set_time("frame", sequence=written)
        rr.set_time("time", duration=float(t[i]))
        rr.log(
            "camera/head_left_rgb",
            rr.Image(teleop_file["obs/images/head_left_rgb"][i]).compress(jpeg_quality),
        )
        rr.log(
            "camera/left_wrist_rgb",
            rr.Image(teleop_file["obs/images/left_wrist_rgb"][i]).compress(jpeg_quality),
        )
        rr.log(
            "camera/right_wrist_rgb",
            rr.Image(teleop_file["obs/images/right_wrist_rgb"][i]).compress(jpeg_quality),
        )

        matched = (-1, -1)
        gaps = (float("nan"), float("nan"))
        if static_file is not None:
            assert static_idx is not None and static_gap is not None
            matched_list, gap_list = [], []
            for cam in (0, 1):
                j = int(static_idx[cam][i])
                gap = float(static_gap[cam][i])
                matched_list.append(j)
                gap_list.append(gap)
                path = f"camera/static_{cam}"
                if j < 0:
                    rr.log(path, rr.Clear(recursive=True))
                else:
                    rr.log(
                        path,
                        rr.Image(
                            static_file[f"observations/images/camera_{cam}_color"][j]
                        ).compress(jpeg_quality),
                    )
            matched = (matched_list[0], matched_list[1])
            gaps = (gap_list[0], gap_list[1])

        for slug, (_, series) in zip(_RERUN_PLOT_SLUGS, rows, strict=True):
            values = []
            for _label, _state_key, state_i, _action_key, action_i in series:
                if state_i is not None:
                    values.append(float(states[i, state_i]))
                if action_i is not None:
                    values.append(float(actions[i, action_i]))
            rr.log(f"plot/{slug}", rr.Scalars(values))

        rr.log(
            "frame_values",
            rr.TextDocument(
                _format_rerun_values(i, n, float(t[i]), states[i], actions[i], matched, gaps),
                media_type="text/markdown",
            ),
        )
        written += 1
        if written % 25 == 0:
            print(f"  logged {written} frames", flush=True)
        if max_frames is not None and written >= max_frames:
            break

    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--teleop_dir",
        type=Path,
        required=True,
        help="directory of recorded episode_<N>.hdf5[.partial] files",
    )
    ap.add_argument(
        "--static_dir",
        type=Path,
        default=None,
        help="directory of record_static_cams.py episode_<N>.hdf5[.partial] files paired with --teleop_dir",
    )
    ap.add_argument(
        "--episode",
        type=int,
        required=True,
        help="episode index N; loads episode_N.hdf5, or .hdf5.partial if that is all that exists",
    )
    output = ap.add_mutually_exclusive_group(required=True)
    output.add_argument("--out", type=Path, help="output .mp4")
    output.add_argument(
        "--interactive",
        action="store_true",
        help="spawn a native Rerun viewer instead of writing an MP4",
    )
    ap.add_argument(
        "--preview-png",
        type=Path,
        default=None,
        help="also write the composed frame at --preview-index as a PNG",
    )
    ap.add_argument("--preview-index", type=int, default=None, help="default: middle row")
    ap.add_argument("--state-frame", choices=("base", "world"), default="base")
    ap.add_argument(
        "--max-gap-ms",
        type=float,
        default=70.0,
        help="blank static frames beyond this alignment gap (default: 70 ms)",
    )
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--last-frames",
        type=int,
        default=None,
        help="visualize only the last N episode rows (labels still use full-episode indices)",
    )
    ap.add_argument("--fps", type=float, default=None, help="default: the record rate")
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
        help="JPEG quality for camera frames sent to Rerun (1-100; default: 85)",
    )
    args = ap.parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.episode < 0:
        raise ValueError("--episode must be >= 0")
    if args.last_frames is not None and args.last_frames < 1:
        raise ValueError("--last-frames must be >= 1")
    if args.interactive and (args.preview_png is not None or args.preview_index is not None):
        ap.error("--preview-png/--preview-index are only available with --out")
    if not 1 <= args.jpeg_quality <= 100:
        ap.error("--jpeg-quality must be between 1 and 100")

    teleop = _episode_path(args.teleop_dir, args.episode)
    static = None if args.static_dir is None else _episode_path(args.static_dir, args.episode)
    if teleop.name.endswith(".partial"):
        print(f"teleop {teleop} is a quarantined .partial (inspect/cut only)", flush=True)
    if static is not None and static.name.endswith(".partial"):
        print(f"static {static} is a quarantined .partial (inspect/cut only)", flush=True)

    print(f"building policy vectors from {teleop} ...", flush=True)
    states, actions = build_policy_vectors(teleop, args.state_frame)
    n = states.shape[0]
    start_row = 0 if args.last_frames is None else max(0, n - args.last_frames)
    if start_row > 0:
        print(
            f"visualizing last {n - start_row} rows "
            f"(episode indices {start_row}..{n - 1} of {n})",
            flush=True,
        )

    align = _load_script("align_static_teleop")
    static_idx = None
    if static is not None:
        head_ns = align.head_on_teleop_clock(teleop)
        if head_ns.shape[0] != n:
            raise RuntimeError(f"teleop has {n} rows but {head_ns.shape[0]} head stamps")
        gap_ns = int(args.max_gap_ms * 1e6)
        static_idx, static_gap = {}, {}
        for cam in (0, 1):
            m = align.match(align.static_on_teleop_clock(static, cam), head_ns, gap_ns)
            static_idx[cam], static_gap[cam] = m["static_idx"], m["gap_ns"] / 1e6
            print(
                f"static camera_{cam}: matched {(static_idx[cam] >= 0).sum()}/{n} rows "
                f"within {args.max_gap_ms:.0f} ms",
                flush=True,
            )

    with h5py.File(teleop, "r") as tf:
        ts = tf["timestamp_ns"][:].astype(np.int64)
        t = (ts - ts[0]) / 1e9
        dt = np.diff(t)
        fps = args.fps if args.fps else 1.0 / float(np.median(dt))
        if args.interactive:
            sf = h5py.File(static, "r") if static is not None else None
            print(
                f"{n} rows, {t[-1]:.1f} s, opening Rerun at {fps / args.stride:.2f} fps",
                flush=True,
            )
            try:
                written = _visualize_rerun(
                    tf,
                    sf,
                    static_idx,
                    static_gap if static_idx is not None else None,
                    t,
                    states,
                    actions,
                    args.stride,
                    args.max_frames,
                    start_row,
                    fps / args.stride,
                    args.jpeg_quality,
                )
            finally:
                if sf is not None:
                    sf.close()
            print(f"logged {written} frames to Rerun")
            return 0

        W = 3 * TILE_W
        H = 2 * TILE_H + TEXT_H
        panel, x_of = render_plot_panel(t, states, actions, TILE_W, TILE_H)
        assert args.out is not None
        print(
            f"{n} rows, {t[-1]:.1f} s, writing {W}x{H} @ {fps:.2f} fps -> {args.out}",
            flush=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            f"{fps / args.stride:.4f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(args.out),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        sf = h5py.File(static, "r") if static is not None else None
        preview_i = n // 2 if args.preview_index is None else args.preview_index
        if args.preview_index is None and start_row > 0:
            preview_i = start_row + (n - start_row) // 2
        written = 0
        try:
            for i in range(start_row, n, args.stride):
                head = _fit(tf["obs/images/head_left_rgb"][i], TILE_W, TILE_H)
                lw = _fit(tf["obs/images/left_wrist_rgb"][i], TILE_W, TILE_H)
                rw = _fit(tf["obs/images/right_wrist_rgb"][i], TILE_W, TILE_H)
                _label(head, f"head_left_rgb  row {i}/{n}  t={t[i]:.2f}s")
                _label(lw, "left_wrist_rgb")
                _label(rw, "right_wrist_rgb")
                matched, gaps = (-1, -1), (float("nan"), float("nan"))
                if sf is not None:
                    tiles = []
                    for cam in (0, 1):
                        assert static_idx is not None and static_gap is not None
                        j = int(static_idx[cam][i])
                        if j < 0:
                            tiles.append(
                                _blank(
                                    f"static camera_{cam}: no frame within {args.max_gap_ms:.0f} ms"
                                )
                            )
                        else:
                            tile = _fit(
                                sf[f"observations/images/camera_{cam}_color"][j],
                                TILE_W,
                                TILE_H,
                            )
                            _label(
                                tile,
                                f"static camera_{cam}  #{j}  gap {static_gap[cam][i]:.0f} ms",
                            )
                            tiles.append(tile)
                    matched = (int(static_idx[0][i]), int(static_idx[1][i]))
                    gaps = (float(static_gap[0][i]), float(static_gap[1][i]))
                else:
                    tiles = [_blank("no --static_dir"), _blank("no --static_dir")]
                plot = panel.copy()
                x = x_of(float(t[i]))
                cv2.line(plot, (x, 0), (x, TILE_H - 1), (255, 0, 0), 1)
                row1 = np.concatenate([head, lw, rw], axis=1)
                row2 = np.concatenate([tiles[0], tiles[1], plot], axis=1)
                text = render_text(states[i], actions[i], matched, gaps, W)
                canvas = np.concatenate([row1, row2, text], axis=0)
                if args.preview_png is not None and i == preview_i:
                    cv2.imwrite(str(args.preview_png), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                    print(f"preview -> {args.preview_png}", flush=True)
                proc.stdin.write(np.ascontiguousarray(canvas).tobytes())
                written += 1
                if written % 100 == 0:
                    print(f"  {written} frames", flush=True)
                if args.max_frames is not None and written >= args.max_frames:
                    break
        except BrokenPipeError as exc:
            stderr = proc.stderr.read().decode(errors="replace")
            raise RuntimeError("ffmpeg pipe broke:\n" + stderr) from exc
        finally:
            if sf is not None:
                sf.close()
            if not proc.stdin.closed:
                proc.stdin.close()
            err = proc.stderr.read()
            proc.wait()
        if proc.returncode != 0:
            stderr = err.decode(errors="replace")
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{stderr}")
    print(f"wrote {args.out} ({written} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
