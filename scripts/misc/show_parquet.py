#!/usr/bin/env python3
"""Print LeRobot parquet dataset layout as an indented tree.

Usage::

    python scripts/show_parquet.py /path/to/dexmate_eef_eef
    python scripts/show_parquet.py /path/to/dexmate_eef_eef --parquet data/chunk-000/file-000.parquet
    python scripts/show_parquet.py /path/to/dexmate_eef_eef --sample

Requires pyarrow (e.g. ``conda activate dexmate_lerobot``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "pyarrow is required. Try: conda activate dexmate_lerobot"
    ) from exc


def _load_info(root: Path) -> dict[str, Any] | None:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return None
    return json.loads(info_path.read_text())


def _feature_meta(info: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if info is None:
        return {}
    return info.get("features", {}).get(key, {})


def _format_vector_axes(meta: dict[str, Any], length: int) -> str:
    names = meta.get("names")
    if isinstance(names, dict) and "axes" in names:
        axes = names["axes"]
        if len(axes) == length:
            return ", ".join(axes)
        return ", ".join(axes[:6]) + (", ..." if len(axes) > 6 else "")
    return f"{length} floats"


def _describe_parquet_column(
    col_name: str,
    arrow_type: str,
    info: dict[str, Any] | None,
    sample_stats: dict[str, str] | None,
) -> str:
    meta = _feature_meta(info, col_name)
    dtype = meta.get("dtype")
    shape = meta.get("shape")

    if "fixed_size_list" in arrow_type:
        length = int(arrow_type.split("[")[-1].rstrip("]"))
        axes = _format_vector_axes(meta, length)
        line = f"{col_name}: float32[{length}]"
        if axes:
            line += f"  ({axes})"
    elif dtype == "video":
        line = f"{col_name}: video (not in parquet)"
        if shape:
            h, w, c = shape
            line += f"  stored as MP4 {h}x{w}x{c}"
    else:
        line = f"{col_name}: {arrow_type}"
        if dtype:
            line += f"  (meta dtype={dtype})"
        if shape:
            line += f"  shape={shape}"

    if sample_stats and col_name in sample_stats:
        line += f"  [{sample_stats[col_name]}]"
    return line


def _collect_sample_stats(parquet_path: Path, columns: list[str]) -> dict[str, str]:
    import numpy as np

    pf = pq.ParquetFile(parquet_path)
    table = pf.read(columns=columns)
    stats: dict[str, str] = {}
    n = table.num_rows

    for col in columns:
        if col not in table.column_names:
            continue
        arr = table[col].to_numpy(zero_copy_only=False)
        if col in ("observation.state", "action"):
            stacked = np.stack([np.asarray(x, dtype=np.float32) for x in arr])
            stats[col] = f"rows={n}, dim={stacked.shape[1]}"
        elif col == "episode_index":
            uniq = sorted({int(x) for x in arr})
            stats[col] = f"unique={uniq}"
        elif col in ("frame_index", "index"):
            stats[col] = f"range={int(arr.min())}..{int(arr.max())}"
        elif col == "timestamp":
            stats[col] = f"range={float(arr.min()):.3f}..{float(arr.max()):.3f}s"
        elif col == "task_index":
            stats[col] = f"unique={sorted({int(x) for x in arr})}"
        else:
            stats[col] = f"rows={n}"
    return stats


def _parquet_columns(parquet_path: Path) -> list[tuple[str, str]]:
    schema = pq.read_schema(parquet_path)
    return [(field.name, str(field.type)) for field in schema]


def _print_tree_line(prefix: str, connector: str, text: str) -> None:
    print(f"{prefix}{connector}{text}")


def _walk_and_print(
    root: Path,
    info: dict[str, Any] | None,
    *,
    parquet_rel: str | None,
    with_sample: bool,
    show_siblings: bool,
) -> None:
    root_name = root.name if root.name else str(root)
    print(f"{root_name}/")

    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []
    if parquet_rel is not None:
        parquet_files = [root / parquet_rel]

    if not parquet_files:
        _print_tree_line("", "├── ", "data/  (no parquet found)")
    else:
        for i, pq_path in enumerate(parquet_files):
            rel = pq_path.relative_to(root)
            is_last_data = (i == len(parquet_files) - 1) and not show_siblings
            branch = "└── " if is_last_data else "├── "
            pf = pq.ParquetFile(pq_path)
            n_rows = pf.metadata.num_rows
            _print_tree_line("", branch, f"{rel}  ({n_rows} rows)")

            cols = _parquet_columns(pq_path)
            sample_stats = (
                _collect_sample_stats(pq_path, [c for c, _ in cols])
                if with_sample
                else None
            )
            child_prefix = "    " if is_last_data else "│   "
            for j, (col_name, arrow_type) in enumerate(cols):
                last_col = j == len(cols) - 1
                conn = "└── " if last_col else "├── "
                desc = _describe_parquet_column(
                    col_name, arrow_type, info, sample_stats
                )
                _print_tree_line(child_prefix, conn, desc)

    if not show_siblings:
        return

    if info:
        video_keys = [
            k
            for k, v in info.get("features", {}).items()
            if v.get("dtype") == "video"
        ]
        if video_keys:
            _print_tree_line("", "├── ", "videos/  (external, joined at load)")
            for i, vk in enumerate(video_keys):
                vmeta = info["features"][vk]
                shape = vmeta.get("shape", [])
                vinfo = vmeta.get("info", {})
                codec = vinfo.get("video.codec", "?")
                fps = vinfo.get("video.fps", info.get("fps", "?"))
                rel_glob = f"videos/{vk}/chunk-*/file-*.mp4"
                last = i == len(video_keys) - 1
                conn = "└── " if last else "├── "
                shape_s = (
                    f"{shape[0]}x{shape[1]}x{shape[2]}"
                    if len(shape) == 3
                    else str(shape)
                )
                _print_tree_line("│   ", conn, f"{rel_glob}  ({codec}, {fps} fps, {shape_s})")

    meta_dir = root / "meta"
    if meta_dir.is_dir():
        _print_tree_line("", "├── ", "meta/")
        for name in ("info.json", "stats.json", "tasks.parquet"):
            if (meta_dir / name).exists():
                _print_tree_line("│   ", "├── ", name)
        ep_dir = meta_dir / "episodes"
        if ep_dir.is_dir():
            for ep_pq in sorted(ep_dir.rglob("*.parquet")):
                rel = ep_pq.relative_to(root)
                _print_tree_line("│   ", "├── ", str(rel))

    debug_dir = root / "debug"
    if debug_dir.is_dir():
        _print_tree_line("", "├── ", "debug/  (not in parquet)")
        for sub in sorted(debug_dir.iterdir()):
            if sub.is_dir():
                npz = sorted(sub.glob("*.npz"))
                _print_tree_line("│   ", "├── ", f"{sub.name}/  ({len(npz)} episode npz)")

    dexmate_meta = root / "dexmate_meta.json"
    if dexmate_meta.is_file():
        dm = json.loads(dexmate_meta.read_text())
        _print_tree_line(
            "",
            "└── ",
            f"dexmate_meta.json  ({dm.get('robot_type', '?')}, task={dm.get('task', '?')})",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print LeRobot dataset / parquet structure as a tree."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=Path(
            "/home/yixuan/Dexmate/data/processed_data/train/dexmate_eef_eef"
        ),
        help="LeRobot variant root (contains data/, meta/, videos/)",
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Only describe this parquet path relative to dataset_root",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Add row counts / index ranges from the parquet file",
    )
    parser.add_argument(
        "--no-siblings",
        action="store_true",
        help="Only print data/*.parquet, skip videos/meta/debug",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        raise SystemExit(1)

    info = _load_info(root)
    if info:
        version = info.get("codebase_version", "?")
        if version != "?" and not str(version).startswith("v"):
            version = f"v{version}"
        print(
            f"# LeRobot {version}, "
            f"fps={info.get('fps')}, "
            f"episodes={info.get('total_episodes')}, "
            f"frames={info.get('total_frames')}\n"
        )

    _walk_and_print(
        root,
        info,
        parquet_rel=args.parquet,
        with_sample=args.sample,
        show_siblings=not args.no_siblings,
    )


if __name__ == "__main__":
    main()
