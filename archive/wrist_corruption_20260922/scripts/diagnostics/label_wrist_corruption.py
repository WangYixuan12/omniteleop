#!/usr/bin/env python3
"""Label *.hdf5 and stable *.hdf5.partial files, preserving human edits.

python scripts/diagnostics/label_wrist_corruption.py
python scripts/diagnostics/label_wrist_corruption.py --watch 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from annotate_wrist_corruption import extract

from omniteleop.wrist_corruption import (
    ANNOTATION_KEY,
    FEATURE_VERSION,
    annotate_episode,
    episode_paths,
    forest_scores,
    make_annotations,
)

REPO = Path(__file__).resolve().parents[2]


def run_batch(args) -> int:
    model_bytes = args.model.read_bytes()
    model = json.loads(model_bytes)
    results = []
    for path in episode_paths(args.data_dir):
        try:
            with h5py.File(path, "r") as f:
                if not bool(f.attrs.get("complete", False)) and not path.name.endswith(
                    ".hdf5.partial"
                ):
                    print(f"{path.name}: skip incomplete", flush=True)
                    continue
                if ANNOTATION_KEY in f:
                    print(f"{path.name}: keep existing annotations and human edits", flush=True)
                    continue
            features, identity = extract(path, args.work_dir / "cache", args.workers)
            arrays = make_annotations(forest_scores(features, model), [], model)
            provenance = {
                "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                "feature_version": FEATURE_VERSION,
                "threshold": model["threshold"],
                "review_threshold": model["review_threshold"],
                "source_before_annotation": identity,
            }
            annotate_episode(path, identity, arrays, provenance)
            count = int(arrays["corrupted"].sum())
            review = int(arrays["review_candidate"].sum())
            print(f"{path.name}: saved {count} flagged, {review} review candidates", flush=True)
            results.append({"file": str(path), "flagged": count, "review_candidates": review})
        except (OSError, ValueError, KeyError) as exc:
            print(f"{path.name}: ERROR {exc}", file=sys.stderr, flush=True)
            results.append({"file": str(path), "error": str(exc)})
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report = args.work_dir / f"batch_{time.time_ns()}.json"
    report.write_text(json.dumps(results, indent=2) + "\n")
    return int(any("error" in item for item in results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/Dexmate/data"))
    parser.add_argument(
        "--model", type=Path, default=REPO / "configs/camera_quality/left_wrist_v1.json"
    )
    parser.add_argument("--work-dir", type=Path, default=REPO / "tmp/wrist_corruption/batch")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        metavar="SECONDS",
        help="Repeat periodically; 0 means run once",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.watch < 0 or not args.data_dir.is_dir():
        parser.error("need an existing data directory, workers >=1, watch >=0")
    try:
        while True:
            status = run_batch(args)
            if not args.watch:
                raise SystemExit(status)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
