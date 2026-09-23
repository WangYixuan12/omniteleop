#!/usr/bin/env python3
"""Train, scan, and annotate completed raw episodes without removing any data.

See docs/wrist_corruption.md. `scan` is read-only; only `apply` modifies HDF5,
and it only adds quality/left_wrist. Partial recordings are never included.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from omniteleop.wrist_corruption import (
    ANNOTATION_KEY,
    FEATURE_VERSION,
    IMAGE_KEYS,
    annotate_episode,
    forest_scores,
    frame_features,
    make_annotations,
    read_gray,
    read_labels,
    source_identity,
)

_GRAY = None


def _init_worker(path: str) -> None:
    global _GRAY
    cv2.setNumThreads(1)
    _GRAY = np.load(path, mmap_mode="r")


def _features_chunk(bounds: tuple[int, int]) -> np.ndarray:
    return np.asarray([frame_features(_GRAY, i) for i in range(*bounds)], dtype=np.float32)


def extract(path: Path, directory: Path, workers: int) -> tuple[np.ndarray, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    identity = source_identity(path)
    feature_file = directory / f"{path.stem}.features.npz"
    if feature_file.exists():
        with np.load(feature_file, allow_pickle=False) as cached:
            if (
                str(cached["feature_version"]) == FEATURE_VERSION
                and json.loads(str(cached["identity"])) == identity
            ):
                print(f"{path.name}: reuse matching feature cache", flush=True)
                return cached["features"], identity
    gray_file = directory / f"{path.stem}.gray.npy"
    print(f"{path.name}: reading {identity['frames']} stereo frames", flush=True)
    gray = read_gray(path, gray_file)
    del gray
    bounds = [(i, min(i + 64, identity["frames"])) for i in range(0, identity["frames"], 64)]
    chunks = []
    # Spawn avoids inheriting HDF5 handles and OpenCV thread pools.
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(str(gray_file),),
    ) as pool:
        for i, chunk in enumerate(pool.map(_features_chunk, bounds)):
            chunks.append(chunk)
            if (i + 1) % 16 == 0:
                print(f"{path.name}: features {bounds[i][1]}/{identity['frames']}", flush=True)
    if source_identity(path) != identity:
        raise ValueError(f"{path}: changed during feature extraction")
    features = np.concatenate(chunks)
    np.savez_compressed(
        feature_file,
        features=features,
        identity=json.dumps(identity),
        feature_version=FEATURE_VERSION,
    )
    gray_file.unlink()
    return features, identity


def evaluate_labels(
    labels: np.ndarray, scores: np.ndarray, threshold: float, offset: int = 0
) -> dict:
    predicted = scores >= threshold
    found = int(np.sum(labels & predicted))
    return {
        "labeled_positive_count": int(labels.sum()),
        "labeled_positives_detected": found,
        "labeled_positive_recall": found / int(labels.sum()) if labels.any() else None,
        "missed_label_indices": (np.flatnonzero(labels & ~predicted) + offset).tolist(),
        "additional_detection_indices": (np.flatnonzero(~labels & predicted) + offset).tolist(),
        "note": "Unlabeled frames are not verified negatives; precision is not established.",
    }


def export_forest(estimator, threshold: float, review_threshold: float) -> dict:
    trees = []
    for tree_estimator in estimator.estimators_:
        tree = tree_estimator.tree_
        values = tree.value[:, 0, :]
        trees.append(
            {
                "left": tree.children_left.tolist(),
                "right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "positive": (values[:, 1] / values.sum(axis=1)).tolist(),
            }
        )
    return {
        "feature_version": FEATURE_VERSION,
        "feature_count": estimator.n_features_in_,
        "threshold": threshold,
        "review_threshold": review_threshold,
        "trees": trees,
    }


def train(args) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    features, identity = extract(args.episode, args.work_dir, args.workers)
    labels_by_episode = read_labels(args.labels, args.index_base)
    indices = labels_by_episode.get(args.episode.stem, [])
    if not indices or max(indices) >= len(features):
        raise ValueError("no matching labels or an out-of-range label")
    labels = np.isin(np.arange(len(features)), indices)
    split = 2 * len(features) // 3
    # The final third is evaluated once, before the production model is refit on
    # all labels. Purge temporal context at the train/test boundary.
    fit_end = split - 3
    if labels[:fit_end].sum() < 6 or labels[split:].sum() < 2:
        raise ValueError("need at least six training positives and two held-out positives")
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        max_features=0.6,
        n_jobs=args.workers,
        random_state=42,
    )
    oof = np.zeros(fit_end, dtype=np.float32)
    for tr, va in GroupKFold(3).split(
        features[:fit_end], labels[:fit_end], groups=np.arange(fit_end) // 350
    ):
        allowed = np.ones(fit_end, dtype=bool)
        for offset in range(-3, 4):
            allowed[np.clip(va + offset, 0, fit_end - 1)] = False
        tr = tr[allowed[tr]]
        model.fit(features[tr], labels[tr])
        oof[va] = model.predict_proba(features[va])[:, 1]
    model.fit(features[:fit_end], labels[:fit_end])
    held_out = model.predict_proba(features[split:])[:, 1]
    evaluation = {
        "source": identity,
        "feature_version": FEATURE_VERSION,
        "index_base_in_label_file": args.index_base,
        "threshold": args.threshold,
        "review_threshold": args.review_threshold,
        "development_blocked_cv": evaluate_labels(labels[:fit_end], oof, args.threshold),
        "held_out_start_index": split,
        "held_out": evaluate_labels(labels[split:], held_out, args.threshold, split),
        "held_out_including_review": evaluate_labels(
            labels[split:], held_out, args.review_threshold, split
        ),
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    np.savez_compressed(
        args.work_dir / "evaluation_scores.npz",
        dev_oof=oof,
        held_out=held_out,
        held_out_start=split,
    )
    model.fit(features, labels)
    exported = export_forest(model, args.threshold, args.review_threshold)
    exported["training"] = {
        "episode": identity,
        "labels": indices,
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "evaluation": evaluation,
        "assumption": "unlabeled training rows are provisional negatives",
    }
    if not np.allclose(
        forest_scores(features, exported), model.predict_proba(features)[:, 1], atol=1e-6
    ):
        raise RuntimeError("JSON forest export does not match sklearn inference")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(exported, separators=(",", ":")) + "\n")
    print(json.dumps(evaluation, indent=2), flush=True)
    print(f"Model saved to {args.model}; production fit includes all supplied labels", flush=True)


def gallery(path: Path, scores: np.ndarray, arrays: dict, directory: Path) -> None:
    """Self-contained local HTML review: original row indices, scores, and both eyes."""
    out = directory / path.stem
    out.mkdir(parents=True, exist_ok=True)
    indices = np.flatnonzero(arrays["corrupted"] | arrays["review_candidate"])
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Wrist corruption review</title>",
        "<style>body{font:16px sans-serif;max-width:1050px;margin:24px auto}"
        "img{max-width:100%}article{margin-bottom:32px}code{background:#eee}</style>",
        f"<h1>{html.escape(path.name)}</h1><p>Zero-based original HDF5 rows. "
        "Each image shows the two stereo eyes of the left wrist. "
        "False in the mask does not certify an image as clean.</p>",
    ]
    with h5py.File(path, "r") as f:
        for i in indices:
            tags = [key for key in ("auto_corrupted", "review_candidate") if arrays[key][i]]
            images = [cv2.resize(f[key][int(i)], (480, 270)) for key in IMAGE_KEYS]
            filename = f"{i:06d}.jpg"
            cv2.imwrite(
                str(out / filename), cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_RGB2BGR)
            )
            parts.append(
                f"<article id='frame-{i}'><h2>Frame {i} · score {scores[i]:.3f}</h2>"
                f"<p>{', '.join(tags)}</p><img loading='lazy' src='{filename}'></article>"
            )
    (out / "index.html").write_text("\n".join(parts))


def scan(args) -> None:
    model_bytes = args.model.read_bytes()
    model = json.loads(model_bytes)
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    labels = read_labels(args.labels, args.index_base) if args.labels else {}
    paths = sorted(
        args.data_dir.glob("episode_*.hdf5"),
        key=lambda path: int(path.stem.removeprefix("episode_")),
    )
    if args.episode_ids:
        selected = set(args.episode_ids)
        paths = [path for path in paths if int(path.stem.removeprefix("episode_")) in selected]
    if args.skip_annotated:
        remaining = []
        for path in paths:
            with h5py.File(path, "r") as source:
                if ANNOTATION_KEY in source:
                    print(f"{path.name}: skipping existing annotations", flush=True)
                else:
                    remaining.append(path)
        paths = remaining
    if not paths:
        print("No matching unannotated completed episodes.", flush=True)
        return
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "wrist_corruption_scan_v1",
        "scan_complete": False,
        "model_sha256": model_hash,
        "model_path": str(args.model.resolve()),
        "feature_version": FEATURE_VERSION,
        "threshold": model["threshold"],
        "review_threshold": model["review_threshold"],
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest()
        if args.labels
        else None,
        "label_index_base": args.index_base,
        "excluded_partial_files": [p.name for p in args.data_dir.glob("*.hdf5.partial")],
        "episodes": [],
    }
    for path in paths:
        features, identity = extract(path, args.report_dir / "cache", args.workers)
        scores = forest_scores(features, model)
        manual = labels.get(path.stem, [])
        arrays = make_annotations(scores, manual, model)
        filename = f"{path.stem}.annotations.npz"
        np.savez_compressed(args.report_dir / filename, **arrays)
        with (args.report_dir / f"{path.stem}.csv").open("w") as output:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "frame_index",
                    "score",
                    "corrupted",
                    "auto_corrupted",
                    "review_candidate",
                ]
            )
            for i in range(len(scores)):
                writer.writerow(
                    [
                        i,
                        f"{scores[i]:.6f}",
                        *[
                            int(arrays[k][i])
                            for k in (
                                "corrupted",
                                "auto_corrupted",
                                "review_candidate",
                            )
                        ],
                    ]
                )
        item = {
            "source": identity,
            "annotations_file": filename,
            "annotations_file_sha256": hashlib.sha256(
                (args.report_dir / filename).read_bytes()
            ).hexdigest(),
            "counts": {key: int(values.sum()) for key, values in arrays.items() if key != "score"},
            "corrupted_indices": np.flatnonzero(arrays["corrupted"]).tolist(),
            "review_indices": np.flatnonzero(arrays["review_candidate"]).tolist(),
        }
        manifest["episodes"].append(item)
        if args.gallery:
            gallery(path, scores, arrays, args.report_dir / "review")
        print(f"{path.name}: {item['counts']}", flush=True)
        # Incremental manifest makes completed scans reviewable if a later episode fails.
        (args.report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["scan_complete"] = True
    (args.report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_index(manifest, args.report_dir, args.gallery)


def write_index(manifest: dict, report_dir: Path, has_gallery: bool) -> None:
    table = []
    for item in manifest["episodes"]:
        name = Path(item["source"]["path"]).stem
        counts = item["counts"]
        link = f"<a href='review/{name}/index.html'>{name}</a>" if has_gallery else name
        table.append(
            f"<tr><td>{link}</td><td>{item['source']['frames']}</td>"
            f"<td>{counts['corrupted']}</td><td>{counts.get('manual_corrupted', 0)}</td>"
            f"<td>{counts['review_candidate']}</td><td><a href='{name}.csv'>CSV</a></td></tr>"
        )
    (report_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Left wrist quality scan</title>"
        "<style>body{font:16px sans-serif;margin:32px}td,th{padding:8px;text-align:left}"
        "table{border-collapse:collapse}tr{border-bottom:1px solid #ccc}</style>"
        "<h1>Left wrist quality scan</h1><p>Original zero-based frame indices. "
        "Corrupted = manual or automatic flags. Review candidates are separate. "
        "Automatic flags can be wrong; unflagged frames are not certified clean. "
        "No original frames have been removed.</p>"
        "<table><tr><th>Episode</th><th>Frames</th><th>Flagged</th><th>Manual</th>"
        "<th>Review</th><th>Scores</th></tr>" + "".join(table) + "</table>"
        "<p>See <a href='manifest.json'>manifest.json</a> for source identity and provenance.</p>"
    )


def apply(args) -> None:
    manifest = json.loads((args.report_dir / "manifest.json").read_text())
    if manifest["schema"] != "wrist_corruption_scan_v1":
        raise ValueError("unknown scan manifest schema")
    if not manifest.get("scan_complete", False):
        raise ValueError("scan did not complete; resume scan before applying annotations")
    results = []
    for item in manifest["episodes"]:
        path = Path(item["source"]["path"])
        if path.parent.resolve() != args.data_dir.resolve():
            raise ValueError(f"manifest source is outside --data-dir: {path}")
        annotation_path = args.report_dir / item["annotations_file"]
        if (
            hashlib.sha256(annotation_path.read_bytes()).hexdigest()
            != item["annotations_file_sha256"]
        ):
            raise ValueError(f"{annotation_path}: changed since scan")
        with np.load(annotation_path, allow_pickle=False) as f:
            arrays = {key: f[key] for key in f.files}
        provenance = {
            key: manifest[key]
            for key in (
                "model_sha256",
                "feature_version",
                "threshold",
                "review_threshold",
                "labels_sha256",
                "label_index_base",
            )
        }
        provenance["source_before_annotation"] = item["source"]
        status = annotate_episode(path, item["source"], arrays, provenance)
        results.append({"path": str(path), "status": status, "counts": item["counts"]})
        print(f"{path.name}: {status}", flush=True)
        (args.report_dir / "applied.json").write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--episode", type=Path, required=True)
    training.add_argument("--labels", type=Path, required=True)
    training.add_argument("--model", type=Path, required=True)
    training.add_argument("--work-dir", type=Path, required=True)
    training.add_argument("--threshold", type=float, default=0.35)
    training.add_argument("--review-threshold", type=float, default=0.1)
    training.add_argument("--index-base", type=int, choices=(0, 1), default=0)
    training.add_argument("--workers", type=int, default=4)
    scanning = commands.add_parser("scan")
    scanning.add_argument("--data-dir", type=Path, required=True)
    scanning.add_argument("--model", type=Path, required=True)
    scanning.add_argument("--report-dir", type=Path, required=True)
    scanning.add_argument("--labels", type=Path)
    scanning.add_argument("--index-base", type=int, choices=(0, 1), default=0)
    scanning.add_argument("--episode-ids", nargs="+", type=int)
    scanning.add_argument("--workers", type=int, default=4)
    scanning.add_argument("--gallery", action="store_true")
    scanning.add_argument(
        "--skip-annotated",
        action="store_true",
        help="Only scan new completed episodes; keep existing annotations",
    )
    applying = commands.add_parser("apply")
    applying.add_argument("--data-dir", type=Path, required=True)
    applying.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(args, "workers") and args.workers < 1:
        parser.error("--workers must be >=1")
    if args.command == "train" and not 0 <= args.review_threshold < args.threshold <= 1:
        parser.error("need 0 <= review-threshold < threshold <= 1")
    {"train": train, "scan": scan, "apply": apply}[args.command](args)


if __name__ == "__main__":
    main()
