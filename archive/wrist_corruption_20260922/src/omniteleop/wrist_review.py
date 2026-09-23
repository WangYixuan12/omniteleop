"""Human review decisions layered over immutable detector predictions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from omniteleop.wrist_corruption import ANNOTATION_KEY, validate_recording


def load_review(path: Path) -> dict:
    """Read a fresh snapshot without retaining a file handle between GUI actions."""
    with h5py.File(path, "r") as f:
        validate_recording(f, path)
        group = f[ANNOTATION_KEY]
        if "pending_review_json" in group.attrs:
            raise ValueError("Interrupted review write; run reviewer with --recover first")
        n = len(f["timestamp_ns"])
        values = {
            k: group[k][:]
            for k in (
                "corrupted",
                "auto_corrupted",
                "review_candidate",
                "score",
            )
        }
        values["manual_corrupted"] = (
            group["manual_corrupted"][:] if "manual_corrupted" in group else np.zeros(n, dtype=bool)
        )
        values["review_decision"] = (
            group["review_decision"][:]
            if "review_decision" in group
            else np.full(n, -1, dtype=np.int8)
        )
        if any(a.shape != (n,) for a in values.values()):
            raise ValueError("Annotation lengths do not match episode")
        values["revision"] = int(group.attrs.get("review_revision", 0))
        return values


def candidate_indices(state: dict, mode: str, unreviewed_only: bool) -> np.ndarray:
    if mode == "review":
        mask = state["review_candidate"].copy()
    elif mode == "auto":
        mask = state["auto_corrupted"].copy()
    elif mode == "flagged":
        mask = state["review_candidate"] | state["auto_corrupted"] | state["manual_corrupted"]
    elif mode == "all":
        mask = np.ones(len(state["score"]), dtype=bool)
    else:
        raise ValueError(f"Unknown filter: {mode}")
    if unreviewed_only:
        mask &= state["review_decision"] == -1
    return np.flatnonzero(mask)


def _finish_event(group, event: dict) -> None:
    """Idempotent completion of a journaled edit, including recovery after interruption."""
    n = len(group["score"])
    if "review_decision" not in group:
        ds = group.create_dataset("review_decision", data=np.full(n, -1, dtype=np.int8))
        ds.attrs["values"] = "-1=unreviewed/use original flags; 0=clean; 1=corrupted"
    if "review_updated_ns" not in group:
        group.create_dataset("review_updated_ns", data=np.zeros(n, dtype=np.int64))
    if "review_history" not in group:
        group.create_dataset(
            "review_history",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype("utf-8"),
            chunks=True,
        )
    i, decision = event["frame"], event["decision"]
    group["review_decision"][i] = decision
    group["review_updated_ns"][i] = event["time_ns"]
    group["corrupted"][i] = (
        decision == 1
        if decision != -1
        else bool(
            group["auto_corrupted"][i]
            or (group["manual_corrupted"][i] if "manual_corrupted" in group else False)
        )
    )
    history = group["review_history"]
    index = event["history_index"]
    if len(history) not in (index, index + 1):
        raise ValueError("Review history conflicts with pending event")
    if len(history) == index:
        history.resize(index + 1, axis=0)
    history[index] = json.dumps(event, sort_keys=True)
    group.attrs["review_revision"] = event["revision"]
    group.attrs["semantics"] = "corrupted = human decision if reviewed, else manual OR auto"
    # The initial detector digest no longer describes the human-edited effective mask.
    if "arrays_sha256" in group.attrs:
        group.attrs["initial_arrays_sha256"] = group.attrs["arrays_sha256"]
        del group.attrs["arrays_sha256"]


def save_decision(path: Path, frame: int, decision: int, expected_revision: int) -> int:
    """Save one edit with optimistic concurrency checking and a replayable journal."""
    if decision not in (-1, 0, 1):
        raise ValueError("decision must be -1, 0, or 1")
    with h5py.File(path, "r+") as f:
        validate_recording(f, path)
        group = f[ANNOTATION_KEY]
        if "pending_review_json" in group.attrs:
            raise ValueError("Interrupted review write; use --recover")
        revision = int(group.attrs.get("review_revision", 0))
        if revision != expected_revision:
            raise ValueError("Labels changed in another reviewer. Reload before editing.")
        if not 0 <= frame < len(group["score"]):
            raise ValueError("Frame index out of range")
        previous = int(group["review_decision"][frame]) if "review_decision" in group else -1
        event = {
            "frame": frame,
            "decision": decision,
            "previous": previous,
            "revision": revision + 1,
            "time_ns": time.time_ns(),
            "history_index": len(group["review_history"]) if "review_history" in group else 0,
        }
        group.attrs["pending_review_json"] = json.dumps(event)
        f.flush()
        _finish_event(group, event)
        f.flush()
        del group.attrs["pending_review_json"]
        f.flush()
        return revision + 1


def recover_review(path: Path) -> bool:
    with h5py.File(path, "r+") as f:
        group = f[ANNOTATION_KEY]
        if "pending_review_json" not in group.attrs:
            return False
        _finish_event(group, json.loads(group.attrs["pending_review_json"]))
        f.flush()
        del group.attrs["pending_review_json"]
        f.flush()
        return True
