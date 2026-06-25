"""Small helpers for SAPIEN video recording."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoRecordingConfig:
    """Deferred video writer settings."""

    path: str
    fps: float
    size: tuple[int, int]


def prepare_video_output_path(path: str | os.PathLike[str]) -> str:
    """Prepare ``path`` for a new recording without creating an empty MP4."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    return str(out)


def make_video_recording_config(
    path: str | os.PathLike[str],
    fps: float,
    size: tuple[int, int],
) -> VideoRecordingConfig:
    """Build a deferred recording config and remove any stale output file."""
    return VideoRecordingConfig(
        path=prepare_video_output_path(path),
        fps=float(fps),
        size=(int(size[0]), int(size[1])),
    )
