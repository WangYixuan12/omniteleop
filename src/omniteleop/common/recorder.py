"""Shared episode recorder used by leader (`vr_reader`) and follower
(`policy_rollout`) scripts.

Frames are accumulated in memory as nested dicts of numpy arrays during an
episode and dumped to a single HDF5 file in a daemon thread on `stop()`.
"""

from __future__ import annotations

import pathlib
import re
import threading
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from loguru import logger


def _recursive_np_stack(list_of_dicts: list[dict]) -> dict:
    """Recursively np.stack a list of nested dicts into a single nested dict of arrays."""
    result: dict = {}
    for key in list_of_dicts[0]:
        val = list_of_dicts[0][key]
        if isinstance(val, dict):
            result[key] = _recursive_np_stack([d[key] for d in list_of_dicts])
        else:
            result[key] = np.stack([d[key] for d in list_of_dicts])
    return result


def _count_leaves(d: dict) -> int:
    n = 0
    for v in d.values():
        n += _count_leaves(v) if isinstance(v, dict) else 1
    return n


def _save_dict_with_progress(data: dict, path: str, on_progress: Callable[[float], None]) -> None:
    """Save nested dict-of-ndarrays to HDF5; ``on_progress(frac in [0,1])`` is
    called repeatedly as datasets are written. Large arrays are chunked along
    axis 0 so the bar advances smoothly mid-leaf.
    """
    import h5py

    total = max(1, _count_leaves(data))
    done = [0.0]

    def step(frac_in_leaf: float) -> None:
        on_progress(min(1.0, (done[0] + frac_in_leaf) / total))

    def end_leaf() -> None:
        done[0] += 1.0
        on_progress(min(1.0, done[0] / total))

    def recurse(group: Any, prefix: str, sub: dict) -> None:
        for key, item in sub.items():
            if isinstance(item, np.ndarray):
                dset = group.create_dataset(prefix + key, shape=item.shape, dtype=item.dtype)
                if item.ndim >= 1 and item.shape[0] > 1 and item.nbytes > 32 * 1024 * 1024:
                    n = item.shape[0]
                    step_size = max(1, n // 8)
                    for start in range(0, n, step_size):
                        end = min(start + step_size, n)
                        dset[start:end] = item[start:end]
                        step(end / n)
                else:
                    dset[...] = item
                end_leaf()
            elif isinstance(item, dict):
                recurse(group, prefix + key + "/", item)
            else:
                raise ValueError(f"Cannot save {type(item)}")

    on_progress(0.0)
    with h5py.File(path, "w") as h5file:
        recurse(h5file, "/", data)


def _progress_bar(p: float, width: int = 15) -> str:
    p = max(0.0, min(1.0, p))
    filled = int(round(width * p))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {p * 100:3.0f}%"


class EpisodeRecorder:
    """Accumulates per-frame data and saves to HDF5 on stop()."""

    _ID_RE = re.compile(r"episode_(\d+)\.hdf5$")

    def __init__(self, save_dir: str) -> None:
        self._save_dir = pathlib.Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._frames: list[dict] = []
        self.recording = False
        self.saving = False
        self.save_progress: float = 0.0
        self._save_thread: Optional[threading.Thread] = None
        existing_ids = (
            int(match.group(1))
            for p in self._save_dir.glob("episode_*.hdf5")
            if (match := self._ID_RE.fullmatch(p.name)) is not None
        )
        self.episode_id = max(existing_ids, default=-1) + 1

    def start(self) -> None:
        """Start a new episode recording."""
        self._frames = []
        self.recording = True
        logger.info("EpisodeRecorder: recording started")

    def record(self, frame: dict) -> None:
        """Record a frame (VRJointData as dict)."""
        self._frames.append(frame)

    def stop(self, extra: Optional[dict] = None) -> Optional[str]:
        """Stop recording; spawn background thread to save HDF5. Returns the
        target path immediately (file is written asynchronously).

        Args:
            extra: Optional nested dict-of-ndarrays merged at the top level of
                the saved file before stacked frames. Used for session metadata
                (e.g., NTP offsets, clock-source map) that doesn't fit a
                per-frame shape.
        """
        self.recording = False
        if not self._frames:
            logger.warning("EpisodeRecorder: 0 frames — skipping save")
            return None
        path = self._save_dir / f"episode_{self.episode_id}.hdf5"
        frames = self._frames
        self._frames = []
        self.episode_id += 1
        self.saving = True
        self.save_progress = 0.0
        self._save_thread = threading.Thread(
            target=self._save_worker, args=(frames, str(path), extra), daemon=True
        )
        self._save_thread.start()
        return str(path)

    def _save_worker(self, frames: list[dict], path: str, extra: Optional[dict] = None) -> None:
        try:
            data = _recursive_np_stack(frames)
            if extra:
                for k, v in extra.items():
                    if k in data:
                        raise ValueError(f"extra key {k!r} collides with frame data")
                    data[k] = v

            def on_progress(p: float) -> None:
                self.save_progress = p

            _save_dict_with_progress(data, path, on_progress)
            logger.info(f"EpisodeRecorder: {len(frames)} frames → {path}")
        except Exception:
            logger.exception("EpisodeRecorder: save failed")
        finally:
            self.saving = False

    def num_frames(self) -> int:
        """Return number of frames recorded so far"""
        return len(self._frames)


class FollowerEpisodeRecorder:
    """Per-tick recorder used by `vr_robot_controller` to log apply/obs timing.

    Unlike `EpisodeRecorder`, the episode_id is assigned externally by the
    leader (via VRJointData.episode_id) so leader and follower files share IDs.
    Files are saved as ``episode_<N>_follower.hdf5``.
    """

    def __init__(self, save_dir: str) -> None:
        self._save_dir = pathlib.Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._frames: list[dict] = []
        self.recording = False
        self.saving = False
        self.save_progress: float = 0.0
        self._save_thread: Optional[threading.Thread] = None
        self._episode_id: int = -1

    def start(self, episode_id: int) -> None:
        self._frames = []
        self._episode_id = episode_id
        self.recording = True
        logger.info(f"FollowerEpisodeRecorder: recording started (episode_{episode_id})")

    def record(self, frame: dict) -> None:
        self._frames.append(frame)

    def stop(self, extra: Optional[dict] = None) -> Optional[str]:
        self.recording = False
        if not self._frames:
            logger.warning("FollowerEpisodeRecorder: 0 frames — skipping save")
            return None
        path = self._save_dir / f"episode_{self._episode_id}_follower.hdf5"
        frames = self._frames
        self._frames = []
        self.saving = True
        self.save_progress = 0.0
        self._save_thread = threading.Thread(
            target=self._save_worker, args=(frames, str(path), extra), daemon=True
        )
        self._save_thread.start()
        return str(path)

    def _save_worker(self, frames: list[dict], path: str, extra: Optional[dict] = None) -> None:
        try:
            data = _recursive_np_stack(frames)
            if extra:
                for k, v in extra.items():
                    if k in data:
                        raise ValueError(f"extra key {k!r} collides with frame data")
                    data[k] = v

            def on_progress(p: float) -> None:
                self.save_progress = p

            _save_dict_with_progress(data, path, on_progress)
            logger.info(f"FollowerEpisodeRecorder: {len(frames)} frames → {path}")
        except Exception:
            logger.exception("FollowerEpisodeRecorder: save failed")
        finally:
            self.saving = False

    def num_frames(self) -> int:
        return len(self._frames)
