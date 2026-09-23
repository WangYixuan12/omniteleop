"""Local reviewer HTTP contract; run with loopback networking available."""

import importlib.util
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pytest

from omniteleop.wrist_corruption import annotate_episode, make_annotations, source_identity


def test_http_all_episodes_positive_marks_and_request_protection(tmp_path):
    for name in ("episode_0.hdf5", "episode_1.hdf5", "episode_8.hdf5.partial"):
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            f.attrs.update(complete=not name.endswith(".partial"), n_frames=3)
            f["timestamp_ns"] = np.arange(3)
            for eye in ("left_wrist_rgb", "left_wrist_right_rgb"):
                f["obs/images/" + eye] = np.zeros((3, 36, 64, 3), dtype=np.uint8)
        arrays = make_annotations(
            np.array([0.2, 0.8, 0]), [], {"threshold": 0.35, "review_threshold": 0.1}
        )
        old = time.time() - 20
        os.utime(path, (old, old))
        annotate_episode(path, source_identity(path), arrays, {})
    script = Path(__file__).resolve().parents[1] / "scripts/diagnostics/review_wrist_corruption.py"
    spec = importlib.util.spec_from_file_location("review_http", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.ReviewServer(("127.0.0.1", 0), tmp_path) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def request(route, value=None, token=None):
            headers = {} if token is None else {"X-Review-Token": token}
            data = json.dumps(value).encode() if value is not None else None
            req = urllib.request.Request(base + route, data=data, headers=headers)
            with urllib.request.urlopen(req) as r:
                return r.read()

        try:
            assert len(json.loads(request("/api/episodes"))) == 3
            assert b"__TOKEN__" not in request("/")
            partial_payload = {
                "file": "episode_8.hdf5.partial",
                "frame": 0,
                "marked": True,
                "revision": 0,
            }
            request("/api/mark", partial_payload, server.token)
            with h5py.File(tmp_path / "episode_8.hdf5.partial", "r") as partial:
                assert not partial.attrs["complete"]
                assert partial["quality/left_wrist/corrupted"][0]
            before = (tmp_path / "episode_0.hdf5").stat().st_mtime_ns
            state = json.loads(request("/api/episode?file=episode_0.hdf5"))
            assert "manual_corrupted" not in state
            assert request("/api/image?file=episode_0.hdf5&frame=0").startswith(b"\xff\xd8")
            assert (tmp_path / "episode_0.hdf5").stat().st_mtime_ns == before
            payload = {"file": "episode_0.hdf5", "frame": 0, "marked": True, "revision": 0}
            with pytest.raises(urllib.error.HTTPError) as exc:
                request("/api/mark", payload)
            assert exc.value.code == 403
            assert json.loads(request("/api/mark", payload, server.token))["revision"] == 1
            state = json.loads(request("/api/episode?file=episode_0.hdf5"))
            assert state["review_decision"][0] == 1 and state["corrupted"][0]
            assert state["review_candidate"][0]
            payload.update(marked=False, revision=1)
            request("/api/mark", payload, server.token)
            state = json.loads(request("/api/episode?file=episode_0.hdf5"))
            assert state["review_decision"][0] == -1 and not state["corrupted"][0]
            with pytest.raises(urllib.error.HTTPError):
                request("/api/image?file=../escape.hdf5&frame=0")
            with h5py.File(tmp_path / "episode_0.hdf5", "r") as f:
                assert "manual_corrupted" not in f["quality/left_wrist"]
        finally:
            server.shutdown()
            thread.join(timeout=5)
