#!/usr/bin/env python3
"""Review all episodes in a local browser. X toggles a corrupted mark; navigation never labels.

python scripts/diagnostics/review_wrist_corruption.py
Open http://localhost:8765. No desktop/remote-desktop session or text labels needed.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from omniteleop.wrist_corruption import (
    ANNOTATION_KEY,
    IMAGE_KEYS,
    episode_paths,
    validate_recording,
)
from omniteleop.wrist_review import load_review, recover_review, save_decision


class ReviewServer(HTTPServer):
    """Loopback-only, single-writer server with CSRF protection and confined file access."""

    def __init__(self, address, data_dir: Path, episode: str | None = None):
        self.data_dir = data_dir.resolve()
        self.token = secrets.token_urlsafe(32)
        self.episode = episode
        super().__init__(address, Handler)

    def episode_path(self, name: str) -> Path:
        """Accept only an immediate HDF5 file inside the configured directory."""
        path = (self.data_dir / name).resolve()
        if (
            path.parent != self.data_dir
            or not path.name.endswith((".hdf5", ".hdf5.partial"))
            or not path.is_file()
        ):
            raise ValueError("Invalid episode filename")
        if self.episode is not None and path.name != self.episode:
            raise ValueError("Episode is outside the requested selection")
        return path


class Handler(BaseHTTPRequestHandler):
    """Small JSON/image API; no general-purpose file serving or arbitrary write routes."""

    def log_message(self, *_args):
        """Keep image requests out of the terminal."""

    def respond(self, status: int, body: bytes, mime: str, cache: str = "no-store"):
        """Return explicit lengths, an explicit cache policy, and no cross-origin permissions."""
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, value, status=200):
        """Serialize NumPy-free API results."""
        self.respond(status, json.dumps(value).encode(), "application/json")

    def valid_host(self):
        """Reject DNS rebinding and cross-site requests before accessing data."""
        port = self.server.server_port
        return self.headers.get("Host") in (f"localhost:{port}", f"127.0.0.1:{port}")

    def do_GET(self):
        """List episodes, read frame labels, or retrieve one stereo image."""
        if not self.valid_host():
            self.json_response({"error": "Invalid host"}, 403)
            return
        try:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/":
                page = Path(__file__).with_name("wrist_review.html").read_text()
                self.respond(
                    200,
                    page.replace("__TOKEN__", self.server.token).encode(),
                    "text/html; charset=utf-8",
                )
            elif url.path == "/api/episodes":
                result = []
                for path in episode_paths(self.server.data_dir):
                    if self.server.episode and path.name != self.server.episode:
                        continue
                    if path.resolve().parent != self.server.data_dir:
                        continue
                    try:
                        with h5py.File(path, "r") as f:
                            if not bool(f.attrs.get("complete", False)) and not path.name.endswith(
                                ".hdf5.partial"
                            ):
                                continue
                            if ANNOTATION_KEY not in f:
                                result.append({"name": path.name, "annotated": False})
                                continue
                        state = load_review(path)
                        result.append(
                            {
                                "name": path.name,
                                "annotated": True,
                                "frames": len(state["score"]),
                                "candidates": int(state["review_candidate"].sum()),
                                "auto": int(state["auto_corrupted"].sum()),
                                "marked": int((state["review_decision"] == 1).sum()),
                            }
                        )
                    except (ValueError, OSError, KeyError) as exc:
                        result.append({"name": path.name, "annotated": False, "error": str(exc)})
                self.json_response(result)
            elif url.path == "/api/episode":
                state = load_review(self.server.episode_path(query["file"][0]))
                self.json_response(
                    {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in state.items()
                        if key != "manual_corrupted"
                    }
                )
            elif url.path == "/api/image":
                path = self.server.episode_path(query["file"][0])
                frame = int(query["frame"][0])
                with h5py.File(path, "r") as f:
                    validate_recording(f, path)
                    if not 0 <= frame < len(f["timestamp_ns"]):
                        raise ValueError("Frame out of range")
                    pair = np.concatenate([f[key][frame] for key in IMAGE_KEYS], axis=1)
                ok, encoded = cv2.imencode(
                    ".jpg", cv2.cvtColor(pair, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                if not ok:
                    raise ValueError("Image encoding failed")
                # Pixels for a given (file, frame) never change -- only the labels beside them
                # do -- so let the browser keep them. Every navigation pulls three of these
                # (~64 KB each, 191 KB a step) and "Play context" re-requests eleven on every
                # 2.4 s loop; with no-store that is ~700 KB per pass forever, which is what
                # makes stepping feel stuck over a forwarded port. The server answers in 2-3 ms
                # locally, so the cost is transfer, not decode. Bounded so a re-recorded
                # episode of the same name cannot be served stale for long. JSON stays no-store:
                # labels DO change, and the UI must always see the current ones.
                self.respond(200, encoded.tobytes(), "image/jpeg", cache="private, max-age=300")
            else:
                self.json_response({"error": "Not found"}, 404)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            self.json_response({"error": str(exc)}, 400)

    def do_POST(self):
        """Toggle only positive human marks. Clean frames require no write."""
        if (
            not self.valid_host()
            or self.headers.get("X-Review-Token") != self.server.token
            or self.headers.get("Sec-Fetch-Site") == "cross-site"
        ):
            self.json_response({"error": "Invalid review token or origin"}, 403)
            return
        if self.path != "/api/mark":
            self.json_response({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 4096:
                raise ValueError("Invalid request size")
            value = json.loads(self.rfile.read(length))
            if (
                type(value["marked"]) is not bool
                or type(value["frame"]) is not int
                or type(value["revision"]) is not int
            ):
                raise ValueError("Expected boolean marked and integer frame/revision")
            path = self.server.episode_path(value["file"])
            revision = save_decision(
                path, value["frame"], 1 if value["marked"] else -1, value["revision"]
            )
            self.json_response({"revision": revision})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.json_response({"error": str(exc)}, 409)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/Dexmate/data"))
    parser.add_argument("--episode", help="Optional episode number or filename; default lists all")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not attempt to open a browser")
    parser.add_argument("--recover", type=Path, help="Finish a pending review write and exit")
    args = parser.parse_args()
    if args.recover:
        print("Recovered" if recover_review(args.recover) else "No pending review write")
        return
    if not args.data_dir.is_dir():
        parser.error("Data directory does not exist")
    episode = args.episode
    if episode and not episode.endswith((".hdf5", ".hdf5.partial")):
        episode = f"episode_{episode}.hdf5" if episode.isdigit() else episode + ".hdf5"
    if (
        episode
        and not (args.data_dir / episode).exists()
        and (args.data_dir / (episode + ".partial")).exists()
    ):
        episode += ".partial"
    with ReviewServer(("127.0.0.1", args.port), args.data_dir, episode) as server:
        url = f"http://localhost:{server.server_port}"
        print(f"Review all episodes at {url} (Ctrl+C to stop)", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
