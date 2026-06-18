"""Play two videos side by side in a single synchronized OpenCV window.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEO_1 = Path("/home/yixuan/Dexmate/wbc/wbc_waypoint_closedloop.mp4")
DEFAULT_VIDEO_2 = Path("/home/yixuan/Dexmate/wbc/wbc_waypoint_openloop.mp4")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video_1",
        type=Path,
        default=DEFAULT_VIDEO_1,
        help="Video to display on the left.",
    )
    parser.add_argument(
        "--video_2",
        type=Path,
        default=DEFAULT_VIDEO_2,
        help="Video to display on the right.",
    )
    parser.add_argument(
        "--window-name",
        default="kinematics (left) | wheel (right)",
        help="OpenCV window title.",
    )
    parser.add_argument(
        "--video_result",
        type=Path,
        default=None,
        help="Optional output MP4 path for the side-by-side video.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="Maximum display height in pixels. Use 0 to keep original height.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Restart both videos when either reaches the end.",
    )
    return parser.parse_args()


def open_video(path: Path) -> cv2.VideoCapture:
    """Open a video path and raise a helpful error if it cannot be read."""
    if not path.exists():
        raise FileNotFoundError(f"Video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return capture


def target_height(frame_1: np.ndarray, frame_2: np.ndarray, max_height: int) -> int:
    """Return the display height used for both frames."""
    height = min(frame_1.shape[0], frame_2.shape[0])
    if max_height > 0:
        height = min(height, max_height)
    return height


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    """Resize a frame to a target height while preserving aspect ratio."""
    if frame.shape[0] == height:
        return frame
    width = round(frame.shape[1] * height / frame.shape[0])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Draw a compact label in the top-left corner of a frame."""
    output = frame.copy()
    cv2.rectangle(output, (8, 8), (210, 42), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        label,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def ensure_even_dimensions(frame: np.ndarray) -> np.ndarray:
    """Pad the frame if needed because MP4 encoders commonly require even sizes."""
    height, width = frame.shape[:2]
    pad_bottom = height % 2
    pad_right = width % 2
    if not (pad_bottom or pad_right):
        return frame
    return cv2.copyMakeBorder(
        frame,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def playback_delay_seconds(video_1: cv2.VideoCapture, video_2: cv2.VideoCapture) -> float:
    """Use the slower reported FPS so neither clip is advanced too quickly."""
    fps_values = [
        fps
        for fps in (
            video_1.get(cv2.CAP_PROP_FPS),
            video_2.get(cv2.CAP_PROP_FPS),
        )
        if fps and fps > 0
    ]
    if not fps_values:
        return 1.0 / 30.0
    return 1.0 / min(fps_values)


def open_result_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    """Open an MP4 writer sized to the rendered side-by-side frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open result video for writing: {path}")
    return writer


def main() -> None:
    """Run side-by-side video playback."""
    args = parse_args()
    video_1 = open_video(args.video_1)
    video_2 = open_video(args.video_2)
    delay = playback_delay_seconds(video_1, video_2)
    writer: cv2.VideoWriter | None = None

    try:
        while True:
            loop_start = time.monotonic()
            ok_1, frame_1 = video_1.read()
            ok_2, frame_2 = video_2.read()

            if not (ok_1 and ok_2):
                if not args.loop:
                    break
                video_1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            height = target_height(frame_1, frame_2, args.max_height)
            left = add_label(resize_to_height(frame_1, height), "kinematics")
            right = add_label(resize_to_height(frame_2, height), "wheel")
            combined = ensure_even_dimensions(np.hstack((left, right)))

            if args.video_result is not None:
                if writer is None:
                    writer = open_result_writer(args.video_result, combined, fps=1.0 / delay)
                writer.write(combined)

            cv2.imshow(args.window_name, combined)
            elapsed = time.monotonic() - loop_start
            wait_ms = max(1, round((delay - elapsed) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if writer is not None:
            writer.release()
        video_1.release()
        video_2.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
