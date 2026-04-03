"""Record VR headset and controller poses to HDF5.

A=start recording, B=stop and save.  No robot, no IK, no Zenoh.

Usage::

    python -m omniteleop.leader.vr_pose_recorder
    python -m omniteleop.leader.vr_pose_recorder --save-dir my_poses --port 5067
"""

from __future__ import annotations

import pathlib
import time
from typing import Optional

import numpy as np
import tyro
from loguru import logger
from rich.console import Console
from yixuan_utilities.hdf5_utils import save_dict_to_hdf5

from omniteleop.common.logging import setup_logging
from omniteleop.leader.communication.webxr_vr_reader import VRFrame, WebXRVRReader

console = Console()


def _recursive_np_stack(list_of_dicts: list[dict]) -> dict:
    result: dict = {}
    for key in list_of_dicts[0]:
        val = list_of_dicts[0][key]
        if isinstance(val, dict):
            result[key] = _recursive_np_stack([d[key] for d in list_of_dicts])
        else:
            result[key] = np.stack([d[key] for d in list_of_dicts])
    return result


def main(
    save_dir: str = "vr_poses",
    port: int = 5067,
) -> None:
    setup_logging()
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    quest = WebXRVRReader(port=port)
    quest.start()
    console.rule("[bold cyan]Waiting for Quest browser to connect…")
    quest.wait_for_data()
    console.rule("[bold green]Connected — press A to start recording, B to stop & save")

    frames: list[dict] = []
    recording = False
    prev_a = False
    prev_b = False

    try:
        while True:
            tf: Optional[VRFrame] = quest.get_latest_transformation()
            if tf is None:
                time.sleep(0.01)
                continue

            a_now = tf["right_a_button"]
            b_now = tf["right_b_button"]

            if a_now and not prev_a and not recording:
                frames = []
                recording = True
                console.print("[bold green]● REC started[/]")

            if b_now and not prev_b and recording:
                recording = False
                if frames:
                    out = save_path / f"poses_{int(time.time_ns())}.hdf5"
                    save_dict_to_hdf5(_recursive_np_stack(frames), {}, str(out))
                    console.print(f"[bold yellow]Saved {len(frames)} frames → {out}[/]")
                else:
                    console.print("[yellow]No frames recorded — skipped save[/]")
                frames = []

            prev_a = a_now
            prev_b = b_now

            if recording:
                frames.append(
                    {
                        "timestamp_ns": np.int64(time.time_ns()),
                        "head": tf["head"].astype(np.float32),
                        "left_wrist": tf["left_wrist"].astype(np.float32),
                        "right_wrist": tf["right_wrist"].astype(np.float32),
                        "left_index_trigger": np.float32(tf["left_index_trigger"]),
                        "right_index_trigger": np.float32(tf["right_index_trigger"]),
                        "left_squeeze": np.float32(tf["left_hand_trigger"]),
                        "right_squeeze": np.float32(tf["right_hand_trigger"]),
                        "left_thumbstick": np.array(tf["left_thumbstick"], dtype=np.float32),
                        "right_thumbstick": np.array(tf["right_thumbstick"], dtype=np.float32),
                    }
                )

            time.sleep(0.01)  # ~100 Hz poll

    except KeyboardInterrupt:
        logger.info("Stopped.")
        if recording and frames:
            out = save_path / f"poses_{int(time.time_ns())}.hdf5"
            save_dict_to_hdf5(_recursive_np_stack(frames), {}, str(out))
            console.print(f"[bold yellow]Auto-saved {len(frames)} frames → {out}[/]")
    finally:
        quest.close()


if __name__ == "__main__":
    tyro.cli(main)
