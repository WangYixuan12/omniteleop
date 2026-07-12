"""Shared plumbing for the LIVE SceneDiff position-condition path.

Both the tabletop entry script (``follower/live_scenediff_rollout.py``) and the mobile
WBC rollout (``scripts/wbc_policy_rollout.py``) do the same three things at deploy time:
capture ONE head frame, write it as a single-frame "before" HDF5, and shell out to
``scene_diff/run_live_pos_condition.sh`` in the ``.venv-merged`` interpreter to reduce
the change masks to per-object world positions. Two of those steps are cross-repo
CONTRACTS, so they live here in ONE place:

* :func:`write_live_capture_hdf5` — the single-frame HDF5 layout (dataset keys + the
  leading frame axis on rgb/depth) that ``scene_diff/scripts/make_deploy_before_hdf5.py``
  and ``extract_first_hdf5_rgb_depth.py`` read back;
* :func:`invoke_scenediff` — the environment-variable interface consumed by
  ``run_live_pos_condition.sh`` (``REFERENCE``/``SAM``/``OBJ_NUM``/``POS_COND_MATCHING``/
  ``CONFIG``/``PYTHON``).

The FK that produces the ``world_T_zed`` extrinsic is deliberately NOT shared: the
tabletop camera is fixed whereas the mobile head pose needs base-odometry composition,
so each caller computes its own extrinsic and passes it in.

This module is intentionally dependency-light (only os/subprocess/pathlib/h5py/numpy) so
importing it never drags in a controller stack — that is exactly why the WBC rollout can
reuse it while ``live_scenediff_rollout.py`` (which imports the tabletop controller at
module load) cannot be imported from the WBC side.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Callable, Optional

import h5py
import numpy as np

# Single-frame "before" HDF5 dataset keys — the contract with make_deploy_before_hdf5.py
# / extract_first_hdf5_rgb_depth.py. Keep in lock-step with those readers.
RGB_KEY = "obs/images/head_left_rgb"
DEPTH_KEY = "obs/images/head_depth"
EXTRINSIC_KEY = "obs/images/extrinsic"
INTRINSIC_KEY = "obs/images/intrinsic"

# The scene_diff driver script (lives in the scene_diff repo, run in .venv-merged).
RUN_SCRIPT_NAME = "run_live_pos_condition.sh"

_Log = Callable[[str], None]


def write_live_capture_hdf5(
    path: "os.PathLike[str] | str",
    rgb: np.ndarray,
    depth: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> None:
    """Write one live head frame as a single-frame deploy-style "before" HDF5.

    ``extrinsic`` (``world_T_head_camera``, ``(4,4)`` float32) and ``intrinsic``
    (``(3,3)`` float32) are embedded so ``make_deploy_before_hdf5.py`` uses the ACTUAL
    camera pose at capture time rather than copying the reference scene's calibration. A
    leading frame axis of 1 is added to rgb/depth so the reader's ``ds[frame]`` indexing
    works; calibration is stored without a frame axis (constant for the capture, and
    ``extract_first_hdf5_rgb_depth`` accepts both).
    """
    path = pathlib.Path(path)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"head rgb must be (H, W, 3), got {rgb.shape}")
    if depth.ndim != 2:
        raise ValueError(f"head depth must be (H, W), got {depth.shape}")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"rgb {rgb.shape[:2]} != depth {depth.shape} (must match)")
    if extrinsic.shape != (4, 4):
        raise ValueError(f"extrinsic must be (4, 4), got {extrinsic.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must be (3, 3), got {intrinsic.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(RGB_KEY, data=np.ascontiguousarray(rgb, dtype=np.uint8)[None, ...])
        f.create_dataset(DEPTH_KEY, data=np.ascontiguousarray(depth, dtype=np.uint16)[None, ...])
        f.create_dataset(EXTRINSIC_KEY, data=extrinsic.astype(np.float32))
        f.create_dataset(INTRINSIC_KEY, data=intrinsic.astype(np.float32))


def show_slot_overlay(overlay_path: "os.PathLike[str] | str", *, log: _Log = print) -> None:
    """Best-effort, NON-BLOCKING preview of the size-slot overlay.

    Always prints the path. If a desktop display is present, also fires the OS image
    viewer in the BACKGROUND so it cannot stall the prompt. cv2 HighGUI is deliberately
    avoided: ``imshow``/``waitKey`` can HANG indefinitely over SSH / headless or with a
    conflicting conda Qt, which would freeze the whole rollout.
    """
    overlay_path = pathlib.Path(overlay_path)
    print(f"\n>>> Size-slot overlay: {overlay_path} <<<\n", flush=True)
    if not overlay_path.exists():
        log(f"slot overlay not found at {overlay_path}; open it manually if it exists.")
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        log("no DISPLAY (headless/SSH) -- open the overlay PNG above on your workstation "
            "(e.g. scp it), then answer the prompts below.")
        return
    for opener in ("xdg-open", "eog", "feh", "display"):
        try:
            subprocess.Popen([opener, str(overlay_path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"opened the overlay with {opener} (background window).")
            return
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001 - viewer is strictly best-effort
            log(f"{opener} failed ({exc}); open {overlay_path} manually.")
            return
    log(f"no image viewer found on PATH; open {overlay_path} manually.")


def slot_names(object_nums: int) -> list[str]:
    """Task-slot names ``[s1_src, s1_dst, s2_src, ...]`` for an even ``object_nums``.

    Falls back to generic ``obj{i}`` labels when ``object_nums`` is odd (no src/dst
    pairing), so the prompt still works for any object count.
    """
    n_stages = object_nums // 2
    names = [f"s{s + 1}_{role}" for s in range(n_stages) for role in ("src", "dst")]
    if len(names) != object_nums:
        names = [f"obj{i}" for i in range(object_nums)]
    return names


def prompt_for_slot_order(
    object_nums: int,
    overlay_path: "os.PathLike[str] | str",
    *,
    log: _Log = print,
    hint: Optional[str] = None,
    show: bool = True,
) -> list[int]:
    """Operator picks the task order from the size-slot overlay; re-asks until a permutation.

    Returns ``order`` (length ``object_nums``): ``order[k]`` is the size-slot index assigned
    to task slot ``k`` of ``[s1_src, s1_dst, ...]``. ``hint`` is an optional task-specific
    line shown before the prompts (e.g. "s1_src = the BOX, s1_dst = the CLOTH").
    """
    names = slot_names(object_nums)
    if show:
        show_slot_overlay(overlay_path, log=log)
    log(f"the overlay labels each detected object 0..{object_nums - 1}; for each task slot, "
        "type the label of the matching object.")
    if hint:
        log(hint)
    while True:
        chosen: list[int] = []
        ok = True
        for name in names:
            raw = input(f"  {name} = which labelled object [0..{object_nums - 1}]? ").strip()
            try:
                chosen.append(int(raw))
            except ValueError:
                log("  not an integer; start over.")
                ok = False
                break
        if ok and sorted(chosen) == list(range(object_nums)):
            log(f"order = {chosen}  ({list(zip(names, chosen))})")
            return chosen
        if ok:
            log(f"  {chosen} is not a permutation of 0..{object_nums - 1} "
                "(use each object exactly once); start over.")


def invoke_scenediff(
    *,
    repo: "os.PathLike[str] | str",
    python: "os.PathLike[str] | str",
    live_hdf5: "os.PathLike[str] | str",
    out_dir: "os.PathLike[str] | str",
    reference: "os.PathLike[str] | str",
    sam: str,
    obj_num: int,
    matching: str,
    config: Optional[str] = None,
    timeout: Optional[float] = None,
    script_name: str = RUN_SCRIPT_NAME,
    frame_label: str = "robot_base",
    log: _Log = print,
) -> pathlib.Path:
    """Run ``run_live_pos_condition.sh`` in the SceneDiff ``.venv-merged`` and return the npz.

    Passes the live capture + output dir positionally and the reference/SAM/obj-count/
    matching/config through the environment (the interface ``run_live_pos_condition.sh``
    reads). A clean exit with no ``episode_0.npz`` means SceneDiff's detection gate failed
    (wrong object count / too few valid pixels) — surfaced as an error before any motion.
    """
    repo = pathlib.Path(repo)
    python = pathlib.Path(python)
    out_dir = pathlib.Path(out_dir)
    script = repo / script_name
    if not script.exists():
        raise FileNotFoundError(f"SceneDiff driver not found: {script}")
    if not python.exists():
        raise FileNotFoundError(
            f"SceneDiff interpreter not found: {python} (expected scene_diff/.venv-merged)")
    if not pathlib.Path(reference).exists():
        raise FileNotFoundError(f"position-condition reference not found: {reference}")

    # Clean env: drop the rollout's PYTHONPATH (e.g. lerobot_original/src) and PYTHONHOME
    # so they cannot shadow the .venv-merged interpreter's modules.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHON"] = str(python)
    env["REFERENCE"] = str(reference)
    env["SAM"] = str(sam)
    env["OBJ_NUM"] = str(obj_num)
    env["POS_COND_MATCHING"] = str(matching)
    # The npz's frame field is METADATA stamped by extract_object_positions.py -- it must
    # name the frame of the extrinsic embedded in live_hdf5 (tabletop base_T_cam ->
    # "robot_base"; mobile WBC world_T_zed -> "world"). Downstream consumers fail closed
    # on the wrong label rather than mixing frames.
    env["FRAME_LABEL"] = str(frame_label)
    if config:
        env["CONFIG"] = str(config)

    cmd = ["bash", str(script), str(live_hdf5), str(out_dir)]
    log(f"running SceneDiff: PYTHON={python} {' '.join(cmd)}")
    # Inherit stdio so SAM3 progress is visible; enforce a wall-clock timeout.
    result = subprocess.run(cmd, env=env, cwd=str(repo), timeout=timeout)
    npz = out_dir / "episode_0.npz"
    if result.returncode != 0:
        raise RuntimeError(
            f"SceneDiff failed (exit {result.returncode}); see output above and scratch "
            f"under {out_dir / '_work'}")
    if not npz.exists():
        raise FileNotFoundError(
            "SceneDiff produced no position condition -- the detection gate failed (wrong "
            f"object count / too few valid pixels). Inspect "
            f"{out_dir / 'deploy_before_overlay.png'} and {out_dir / '_work' / 'skip.json'}, "
            "fix the scene/params, and rerun.")
    return npz
