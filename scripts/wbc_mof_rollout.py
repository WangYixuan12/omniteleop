#!/usr/bin/env python3
"""Live rollout of a MoF (Mixture of Frames) policy on the Vega WBC robot.

The MoF sibling of ``scripts/wbc_policy_rollout.py`` and
``scripts/wbc_maniflow_rollout.py``. It does NOT fork the hardware loop: it
imports the shared rollout and swaps in :class:`omniteleop.wbc_mof_bundle
.MoFBundle` via ``_run_rollout(..., policy_factory=)``. Engage, reference
alignment, the asynchronous chunk scheduler, the stale-action drop, the 100 Hz
WBC tick, the hold watchdog, recording and shutdown are all THE SAME CODE — a
second copy of that loop is exactly the divergence that gets a robot hurt.

Unlike the ManiFlow sibling, not even the inference worker is replaced: the
MoF observation is the same raw ``(world state, head_rgb, wrist_rgb)`` the
shared worker already gathers with its freshness gates; the bundle converts it
with the porter's exact preprocessing (224x224 anisotropic squash, per-entity
world pos + wxyz quats from the state blocks).

Deploy gates fire inside the bundle at load time, BEFORE engage: WBC action
layout, ``external_rot6d_convention: column`` (predictions come out in the
recorded action convention ``split_policy_action`` decodes), the training
port's ``meta.json``, and fitted normalizer statistics.

Environment: the ``dexmate_mof`` conda env (a dexmate_lerobot clone + editable
mofpo + robomimic; see the README "MoF" section for the recipe).

Usage::

    python scripts/wbc_mof_rollout.py \
      --policy-path ~/mofpo/data/outputs/wbc_v1_mof_moe_500ep/checkpoints/latest.ckpt \
      --save-dir ~/Dexmate/data/rollout/mof \
      --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
      --max-seconds 100
    # if the buffer warns it will run dry (16-step DDIM x 5 experts), lower
    # --policy-interval or pass --num-inference-steps 8

    # replay a recorded take through the identical actuation path (no policy):
    python scripts/wbc_mof_rollout.py \
      --replay-episode ~/Dexmate/data/raw_data/episode_1.hdf5 \
      --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5
"""

from __future__ import annotations

import functools
import importlib.util
import sys
import types
from pathlib import Path

from omniteleop.wbc_mof_bundle import MoFBundle


def _load_rollout() -> types.ModuleType:
    """Import the shared rollout by path (``scripts/`` is not a package)."""
    path = Path(__file__).resolve().parent / "wbc_policy_rollout.py"
    spec = importlib.util.spec_from_file_location("wbc_policy_rollout", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the shared rollout at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rollout = _load_rollout()

DEFAULT_SAVE_DIR = str(Path("~/Dexmate/data/rollout/mof").expanduser())


def main() -> None:
    """Parse the shared rollout CLI plus MoF flags and run the shared loop."""
    parser = rollout.build_parser()
    parser.description = __doc__
    mof = parser.add_argument_group("mof (mixture-of-frames policy)")
    mof.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="override the checkpoint's DDIM step count (default: checkpoint's, "
        "16 for the paper recipe). Fewer steps cut the ~5-expert inference "
        "cost if the chunk buffer runs dry.",
    )
    mof.add_argument(
        "--no-ema",
        action="store_true",
        help="load the raw 'model' weights instead of the EMA copy (default: EMA).",
    )
    mof.add_argument(
        "--port-meta",
        default=None,
        help="override the training port's meta.json, which records the exact "
        "preprocessing (224x224 anisotropic squash, column rot6d, wxyz quats) "
        "the live observations MUST reproduce. Default: <cfg.demos_dir>/meta.json.",
    )
    args = parser.parse_args()

    if args.save_dir is None and not args.replay_episode:
        args.save_dir = DEFAULT_SAVE_DIR
    rollout.finalize_args(parser, args)

    policy_factory = functools.partial(
        MoFBundle,
        use_ema=not args.no_ema,
        num_inference_steps=args.num_inference_steps,
        port_meta_path=args.port_meta,
    )
    rollout._run_rollout(args, policy_factory=policy_factory)  # noqa: SLF001 -- shared loop by design


if __name__ == "__main__":
    main()
