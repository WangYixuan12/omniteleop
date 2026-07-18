#!/usr/bin/env python3
"""Offline GT-vs-predicted validation of a WBC MoF checkpoint.

The MoF sibling of ``scripts/vis_wbc_policy_prediction.py``: runs the deploy
bundle's EXACT inference path (``MoFBundle.predict_chunk`` on the stored port
observations) over a ported safetensors episode and compares the predicted
29-D world actions against the recorded ground truth. This is the gate before
any hardware rollout.

Outputs, per entity (left / right / head):
- translation MAE (m) and geodesic rotation error (deg) over all predicted
  steps, printed to stdout;
- an overlay figure: GT (solid) vs 1-step-ahead prediction (dashed) position
  traces per axis, rotation-error series, and the gripper command series.

Environment: ``dexmate_mof``.

Usage::

    python scripts/vis_wbc_mof_prediction.py \
      --policy-path ~/mofpo/data/outputs/wbc_v1_mof_moe_500ep/checkpoints/latest.ckpt \
      --episode .../processed_wbc/mof/dexmate_wbc_mof/test/episode_0.safetensors \
      --output ~/Dexmate/data/visualization/mof_pred_test0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from omniteleop.wbc_mof_bundle import MoFBundle
from omniteleop.wbc_policy_format import GRIPPER_BINARY_THRESHOLD

ENTITIES = {
    "left": (slice(0, 3), slice(3, 9), 9),
    "right": (slice(10, 13), slice(13, 19), 19),
    "head": (slice(20, 23), slice(23, 29), None),
}
DEFAULT_EPISODE = Path(
    "~/Dexmate/data/box2cloth/processed_wbc/mof/dexmate_wbc_mof/test/episode_0.safetensors"
).expanduser()


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) column rot6d -> (..., 3, 3); columns are stored orthonormal."""
    c1, c2 = rot6d[..., 0:3], rot6d[..., 3:6]
    c1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
    c2 = c2 - np.sum(c1 * c2, axis=-1, keepdims=True) * c1
    c2 = c2 / np.linalg.norm(c2, axis=-1, keepdims=True)
    return np.stack([c1, c2, np.cross(c1, c2)], axis=-1)


def _geodesic_deg(rot6d_a: np.ndarray, rot6d_b: np.ndarray) -> np.ndarray:
    Ra, Rb = _rot6d_to_matrix(rot6d_a), _rot6d_to_matrix(rot6d_b)
    tr = np.einsum("...ij,...ij->...", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def evaluate(
    bundle: MoFBundle, arrays: dict[str, np.ndarray], stride: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    T = arrays["action"].shape[0]
    n_obs, n_act = bundle.n_obs_steps, bundle.n_action_steps
    records = []  # (t, gt (n_act, 29), pred (n_act, 29))
    for t in range(n_obs - 1, T - n_act, stride):
        history = [bundle.obs_from_port_arrays(arrays, i) for i in range(t - n_obs + 1, t + 1)]
        pred = bundle.predict_chunk(history)
        gt = arrays["action"][t : t + n_act]
        records.append((t, gt, pred))
    if not records:
        raise SystemExit(f"episode too short ({T} frames) for a single chunk")
    return records


def report(records: list[tuple[int, np.ndarray, np.ndarray]]) -> dict:
    gt = np.stack([r[1] for r in records])  # (K, n_act, 29)
    pred = np.stack([r[2] for r in records])
    metrics: dict[str, float] = {}
    print(f"chunks evaluated: {len(records)} x {gt.shape[1]} steps")
    for name, (pos_sl, rot_sl, grip_ch) in ENTITIES.items():
        t_mae = float(np.abs(pred[..., pos_sl] - gt[..., pos_sl]).mean())
        r_err = float(_geodesic_deg(pred[..., rot_sl], gt[..., rot_sl]).mean())
        metrics[f"{name}/translation_mae_m"] = t_mae
        metrics[f"{name}/rotation_err_deg"] = r_err
        line = f"  {name:>5}: translation MAE {t_mae * 100:6.2f} cm | rotation err {r_err:6.2f} deg"
        if grip_ch is not None:
            acc = float(
                (
                    (pred[..., grip_ch] >= GRIPPER_BINARY_THRESHOLD)
                    == (gt[..., grip_ch] >= GRIPPER_BINARY_THRESHOLD)
                ).mean()
            )
            metrics[f"{name}/gripper_acc"] = acc
            line += f" | gripper acc {acc * 100:5.1f}%"
        print(line)
    return metrics


def render(records: list[tuple[int, np.ndarray, np.ndarray]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.array([r[0] for r in records])
    gt1 = np.stack([r[1][0] for r in records])  # first predicted step vs GT
    pred1 = np.stack([r[2][0] for r in records])

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    for row, (name, (pos_sl, rot_sl, _grip_ch)) in enumerate(ENTITIES.items()):
        for col, axis_name in enumerate("xyz"):
            ax = axes[row][col]
            ch = pos_sl.start + col
            ax.plot(ts, gt1[:, ch], "-", color="tab:red", label="GT")
            ax.plot(ts, pred1[:, ch], "--", color="tab:purple", label="pred")
            ax.set_ylabel(f"{name} {axis_name} [m]")
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)
        rot_err = _geodesic_deg(pred1[:, rot_sl], gt1[:, rot_sl])
        axes[3][list(ENTITIES).index(name)].plot(ts, rot_err, color="tab:orange")
        axes[3][list(ENTITIES).index(name)].set_ylabel(f"{name} rot err [deg]")
        axes[3][list(ENTITIES).index(name)].set_xlabel("frame")
    grip_ax = axes[0][2].twinx()
    for name, (_, _, grip_ch) in ENTITIES.items():
        if grip_ch is None:
            continue
        grip_ax.step(ts, gt1[:, grip_ch], where="post", alpha=0.35, label=f"{name} grip GT")
        grip_ax.step(
            ts, pred1[:, grip_ch], where="post", alpha=0.35, ls="--", label=f"{name} grip pred"
        )
    grip_ax.set_ylim(-0.1, 1.1)
    grip_ax.legend(loc="lower right", fontsize=6)
    fig.suptitle("WBC MoF: GT vs 1-step-ahead predicted world action")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"overlay figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy-path",
        required=True,
        help="mofpo workspace .ckpt (e.g. .../checkpoints/latest.ckpt)",
    )
    parser.add_argument(
        "--episode",
        type=Path,
        default=DEFAULT_EPISODE,
        help=f"ported safetensors episode (default {DEFAULT_EPISODE})",
    )
    parser.add_argument(
        "--port-meta", default=None, help="override the training port's meta.json path"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="frames between evaluated chunks (default n_action_steps)",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output", type=Path, default=None, help="overlay PNG path (default alongside the episode)"
    )
    args = parser.parse_args()

    from safetensors.numpy import load_file

    bundle = MoFBundle(
        args.policy_path,
        device=args.device,
        use_ema=not args.no_ema,
        num_inference_steps=args.num_inference_steps,
        port_meta_path=args.port_meta,
    )
    print(bundle.describe())
    arrays = load_file(str(args.episode))
    stride = args.stride if args.stride is not None else bundle.n_action_steps
    records = evaluate(bundle, arrays, stride)
    report(records)
    out = args.output or args.episode.with_suffix(".mof_pred.png")
    render(records, Path(out))


if __name__ == "__main__":
    main()
