#!/usr/bin/env python3
"""Read-only live monitor for WBC self-collision safety on the real robot.

Run this in one shell while manually moving the arms, for example with
``scripts/admittance_control.py`` in another shell. The monitor does not command the
robot. It reads measured joints, uses the same WBC configuration and thresholds as
``wbc_vr_leader.py`` / ``wbc_vr_record.py`` via ``VegaWholeBodyIK(WBCConfig())``,
but computes monitor distances on the authentic URDF collision meshes instead of
the simplified collision-sphere URDF. Safe lines never print ``OK``; below-threshold
lines print ``WARN`` or a louder violation.

Example::

    /home/yixuan/miniforge3/envs/dexmate/bin/python \
        scripts/diagnostics/monitor_wbc_collision_safety.py --rate 20
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pinocchio as pin
from pink import Configuration

from omniteleop.follower import wbc_safety
from omniteleop.follower.whole_body_ik import VegaWholeBodyIK, WBCConfig

DEFAULT_COMPONENTS = ("head", "left_arm", "right_arm", "torso")


@dataclass(frozen=True)
class CollisionThresholds:
    """Warning and hard-floor self-collision distances in meters."""

    warn: float
    floor: float


@dataclass(frozen=True)
class CollisionPairDistance:
    """Closest kept collision pair and its surface distance in meters."""

    first_name: str
    second_name: str
    distance: float


@dataclass(frozen=True)
class CollisionCheckContext:
    """Geometry model/data pair used by this monitor."""

    label: str
    geometry_model: pin.GeometryModel
    geometry_data: pin.GeometryData


def build_measured_q(
    nominal_q: np.ndarray,
    idx_q: Mapping[str, int],
    measured_joints: Mapping[str, float],
) -> np.ndarray:
    """Overlay measured joint positions onto a WBC nominal configuration.

    Unknown joint names are ignored so the script can tolerate robot API variants.
    Missing known joints remain at the WBC nominal posture, which is safer than
    inventing zeros for torso/head joints.
    """
    q = np.asarray(nominal_q, dtype=float).copy()
    for name, value in measured_joints.items():
        idx = idx_q.get(name)
        if idx is not None:
            q[idx] = float(value)
    return q


def build_authentic_collision_context(ik: VegaWholeBodyIK) -> CollisionCheckContext:
    """Build kept cross-group pairs on the authentic reduced URDF collision model."""
    gm = ik.collision_model
    if gm is None or gm.ngeoms == 0:
        raise RuntimeError("authentic URDF collision model is empty; cannot monitor safety")

    gm.removeAllCollisionPairs()
    group_of = [wbc_safety.collision_group(gm.geometryObjects[i].name) for i in range(gm.ngeoms)]
    candidates = wbc_safety.cross_group_index_pairs(group_of)
    if not candidates:
        raise RuntimeError("authentic URDF collision model has no cross-group pairs")

    for i, j in candidates:
        gm.addCollisionPair(pin.CollisionPair(i, j))

    nominal_data = gm.createData()
    Configuration(
        ik.model,
        ik.model.createData(),
        ik.nominal_q(),
        collision_model=gm,
        collision_data=nominal_data,
    )
    dist_map = {
        candidates[k]: float(nominal_data.distanceResults[k].min_distance)
        for k in range(len(candidates))
    }
    kept = wbc_safety.filter_nominal_overlaps(
        candidates,
        lambda i, j: dist_map[(i, j)],
        ik.config.nominal_pair_keep_dist,
    )

    gm.removeAllCollisionPairs()
    for i, j in kept:
        gm.addCollisionPair(pin.CollisionPair(i, j))
    if not gm.collisionPairs:
        raise RuntimeError("authentic URDF collision model has no kept cross-group pairs")
    return CollisionCheckContext("authentic_urdf", gm, gm.createData())


def closest_collision_pair(context: CollisionCheckContext) -> CollisionPairDistance:
    """Return the closest kept cross-group pair in the current monitor state."""
    gm = context.geometry_model
    gd = context.geometry_data
    if gm is None or gd is None or not gm.collisionPairs:
        return CollisionPairDistance("no_collision_pairs", "no_collision_pairs", float("inf"))

    distances = [float(r.min_distance) for r in gd.distanceResults[: len(gm.collisionPairs)]]
    k = int(np.argmin(distances))
    pair = gm.collisionPairs[k]
    return CollisionPairDistance(
        gm.geometryObjects[pair.first].name,
        gm.geometryObjects[pair.second].name,
        distances[k],
    )


def format_collision_alert(
    pair: CollisionPairDistance,
    thresholds: CollisionThresholds,
) -> str:
    """Format the closest-pair status line for safe/warn/violation states."""
    dist = float(pair.distance)
    pair_text = f"{pair.first_name} <-> {pair.second_name}"
    dist_text = f"{dist * 100:.2f}cm"
    warn_text = f"warn={thresholds.warn * 100:.2f}cm floor={thresholds.floor * 100:.2f}cm"
    if dist >= thresholds.warn:
        return f"SAFE closest={dist_text} {pair_text} ({warn_text})"
    if dist < thresholds.floor or dist < 0.0:
        return (
            "\a\a\a!!! COLLISION AVOIDANCE VIOLATION !!! "
            f"closest={dist_text} {pair_text} ({warn_text})"
        )
    return f"WARN self-collision close: closest={dist_text} {pair_text} ({warn_text})"


def _read_joint_positions(bot, components: Sequence[str]) -> dict[str, float]:
    """Read a flat joint-name -> position mapping from dexcontrol."""
    raw = bot.get_joint_pos_dict(component=list(components))
    if not isinstance(raw, Mapping):
        raise TypeError(f"get_joint_pos_dict returned {type(raw).__name__}, expected mapping")
    out: dict[str, float] = {}
    for name, value in raw.items():
        try:
            out[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _sleep_until_next_tick(last_tick: float, period: float) -> float:
    now = time.monotonic()
    delay = period - (now - last_tick)
    if delay > 0.0:
        time.sleep(delay)
    return time.monotonic()


def monitor(
    *,
    rate: float,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    config_path: str | None = None,
    repeat_violation: bool = True,
) -> None:
    """Run the live read-only monitor until interrupted."""
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"rate must be finite and > 0, got {rate}")

    from dexcontrol.robot import Robot  # noqa: PLC0415 -- hardware dependency

    ik = VegaWholeBodyIK(config_path=config_path) if config_path else VegaWholeBodyIK(WBCConfig())
    collision_context = build_authentic_collision_context(ik)

    thresholds = CollisionThresholds(
        warn=ik.config.self_collision_safe_dist,
        floor=ik.config.self_collision_floor,
    )
    nominal_q = ik.nominal_q()
    monitor_configuration = Configuration(
        ik.model,
        ik.model.createData(),
        nominal_q,
        collision_model=collision_context.geometry_model,
        collision_data=collision_context.geometry_data,
    )
    period = 1.0 / rate
    bot = Robot()
    last_tick = time.monotonic()
    last_alert: str | None = None

    print(
        "Monitoring WBC collision safety: "
        f"model={collision_context.label} "
        f"warn={thresholds.warn * 100:.2f}cm floor={thresholds.floor * 100:.2f}cm "
        f"rate={rate:g}Hz components={','.join(components)}"
    )
    print("Printing closest authentic-URDF collision pair distance; safe lines do not print OK.")

    try:
        while True:
            measured = _read_joint_positions(bot, components)
            q = build_measured_q(nominal_q, ik._idx_q, measured)  # noqa: SLF001
            monitor_configuration.update(q)
            pair = closest_collision_pair(collision_context)
            alert = format_collision_alert(pair, thresholds)
            if alert is not None and (repeat_violation or alert != last_alert):
                print(f"{time.strftime('%H:%M:%S')} {alert}", flush=True)
            last_alert = alert
            last_tick = _sleep_until_next_tick(last_tick, period)
    except KeyboardInterrupt:
        print("\nStopped WBC collision safety monitor.")
    finally:
        try:
            bot.shutdown()
        except Exception:
            pass


def _parse_components(text: str) -> tuple[str, ...]:
    components = tuple(s.strip() for s in text.split(",") if s.strip())
    if not components:
        raise argparse.ArgumentTypeError("component list cannot be empty")
    return components


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rate", type=float, default=20.0, help="monitor frequency in Hz")
    p.add_argument(
        "--components",
        type=_parse_components,
        default=DEFAULT_COMPONENTS,
        help="comma-separated robot components to read",
    )
    p.add_argument(
        "--config-path",
        default=None,
        help="optional alternate WBC config YAML; defaults to the repo wbik.yaml",
    )
    p.add_argument(
        "--suppress-repeat",
        action="store_true",
        help="print only when the alert text changes",
    )
    args = p.parse_args()
    monitor(
        rate=args.rate,
        components=args.components,
        config_path=args.config_path,
        repeat_violation=not args.suppress_repeat,
    )


if __name__ == "__main__":
    main()
