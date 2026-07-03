#!/usr/bin/env python3
"""Shared infrastructure for the whole-body VR teleop followers (sim + real robot).

The SAPIEN sim follower (``scripts/wbc_vr_record.py``) and the real-robot follower
(``scripts/wbc_vr_robot.py``) share two things that MUST stay in lock-step, so a take
recorded and visualized in sim drives the robot the same way it looked:

* :class:`VRTeleopConfig` -- the control-loop tunables (control/command rates,
  head-target low-pass/deadband, base PD gains, base command slew / clamp / post-deadband,
  replay speed) that shape the low-rate leader command stream before/around the
  whole-body IK. They live in the
  ``vr_teleop:`` block of the canonical ``follower/wbik.yaml`` (the single follower
  config file, beside the WBC *solver* tunables) and are loaded here so BOTH followers
  bind identical argparse defaults to them. ``WBCConfig.from_yaml`` ignores that block
  (it is a follower-loop parameter, not a solver parameter -- see ``_NON_WBC_SECTIONS``
  in ``omniteleop.follower.whole_body_ik``).
* :class:`VRJointSubscriber` -- the live Zenoh source for the leader's ``vr/joints``
  ``VRJointData`` stream. It mirrors :class:`omniteleop.wbc_record.ReplaySource`'s
  ``latest`` / ``done`` / ``segment_duration`` / ``start`` / ``close`` surface so the
  follower loop is source-agnostic (live subscriber vs file replay).

Depends only on ``yaml`` + ``dexcomm`` + ``omniteleop.common`` -- no sapien / pinocchio
and, crucially, no ``omniteleop.follower`` package ``__init__`` (heavy, hardware deps).
That keeps it safe to import at the top of both followers and under the stub-loaded
follower tests, the same discipline as :mod:`omniteleop.wbc_stream`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Optional, Union

import yaml
from dexcomm import Node
from dexcomm.codecs import DictDataCodec

from omniteleop.common import get_config
from omniteleop.common.schemas import VRJointData

# The shared follower config file: the vr_teleop: block sits beside the WBC solver
# tunables in follower/wbik.yaml (the single source of truth). VRTeleopConfig reads ONLY
# that block; WBCConfig reads only the solver keys and skips vr_teleop (see its
# from_yaml / _NON_WBC_SECTIONS). Absolute path => loadable from any working directory.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("follower") / "wbik.yaml"
VR_TELEOP_SECTION = "vr_teleop"
# Fields the followers require to be STRICTLY positive (their argparse validation rejects
# <= 0): a replay clock speed, an IK loop rate, a base-twist slew rate, and a base-velocity
# clamp of 0 are degenerate. The rest are disable-at-zero tunables (deadbands, the LPF tau,
# PD gains, and cmd_rate -- 0 disables interpolation) and may be 0.
_STRICTLY_POSITIVE_FIELDS = frozenset(
    {"replay_speed", "ik_rate", "base_accel", "base_max_speed"}
)
_BOOLEAN_FIELDS = frozenset({"base_yaw_hold_in_xy"})


@dataclass(frozen=True)
class VRTeleopConfig:
    """Shared command-stream tunables for the whole-body VR followers.

    Loaded from the ``vr_teleop:`` block of ``follower/wbik.yaml`` (see
    :meth:`from_yaml`); both followers bind their argparse defaults to these so the
    head/base command shaping is identical in sim and on hardware. The fields have no
    in-code defaults on purpose -- the YAML is the single source of truth, so a missing
    key is an error rather than a silent fallback.
    """

    replay_speed: float                # replay speed multiplier (<1 = slower)
    ik_rate: float                     # whole-body IK / control loop rate (Hz)
    cmd_rate: float                    # leader command rate (Hz), interpolated to ik_rate
    head_lpf_tau: float                # head-target low-pass time constant (s); 0 disables
    head_planar_pos_deadband: float    # radial x/y head-target deadband (m)
    head_planar_yaw_deadband: float    # heading-yaw head-target deadband (rad)
    base_kp_xy: float                  # closed-loop base PD gain on planar pose error (1/s)
    base_kp_yaw: float                 # closed-loop base PD gain on yaw error (1/s)
    base_yaw_hold_in_xy: bool          # allow yaw feedback when WBC base_dofs hard-locks yaw
    base_deadband: float               # closed-loop base-twist pre-deadband (m/s; angular = 2x)
    base_accel: float                  # base-twist slew limit (m/s^2; angular = 2x)
    base_max_speed: float              # base linear-velocity clamp (m/s; angular = 2x)
    base_post_linear_deadband: float   # per-axis vx/vy post-shaping deadband (m/s)
    base_post_angular_deadband: float  # per-axis wz post-shaping deadband (rad/s)

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "VRTeleopConfig":
        """Load the ``vr_teleop:`` block from ``wbik.yaml`` (or ``path``).

        Every field must be present (the YAML is the single source of truth, no silent
        defaults); unknown keys in the block raise (typo guard). Each value must be a
        finite number that is ``>= 0`` -- or ``> 0`` for the strictly-positive fields
        (``replay_speed``, ``base_accel``; see ``_STRICTLY_POSITIVE_FIELDS``) -- except
        the explicit boolean fields. All raises are ``ValueError`` carrying the
        ``vr_teleop.<field>`` context (never a bare ``float()`` ``TypeError``).
        """
        cfg_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or VR_TELEOP_SECTION not in data:
            raise ValueError(f"{cfg_path}: missing required '{VR_TELEOP_SECTION}:' section")
        block = data[VR_TELEOP_SECTION]
        if not isinstance(block, dict):
            raise ValueError(
                f"{cfg_path}: '{VR_TELEOP_SECTION}' must be a mapping, got "
                f"{type(block).__name__}"
            )
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(block) - names)
        if unknown:
            raise ValueError(f"{cfg_path}: unknown {VR_TELEOP_SECTION} keys {unknown}")
        missing = sorted(names - set(block))
        if missing:
            raise ValueError(f"{cfg_path}: missing {VR_TELEOP_SECTION} keys {missing}")
        values = {}
        for name in names:
            raw = block[name]
            if name in _BOOLEAN_FIELDS:
                if not isinstance(raw, bool):
                    raise ValueError(
                        f"{cfg_path}: {VR_TELEOP_SECTION}.{name} must be a boolean, "
                        f"got {raw!r}"
                    )
                values[name] = raw
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{cfg_path}: {VR_TELEOP_SECTION}.{name} must be a number, got {raw!r}"
                ) from None
            positive = name in _STRICTLY_POSITIVE_FIELDS
            invalid = not math.isfinite(value) or (value <= 0.0 if positive else value < 0.0)
            if invalid:
                bound = "> 0" if positive else ">= 0"
                raise ValueError(
                    f"{cfg_path}: {VR_TELEOP_SECTION}.{name} must be finite and {bound}, "
                    f"got {raw!r}"
                )
            values[name] = value
        return cls(**values)


class VRJointSubscriber:
    """Subscribe to ``vr/joints``; keep the latest frame and optionally tee every frame.

    The live counterpart to :class:`omniteleop.wbc_record.ReplaySource`: it exposes the
    same ``latest`` / ``done`` / ``segment_duration`` / ``start`` / ``close`` surface so
    both followers run the SAME loop over either source. ``sink`` (optional) is called
    with every received frame for lightweight instrumentation (e.g. the sim follower's
    stream recording) without affecting the follower loop -- a sink exception never drops
    the subscription. ``name`` is the Zenoh node name; the two followers pass distinct
    names (``wbc_vr_record`` / ``wbc_vr_robot``).
    """

    def __init__(
        self,
        namespace: str = "",
        sink: Optional[Callable[[VRJointData], None]] = None,
        *,
        name: str = "wbc_vr_follower",
    ) -> None:
        self.node = Node(name=name, namespace=namespace)
        self.topic = get_config().get_topic("vr_joints", "vr/joints")
        self._latest: Optional[VRJointData] = None
        self._sink = sink
        self.sub = self.node.create_subscriber(
            self.topic, self._on_msg, decoder=DictDataCodec.decode
        )

    def _on_msg(self, data: dict) -> None:
        try:
            vr = VRJointData(**data)
        except (TypeError, ValueError):
            return  # ignore malformed frames
        self._latest = vr
        if self._sink is not None:
            try:
                self._sink(vr)
            except Exception:  # never let a sink error drop the subscription
                pass

    @property
    def latest(self) -> Optional[VRJointData]:
        """Most recently received ``VRJointData`` (None before the first frame)."""
        return self._latest

    @property
    def done(self) -> bool:
        """Live sources never self-terminate (parity with ReplaySource.done)."""
        return False

    @property
    def segment_duration(self) -> Optional[float]:
        """Live mode uses the interpolator's fixed 1/cmd-rate glide (no override)."""
        return None

    def start(self) -> None:
        """No-op (the subscriber is live the moment it is created)."""

    def close(self) -> None:
        """Close the Zenoh node (best-effort)."""
        try:
            self.node.close()
        except Exception:  # best-effort cleanup
            pass
