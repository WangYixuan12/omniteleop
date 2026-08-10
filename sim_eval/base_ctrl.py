"""Bridge to the REAL follower control code, so sim and hardware run the same functions.

This file used to be a hand-written pure-numpy PORT of
``omniteleop/follower/base_closed_loop.py``. A port is a snapshot: it drifts silently, and it
did -- an audit against the hardware follower found the sim running a dispatch deadband of 0.02
where ``wbik.yaml`` says 0.0, purely because the number had been retyped here instead of read
from the config both sides share. Nothing about the sim justifies a second implementation of the
controller, so there no longer is one: every function below IS the object the real follower
imports, loaded from the same file on disk.

Loading it takes two small tricks, both of which exist because the behavior conda env
deliberately does not have the robot comms stack:

  * ``omniteleop.follower.__init__`` imports ``dexcomm``, so importing
    ``omniteleop.follower.base_closed_loop`` by name fails. The module itself is numpy-only, so
    it is loaded straight from its file and the package ``__init__`` is never executed.
  * The tunables are read straight out of the ``vr_teleop:`` block of ``wbik.yaml`` rather than
    through ``VRTeleopConfig``. That class is the natural home for them and is what
    ``scripts/wbc_vr_robot.py`` binds its CLI defaults from -- but it lives in
    ``omniteleop.wbc_teleop``, which imports ``dexcomm`` for the VR transport and ``loguru``
    through ``omniteleop.common``, neither of which the behavior env has. Stubbing a chain of
    unrelated libraries to reach a yaml parser trades one snapshot risk for another, so the file
    both sides already share is read directly instead, and EVERY field the sim uses must be
    present in it (`_vr_teleop` raises otherwise). A field that gains a code-side default without
    appearing in the yaml therefore fails loudly here rather than diverging quietly.

``omniteleop.wbc_stream`` and ``omniteleop.wbc_robot_util`` are numpy-only and import normally.

What that buys, concretely: every control constant in the sim has exactly one definition, in
``wbik.yaml``, and a hardware retune reaches the sim for free.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys

import yaml

#: Checkout that owns the real follower code. The WBC service already hardcodes this tree
#: (see `wbc_service.DEFAULT_CONFIG`); the env var is here so a second checkout can be pointed at.
OMNITELEOP_SRC = os.environ.get("OMNITELEOP_SRC", "/home/yixuan/omniteleop/src")
if not os.path.isdir(OMNITELEOP_SRC):
    raise RuntimeError(
        f"OMNITELEOP_SRC={OMNITELEOP_SRC!r} is not a directory; the sim cannot reach the real "
        "follower control code, and running a re-implementation instead is exactly what this "
        "module exists to prevent"
    )
if OMNITELEOP_SRC not in sys.path:
    sys.path.insert(0, OMNITELEOP_SRC)


def _load_by_path(name, relpath):
    """Import a numpy-only omniteleop module from its FILE, skipping the package __init__."""
    path = os.path.join(OMNITELEOP_SRC, relpath)
    if not os.path.isfile(path):
        raise RuntimeError(f"expected real follower module at {path}, which does not exist")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # required before exec for the module's own self-refs
    spec.loader.exec_module(module)
    return module


#: The follower config both sides share. `wbc_service.py` loads its SOLVER half from this exact
#: file; this module loads the control-loop half (`vr_teleop:`) from it.
WBIK_YAML = os.environ.get(
    "OMNITELEOP_WBIK_YAML", os.path.join(OMNITELEOP_SRC, "omniteleop/follower/wbik.yaml"))


class _YamlBlock:
    """One block of the shared follower config, with every access checked against the file."""

    def __init__(self, path, section=None):
        if not os.path.isfile(path):
            raise RuntimeError(f"follower config {path!r} not found")
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        block = doc if section is None else doc.get(section)
        if not isinstance(block, dict):
            raise RuntimeError(f"{path!r} has no `{section}:` block to share with the follower")
        self._block, self._path = block, path
        self._label = "" if section is None else f"{section}."

    def __getattr__(self, name):
        block = object.__getattribute__(self, "_block")
        if name not in block:
            raise AttributeError(
                f"{object.__getattribute__(self, '_label')}{name} is not in "
                f"{object.__getattribute__(self, '_path')!r}; the sim will not substitute a local "
                "default for a shared control parameter"
            )
        return block[name]


_bcl = _load_by_path("omniteleop.follower.base_closed_loop",
                     "omniteleop/follower/base_closed_loop.py")
_stream = importlib.import_module("omniteleop.wbc_stream")
_robot_util = importlib.import_module("omniteleop.wbc_robot_util")

# ---- the real controller, re-exported under the names the sim already calls ----
wrap_pi = _bcl.wrap_pi
limit_twist = _bcl.limit_twist
pose_error_body = _bcl.pose_error_body
pd_twist = _bcl.pd_twist
shape_twist = _bcl.shape_twist
shape_project_twist = _bcl.shape_project_twist
mask_planar_twist_for_base_dofs = _bcl.mask_planar_twist_for_base_dofs
project_planar_twist_single_axis = _bcl.project_planar_twist_single_axis
integrate_se2 = _bcl.integrate_se2
BASE_DOF_MODES = _bcl.BASE_DOF_MODES

TargetInterpolator = _stream.TargetInterpolator
HeadTargetLowPassFilter = _stream.HeadTargetLowPassFilter
HeadTargetPlanarDeadbandFilter = _stream.HeadTargetPlanarDeadbandFilter

clamp_joint_step = _robot_util.clamp_joint_step
base_quiet_dispatch = _robot_util.base_quiet_dispatch

#: The `vr_teleop:` block of wbik.yaml -- the same values the hardware follower binds its CLI
#: defaults from (`scripts/wbc_vr_robot.py: _VR_TELEOP = VRTeleopConfig.from_yaml()`).
VR_TELEOP = _YamlBlock(WBIK_YAML, "vr_teleop")
#: The solver block (top level) -- the single-axis projection caps and thresholds the follower
#: reads off `WBCConfig`. The sim needs them follower-side too, for the same projection.
WBIK = _YamlBlock(WBIK_YAML)

# Fail at import rather than at the first control tick if the real modules ever move a name.
_REQUIRED = (
    "wrap_pi", "limit_twist", "pd_twist", "shape_twist", "shape_project_twist",
    "mask_planar_twist_for_base_dofs", "TargetInterpolator", "HeadTargetLowPassFilter",
    "HeadTargetPlanarDeadbandFilter", "clamp_joint_step", "base_quiet_dispatch",
)
_missing = [n for n in _REQUIRED if globals().get(n) is None]
if _missing:
    raise RuntimeError(f"real follower control code is missing {_missing}")
_REQUIRED_CFG = (
    "ik_rate", "cmd_rate", "base_kp_xy", "base_kp_yaw", "base_deadband", "base_accel",
    "base_max_speed", "base_post_linear_deadband", "base_post_angular_deadband",
    "head_lpf_tau", "head_planar_pos_deadband", "head_planar_yaw_deadband",
    "base_yaw_hold_in_xy",
)
_missing_cfg = [n for n in _REQUIRED_CFG if not hasattr(VR_TELEOP, n)]
if _missing_cfg:
    raise RuntimeError(f"wbik.yaml vr_teleop block is missing {_missing_cfg}")
