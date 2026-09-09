"""Read the follower's robot model without importing the hardware or IK stack."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

OMNITELEOP_SRC = Path(os.environ.get(
    "OMNITELEOP_SRC", Path(__file__).resolve().parents[1] / "src")).expanduser()
WBIK_YAML = Path(os.environ.get(
    "OMNITELEOP_WBIK_YAML", OMNITELEOP_SRC / "omniteleop/follower/wbik.yaml")).expanduser()
OPTICAL_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])


def follower_urdf(config=WBIK_YAML):
    """Resolve wbik.yaml's package prefix in the same order as whole_body_ik."""
    with Path(config).open() as stream:
        value = yaml.safe_load(stream)["urdf_path"]
    match = re.match(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
    if match:
        package = match[1]
        override = os.environ.get(f"{package.upper()}_ROOT")
        if override:
            root = Path(override).expanduser()
        else:
            try:
                module = importlib.import_module(package)
            except ImportError:
                root = None
            else:
                root = Path(module.__file__).parent if module.__file__ else None
            if root is None and package == "yixuan_utilities":
                root = Path.home() / package / "src" / package
        if root is None or not root.is_dir():
            raise FileNotFoundError(f"Cannot resolve {package}; set {package.upper()}_ROOT")
        value = str(root) + value[match.end():]
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Follower URDF does not exist: {path}. Point YIXUAN_UTILITIES_ROOT at the "
            "same updated package used by the WBC service."
        )
    return path


def origin_matrix(element):
    """URDF origin to a column-vector transform (fixed-axis roll, pitch, yaw)."""
    origin = element.find("origin")
    transform = np.eye(4)
    if origin is not None:
        transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return transform


def fixed_transform(root, parent, child):
    """Return parent_T_child, rejecting frames not rigidly attached to parent."""
    joints = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
    transform, seen = np.eye(4), set()
    while child != parent:
        if child in seen or child not in joints:
            raise ValueError(f"No fixed chain from {parent} to {child}")
        seen.add(child)
        joint = joints[child]
        if joint.get("type") != "fixed":
            raise ValueError(f"{joint.get('name')} is not fixed")
        transform = origin_matrix(joint) @ transform
        child = joint.find("parent").get("link")
    return transform


def wrist_camera_poses(urdf=None):
    """EEF-local USD camera poses from the URDF optical frames, for both wrists."""
    root = ET.parse(follower_urdf() if urdf is None else urdf).getroot()
    return {
        side: fixed_transform(root, f"{prefix}_ee", f"{prefix}_wrist_camera") @ OPTICAL_TO_USD
        for side, prefix in (("left", "L"), ("right", "R"))
    }
