"""Robot mounting and optical-frame checks that do not require Isaac Sim."""
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from port_vega_wrist_cameras import port_graft
from robot_model import fixed_transform, wrist_camera_poses


def _joint(root, name, parent, child, xyz="0 0 0", rpy="0 0 0", kind="fixed"):
    joint = ET.SubElement(root, "joint", name=name, type=kind)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
    return joint


def _reference():
    root = ET.Element("robot", name="fixture")
    for p in ("L", "R"):
        ET.SubElement(root, "link", name=f"{p}_ee")
        rigid = ET.SubElement(root, "link", name=f"{p}_robotiq")
        visual = ET.SubElement(rigid, "visual")
        ET.SubElement(visual, "origin", xyz="0 0.0934 -0.24", rpy="1.5707963 0 0")
        _joint(root, f"{p}_robotiq_j0", f"{p}_ee", f"{p}_robotiq", "0 0 0.24", "0 0 1.5708")
    return root


def _current(reference):
    root = deepcopy(reference)
    for p in ("L", "R"):
        root.find(f"joint[@name='{p}_robotiq_j0']/origin").set("xyz", "0 0 0.2341892")
        root.find(f"link[@name='{p}_robotiq']/visual/origin").set("xyz", "0 0.0934 -0.2005")
        for suffix in ("gripper_connector", "wrist_cam_bracket", "wrist_zed_mini", "wrist_camera"):
            ET.SubElement(root, "link", name=f"{p}_{suffix}")
            _joint(root, f"{p}_{suffix}_mount", f"{p}_ee", f"{p}_{suffix}")
    return root


def _graft():
    root = ET.Element("robot", name="graft")
    for p in ("L", "R"):
        ET.SubElement(root, "link", name=f"{p}_ee")
        ET.SubElement(root, "link", name=f"{p}_gripper_base")
        _joint(root, f"{p}_mount", f"{p}_ee", f"{p}_gripper_base")
        for i in range(8):
            ET.SubElement(root, "link", name=f"{p}_knuckle{i}")
            joint = _joint(root, f"{p}_knuckle_joint{i}", f"{p}_gripper_base", f"{p}_knuckle{i}", kind="revolute")
            if i >= 2:
                ET.SubElement(joint, "mimic", joint=f"{p}_knuckle_joint0", multiplier="-1")
    return root


def test_port_moves_whole_gripper_without_changing_linkage():
    reference, graft = _reference(), _graft()
    current = _current(reference)
    before = ET.tostring(graft)
    result, offsets = port_graft(graft, reference, current)
    assert ET.tostring(graft) == before  # original is the rebuild/rollback source
    for p in ("L", "R"):
        # The reference-frame shortening offsets part of the 39.5 mm visual shift.
        np.testing.assert_allclose(offsets[p]["mount"], np.array([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, .0336892], [0, 0, 0, 1]]), atol=1e-12)
        for i in range(8):
            selector = f"joint[@name='{p}_knuckle_joint{i}']"
            assert ET.tostring(result.find(selector)) == ET.tostring(graft.find(selector))
        assert result.find(f"link[@name='{p}_gripper_connector']") is not None
        assert result.find(f"link[@name='{p}_wrist_camera']") is not None
    assert len(result.findall("joint/mimic")) == 12
    with pytest.raises(ValueError, match="already has wrist cameras"):
        port_graft(result, reference, current)


def test_camera_uses_composed_optical_pose_and_usd_axis_convention(tmp_path):
    root = ET.Element("robot")
    for p in ("L", "R"):
        _joint(root, f"{p}_body", f"{p}_ee", f"{p}_camera_body", "0.02 -0.03 0.04", "0.4 -0.2 0.7")
        _joint(root, f"{p}_lens", f"{p}_camera_body", f"{p}_wrist_camera", "0.0065 0.03025 0", "1.5707963267948966 0 0")
    path = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(path)
    poses = wrist_camera_poses(path)
    body = Rotation.from_euler("xyz", [.4, -.2, .7]).as_matrix()
    optical = body @ Rotation.from_euler("x", np.pi / 2).as_matrix()
    for pose in poses.values():
        np.testing.assert_allclose(pose[:3, 3], [.02, -.03, .04] + body @ np.array([.0065, .03025, 0]))
        np.testing.assert_allclose(pose[:3, :3] @ [0, 0, -1], optical @ [0, 0, 1])
        np.testing.assert_allclose(pose[:3, :3] @ [0, 1, 0], optical @ [0, -1, 0])
    root.find("joint").set("type", "revolute")
    with pytest.raises(ValueError, match="not fixed"):
        fixed_transform(root, "L_ee", "L_wrist_camera")
