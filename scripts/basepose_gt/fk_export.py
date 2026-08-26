"""Export per-frame base_T_cam + arm-chain points (base frame) for the ICP ground truth.

Runs in the dexmate env (the only one with real pinocchio). The head camera cloud is
lifted into the base frame with base_T_cam; the arm points are the capsule skeleton used
to delete the robot's own arms from the cloud before ICP.
"""
import os
import sys
import h5py
import numpy as np
import pinocchio as pin

EP, OUT, STRIDE = sys.argv[1], sys.argv[2], int(sys.argv[3])
URDF = os.path.expanduser(
    "~/yixuan_utilities/src/yixuan_utilities/assets/robot/"
    "vega-urdf/vega_with_robotiq.urdf")
TORSO = ["torso_j1", "torso_j2", "torso_j3"]
HEAD = ["head_j1", "head_j2", "head_j3"]
LARM = [f"L_arm_j{i}" for i in range(1, 8)]
RARM = [f"R_arm_j{i}" for i in range(1, 8)]
# Capsule skeleton: every arm link origin plus the gripper, per arm.
ARM_FRAMES = ([f"L_arm_l{i}" for i in range(1, 9)] + ["L_ee", "L_robotiq"]
              + [f"R_arm_l{i}" for i in range(1, 9)] + ["R_ee", "R_robotiq"])

m = pin.buildModelFromUrdf(URDF)
d = m.createData()
cam_fid = m.getFrameId("zed_left_camera")
arm_fids = [m.getFrameId(n) for n in ARM_FRAMES]

with h5py.File(EP, "r") as f:
    n_all = f.attrs["n_frames"]
    idx = np.arange(0, n_all, STRIDE)
    torso = np.asarray(f["obs/joint/torso"], np.float64)[idx]
    head = np.asarray(f["obs/joint/head"], np.float64)[idx]
    larm = np.asarray(f["obs/joint/left_arm"], np.float64)[idx]
    rarm = np.asarray(f["obs/joint/right_arm"], np.float64)[idx]
    out = {"frames": idx,
           "odom": np.asarray(f["obs/base/pose"], np.float64)[idx],
           "arkit": np.asarray(f["obs/base/pose_arkit"], np.float64)[idx],
           "arkit_world": np.asarray(f["obs/base/pose_arkit_world"], np.float64)[idx],
           "t_ns": np.asarray(f["timestamp_ns"], np.int64)[idx],
           "K": np.asarray(f["meta/head_stereo/left_K"], np.float64),
           "baseline_m": float(np.asarray(f["meta/head_stereo/baseline_m"]))}

base_T_cam = np.zeros((len(idx), 4, 4))
arm_pts = np.zeros((len(idx), len(arm_fids), 3))
for k in range(len(idx)):
    q = pin.neutral(m)
    for names, vals in ((TORSO, torso[k]), (HEAD, head[k]), (LARM, larm[k]), (RARM, rarm[k])):
        for nm, v in zip(names, vals):
            q[m.joints[m.getJointId(nm)].idx_q] = v
    pin.framesForwardKinematics(m, d, q)
    M = d.oMf[cam_fid]
    base_T_cam[k, :3, :3], base_T_cam[k, :3, 3], base_T_cam[k, 3, 3] = M.rotation, M.translation, 1.0
    for j, fid in enumerate(arm_fids):
        arm_pts[k, j] = d.oMf[fid].translation

out["base_T_cam"] = base_T_cam
out["arm_pts_base"] = arm_pts
np.savez_compressed(OUT, **out)
print(f"{len(idx)} frames -> {OUT}")
print(f"base_T_cam[0] t={base_T_cam[0,:3,3].round(4)}  "
      f"spread over take: {np.ptp(base_T_cam[:,:3,3],0).round(5)} m")
