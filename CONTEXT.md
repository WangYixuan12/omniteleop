# Omniteleop WBC Policy Pipeline

Vocabulary for the Dexmate Vega whole-body teleoperation + learned-policy pipeline
(record → port → train → roll out), shared across the DP/ACT (LeRobot), ManiFlow,
and MoF policy integrations.

## Language

### Frames & poses

**Engage-origin world**:
The fixed world frame anchored where the base odometry was zeroed at engage.
Actions, `obs/base/pose`, SceneDiff positions, and ManiFlow clouds all live here.
_Avoid_: odom frame, global frame

**Base frame**:
The robot's planar (x, y, yaw) frame at a given timestep, from wheel odometry.
LeRobot `observation.state[:29]` is expressed here (egocentric FK).
_Avoid_: robot frame, chassis frame

**Pose entity**:
One of the three world-pose targets in the 29-D action: left EEF, right EEF, head.
Each is position (3) + column-major rot6d (6); grippers at dims 9/19 are not entities.

**Head target**:
The 9-D world-frame pose commanded to `VegaWholeBodyIK.solve(..., head_target=...)`
(dims 20–28). A full pose, not a look-at point (that is mof_hommi's convention).

### MoF (Mixture of Frames)

**Frame expert**:
One denoising branch of the MoF policy that predicts the whole action chunk
expressed in one reference frame. WBC run enables five:
`base`, `base_rel_trans`, `left`, `right`, `rel_traj`.
_Avoid_: head expert (entities are not experts)

**Canonical space**:
The single frame the shared diffusion target/noise lives in and into which every
expert's prediction is mapped before fusing. WBC run: `base_rel_trans`.

**base_rel_trans**:
Action representation with rotations in the base frame and each entity's
translation relative to that entity's own obs-time position.

**rel_traj**:
Trajectory-relative representation: each entity's action pose expressed relative
to that entity's own obs-time pose (in the base frame).

### Data

**Raw take / episode**:
One teleoperated recording (`episode_N.hdf5`) under a task's `raw_data/`
(+ `recovery/` for recovery takes, whose training window the porters trim).

**box2cloth**:
The completed 41-raw + 7-recovery box→cloth task dataset under
`~/Dexmate/data/box2cloth/` — the common training set for DP, ACT, ManiFlow, MoF.

**Porter**:
A script in `omniteleop/scripts/` that converts raw takes into one policy stack's
training format, replicating FK/world composition, recovery trimming, and the
`split.csv` train/test split. One porter per stack (LeRobot hdf5, ManiFlow zarr,
MoF safetensors).

**Position condition / env_state**:
The per-episode-constant 6-D `[box, cloth]` world positions from SceneDiff,
attached by every porter; whether a policy consumes it is a training-config choice.
_Avoid_: scene condition, object prior
