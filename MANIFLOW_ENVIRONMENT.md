# ManiFlow Conda environment

This document owns the detailed environment setup that used to live in the main README. The active
contract is deliberately simple: **all integrated ManiFlow action-frame modes use
`dexmate_maniflow`**.

## Environment ownership

| Environment | Role | Status |
| --- | --- | --- |
| `dexmate_maniflow` | Build/inspect the zarr; train, inspect, and deploy `action_frame_mode=world` or `action_frame_mode=mof` | Active |
| `lerobot` | SceneDiff DINO feature extraction before the zarr build | Active preprocessing |
| `dexmate_lerobot` | SAM3.1 mask tracking before the zarr build | Active preprocessing |
| `dexmate_mof` | Offline visualization and robot rollout for old standalone mofpo checkpoints | Legacy reproduction only |
| `mof` | Training old standalone mofpo checkpoints | Legacy reproduction only |
| `maniflow` | Generic upstream ManiFlow environment | Removed 2026-07-19 |

Do not create a separate environment for integrated MoF. `action_frame_mode` is a Hydra/config and
checkpoint choice, not a dependency boundary. The integrated code vendors the required frame
transforms and does not import the standalone `mof` package.

The currently verified `dexmate_maniflow` stack uses Python 3.12.13, PyTorch 2.7.1+cu126,
`timm` 1.0.27, `zarr` 2.18.7, and Hydra 1.3.2. Both integrated modes share those versions.

## Rebuilding `dexmate_maniflow`

`dexmate_maniflow` is a clone of `dexmate_lerobot`. The clone preserves the editable
dexcontrol/omniteleop installs and supplies pinocchio/pink, LeRobot, Rerun, and `fpsample`.

```bash
conda create -n dexmate_maniflow --clone dexmate_lerobot -y
MANIFLOW_PY=~/miniforge3/envs/dexmate_maniflow/bin/python

# The rollout policy import chain needs timm. Keep the clone's NumPy pin.
$MANIFLOW_PY -m pip install --no-deps timm

# Training needs numba and the zarr-v2 API.
$MANIFLOW_PY -m pip install "zarr<3" numba

# Upstream's package layout is imported through a path file.
echo /home/yixuan/ManiFlow_Policy/ManiFlow \
  > ~/miniforge3/envs/dexmate_maniflow/lib/python3.12/site-packages/maniflow_root.pth

# PyTorch3D is a hard import in pointnet_extractor. Build main against Torch 2.7/CUDA 12.6.
export TORCH_CUDA_ARCH_LIST=8.9 FORCE_CUDA=1 MAX_JOBS=$(nproc) CUDA_HOME=/usr
$MANIFLOW_PY -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@main"
```

Use PyTorch3D `main` rather than `stable`: release 0.7.9 only claims PyTorch through 2.4, while
`main` was verified against PyTorch 2.7.1+cu126 and Python 3.12. Its CUDA
`sample_farthest_points` implementation is called on every DP3Encoder forward.

ManiFlow's ReplayBuffer is bound to zarr 2: it calls `zarr.open(path, mode)` positionally and
`zarr.group(read_only_store)`, both rejected by zarr 3. The zarr porter itself is version-agnostic
and always writes format-2 stores.

The upstream `scripts/requirements.txt` is not the source of truth for this WBC environment. Its
`numpy==1.23.5` conflicts with `numba==0.61.2`, which requires NumPy ≥1.24, and its simulation/cloud
dependencies (`sapien`, `mplib`, `dm_control`, `open3d`, `azure`, `deepspeed`) are not required by
this pipeline.

## Verification

```bash
conda activate dexmate_maniflow
cd ~/ManiFlow_Policy/ManiFlow

python -c "from maniflow.model.diffusion.mof_mixture import MofMixture; \
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy; \
print('integrated ManiFlow imports OK')"

python tests/test_frame_transforms.py
```

The frame-transform suite should report `8 tests passed`. The Hydra mode switch should compose into
both the policy and dataset:

```bash
cd ~/ManiFlow_Policy
bash scripts/train_dexmate_wbc.sh <tag> action_frame_mode=mof \
  training.num_epochs=1 training.max_train_steps=5 \
  logging.mode=offline checkpoint.save_ckpt=False
```

The last command is a GPU training smoke, not an environment-install step; run it only when a training
smoke is desired.

## Retired generic `maniflow` environment

The generic Python-3.10 `maniflow` environment was removed on 2026-07-19 after confirming that it
lacked the Dexmate/omniteleop stack and was not used by the integrated WBC pipeline. Its complete
Conda/pip export is retained at:

```text
/home/yixuan/conda-env-backups/maniflow-2026-07-19.yml
SHA-256: 12c1471d8b723c49c290cb9fc4bbae43599bcaca3796d5c99f69ce25edcb2caf
```

Restore it only for generic upstream/legacy simulation work:

```bash
conda env create -f /home/yixuan/conda-env-backups/maniflow-2026-07-19.yml
```

Restoring or removing that legacy environment does not modify the ManiFlow source tree because its
`maniflow` installation points back to `/home/yixuan/ManiFlow_Policy/ManiFlow`.

## Verified data-path invariants

In this stack, enriching a zarr with masks and DINO leaves `point_cloud`, `state`, `action`, and
`env_state` bit-identical to the pre-mask build. The porter recomputes frame 0 and checks the cloud
and mask labels through the same row lineage. The current rollout rebuilds the maskless live cloud
bit-identically to training for its supported conditioning modes; see the live limitation in the main
README.
