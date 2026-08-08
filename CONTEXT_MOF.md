# MoF WBC Pipeline — Session Briefing

**Status (2026-07-18):** v1 MoF-MoE on box2cloth is complete (port → train →
offline vis → rollout wrapper). Hardware rollout not yet exercised.
**Position conditioning is deliberately out of scope for v1** — the porter
always writes `env_state`, but the task `shape_meta` does not consume it.

Primary sources (local, some gitignored):


| Doc                            | Path                                               |
| ------------------------------ | -------------------------------------------------- |
| Spec + completion              | `.scratch/mof-wbc-integration/spec.md`             |
| 7 resolved tickets             | `.scratch/mof-wbc-integration/issues/0{1..7}-*.md` |
| ADR (sim stack, not mof_hommi) | `docs/adr/0001-mof-sim-stack-for-wbc.md`           |
| Shared WBC glossary            | `CONTEXT.md`                                       |
| Runbook (copy-paste commands)  | `README.md` → `## MoF (Mixture of Frames)`         |
| ManiFlow condition vocab       | `/home/yixuan/ManiFlow_Policy/CONTEXT.md`          |


---



## 1. What MoF is (method)

**Mixture of Frames (MoF)** is a diffusion policy that denoises the *same*
action chunk in several reference frames at once, then fuses the per-frame
predictions.

- Paper / site: [https://mofpo.github.io/](https://mofpo.github.io/)
- Fork used here: `/home/yixuan/mofpo` branch `wbc` (origin `Crdr2/mofpo`,
upstream `pointW/mofpo`)
- Carrier stack: the paper's `mof/` **simulation** stack (UNet + ResNet18
spatial-softmax, absolute world actions) — **not** `mof_hommi` (DiT +
pointmap + relative look-at head). See ADR 0001.



### Frame expert vs pose entity (do not confuse)


| Term                | Meaning in WBC MoF                                                                                                                                               |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pose entity**     | One of three *targets* in the 29-D action: left EEF, right EEF, **head**. Each is pos(3) + column-major rot6d(6). Grippers (dims 9, 19) are **not** entities.    |
| **Frame expert**    | One *denoising branch* that predicts the whole chunk expressed in one reference frame. WBC enables five: `base`, `base_rel_trans`, `left`, `right`, `rel_traj`.  |
| **Canonical space** | The single frame where shared diffusion noise / target lives, and into which every expert's prediction is mapped before the router fuses. WBC: `base_rel_trans`. |


The head is a **third pose entity in every expert transform**. It is **not** a
frame expert (there is no `head` expert). Avoid saying "head expert."

### Representations (WBC)

- `base`: action expressed in the base (odometry) frame.
- `base_rel_trans`: rotations in the base frame; each entity's translation
relative to that entity's own **obs-time** world position.
- `left` **/** `right`: action expressed in the left / right EEF frame at obs time.
- `rel_traj`: each entity's action pose relative to that entity's own
obs-time pose (in the base frame).

Router: **learned** (MoF-MoE). Expert auxiliary loss on.

---



## 2. Why this stack (ADR summary)

Integrate ManiFlow-style: porter → safetensors → mofpo dataset → env-runner-free
train → thin rollout wrapper.

Rejected `mof_hommi` because it changes *everything at once* vs DP/ACT/ManiFlow
baselines (DiT, DINOv3, 512² pointmap, relative actions, 3-D look-at head,
UMI 60 Hz, 8-GPU) and **disables base-frame experts** (iPhone demos lack
odometry). The Vega has real wheel odometry — base experts are exactly what we
want. 

---



## 3. End-to-end pipeline

Shared raw takes with DP/ACT/ManiFlow: **box2cloth**
(`~/Dexmate/data/box2cloth/`, 41 raw + 7 recovery, `split.csv` → 46 train / 2
val). All porters reuse the same FK / world composition / recovery trim /
split.

```text
raw HDF5 (10 Hz, engage-origin world actions)
    │
    ├─ SceneDiff positions (always required for MoF port)
    │     ~/Dexmate/data/box2cloth/scene_diff/positions/{raw,recovery}/episode_N.npz
    │
    ▼  scripts/port_wbc_mobile_mof.py          env: dexmate_lerobot
safetensors + meta.json
    ~/Dexmate/data/box2cloth/processed_wbc/mof/dexmate_wbc_mof/
    │
    ▼  mofpo train.py --config-name=train_mof_moe_wbc   env: mof
checkpoint (.ckpt, EMA)
    ~/mofpo/data/outputs/.../checkpoints/latest.ckpt
    (archived: ~/Dexmate/model/mof/wbc_v1_mof_moe/)
    │
    ├─ offline: scripts/vis_wbc_mof_prediction.py       env: dexmate_mof
    └─ robot:   scripts/wbc_mof_rollout.py              env: dexmate_mof
         └─ injects MoFBundle into shared wbc_policy_rollout loop
```

Commands (canonical copy in README MoF section):

```bash
# 1 Port
(dexmate_lerobot) python scripts/port_wbc_mobile_mof.py \
  --raw-dir ~/Dexmate/data/box2cloth/raw_data \
  --out-root ~/Dexmate/data/box2cloth/processed_wbc/mof \
  --include-recovery-data ~/Dexmate/data/box2cloth/raw_data/recovery \
  --positions-dir ~/Dexmate/data/box2cloth/scene_diff/positions

# 2 Train (no simulator; task.env_runner: null → val_loss top-k)
(mof) cd ~/mofpo && python train.py --config-name=train_mof_moe_wbc \
  logging.mode=offline hydra.run.dir=data/outputs/wbc_v1_mof_moe_500ep

# 3 Offline validate (exact deploy inference path)
(dexmate_mof) python scripts/vis_wbc_mof_prediction.py \
  --policy-path ~/mofpo/data/outputs/wbc_v1_mof_moe_500ep/checkpoints/latest.ckpt \
  --episode ~/Dexmate/data/box2cloth/processed_wbc/mof/dexmate_wbc_mof/test/episode_0.safetensors \
  --output ~/Dexmate/data/visualization/mof_pred_test0.png

# 4 Hardware (shared loop; measure interval on a free GPU)
(dexmate_mof) python scripts/wbc_mof_rollout.py \
  --policy-path ~/mofpo/data/outputs/wbc_v1_mof_moe_500ep/checkpoints/latest.ckpt \
  --save-dir ~/Dexmate/data/rollout/mof \
  --align-reference ~/Dexmate/data/raw_data_reference/reference.hdf5 \
  --policy-interval 0.6 --max-seconds 100
# or --num-inference-steps 8 (~171 ms/chunk, same offline MAE to 0.01 cm)
```



### Environments


| Env               | Role                                                                                                                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dexmate_lerobot` | MoF porter (FK, safetensors, SceneDiff npz)                                                                                                                                                                 |
| `mof`             | Training only (mujoco 3.3.5 / mink pins — do not perturb)                                                                                                                                                   |
| `dexmate_mof`     | Offline vis + robot rollout. Clone of `dexmate_lerobot` + `mofpo_root.pth` + robomimic/threadpoolctl + ManiFlow env's pytorch3d. mofpo has no `__init__.py`; `.pth` is required (same pattern as ManiFlow). |


---



## 4. Data contract (porter → dataset)

Schema id: `wbc_mof_safetensors_v1`. Sidecar
`<dataset_root>/meta.json` is the **train + deploy source of truth** (resolution,
anisotropic squash, fps, column rot6d, wxyz quats, tensor schema, episode
split list). Bundle load refuses a mismatch before engage.

### Per-episode safetensors keys


| Key                                                      | Shape                 | Notes                                                                                                                                            |
| -------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `head_image`, `wrist_image`                              | `(T,224,224,3)` uint8 | Anisotropic squash from 240×320 — **full FOV**, no porter crop. Train/infer apply paper chain: resize 224 → crop 192 → resize 76, ImageNet norm. |
| `left_ee_pos/quat`, `right_ee_pos/quat`, `head_pos/quat` | `(T,3)` / `(T,4)`     | Engage-origin **world**; quat **wxyz**                                                                                                           |
| `base_pos`, `base_quat`                                  | `(T,3)` / `(T,4)`     | Planar odometry `(x,y,yaw)` lifted to 3-D pose                                                                                                   |
| `proprioception_grippers`                                | `(T,2)`               | Raw FC03 (state dims 9/19)                                                                                                                       |
| `action`                                                 | `(T,29)`              | Verbatim `ik.solve` targets, **column** rot6d, world                                                                                             |
| `env_state`                                              | `(T,6)`               | Constant `[box=src, cloth=dst]` world XYZ from SceneDiff. **Always written.**                                                                    |


29-D action layout (same as DP/ACT/ManiFlow):

```text
0-8 left EEF (pos3 + rot6d) | 9 left grip
10-18 right EEF             | 19 right grip
20-28 head EEF (full pose target — NOT look-at)
```

**Rotation rule:** port keeps column rot6d. Checkpoint sets
`external_rot6d_convention: column` so BiGym's row↔column boundary conversion
must **not** run. A row-convention checkpoint would emit silently transposed
rotations — the bundle gates this at load.

**Action bit-identity:** MoF `action` matches LeRobot / ManiFlow porters
(shared `compute_frame_state_action` / `wbc_policy_format`). Obs poses match
ManiFlow's world-frame `agent_pos` entity blocks (not DP's base-frame
`observation.state[:29]`).

### Frames vs sibling stacks


|              | DP / LeRobot               | ManiFlow          | MoF                                                 |
| ------------ | -------------------------- | ----------------- | --------------------------------------------------- |
| Action       | world 29-D column rot6d    | same              | same                                                |
| Obs EEF/head | **base** (egocentric FK)   | **world**         | **world** pos+quat keys                             |
| Visual       | RGB 120×160 (train resize) | world point cloud | RGB 224² squash → 76                                |
| `env_state`  | optional parquet ENV       | optional zarr     | **always** in safetensors; v1 `shape_meta` omits it |


---



## 5. Training / policy wiring (mofpo)


| Piece        | Path                                                                                     |
| ------------ | ---------------------------------------------------------------------------------------- |
| Task config  | `mof/config/task/dexmate_wbc.yaml` (`env_runner: null`, `val_ratio: 0`)                  |
| Train config | `mof/config/train_mof_moe_wbc.yaml`                                                      |
| Dataset      | `mof/dataset/dexmate_wbc_replay_dataset_mof.py`                                          |
| Layout       | `mof.common.action_layout.wbc_layout()` (3 entities; BiGym default stays byte-identical) |
| Transforms   | `mof.common.mof_transform_util` (generic N-entity paths)                                 |


v1 recipe (paper defaults, WBC overrides only):

- batch 128, DDIM 50 train / 16 infer, EMA, ~500 epochs
- horizon 16, 8 action steps, 2 obs steps, 10 Hz
- experts: `[base, base_rel_trans, left, right, rel_traj]`
- canonical: `base_rel_trans`
- `action_layout: wbc`, `external_rot6d_convention: column`
- checkpoint top-k on `val_loss` **(min)** — not sim score
- `n_demo: null` (all 46 train episodes); val = porter `test/` only

Dataset precomputes all five expert action representations over the three
entities; grippers stay absolute/untouched in those transforms. Round-trip
tests: each expert → inverse/canonical decode → porter world action.

**Important:** `env_state` is loaded into the replay buffer but exposed to the
policy **only if listed in** `shape_meta.obs`. v1 task config omits it →
unconditioned. Conditioned MoF = add `env_state` to `shape_meta` + wire the
policy's low-dim / global cond path (config flip; no re-port).

---



## 6. Deploy (omniteleop)


| Piece           | Path                                                                          |
| --------------- | ----------------------------------------------------------------------------- |
| Bundle          | `src/omniteleop/wbc_mof_bundle.py` (`MoFBundle`)                              |
| Offline vis     | `scripts/vis_wbc_mof_prediction.py`                                           |
| Rollout wrapper | `scripts/wbc_mof_rollout.py` → shared `wbc_policy_rollout` via policy-factory |


Bundle surface (same as DP / ManiFlow bundles):

- `reset()`, `predict_chunk(obs_history)` → `(n_action_steps, 29)` **absolute
world** actions (de-canonicalized from fused `base_rel_trans`)
- `obs_from_port_arrays` / `obs_from_live` (world entity poses + cameras)
- `use_env_state = False`, `position_condition_mode = "none"` for v1
- Load gates: WBC layout, column convention, `meta.json` match, fitted normalizer

Do **not** pass `--position-condition` to `wbc_mof_rollout.py` for v1 ckpts.

---



## 7. v1 results (held-out MAE)

Hardware candidate: `~/Dexmate/model/mof/wbc_v1_mof_moe/checkpoints/latest.ckpt`
(prefer **latest** over best-val_loss — val_loss is a noisy epsilon proxy;
action MAE kept improving). Overlays under
`~/Dexmate/data/visualization/mof_pred_test{0,1}_*.png`.


| ckpt       | take | left cm/deg/grip        | right cm/deg/grip       | head cm/deg     |
| ---------- | ---- | ----------------------- | ----------------------- | --------------- |
| epoch 70   | ep0  | 2.68 / 4.97 / 88.7%     | 2.53 / 3.65 / 96.6%     | 2.11 / 1.58     |
| epoch 70   | ep1  | 2.81 / 4.37 / 95.8%     | 2.88 / 4.22 / 95.5%     | 2.30 / 1.66     |
| **latest** | ep0  | **2.24 / 4.71 / 85.7%** | **2.15 / 3.43 / 95.4%** | **1.56 / 1.29** |
| **latest** | ep1  | **2.41 / 3.74 / 95.8%** | **2.50 / 4.74 / 96.6%** | **2.03 / 1.27** |


---



## 8. Ticket map (already done)


| #   | What                                                    | Repo / commit note                    |
| --- | ------------------------------------------------------- | ------------------------------------- |
| 01  | N-entity frame transforms (BiGym bit-identical default) | mofpo `74dbc80`                       |
| 02  | MoF porter + meta.json + contract tests                 | omniteleop `80b35c3`                  |
| 03  | WBC dataset + `dexmate_wbc` task + round-trips          | mofpo `2b480af`                       |
| 04  | Env-runner-free training + micro-train smoke            | mofpo `2c68098`                       |
| 05  | `MoFBundle` + offline vis + `dexmate_mof` env           | omniteleop `4586e77`                  |
| 06  | `wbc_mof_rollout` thin wrapper                          | omniteleop                            |
| 07  | 500-ep v1 run + README                                  | archived under `~/Dexmate/model/mof/` |


---



## 9. ManiFlow comparison



### What already exists on the MoF side

1. **Porter always attaches** `env_state` (6-D task-slot order `[src, dst]`,
  same SceneDiff contract as ManiFlow / LeRobot). No re-port needed for a
   position-conditioned MoF run.
2. **Dataset already loads** `env_state` **into the replay buffer**; gating is
  solely `shape_meta.obs`.
3. **Bundle has** `set_env_state` / `use_env_state` knobs stubbed toward the
  shared rollout's `--position-condition` path (v1 keeps them off).
4. Shared vocabulary: task slots, `env_state`, engage-origin world — see
  `ManiFlow_Policy/CONTEXT.md` + this repo's `CONTEXT.md`.

MoF v1 is RGB + low-dim UNet diffusion — **no point cloud, no DiTX, no mask**  
**channels**. 

### Invariants that must not break

- Engage-origin **world** actions; column rot6d end-to-end for WBC MoF.
- Head = full 9-D pose entity, never look-at.
- `meta.json` schema / squash / fps / convention gates at deploy.
- `split.csv` train/val identity with other baselines.
- BiGym / DexMimicGen paths in mofpo stay byte-identical (additive WBC files +  
gated edits only).
- Do not adopt `mof_hommi` as a shortcut for "real robot."

---



## 10. Out of scope

- `mof_hommi` DiT / pointmap stack
- New task collections outside box2cloth
- MoF-Ensemble / single-frame ablations (after a conditioned or hardware
baseline)

---



## 11. Agent workflow reminders

- Vocabulary: use `CONTEXT.md` terms (engage-origin world, pose entity, frame
expert, canonical space, porter, env_state, box2cloth).
- Validate dims aggressively; never pad with zeros to "make it run."
- Prefer surgical diffs; keep upstream sim behavior identical.
- Verify with: porter contract tests, expert round-trips
(`tests/test_wbc_dataset.py`), optional
`MOF_RUN_TRAIN_SMOKE=1 pytest tests/test_wbc_training_smoke.py`, then
offline `vis_wbc_mof_prediction.py` before any robot run.
- Issues for new work: markdown under `.scratch/<feature-slug>/` (GitHub
Issues disabled). See `docs/agents/issue-tracker.md`.

