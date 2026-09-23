# Left-wrist corruption labeling — retired 2026-09-22

Wrist frames stopped tearing once the wrist cameras moved to a USB hub; Yifan verified
the 2026-09-20 collection (`/data/Dexmate/data_mobile/{basket,pillow}`) has no corrupted
frames. Corruption labeling was removed from the recording runbook
(`/home/yifan/workspace/wm/README.md` §4d). Do not label new collections.

The files are kept here, at their original repo-relative paths, for reference:

| path | role |
|---|---|
| `scripts/diagnostics/label_wrist_corruption.py` | batch auto-label: writes `quality/left_wrist` into teleop HDF5 |
| `scripts/diagnostics/review_wrist_corruption.py` + `wrist_review.html` | browser review (X marks a frame corrupted) |
| `scripts/diagnostics/annotate_wrist_corruption.py` | feature extraction, forest training, scan/apply |
| `src/omniteleop/wrist_corruption.py`, `src/omniteleop/wrist_review.py` | schema, features, annotation/review writes |
| `configs/camera_quality/left_wrist_v1.json` | trained forest (default `--model`) |
| `tests/test_wrist_*.py` | unit/HTTP tests |
| `docs/wrist_corruption.md` | full workflow and `quality/left_wrist` schema |

Training data, feature caches and batch reports stay in `tmp/wrist_corruption/`.

## Still reading `quality/left_wrist`

- `src/omniteleop/common/capture_timeline.py` (`quality_mask`): honors labels when present
  (older `data_ws`/`data_lh` batches); a missing group now means every frame is accepted
  and adds no "Wrist review incomplete" blocker.
- robo02 segmentation/packing (`frame_quality.py`, `segment_episode.py`, `pack_episode*.py`
  under `mobile_wm/ws_scans_20260913` and the `unet3d_handoff_20260919` runtimes) refuse
  unlabeled episodes unless run with `--allow-unlabeled`.

## Restore

Move each file back to the same relative path under the repo root, then run
`PYTHONPATH=src pytest tests/test_wrist_corruption.py tests/test_wrist_review.py tests/test_wrist_review_http.py`.
