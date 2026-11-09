# Changes

## Team handoff — 2026-11-09
- Initialized git repository; added `CONTRIBUTING.md` (main/develop/feature branch flow), PR template, CI workflow and `reviews/` for human decision CSVs.
- Added `docs/21-human-tasks-handoff.md` (tasks that need people) and `docs/22-training-and-model-guide.md` (what was trained, backbone choice, how to train more).
- Rewrote `docs/04-implementation-status.md` to match the finished experiments; marked older status docs (05, 12, 20) as superseded.
- No training or evaluation was rerun; v2 results are unchanged.

## Research follow-up — 2026-11-02
- Added explicit robustness comparison axes and bound evidence to provenance and the selected validation checkpoint; robustness now defaults to validation.
- Added guarded orphan recovery with process lock, PID checks and recovery events; suite manifests track attempts, retries, completed-run evidence and frozen dataset split hashes.
- Added explicit incomplete-label exclusion with an audit trail, pretrained verification, train-only benchmarking and exhaustive source-train dHash audit tools.
- Fixed checkpoint temporary filenames on deep Windows paths. Full screening remains blocked pending dataset label and duplicate review.

## 0.2 — 2026-11-02 continued
- Fixed the training augmentation dictionary being converted to a boolean; old runs without the post-fix protocol marker cannot be ranked.
- Added parameterized Albumentations and occlusion, source holdout protections and dataset audit, and pinned the inspected Hugging Face revision.
- Added single-task ablations, controlled suite planning, screening selection, AMP/gradient accumulation, epoch-boundary resume and run status/provenance.
- Added robustness, room-domain and report tools, and verified model checkpoint round trips with random weights.
- Verified pretrained ResNet18 and a small source-preview smoke on CUDA; no full training or real-room evaluation is claimed.

## 0.2 — 2026-11-02
- Separate project/dataset/model/training/augmentation/inference/experiment configs.
- Config validation and E1/E2/E3 controlled comparison runner.
- CSV/Hugging Face imports with explicit column mapping, source/product groups and previous split preservation.
- Train-only augmentation and preview command; detector augmentation uses its own box-aware config.
- Snapshot actual configs, code, label mapping, dataset hashes and separate task metrics per run.
- Warm-start full weights or backbone-only; verify label mapping and split membership.
- New CLI: prepare-data, audit-data, train, experiments, evaluate-run, room.
- Current rubric and deadline included in planning documents.

## 0.1
Initial training/demo scaffold and CPU artificial-data smoke tests.
# 2026-11-02 — controlled experiment groundwork

- Added ten torchvision backbone choices, checkpoint backbone metadata, C1–C8 configuration examples, augmentation/occlusion presets, and suite configuration dry-run.
- Preserved legacy E1–E3 configuration and fixed head learning rate being reset when unfreezing the backbone.
- See `docs/04-implementation-status.md` for verification boundaries and remaining work; this is not a completed research evaluation.

