# Conservative ZIP snapshot and validation-only pilot

This snapshot uses `D:\Download\furniture_dataset.zip` directly with `new_data/zip_dataset.py`. It is separate from the v2 suite. ZIP SHA-256 is `dce23f8f54b1317d31805564a4d58ab0eff2ed3ce00e06656df188440563f79a`.

## Snapshot and leakage audit

`artifacts/zip-conservative-snapshot-v2` is a new immutable set of manifests and evidence; existing v2 files were not overwritten. From 7,650 ZIP train originals, 5,029 images in 4,936 conservative groups remain, and 2,621 images in 2,135 groups are excluded. Offline augmented copies were not included. Counts are of actual IDs, not pair-count subtraction. `class-counts.csv` provides before/after type × style counts; no type or style class, and no previously present type × style combination, disappeared. `train.csv` SHA-256 is `51adf5181fd6f687a6b41e91f0f18fe8e7d9f1cc7fa363e4a061e7b841e88bbb`; hashes of all manifests, decisions, holdout manifests, source audit inputs and scripts are in `audit.json` and `snapshot-provenance.json`.

The grouping uses source IDs plus exact JPEG bytes reproduced from v2 PNGs, SHA-256 of ZIP members, existing v2 candidate components, and all ZIP-train dHash ≤4 candidate edges both within train and to all three frozen v2 holdouts. Candidate edges are conservative grouping evidence, not an automatic identity conclusion. Group connections are transitive: any group touching a holdout, existing quarantine or unverified source match is excluded, as is a group containing a conflicting raw label. Each link and reason is in `group-evidence.csv`; exclusions and reasons are in `excluded.csv`. The independently run `verification-v2.json` rechecks provenance, graph boundaries, train-only class mapping and all 5,029 retained ZIP images against JPEGs reconstructed from v2 train; it found no retained group touching a holdout or quarantine. Recompression-based equality is tied to the recorded PIL/JPEG operation and source files, not proof of the ZIP's original unpinned revision.

All 5,029 retained JPEGs exactly match reconstructions of v2 train images; **zero additional source images have been established**. This ZIP is useful for direct loading, not evidence that additional data improved coverage. The selected train mapping has the same 4 type and 37 style names/indices as v2, with no unseen validation classes. The source ZIP's prepared mapping has 34 style names because it used aliases. We use each source row's `style_raw` only when the corresponding v2 train source ID and reconstructed JPEG match and its raw label agrees with v2. `mapping-comparison.csv` lists exact-name matches and non-exact names; it does not silently equate aliases or reuse ZIP class indices.

All 19 real-source dHash candidate pairs were visually inspected from `artifacts/zip-batch-audit-20261104/real-source-candidates-*.png` and recorded as preliminary `different` in `real-source-ai-review.csv` with an item-specific reason. These decisions are AI observations, not independent human verification; none was used to lift a quarantine. All groups containing those candidates remain excluded. This preserves the real-source holdout and makes the conservative policy explicit.

One deterministic example per style, rotating through the four types, is in `artifacts/zip-label-review-v1/samples.csv` and `label-samples-*.png`, with AI observations in `ai-review.csv`. Most type labels look plausible, but style cannot be verified from a single image alone. Examples requiring review include a gaming-style chair labeled `classical`, a generic sofa labeled `gothic`, and a canopy bed labeled `scandinavian`. No labels were automatically changed. These synthetic/source annotations are not expert ground truth.

## Smoke and pilot

`artifacts/zip-pilot-smoke-v6/status.json` records a successful four-image CPU optimizer-step smoke using the existing DenseNet121/ImageNet initialization, C6 augmentation, and training loss/optimizer primitives. It is a loader/model integration test, not a research result. The relevant `new_data` tests passed: 9 tests covering the candidate grouping, whole-group exclusion, ZIP loader, label/error handling, worker isolation and failure-event recording.

`new_data/pilot-config.json` specifies a separate, three-epoch pilot: the validation-selected baseline DenseNet121/C6, matching baseline preprocessing functions, augmentation, optimizer, loss weights, batch size, seed and two frozen epochs, then one fine-tuning epoch. Images in the ZIP have already been resized and JPEG-compressed, unlike the v2 PNG inputs; therefore input fidelity, dataset size (5,029 vs 5,956 train images), and compute budget (3 vs up to 30 epochs) differ. Do not interpret differences in validation metrics as an effect of additional training data. The mapping covers 100% of the frozen v2 validation labels for each task. No test predictions or test-based selection are part of this pilot.

The pilot is in `artifacts/zip-pilot-c6-v3`. At the time of writing, `status.json` reports `waiting_for_v2`: the separate worker waits for both the existing v2 workflow and post-training audit to complete before using the GPU. It then rechecks ZIP, snapshot, holdout, selected baseline, code, config and weights hashes. Training is **not yet complete**, so pilot Macro F1, data-wait and throughput results do not exist yet. Once complete, per-epoch data-wait, wall time, optimizer steps, train sample count, and validation type/style macro F1 will be in `history.json`; checkpoints and their provenance remain separate from v2. Earlier `zip-pilot-c6-v1` and `zip-pilot-c6-v2` workers were stopped while waiting to add post-wait hash revalidation and failure events; their folders are preserved and do not represent completed runs. No v2 process was stopped.

## Commands

With the project `.venv` active and sufficient working space, a fresh snapshot and independent verification can be reproduced without extracting ZIP images:

```powershell
$env:FURNITURE_ARCHIVE = 'D:\Download\furniture_dataset.zip'
$env:TEMP = (Resolve-Path '.pip-temp').Path
$env:TMP = $env:TEMP
$env:OMP_NUM_THREADS = '1'
.venv\Scripts\python.exe -m new_data.build_conservative_snapshot --config new_data\snapshot-config.example.json --output artifacts\zip-conservative-snapshot-next
.venv\Scripts\python.exe -m new_data.verify_snapshot --config new_data\snapshot-config.example.json --snapshot artifacts\zip-conservative-snapshot-next --output verification-v2.json
```

To reproduce the four-image smoke against the recorded snapshot:

```powershell
.venv\Scripts\python.exe -m new_data.pilot_zip --config new_data\pilot-config.json --output artifacts\zip-pilot-smoke-next --mode smoke
```

The existing pilot process was started with `--mode pilot --wait-for-v2 --workflow-pid 30880 --post-pid 34868`. Inspect `status.json` and process/lock state before restarting. If interrupted, use the same config and output with `--resume`; provenance must match, and it resumes the last complete epoch when a checkpoint exists. Do not manually change the status or start a duplicate worker. The source, selected checkpoint and weights are checked again before and during training; a Python exception records a `failure-*.json` event and failed status under the run lock.
