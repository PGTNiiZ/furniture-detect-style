# V2 suite and large-source intake status — 2026-11-04

> **อัปเดต 2026-11-09:** รายงานนี้ถูกต้อง ณ 2026-11-04 ปัจจุบัน repeated seeds เสร็จ 8/8, post-training audit, robustness, synthetic/real-source test และ ZIP ส่งมอบสุดท้ายเสร็จแล้ว (`artifacts/model-only-release/final-delivery.json`) และ ZIP pilot เสร็จ (`artifacts/zip-pilot-c6-v3/status.json`, mean macro-F1 validation 0.588) ส่วนปัญหา `furniture_sources.zip` ขนาดใหญ่ยังไม่ได้รับการแก้


This is a point-in-time report. The suite, post-training audit, finalizer and ZIP pilot are separate processes; consult their status files for the latest state.

| Work | Observed status | Evidence / limitation |
| --- | --- | --- |
| V2 screening | 10/10 complete | `artifacts/runs/suites/furniture-v2-screening.json` |
| Controlled C1–C8 | 24/24 complete | `artifacts/runs/suites/furniture-v2-model-only-controlled.json` |
| Single-task baselines | 2/2 complete | `artifacts/runs/suites/furniture-v2-model-only-ablation.json` |
| Repeated seeds | 7/8 complete, one running | Last run: `repeated-densenet121-C5-2026-type-style`, at epoch 13 when last checked. Repeated-seed summary awaits its completion. |
| Selection | Frozen from validation | DenseNet121/C6, epoch 30, checkpoint SHA-256 `24e413586f98537fff9f56eb66c71a10c30f11154d3eaa58904ce5e50dc78328`. |
| Prediction audit, robustness, two test reports, final package | Pending | Existing post-training/finalizer workers are waiting for the suite. Do not report them as completed. |
| ZIP pilot | Waiting for V2 and post-training | One three-epoch integration pilot remains queued in `artifacts/zip-pilot-c6-v3/status.json`. No pilot validation score or throughput exists yet. |
| Large ZIP local access | Not found in checked locations | No complete `furniture_sources.zip` found in searched D:, E:, or Downloads paths. One unrelated/unidentified partial `.crdownload` was preserved. |
| Large ZIP source access | Unavailable during read-only check | A range request for the previously supplied Drive file returned HTTP 404. No current ZIP bytes were read. |
| New independent images eligible for training | Unknown | The large ZIP, annotations and holdout overlap have not been inspected locally. Do not equate unknown with zero or with the 633,752 image-extension entries in the old inventory. |

The selected checkpoint at `artifacts/runs/furniture-v2-conservative/controlled-densenet121-C6-42-type-style/20261103T003152Z-98889263/best.pt` was rehashed and matched the frozen selection. Its saved **validation** metrics are:

| Task | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Type | 0.995367 | 0.995424 |
| Style | 0.227027 | 0.239979 |

The source values are recorded in `artifacts/v2-selected-validation-20261104.json`. These are not synthetic-test or real-source-test scores. The selected-checkpoint room demo is also complete, but remains an inference demonstration, not a quantitative room evaluation.

The completed DenseNet121/C3 runs also contain the following saved validation metrics. The checkpoints' checksums were verified; independent prediction audit is still pending. Single-task checkpoints use their task's selection metric, while the joint run uses mean macro F1, so this table alone does not establish a statistically significant benefit.

| C3 run | Type accuracy / Macro F1 | Style accuracy / Macro F1 |
| --- | ---: | ---: |
| Type-only | 0.996911 / 0.996935 | — |
| Style-only | — | 0.220077 / 0.223050 |
| Joint type + style | 0.996139 / 0.996126 | 0.230116 / 0.237118 |

Values, epoch numbers and checkpoint hashes are in `artifacts/v2-c3-single-task-validation-20261104.json`.

The selected backbone's completed C1–C8 clean-validation results, including each training augmentation/occlusion preset and verified checkpoint hash, are in `artifacts/v2-densenet-c1-c8-validation-20261104.csv`. They are not the planned clean/light/strong validation robustness evaluation, which still awaits the post-training workflow.

The combined `new_data` tests passed **12 tests** using `python -m pytest -q new_data --basetemp .pip-temp/source-inventory-and-label-tests-1104`. They include a regression case where a ZIP folder is named `nordic` but its manifest style is `scandinavian`: the loader returns the manifest label through the snapshot mapping. This test run is separate from the frozen v2 release's existing test binding.

A subsequent combined run, `python -m pytest -q tests new_data`, passed **166 tests**, with 65 warnings, in 165.40 seconds. The hashes of 144 code/config files matched before and after the run. See `docs/evidence/v2-and-new-tests-20261104.txt` and `docs/evidence/v2-and-new-tests-binding-20261104.json`. This is additional combined evidence; it does not replace or modify the earlier v2 finalizer binding.

## Large ZIP: prior cached directory only

The previously captured ZIP central directory had 638,840 outer file entries, of which 633,752 had image extensions. It included nested archives and duplicate directory representations, and is not a current local archive or a count of unique trainable images. All 50 recorded three-component prefixes together account for all 638,840 outer entries; `artifacts/large-sources-cached-inventory.csv` aggregates them into eight sources. Its file-entry counts are reproducible from `artifacts/drive-archive-inventory.jsonl` using `artifacts/build_cached_large_inventory.py`. Source-specific image counts, distinct image counts and annotation coverage cannot be reconstructed from the saved aggregate.

| Source | Outer file entries | Evidence currently available |
| --- | ---: | --- |
| Amazon-Berkeley-Objects | 398,234 | Image tree, compressed listings, and CC-BY-4.0 text; listing schema and per-item terms not checked. |
| iMaterialist-Furniture-2018 | 145,938 | Train/validation/test paths; annotations and label meanings not checked. |
| DeepFurniture | 49,514 | Category and metadata JSON files; bundled dataset info contains placeholder license/citation. |
| Pix3D | 22,446 | Data plus nested ZIPs; annotation coverage and duplicate copies not checked. |
| IKEA-Dataset | 11,754 | Room-named directories plus nested ZIPs; labels and terms not established. |
| Darebal_furniture-catalog | 5,558 | Category-like folder names, not yet validated as annotations. |
| HomeObjects-3K | 5,381 | YOLO-format label examples and YAML mapping, including bed/sofa/chair/table. YAML states 2,285 train and 404 validation images; file counts and box quality remain unverified. No style labels established. |
| Furniture-Synthetic-Dataset | 15 | Parquet files and README for the same named source as v2; exact content/revision not verified. |

`artifacts/drive-source-annotations.json` preserves the selected metadata retrieved in the earlier read-only inspection. A source is not approved for detector training, category-only training, or joint type/style training until its actual records, licenses, labels, groups and overlap with v2 splits are audited. In particular no number of new images per type × style can yet be reported.

The search for extracted HomeObjects/Pix3D/ABO artifacts found only `HomeObjects-3K.yaml` shipped inside the isolated Ultralytics environment, not an extracted room-image dataset. It is a library example configuration, not evidence that the dataset is locally available.

`new_data/masked-loss-proposal.yaml` is a non-runnable proposal for a separate category-only experiment if that kind of data is actually found. It requires task-specific valid-label normalization, a skipped task when its batch has no labels, and no fabricated unknown style class. It has not changed the current trainer or experiment protocol.

## Next concrete input

Provide a complete `furniture_sources.zip` on a reliably readable drive, plus a trusted SHA-256 if available, or restore access to the supplied Drive file. The historical ZIP byte size was 42,671,832,118 bytes. On the last space check D: had about 27.6 GB free, insufficient for the ZIP alone; E: was exFAT with about 122 GB free but its volume health still read `Warning` and a previous read-back failed. Do not use the failed E: copy as evidence of a valid source. A trustworthy destination needs the ZIP size plus temporary/cache reserve; extracting all outer files would require an additional roughly 44.7 GB and nested archives may need more.

When a complete local ZIP is supplied, the read-only inventory below computes the archive SHA-256, verifies the expected size and every member's CRC, and counts file/image-extension/annotation-extension entries by source without extracting images. A supplied trusted SHA verifies identity; without one, the computed SHA and CRC prove neither upstream identity nor licensing. The tool does not count independent images or validate annotation semantics. Subsequent source-specific image/annotation and cross-split duplicate audits are still required.

```powershell
$env:TEMP = (Resolve-Path '.pip-temp').Path
$env:TMP = $env:TEMP
.venv\Scripts\python.exe -m new_data.inventory_local_sources --archive '<reliable-drive>:\furniture-data\archives\furniture_sources.zip' --expected-size 42671832118 --expected-sha256 '<trusted-sha256>' --output artifacts\large-sources-local-inventory
```

If there is no trusted checksum, omit `--expected-sha256` and record that absence. Keep the source ZIP unchanged; do not use `--skip-crc` when claiming all members were verified.
