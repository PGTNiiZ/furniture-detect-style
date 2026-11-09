# ZIP DataLoader inspection — 2026-11-04

This inspection uses `D:\Download\furniture_dataset.zip` directly, without extracting image files, and does not change the existing v2 suite. It does not approve this ZIP's training split for research training.

## Results

- Archive SHA-256: `dce23f8f54b1317d31805564a4d58ab0eff2ed3ce00e06656df188440563f79a`. It agrees with the locally CRC-verified source in `artifacts/small-d-zip-crc-20261104.json`.
- ZIP train manifest: 7,650 rows. All 7,650 images decoded; `errors.csv` contains zero error rows. Each row's type/style, stored class IDs, and folder names agreed with the ZIP's CSV and `labels.json`. This is structural consistency, **not independent human verification that the depicted furniture has the right semantic labels**.
- The actual loader read 7,650 images in 479 batches, batch size 16, two Windows workers, CPU, `shuffle=false`, `drop_last=false`; last batch contained two images. Every returned label tensor matched its corresponding manifest row. Elapsed loader time was 50.05 seconds, with OS cache uncontrolled and the v2 workflow running; this is not a cold-cache benchmark or training throughput measurement.
- The ZIP label mapping has 4 type and 34 style classes; v2 has its own mapping. Do not substitute one mapping for the other or infer that label canonicalization was approved.
- No model was trained by this inspection. The existing v2 workflow continued independently. Its prior eight-image CPU optimizer-step smoke is documented separately in `docs/17-data-audit-loader-20261104.md`.

Output files in `artifacts/zip-batch-audit-20261104/`: `audit.json`, `errors.csv`, `train-manifest.csv`, `mapping.json`, `loader-config.json`, `holdout-dhash-candidates.csv` and four `real-source-candidates-*.png` contact sheets. The derived manifest is for inspection only; it is not a new audited snapshot.

## Holdout comparison

For each decoded ZIP train image, the script computes dHash and compares it to every recorded v2 validation, synthetic-test and real-source-test dHash. With distance ≤4, it found 1,108 validation pairs (1,100 distinct incoming IDs), 966 synthetic-test pairs (958 incoming IDs) and 19 real-source-test pairs (16 incoming IDs). dHash is a candidate detector, not an identity decision, and resized/recompressed images can alter its value. The CSV records both IDs, paths, split, distance and `unreviewed` decisions. Four contact sheets show the 19 real-source candidates; review them independently before deciding same/related/different/uncertain.

Separately, the prior pixel comparison matched all 1,098 same-source-index validation candidates and 954 synthetic-test candidates closely after resize/JPEG conversion. Those findings already establish substantial apparent overlap; they are not 2,052 additional pairs to add to the dHash totals. The source ZIP did not pin a revision, and the contact-sheet candidates require individual review. Do not train using this manifest against the existing v2 holdouts without a documented group-level leakage resolution and snapshot audit. See `docs/17-data-audit-loader-20261104.md` and `artifacts/small-v2-holdout-image-comparisons.csv`.

## Reproduce on Windows

The config is `new_data/audit-config.example.json`; archive path comes from an environment variable, so it can point to another verified drive without source edits. Its expected SHA is deliberately pinned to this ZIP. Choose a new output directory for every run.

```powershell
$env:FURNITURE_ARCHIVE = 'D:\Download\furniture_dataset.zip'
$env:TEMP = (Resolve-Path '.pip-temp').Path
$env:TMP = $env:TEMP
$env:OMP_NUM_THREADS = '1'
.venv\Scripts\python.exe -m new_data.audit_zip_batches --config new_data\audit-config.example.json --output artifacts\zip-batch-audit-next
```

The command checks the archive SHA, inspects all ZIP train rows and images, writes errors and holdout candidates, then runs DataLoader only when inspection finds no error. The experimental adapter is `new_data/zip_dataset.py`. It opens a ZIP handle per worker and fails visibly on corrupt or missing images. It is separate from the v2 training pipeline and has no disk cache. `python -m pytest -q new_data/test_audit_zip_batches.py new_data/test_zip_dataset.py --basetemp .pip-temp\zip-audit-test-1104` passed six tests, including corrupted image and mismatched-label reporting, holdout candidates, worker isolation and shuffled batches. These are separate from the frozen v2 suite's prior test binding.
