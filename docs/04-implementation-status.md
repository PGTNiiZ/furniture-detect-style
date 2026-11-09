# สถานะและ checklist สำหรับงาน Furniture Multi-task

**อัปเดตล่าสุด 2026-11-09** จากไฟล์สถานะใน `artifacts/` (ฉบับก่อนหน้าเขียนเมื่อ 2026-11-02 ก่อนเริ่มเทรน จึงล้าสมัย) การเทรนจริงเสร็จครบแล้ว รายละเอียดผลดู [RELEASE_STATUS.md](../RELEASE_STATUS.md) และ [docs/22](22-training-and-model-guide.md) งานที่ยังค้างคืองานที่ต้องใช้คน ดู [docs/21](21-human-tasks-handoff.md)

สถานะนี้แยก implementation กับหลักฐานการทดลอง: การผ่าน unit test ไม่เท่ากับพิสูจน์ความแม่นยำ ข้อมูลสังเคราะห์ของผู้เผยแพร่ไม่ใช่ภาพถ่ายจริง และ label ของภาพจริงมาจาก AI ไม่ใช่ผู้เชี่ยวชาญ

## ภาพรวมการทดลอง

| ขั้น | สถานะ | หลักฐาน |
|---|---|---|
| Screening 10 backbones (C3, seed 42) | เสร็จ 10/10 | `artifacts/runs/suites/furniture-v2-screening.json` |
| Controlled C1–C8 (convnext_tiny, resnet50, densenet121) | เสร็จ 24/24 | `furniture-v2-model-only-controlled.json` |
| Single-task baseline (DenseNet121) | เสร็จ 2/2 | `furniture-v2-model-only-ablation.json` |
| Repeated seeds (DenseNet121 C1, C2, C3, C5) | เสร็จ 8/8 | `furniture-v2-model-only-repeated.json` |
| เลือกโมเดลจาก validation | ตรึงแล้ว: DenseNet121 + C6 | `artifacts/model-only-release/selection-frozen.json` |
| Robustness, synthetic test, real-source test | เสร็จ รายงานแยกกัน | `artifacts/model-only-release/` |
| ZIP pilot 3 epoch | เสร็จ (ไม่ใช่ข้อมูลใหม่ ซ้ำกับ v2 ทั้งหมด) | `artifacts/zip-pilot-c6-v3/status.json` |
| Room end-to-end evaluation | **ยังไม่ทำ** ไม่มี annotation อิสระ | `data/room-independent-review-v1/` |

เทสต์: `python -m pytest -q tests new_data` ผ่าน **166 passed** (65 warnings, 179.5 วินาที) รันเมื่อ 2026-11-09 บน Windows + `.venv` ของเครื่องเจ้าของ repo ยังไม่เคยรันบน Linux/CI

ผลสำคัญ (macro-F1): validation type 0.995 / style 0.240, synthetic test type 0.996 / style 0.200, real-source test type 0.831 / style 0.028 สไตล์ยังต่ำมาก ต้องรายงานตรง ๆ

## Implementation checklist

| ข้อกำหนด | สถานะ | หลักฐาน / ข้อจำกัด |
|---|---|---|
| P0 augmentation (เดิมส่ง boolean แทน dict) | แก้แล้ว | `tests/test_augmentation_pipeline.py`; run เก่าที่ไม่มี marker ห้ามใช้สรุป |
| Augmentation presets และ occlusion | เสร็จ | `configs/augmentation`, `docs/05-augmentation.md`; E1–E3 ใช้ torchvision legacy ห้ามเทียบกับ Albumentations |
| Dataset | ใช้ snapshot `data/furniture-v2-conservative` | `docs/10`, `docs/11`; ภาพ 622 ภาพถูก quarantine แบบเหมารวม คู่ภาพซ้ำ 494 คู่ยังไม่มีคนตัดสิน |
| Models | 10 torchvision backbones | `tests/test_backbones.py`, `docs/evidence/pretrained-verification-10.json` |
| Multi/single task | เสร็จ | `train.py`, `tests/test_pipeline.py` |
| Freeze/optimization | BN freeze, แยก LR, accumulation, AMP | `tests/test_training_controls.py` |
| Resume / transfer | เสร็จ | `resume` CLI; resume ได้เฉพาะ checkpoint ที่จบ epoch และ code/config/data เดิม |
| Suites | เสร็จ รันจริงครบ | `suite.py`, `docs/07-suite-recovery.md` |
| Run tracking / recovery | เสร็จ | `docs/07-run-recovery.md`, `tests/test_recovery.py` |
| Clean / robustness | เสร็จ clean/light/strong บน validation | `artifacts/model-only-release/robustness-report/` ไม่มี object mask |
| Benchmark | มี benchmark inference และ training ระยะสั้น | ไม่ใช่ค่าประมาณเวลาเทรนทั้ง suite |
| Detector / recommendation | baseline YOLOv8n (COCO) ไม่ได้เทรนเอง; recommendation เป็น heuristic | `docs/14-room-evaluation-protocol.md` ไม่มีการประเมินเชิงปริมาณ |
| Reports | validation, synthetic test, confusion matrix, error examples | กราฟเปรียบเทียบบางส่วนในแผนเดิมยังไม่ครบ |
| ขอบเขตหัวข้อกับอาจารย์ | **ยังไม่ยืนยัน** | ดูงาน F ใน docs/21 |

## คำสั่ง Windows PowerShell

```powershell
.venv\Scripts\python.exe -m pip install -e ".[data,dev]"
.venv\Scripts\python.exe -m furniture.cli check-device
.venv\Scripts\python.exe -m furniture.cli check-config --config configs/project_v2_conservative.yaml --experiment C6
.venv\Scripts\python.exe -m furniture.cli suite --stage screening --config configs/project_v2_conservative.yaml --dry-run
```

อย่ารัน `suite` โดยไม่ใส่ `--dry-run` กับ output เดิม เพราะ run ของ v2 ถูกตรึงไว้แล้ว ถ้าจะเทรนเพิ่มให้ใช้ `train` กับ `--run-dir` ใหม่ตาม [docs/22](22-training-and-model-guide.md) การ resume ใช้ `resume --run-dir RUN_DIR` เมื่อ run ถูกขัดจังหวะและไม่มีการเปลี่ยน source/config/split/PyTorch version การตั้ง `initialization: checkpoint` เป็น warm-start (optimizer ใหม่) ไม่ใช่ resume

## ภาพรวมการออกแบบ

ภาพ → shared backbone → type/style heads → weighted cross entropy; detector ใช้แยกเพื่อหา crops และ recommendation เป็น heuristic ไม่ใช่หลักฐานว่า multi-task ดีกว่า single-task หรือคำแนะนำสวยงาม งานนี้ใช้ RGB + aspect-preserving resize/pad สีขาว + ImageNet normalization เดียวกันทุก backbone เป็น controlled protocol จึงไม่อ้างว่าเป็นสูตรทางการของทุก weights

Output ของแต่ละ run: `resolved_config.yaml`, `status.json`, `source/`, `provenance.json`, `dataset_metadata.json`, `environment.json`, `history.json`, `best.pt`, `last.pt`, `validation.json` และ predictions ผล test จะมีเฉพาะเมื่อสั่งประเมินแยกหลังตรึง protocol เท่านั้น
