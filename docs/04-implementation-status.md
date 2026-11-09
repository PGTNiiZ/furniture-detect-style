# สถานะและ checklist สำหรับงาน Furniture Multi-task

สถานะนี้แยก implementation กับหลักฐานการทดลอง: การผ่าน unit test ไม่เท่ากับฝึกครบหรือพิสูจน์ความแม่นยำ ข้อมูลสังเคราะห์ของผู้เผยแพร่เป็นข้อมูลจากแหล่งจริง แต่ไม่ใช่ภาพถ่ายจริง

| ข้อกำหนด | ก่อนแก้ / สิ่งที่เปลี่ยน | หลักฐานหรือวิธีตรวจ | ข้อจำกัด |
|---|---|---|---|
| P0 augmentation | เดิม `fit` ส่ง boolean แทน dict; แก้ให้ส่ง config จริงเข้า train | `tests/test_augmentation_pipeline.py`, `src/furniture/train.py` | ผล run เก่าที่ไม่มี marker ห้ามใช้สรุป augmentation |
| Presets และ preview | เดิม preset hardcode; controlled ใช้ Albumentations 2.0.8 และ YAML แยก parameters/occlusion | `configs/augmentation`, `docs/05-augmentation.md`, `preview-occlusion` | E1–E3 ใช้ torchvision legacy; อย่าเปรียบเป็น backend เดียวกัน |
| Dataset | ตรึง revision; อ่าน 10,000 source rows และตรวจ dHash ครบทุกคู่ของ source train | `docs/05-dataset-source-evidence.md`, `docs/evidence` | 1 แถวขาด labels; source test มี labels ที่ train ไม่รู้จัก; train/val มี 240 candidate pairs รอตรวจด้วยคน |
| Models | 10 torchvision backbones, enum weights ชัดเจน, image size และ checkpoint metadata | `model.py`, `tests/test_backbones.py`, `docs/evidence/pretrained-verification-10.json` | โหลด pretrained และ forward ผ่านทั้ง 10 ตัว แต่ยังไม่ได้ฝึก screening |
| Multi/single task | เลือก type/style ผ่าน training.tasks; metrics และ exports เลือกตาม heads | `train.py`, `tests/test_pipeline.py` | room inference ต้องมีสอง heads |
| Freeze/optimization | BN freeze, แยก LR, accumulation และ CUDA AMP | `train.py`, `tests/test_training_controls.py` | BatchNorm ของ microbatch ไม่เท่าการใช้ physical batch ใหญ่ |
| Resume/transfer | last checkpoint เก็บ model/optimizer/scheduler/scaler และ seed ต่อ epoch; warm-start แยกต่างหาก | `resume` CLI, `tests/test_pipeline.py` | Resume เฉพาะ checkpoint ใหม่จาก epoch ที่จบและ code/config/data/version เดิม; GPU บาง operations อาจไม่ bitwise deterministic |
| Suites | Screening 10, controlled C1–C8, ablation และ repeated seeds | `suite.py`, `tests/test_suite_planning.py`; `suite --dry-run` | การเลือก top 3 ใช้ validation; ยังไม่ได้ฝึก 44 runs |
| Run tracking | สถานะ running/completed/failed/interrupted, OS lock, owner และ source/config provenance; checkpoint เขียนแบบ atomic | `pipeline.py`, `docs/07-run-recovery.md`, `tests/test_recovery.py` | run เก่าที่ไม่มี owner และกรณีที่ตรวจ process ไม่ได้ fail closed; การกู้ต้องสั่ง explicit |
| Clean/robustness | clean metrics และ fixed occlusion manifest | `train.py`, `robustness.py`, `docs/05-robustness.md` | ยังไม่ประเมิน room-domain ที่มี ground truth; ไม่มี object masks สำหรับวัดพื้นที่บังวัตถุ |
| Benchmark | วัด forward ของ synthetic tensors พร้อม warmup/sync และ hardware | `benchmark` CLI, `artifacts/source-preview-smoke-20261102/benchmark-v2.json` | ยังไม่ใช่ประมาณการฝึกครบ suite จากข้อมูลเต็ม |
| Detector/recommendation | แยก detector/classifier/heuristics; มี evaluator สำหรับ reference/detector crops และ detection misses | `room_evaluation.py`, `docs/06-room-domain-evaluation.md` | ยังไม่มี annotations สำหรับประเมินภาพห้องจริงหรือ human evaluation ของคำแนะนำ |
| Reports / rubric | comparison, CSV, training curves และ confusion matrices แยกตาม split | `compare`, `compare-robustness`, `report` | กราฟเปรียบเทียบทุกแบบในแผนยังไม่ครบ; ยืนยันขอบเขตหัวข้อกับอาจารย์ |

## แก้ก่อน screening รอบล่าสุด

`compare-robustness --vary backbone` อนุญาต weights ต่างกันตาม backbone แต่ตรวจ controls, data/split hashes และ evidence ที่ผูกกับ best checkpoint; ผล robustness รุ่นเก่าที่ไม่มี evidence v3 ต้องประเมิน validation ใหม่ การประเมิน robustness ใช้ validation เป็น default ไม่เปิด test โดยปริยาย

`suite --stage screening --manifest PATH` บันทึก manifest v2 และ hash ของ split ทั้งสามก่อนฝึก พร้อมตรวจซ้ำระหว่าง suite; ใช้ `suite --continue-manifest PATH --retry-failed` ต่อเฉพาะงานที่ยังไม่สำเร็จ ใช้ `--recover-running` เมื่อต้องกู้ process ที่ตายแล้ว และ `--restart-failed` เฉพาะเมื่อตั้งใจเริ่ม attempt ใหม่โดยรักษาไฟล์เก่า ดู `docs/07-suite-recovery.md` และ `docs/07-run-recovery.md` ผล completed ที่ตรวจหลักฐานผ่านจะไม่ถูกฝึกซ้ำ

ยังไม่เริ่ม screening: source test มี labels ที่ train ไม่รู้จัก และยังต้องตรวจ candidate pairs ระหว่าง train/validation ก่อนตรึง dataset protocol ผลจากชุด smoke และการตรวจ pretrained ไม่ใช่ผลวิจัย ดู `docs/05-dataset-source-evidence.md`

## คำสั่ง Windows PowerShell

```powershell
.venv\Scripts\python.exe -m pip install -e ".[data,dev]"
.venv\Scripts\python.exe -m furniture.cli check-device
.venv\Scripts\python.exe -m furniture.cli check-config --config configs/project_controlled.yaml --experiment C5
.venv\Scripts\python.exe -m furniture.cli suite --stage screening --dry-run
.venv\Scripts\python.exe -m furniture.cli suite --stage controlled --backbones resnet18 resnet50 densenet121 --dry-run
.venv\Scripts\python.exe -m furniture.cli suite --stage ablation --backbones resnet18 --dry-run
.venv\Scripts\python.exe -m furniture.cli suite --stage repeated --backbones resnet18 --conditions C1 C2 C3 C5 --seeds 123 2026 --dry-run
```

Dry-run ตรวจเฉพาะ config และสร้างรายการ output ที่วางแผน ไม่ดาวน์โหลดหรือตรวจข้อมูล/weights เมื่อเตรียม snapshot ที่ audit แล้วจึงเอา `--dry-run` ออกเพื่อฝึกจริง ต้องเลือก backbone จาก validation ก่อนสั่ง controlled/ablation/repeated; `select-screening --runs RUN1 RUN2 RUN3 ...` ใช้เฉพาะ run ที่ completed และมี protocol หลังแก้ P0

เตรียมและตรวจข้อมูลก่อนฝึกด้วย `prepare-data --config configs/project_controlled.yaml` และ `audit-data --config configs/project_controlled.yaml` ตรวจ candidate และ group ด้วยคน ห้ามถือ source test ที่ถูกใช้ฝึกใน snapshot รุ่นเก่าเป็น untouched test ดู `docs/03-new-data-and-augmentation.md` และ `docs/05-dataset-source-evidence.md`

เมื่อ run ถูกขัดจังหวะที่มี `last.pt`, ใช้ `resume --run-dir RUN_DIR` โดยต้องไม่มีการเปลี่ยน source/config/split หรือ PyTorch version และ resume ได้ที่ขอบ epoch; การตั้ง `initialization: checkpoint` เป็น warm-start optimizer ใหม่ ไม่ใช่ resume

## ภาพรวมการออกแบบ

ภาพ → shared backbone → type/style heads → weighted cross entropy ของ tasks ที่เลือก; detector ใช้แยกเพื่อหา crops และ recommendation เป็น heuristic ไม่ใช่หลักฐานว่า model แบบ multi-task ดีกว่า single-task หรือคำแนะนำสวยงาม pretrained recipes ของ torchvision บางรุ่นใช้ resize/crop ต่างกัน งานนี้ใช้ RGB + aspect-preserving resize/pad สีขาว + ImageNet normalization เดียวกันเป็น controlled protocol จึงไม่อ้างว่าเป็น official recipe ของทุก weights

Output `resolved_config.yaml`, `status.json`, `source/`, `provenance.json`, `dataset_metadata.json`, `environment.json`, `history.json`, `best.pt`, `last.pt`, `validation.json` และ predictions อธิบายว่าทำอะไรจริง ผล test จะมีเฉพาะเมื่อสั่งประเมินแยกหลังตรึง protocol เท่านั้น
