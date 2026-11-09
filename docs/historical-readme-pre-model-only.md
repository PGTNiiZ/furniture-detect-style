# Project 2 — Furniture Detection & Target Style Recommendation

สถานะการปรับระบบทดลองและข้อจำกัดปัจจุบันดูที่ [docs/04-implementation-status.md](docs/04-implementation-status.md) ก่อนเริ่มชุดทดลองใหม่

เวอร์ชัน 0.2: แยก config สำหรับข้อมูล โมเดล การเทรน augmentation และการทดลอง เพื่อแก้ไขและฝึกกับข้อมูลชุดใหม่ได้

เป้าหมาย: ภาพห้อง + สไตล์ที่เลือก → ตรวจเฟอร์นิเจอร์ → วิเคราะห์ประเภทและสไตล์รายชิ้น → เสนอเก็บ/ตรวจเพิ่ม/พิจารณาเปลี่ยน → เลือกชิ้นทดแทนประเภทเดียวกัน

สถานะ: โค้ดและการทดสอบ pipeline พร้อมใช้งานกับข้อมูลที่เตรียมตามรูปแบบ; ยังไม่มีผลเทรนเฟอร์นิเจอร์จริง และยังไม่ได้วัดประสิทธิภาพบนภาพห้องจริง

## เปิดอ่านตามลำดับ
1. README นี้: ติดตั้งและเริ่มรัน
2. [คู่มือ config และแต่ละขั้น](docs/02-config-and-workflow.md): แก้ไฟล์ไหนเมื่ออยากเปลี่ยนอะไร
3. [แผนตาม rubric](docs/01-plan.md): การทดลอง หลักฐาน คะแนน และกำหนดส่ง 14 พฤศจิกายน 2569
4. [คู่มือเพิ่มข้อมูล](docs/03-new-data-and-augmentation.md): CSV, version, group, checkpoint และ augmentation
5. `notebooks/01_pipeline_walkthrough.ipynb`: เรียกคำสั่งทีละส่วนจาก notebook (ต้องใช้ environment ที่ติดตั้งแล้ว)

## โครงสร้าง
| ตำแหน่ง | หน้าที่ |
|---|---|
| configs/project.yaml | จุดรวม เลือกว่าจะอ่าน config ย่อยไฟล์ไหน |
| configs/datasets/ | แหล่งข้อมูล paths ชื่อคอลัมน์ group และ split |
| configs/models/ | backbone, initialization, image size, dropout |
| configs/training/ | epochs, batch, learning rate, loss weights, metric |
| configs/augmentation/ | การเพิ่มความหลากหลายของภาพฝึก |
| configs/experiments/ | E1/E2/E3; ค่าที่เปลี่ยนจาก baseline |
| configs/inference/ | detector, thresholds และจำนวนคำแนะนำ |
| src/furniture/ | โมดูลที่แต่ละคำสั่งเรียก |
| tests/ | ตรวจ data leakage, config, augmentation และ training pipeline |
| docs/, notebooks/ | คำอธิบายกับตัวเรียก pipeline |
| data/ | ข้อมูลในเครื่อง ไม่รวมใน ZIP/Git |
| artifacts/ | ผลแต่ละ run แยก dataset/experiment/time ไม่รวมใน ZIP/Git |

## ติดตั้ง
Python 3.10+; เปิด terminal ในโฟลเดอร์ furniture-project2
```bash
python -m venv .venv
```
Windows PowerShell: `.venv\Scripts\Activate.ps1`
macOS/Linux: `source .venv/bin/activate`

ติดตั้ง PyTorch และ torchvision ที่รองรับ GPU/CUDA ตาม https://pytorch.org/get-started/locally/ ก่อน แล้ว:
```bash
python -m pip install -e ".[data,detection,dev,reports]"
python -m pytest -q
```
ตรวจอุปกรณ์ก่อนเทรนด้วย `python -m furniture.cli check-device` โดย `device: auto` ใน `configs/training/default.yaml` และ `configs/inference/room_detector.yaml` เลือก CUDA เมื่อ PyTorch ตรวจพบ มิฉะนั้นใช้ CPU; กำหนด `device: cpu` เพื่อบังคับ CPU ได้ และการประเมินใช้ auto โดยปริยาย (`--device cpu` เพื่อบังคับ CPU) การตรวจนี้ไม่ยืนยันว่า wheel รองรับ GPU ทุกรุ่น ควรทดลองรันจริงด้วย

CPU ใช้ได้กับ smoke; ถ้าใช้ RTX 5070 Ti ตรวจ `torch.cuda.is_available()` และ GPU architecture support ของ wheel จริงก่อนเทรนเต็ม ไม่รับประกัน CUDA จาก smoke CPU

## ทดลองระบบโดยไม่ดาวน์โหลดโมเดลหรือ dataset
```bash
python -m furniture.cli make-smoke-data --output data/smoke
python -m furniture.cli check-config --config configs/project_smoke.yaml
python -m furniture.cli train --config configs/project_smoke.yaml
```
คำสั่งสุดท้ายพิมพ์ `run_dir` ที่บันทึกโมเดล ใช้ path จริงนั้นแทน RUN_DIR:
```bash
python -m furniture.cli evaluate-run --run-dir RUN_DIR --device cpu
```
ภาพสีสังเคราะห์และคะแนน smoke มีไว้ตรวจโค้ด ห้ามอ้างเป็นผลของโครงการ
ถ้าสร้าง `data/smoke` แล้ว ไม่ต้องสร้างซ้ำ; ทุกคำสั่ง train สร้าง run ใหม่ให้เอง

## ใช้ข้อมูล Hugging Face
`configs/datasets/furniture_hf.yaml` ตรึง commit SHA ที่ตรวจพบไว้แล้ว ตรวจแหล่งข้อมูลและ license ตาม `docs/05-dataset-source-evidence.md` ก่อนใช้ และกำหนด SHA ใหม่เฉพาะเมื่อสร้าง dataset version ใหม่
หา SHA จากหน้า Files and versions ของ https://huggingface.co/datasets/filnow/furniture-synthetic-dataset
ไม่ใช้ชื่อ branch `main` แทน SHA เพราะข้อมูลอาจเปลี่ยนภายหลัง
```bash
python -m furniture.cli prepare-data --config configs/project.yaml
python -m furniture.cli audit-data --config configs/project.yaml
```
ตรวจ audit.json, จำนวนคลาส, label-image agreement และ near-duplicate/source groups ก่อนใช้ผลเป็นหลักฐาน
prepare ทำการแบ่งชุดใหม่จาก group ประมาณ 70/15/15 ไม่ใช่รับประกัน split เดิมของผู้เผยแพร่
โปรแกรมจัด exact pixel duplicates กับ group_id ให้อยู่ split เดียว; การหา near-duplicates ยังต้องตรวจเพิ่มและกรอก group_id

## เริ่มการทดลองตามเกณฑ์อาจารย์
```bash
python -m furniture.cli train --config configs/project.yaml --experiment E2
```
รันสามรูปแบบต่อกัน:
```bash
python -m furniture.cli experiments --config configs/project.yaml --ids E1 E2 E3
```
E1: frozen backbone + สอง heads, E2: fine-tune ร่วมกัน, E3: E2 + horizontal flip
ชุดนี้ใช้ dataset, seed, image size, batch, maximum epochs, learning rates และ selection metric เดียวกัน; freeze/augmentation เป็นตัวแปรทดลอง
Early stopping อาจทำให้ actual epochs ต่างกัน ให้รายงาน best epoch และ actual epochs ประกอบ
suite สรุป validation เป็น CSV ให้อัตโนมัติ ไม่รัน test เพื่อเลือกผู้ชนะ
ใช้ `evaluate-run --run-dir RUN_DIR` กับโมเดลที่เลือกหลังตรึง protocol; ถ้ารายงาน test ของทั้งสาม ต้องตรึงวิธี/เกณฑ์ทั้งหมดก่อนเปิด test และห้ามนำ test มาจูน

## ใช้ข้อมูลใหม่จาก CSV
1. ดู header ใน `examples/labels_template.csv`
2. วางภาพและ labels จริง เช่น data/incoming/images กับ data/incoming/labels.csv
3. แก้ `configs/datasets/local_csv.yaml` ให้ชื่อคอลัมน์และ paths ตรงข้อมูล
4. เปลี่ยน id/root เป็นเวอร์ชันใหม่ทุกครั้งที่ข้อมูลเปลี่ยน
```bash
python -m furniture.cli prepare-data --config configs/project_new_data.yaml
python -m furniture.cli audit-data --config configs/project_new_data.yaml
python -m furniture.cli train --config configs/project_new_data.yaml --experiment E2
```
จำนวนคลาสสร้างจาก labels ใน train อัตโนมัติ; type/style เป็นสองงานเดิมที่ต้องมี labels
เมื่อเพิ่มข้อมูลจาก snapshot เดิม ให้กำหนด `split.previous_root` เพื่อรักษาชุด train/val/test เดิม; โปรแกรมหยุดเมื่อพบการเชื่อมกลุ่มข้าม split ที่ขัดกัน

## ดู augmentation ก่อนเทรน
E3 override เฉพาะ horizontal_flip_p=.5 เพื่อเปรียบเทียบกับ E2 อย่างตรงไปตรงมา
```bash
python -m furniture.cli preview-augmentation --config configs/project.yaml --experiment E3 --image path/to/image.jpg --output artifacts/augmentation-preview.png
```
preset `configs/augmentation/furniture_light.yaml` เพิ่ม rotation/brightness/contrast แบบอ่อน เป็นตัวเลือกการทดลองใหม่: เปลี่ยน reference ใน project copy และสร้าง experiment id ใหม่ ห้ามนำผลมาอ้างว่าเป็น E3 horizontal flip only
augmentation ทำ online เฉพาะ train; val/test/room inference ใช้ resize/pad + normalize คงที่ ไม่มี random transforms

## วิเคราะห์ห้องหลังมีโมเดลที่ฝึกแล้ว
```bash
python -m furniture.cli room --config configs/project.yaml --image path/to/room.jpg --checkpoint RUN_DIR/best.pt --target scandinavian --catalog data/furniture-v1/train.csv --output artifacts/room-demo
```
ชื่อ target ต้องอยู่ใน mapping ของ checkpoint; preprocessing ของ classifier อ่านจาก checkpoint เสมอ
output คือ annotated.jpg กับ report.json; ถ้าไม่พบเฟอร์นิเจอร์ประเภทที่รองรับ จะรายงานสถานะนั้น
คำแนะนำเป็น heuristic จากคะแนนสไตล์ที่ยังไม่ calibrate; ต่าง label ไม่ได้พิสูจน์ว่าเข้ากันไม่ได้
replacement baseline กรองประเภทเดียวกันและ label สไตล์เป้าหมาย เรียง ID; ยังไม่มี learned compatibility ranking/งบประมาณ/ขนาดจริง/การสร้างภาพห้องใหม่
Catalog train.csv ใช้ demo ได้; การประเมินต้องแยก query/catalog และภาพใกล้ซ้ำ ไม่เลือกคะแนนจากภาพ query ที่อยู่ใน catalog

## ฝึก Detector เพิ่ม
เตรียม room images และ YOLO boxes ตาม `configs/detection.example.yaml` ก่อน
config การฝึก detector: `configs/inference/room_detector.yaml`
```bash
python -m furniture.cli train-detector --config configs/inference/room_detector.yaml
```
Ultralytics รับผิดชอบการ transform ภาพและ boxes ให้สอดคล้องกัน; ไม่ส่ง image-only augmentation ของ classifier ไปเปลี่ยนภาพ detection
pretrained YOLOv8n ครอบคลุมเฉพาะคลาสเดิม เช่น chair/couch/bed/dining table; cabinet/wardrobe ต้องมีข้อมูลและการฝึกเพิ่ม

## ผลที่เก็บต่อ run
- best.pt, last.pt: โมเดลที่เลือกจาก validation และสถานะล่าสุด
- resolved_config.yaml: ค่าใช้งานจริงและเนื้อหา config ต้นทาง
- source/: สำเนาโค้ดที่ใช้; provenance.json เก็บ config/code hashes
- config.json, environment.txt, checkpoint.sha256
- history.json: loss แยก type/style, metrics แต่ละงาน และสถานะ frozen
- summary.json: metric ที่เลือก best epoch และจำนวน epochs ที่อัปเดต backbone ร่วม
- validation.json, validation_predictions.csv, confusion matrices CSV
- test.json และผลต่อภาพเมื่อสั่ง evaluate-run
- mapping, dataset audit, split hashes และ data membership ใน checkpoint

แก้ config/โค้ดแล้ว train อีกครั้ง จะได้ run ใหม่; config ที่บันทึกใน run เก่าไม่เปลี่ยนตาม
การเรียนรู้ต่อจาก checkpoint เป็น warm-start run ใหม่และ optimizer ใหม่; `python -m furniture.cli resume --run-dir RUN_DIR` ใช้กลับมาฝึก run ที่ interrupted/failed จาก epoch boundary ของ checkpoint รุ่นใหม่เท่านั้น ดูข้อจำกัดใน `docs/04-implementation-status.md`

## เอกสารและคำสั่งเก่า
เวอร์ชันนี้ใช้ `train --config configs/project.yaml` แทน train-attributes และใช้ `room` แทน analyze-room; ดู CHANGELOG.md
ยังไม่ได้ push GitHub เพราะปลายทาง Project 2 ยังไม่ยืนยัน

## ตรวจภาพ Dataset และ augmentation

หลัง `prepare-data` ให้สร้างแกลเลอรีเพื่อดูภาพทุกภาพและกรองตาม split, type และ style:

```bash
python -m furniture.cli inspect-dataset --config configs/project.yaml --experiment E3 --output artifacts/dataset-review --variants 3
```

เปิด `artifacts/dataset-review/index.html` ในเบราว์เซอร์ ภาพแรกผ่าน resize/pad และภาพถัดไปคือตัวอย่าง augmentation แบบสุ่มตาม config ของ experiment เฉพาะ train; val/test แสดงภาพที่ไม่มี augmentation ค่า seed และ config แสดงในหน้าเว็บ จำนวน variants ปรับได้ และแต่ละครั้งต้องเลือก output directory ใหม่ ภาพ augmentation เป็นเพียงตัวอย่าง เพราะขณะฝึกจะสุ่มใหม่ รวมทั้งอาจเหมือนภาพเดิมได้เมื่อการแปลงไม่ถูกสุ่มใช้ ตรวจ label และภาพซ้ำใกล้เคียงด้วยสายตาก่อนฝึก

ทดลองด้วยข้อมูลสังเคราะห์: `python -m furniture.cli make-smoke-data` แล้วใช้ `--config configs/project_smoke.yaml` (ข้อมูล smoke ไม่ใช่ข้อมูลจริง)
