# ตรวจเฟอร์นิเจอร์หลายชิ้นในภาพห้องและจำแนกประเภท/สไตล์ด้วย Multi-task Learning

รองรับ `single`: ภาพเฟอร์นิเจอร์ชิ้นเดียว → classifier และ `room`: ภาพห้อง → detector → crop รายชิ้น → classifier โมเดล classifier ใช้ backbone ร่วมกับ type/category และ style สอง heads และรวม loss เพื่อปรับ backbone ระหว่าง fine-tuning; detector เป็นโมเดลแยก ผลแต่ละงานรายงานแยกกัน ขอบเขตคือโมเดล การทดลอง CLI และ notebook ดูสถานะจริงใน [RELEASE_STATUS.md](RELEASE_STATUS.md)

## เริ่มต้นสำหรับเพื่อนร่วมทีม

- วิธีทำงานด้วย Git/branch/Pull Request: [CONTRIBUTING.md](CONTRIBUTING.md) (`main` = นิ่ง เจ้าของ repo เท่านั้นที่ merge, `develop` = รวมงาน, ทำงานบน branch ของตัวเองแล้วเปิด PR เข้า `develop`)
- เทรนอะไรไปแล้ว ทำไมเลือก DenseNet121 และเทรนต่อยังไง: [docs/22-training-and-model-guide.md](docs/22-training-and-model-guide.md)
- ใบงานที่ต้องใช้คน (ตรวจภาพซ้ำ นโยบายสไตล์ ภาพห้อง): [docs/21-human-tasks-handoff.md](docs/21-human-tasks-handoff.md)
- ภาพ dataset, checkpoint และผลเทรน (`data/`, `artifacts/`) ไม่อยู่ใน git ขอไฟล์จากเจ้าของ repo ผ่าน Drive

## ข้อมูลและการทดลอง

ใช้ `data/furniture-v2-conservative` ที่ตรึง revision, manifests และ hashes ไว้: train 5,956; validation 1,295; synthetic test 1,127; real-source test 1,000 ภาพ ไม่ผสม real-source test เข้าฝึกหรือเลือกโมเดล ภาพ 622 ภาพถูก quarantine เนื่องจากคู่ dHash ข้าม split ยังรอคนตรวจ; dHash ไม่ใช่คำตัดสิน identity ดู [dataset protocol](docs/10-dataset-protocol-v2.md), [audit](docs/11-dataset-audit-v2.md) และ [หน้าตรวจ 494 คู่](artifacts/source-train-cross-split-review/index.html)

Screening เดิมใช้ 10 backbones และ validation เท่านั้น ตามด้วย C1–C8, single-task และ repeated seeds ตาม [protocol ที่ตรึงไว้](docs/13-model-only-experiment-protocol.md) ก่อนเลือก checkpoint และเปิด test โดยรายงาน synthetic test กับ real-source test แยกกัน ห้ามใช้ smoke, benchmark หรือผล pretrained step เป็นคะแนนวิจัย

สถานะล่าสุดดู `artifacts/runs/suites/furniture-v2-screening.json` และ `artifacts/model-only-release/workflow-status.json` ผลเปรียบเทียบที่เสร็จแล้วจะอยู่ใน `artifacts/model-only-release/validation-report/`; ผล synthetic test และ real-source reference แยกอยู่ใน `artifacts/model-only-release/` พร้อมหลักฐานผูก checkpoint และ split

## Windows PowerShell

เปิด PowerShell ที่ root ของโปรเจกต์ ใช้ Python 3.10 ขึ้นไปและ PyTorch/torchvision คู่ที่รองรับเครื่อง ใน workspace นี้ `.venv` มี dependencies และ CUDA อยู่แล้ว **อย่าเปลี่ยนแพ็กเกจใน `.venv` ระหว่าง screening หรือระหว่างรอ recovery** เพราะ environment เป็นส่วนหนึ่งของ provenance. หากต้องติดตั้งใหม่ให้ใช้ environment แยก เช่น:

```powershell
py -3.11 -m venv .venv-release
.venv-release\Scripts\python.exe -m pip install -e '.[data,dev,reports]'
.venv-release\Scripts\python.exe -m pip check
```

เลือก PyTorch wheel สำหรับ CUDA ที่ตรงกับเครื่องก่อนติดตั้ง package หากต้องใช้ GPU; CPU ทำงานได้เช่นกัน คำสั่งติดตั้ง environment แยกเป็นแนวทางสำหรับเครื่องใหม่ ส่วน environment ปัจจุบันและการทดสอบที่รันจริงมีหลักฐานใน `docs/evidence/`

```powershell
.venv\Scripts\python.exe -m furniture.cli check-device
.venv\Scripts\python.exe -m pytest -q
```

## Inference: ภาพเดี่ยว โฟลเดอร์ หรือ CSV

ใช้ `best.pt` ของ run ที่เสร็จแล้วและมี mapping กับ preprocessing metadata อยู่ใน checkpoint ตัว checkpoint ที่ผ่านการเลือกสุดท้ายจะระบุใน `artifacts/model-only-release/selection-frozen.json` หลัง controlled เสร็จ ตั้ง `$Checkpoint` เป็น path จริงก่อนใช้ ตัวอย่างคำสั่งนี้ได้ตรวจด้วย checkpoint จาก screening ที่ completed แล้ว แต่ checkpoint สำหรับ release **ถูกเลือกแล้ว ใช้ release/checkpoint/best.pt และตรวจ SHA-256 ใน selection-frozen.json**

```powershell
$Checkpoint = 'PATH_TO_COMPLETED_RUN\best.pt'
.venv\Scripts\python.exe -m scripts.model_inference --checkpoint $Checkpoint --image 'PATH_TO_IMAGE.jpg' --output predictions.json --device cpu --top-k 3
.venv\Scripts\python.exe -m scripts.model_inference --checkpoint $Checkpoint --folder 'PATH_TO_IMAGE_FOLDER' --output folder-results.csv --device cuda --top-k 5
.venv\Scripts\python.exe -m scripts.model_inference --checkpoint $Checkpoint --csv 'PATH_TO_IMAGES.csv' --output list-results.json --device auto --top-k 3
```

CSV input ต้องมี `image_id,image_path`; relative path อ้างอิงโฟลเดอร์ของ CSV ใช้ `--image-id` กำหนด sample ID ของภาพเดี่ยวได้ ผล JSON/CSV มี sample ID, type/style, confidence และ top-k ต่อ task โดยจำกัด k ตามจำนวนคลาสจริง ภาพเสียถูกบันทึกเป็น `status=error` พร้อมข้อความ และ CLI คืน exit code 1 หากมีภาพเสีย; ผลของภาพอื่นยังถูกเก็บไว้ โปรแกรมไม่เขียนทับ output เดิม

ภาพถูกแก้ EXIF orientation, รวม alpha กับพื้นขาว, resize/pad และ normalize ตาม metadata ใน checkpoint หาก metadata ขาดหรือระบุวิธีที่ไม่รองรับ โปรแกรมจะหยุดแทนการเดาค่า confidence เป็น softmax ของ classifier ที่ **ยังไม่ได้ประเมิน calibration** ไม่ใช่ compatibility score หรือ calibrated probability

ใช้จาก Python โดยโหลดโมเดลเพียงครั้งเดียว:

```python
from scripts.model_inference import FurniturePredictor

predictor = FurniturePredictor('PATH_TO_COMPLETED_RUN/best.pt', device='cpu')
results = predictor.predict([('item-1', 'PATH_TO_IMAGE.jpg'), ('item-2', 'PATH_TO_OTHER.png')], top_k=3)
```

Notebook ตัวอย่างอยู่ที่ [notebooks/02_model_inference.ipynb](notebooks/02_model_inference.ipynb) และเรียกโมดูลเดียวกับ CLI

## ภาพห้อง: detector → crops → classifier

Config เดิมระบุ `yolov8n.pt` แต่ยังไม่พบ trained detector checkpoint ของผู้ใช้ จึงตรวจ baseline YOLOv8n pretrained COCO แยกโดยไม่ฝึก detector ใหม่ ดูที่มา, checksum, การติดตั้ง environment แยก, schema และข้อจำกัดใน [room protocol](docs/14-room-evaluation-protocol.md):

```powershell
.venv-room\Scripts\python.exe -m scripts.detect_then_classify --image 'PATH_TO_ROOM.jpg' --detector-weights 'PATH_TO_EXISTING_YOLO.pt' --checkpoint $Checkpoint --output room-result --device cpu --save-crops
```

ผลในโฟลเดอร์มี JSON, CSV, ภาพวาดกรอบ และ crops หากระบุ option; แต่ละชิ้นเก็บ detector label แยกจาก category/style ของ classifier และบันทึก disagreement. ไม่มี detector annotations จึงยังวัด end-to-end เชิงปริมาณไม่ได้ Notebook: `notebooks/03_room_inference.ipynb` ส่วน catalog หากมีจริงสามารถใช้ `furniture.recommend.replacements`: กรอง type/style แล้วเรียง ID ไม่ใช่ visual similarity หรือโมเดล compatibility

มี [ภาพห้องและรายการ annotation สำหรับตรวจ](data/room-independent-review-v1/README.md) แยกจาก dataset ฝึกแล้ว กรอบเป็นร่างจากการดูภาพโดยไม่ใช้ผลโมเดล ยังต้องให้คนตรวจความครบและกรณีกำกวมก่อนประเมิน; ค่า style ที่ไม่ยืนยันเว้นว่าง

## หลักฐานและข้อจำกัด

- ผล type/style, confusion matrices และ error examples ต้องมาจาก run ที่เสร็จจริง; ส่วนที่ยังไม่รันระบุใน [RELEASE_STATUS.md](RELEASE_STATUS.md)
- real-source annotations มาจากโมเดลของผู้เผยแพร่ ไม่ได้ยืนยันว่าเป็น expert ground truth; unseen/missing labels แยกจาก eligible closed-set metrics ตาม [external protocol](docs/09-external-evaluation-protocol.md)
- Bonn/IKEA/Target เป็น future work จนตรวจการเข้าถึงและ annotations แล้ว ไม่ผสมเข้า suite ปัจจุบัน
- ภาพ dataset เต็มไม่อยู่ใน ZIP; selected checkpoint อยู่ที่ release/checkpoint/best.pt ตรวจ path และ SHA-256 ของ checkpoint ที่เลือกใน selection record ก่อนใช้งาน

README ก่อนปรับขอบเขตเก็บไว้ที่ [historical README](docs/historical-readme-pre-model-only.md); ข้อมูลเก่าในเอกสารประวัติไม่ใช่เงื่อนไข release ปัจจุบัน
