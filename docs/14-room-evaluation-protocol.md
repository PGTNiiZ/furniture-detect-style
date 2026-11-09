# Room inference และ protocol สำหรับการประเมิน

เป้าหมายคือภาพห้อง → detector → boxes → crops → classifier แบบ shared backbone สอง heads; detector และ classifier เป็นคนละโมเดล ไม่ถือเป็นหลักฐานของ joint multi-task learning ตัวเดียว Classifier output ใช้ type และ style จาก checkpoint จริง detector label เก็บแยกและบันทึก disagreement ตามตาราง mapping ใน `configs/inference/room-yolov8n-coco-baseline.yaml`

## Detector ที่ตรวจได้

Config เก่าอ้าง `yolov8n.pt` แต่ไม่พบ user-trained detector checkpoint ใน workspace. Baseline ที่ดาวน์โหลดแยกคือ Ultralytics YOLOv8n detect pretrained บน COCO จาก [release v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt), SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`. หลักฐาน load และ CPU forward จริงอยู่ `artifacts/detector-baselines/yolov8n-coco-v8.3.0/verification.json`; ไม่ใช่ detector ที่ผู้ใช้ฝึกเองและไม่ใช่ accuracy บนภาพห้อง

ZIP ไม่แจกจ่าย detector weights; ดาวน์โหลดจากแหล่งทางการตามคำสั่งด้านล่างและตรวจ [เงื่อนไขลิขสิทธิ์ของ Ultralytics](https://www.ultralytics.com/license) สำหรับการใช้งานที่ต้องการ

COCO labels ที่ map ไป classifier type ได้คือ `bed→bed`, `chair→chair`, `couch→sofa`, `dining table→table`; detector ไม่มี label ทั่วไปสำหรับ table ทุกชนิด และการมีชื่อ class ไม่รับประกันว่าจะตรวจพบวัตถุทุกชิ้น ใช้ Ultralytics inference 640 letterbox และบันทึก config/weight hashes. ตัวอย่าง Pexels เก็บ [ที่มาและข้อจำกัด](../artifacts/room-inference-demo/source-provenance.json); [ภาพวาดกรอบ](../artifacts/room-inference-demo/result-v3/annotated.png), [JSON](../artifacts/room-inference-demo/result-v3/result.json) และ [CSV](../artifacts/room-inference-demo/result-v3/objects.csv) เป็น demonstration ด้วย screening checkpoint ไม่ใช่ ground truth หรือ selected final checkpoint

เมื่อ controlled suite ตรึง selected classifier แล้ว post-training worker จะรันภาพ demonstration เดิมอีกครั้งโดยใช้ selected checkpoint และบันทึก `artifacts/model-only-release/selected-room-demo/` พร้อมสถานะใน `selected-room-demo-status.json`; ภาพนี้ไม่ใช้เลือกโมเดลหรือปรับ threshold. หาก detector หรือ environment หาย ขั้น demo บันทึก failure แยกและไม่หยุดการฝึก; หลังแก้สาเหตุสามารถกู้ post-training worker ด้วย `--retry-demo` โดยไม่แก้ status เองและเก็บ events เดิม

## Room inference บน Windows

ต้องมี `ultralytics` ใน environment แยกที่ไม่เปลี่ยน environment ของ suite ที่กำลังฝึก ดาวน์โหลด detector จาก URL ข้างต้นไปยัง path ใน config และตรวจ SHA-256 ก่อนใช้ หรือระบุ `--detector-weights` เป็นไฟล์จริงของผู้ใช้ แยกตรวจ provenance ของไฟล์นั้นเอง

```powershell
py -3.11 -m venv .venv-room
.venv-room\Scripts\python.exe -m pip install -e '.[detection]'
New-Item -ItemType Directory -Force artifacts\detector-baselines\yolov8n-coco-v8.3.0
Invoke-WebRequest 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt' -OutFile artifacts\detector-baselines\yolov8n-coco-v8.3.0\yolov8n.pt
Get-FileHash -Algorithm SHA256 artifacts\detector-baselines\yolov8n-coco-v8.3.0\yolov8n.pt
# ต้องตรง f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
.venv-room\Scripts\python.exe -m scripts.detect_then_classify --image PATH_TO_ROOM.jpg --checkpoint PATH_TO_CLASSIFIER.pt --config configs/inference/room-yolov8n-coco-baseline.yaml --output artifacts/my-room-result --device cpu --save-crops
```

การตรวจจริงรอบนี้ใช้ environment แยกที่มีอยู่แล้ว `research-clean-env` (CPU) กับ Ultralytics 8.4.198; คำสั่งสร้าง `.venv-room` เป็นแนวทางสำหรับติดตั้งใหม่และยังไม่ได้ทดสอบเป็น clean install ในรอบนี้

คืน `result.json`, `objects.csv`, `annotated.png` และ crops เมื่อเลือก option; ผลและ config/checkpoint hashes ถูกผูกใน bundle ที่กู้ได้ EXIF ถูกแก้ก่อน detector ทั้ง inference และ evaluation ใช้ `scripts.room_crop.prepare_crop` เดียวกัน: ตรวจกรอบ, ปัด floor/ceil, clip, ตรวจ `min_size`, แล้ว pad โดยใช้ขนาดหลัง clip และบันทึกทั้ง detector box กับ crop box; invalid/small crop ถูกนับแยก. เมื่อไม่พบกรอบคืน `no_detections` confidence ไม่ถูกคูณกันและ style softmax ไม่ใช่ compatibility percentage. Notebook: `notebooks/03_room_inference.ipynb`

## แผนประเมินที่ต้องตรึงก่อนเปิด final

เตรียม [ชุดภาพรอตรวจ 2 ภาพ](../data/room-independent-review-v1/README.md) จากคนละผู้ถ่ายพร้อมกรอบร่างอิสระจาก predictions และคำถามรายชิ้นแล้ว แต่ยังไม่มี independent human review หรือ quantitative room score. ต้องมี `images.csv` ตาม `examples/room_images_template.csv` รวมภาพที่ไม่มีวัตถุด้วย และ `objects.csv` ตาม `examples/room_objects_template.csv` หนึ่งแถวต่อวัตถุ; style ที่ไม่ทราบให้ว่าง ห้ามใช้ model output เติม label. image SHA-256 ตรวจไฟล์จริง, `room_id` และ exact image hash ห้ามข้าม development/final; ต้องตรวจความเป็นอิสระจากข้อมูลฝึกและ annotations ให้ครอบคลุมวัตถุด้วยคนด้วย

`examples/room_eval_protocol.example.json` เป็น **draft** ไม่ใช่ threshold ที่ผ่าน calibration. ตัดสิน confidence/IoU/crop padding จาก development data อิสระ, บันทึก `development_reference` กับ `locked_at_utc`, ยืนยัน `independent_from_classifier_training` และเปลี่ยน `status` เป็น `locked` ก่อนคำนวณ final predictions. โปรแกรมตรวจ exact pixel hash ของภาพห้องกับ train/val/test exposure ใน classifier checkpoint; การยืนยันว่าไม่ใช่ภาพหรือห้องที่เกี่ยวข้องกันนอกเหนือจาก exact hash ยังต้องตรวจ provenance เพิ่ม One-to-one matching ใช้ greedy descending IoU แบบ class-agnostic แล้วรายงาน disagreement ของ detector category แยก; นับ misses, false positives รวมภาพว่าง และ unmapped detector labels แยก. Reference-crop metrics และ matched-detector-crop metrics รายงาน eligible counts กับ coverage เทียบวัตถุทั้งหมดต่อ type/style ไม่ถือ matched accuracy เป็น accuracy ของทั้งระบบ

ส่งออก detector CSV ด้วย `python -m scripts.export_room_detections --images ... --image-root ... --weights ... --split final --output NEW_DIRECTORY`; จากนั้นใช้ `python -m scripts.room_evaluation --images ... --objects ... --detections NEW_DIRECTORY/detections.csv --image-root ... --checkpoint ... --detector-checkpoint ... --protocol LOCKED.json --output NEW_REPORT`. Export และ evaluation ตรวจ image, detector, prediction, protocol และ checkpoint hashes พร้อมบันทึก bundle/recovery evidence. ผล final ต้องไม่ถูกใช้ปรับ threshold หรือเลือกโมเดล
