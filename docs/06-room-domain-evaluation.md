# ตรวจระบบด้วยภาพห้องและ annotation จริง

การทดสอบนี้ต้องใช้ภาพห้องจริงที่มีสิทธิ์ใช้ พร้อม bounding boxes และ type/style ที่ตรวจด้วยคน โดยห้องเดียวกันต้องอยู่ split เดียวกัน ไฟล์ตัวอย่าง `examples/room_annotations_template.csv` และ `examples/room_detections_template.csv` มีเฉพาะหัวตาราง ไม่มีข้อมูลหรือผลลัพธ์สมมติ

Annotation หนึ่งแถวต่อวัตถุ: `room_id` ระบุห้อง, `image_id` ระบุภาพ, `image_path` เป็น path ภายใน image root, `object_id` ไม่ซ้ำในชุด, `split` เป็น train/val/test, `type` และ `style` ต้องตรง class mapping ใน checkpoint และ x1,y1,x2,y2 เป็นพิกัดพิกเซลบนภาพหลังแก้ EXIF (ไม่ใช่ YOLO normalized) มุมซ้ายบนรวมถึง มุมขวาล่างไม่รวม อ้างอิงภาพขนาดเต็ม ไม่ใช่พิกัด crop ตรวจ label ทุกชิ้นที่อยู่ในขอบเขต detector หากภาพหนึ่งมีวัตถุที่ไม่ได้ annotate จะนับ detection นั้นเป็น false positive

Detector CSV มี image_id, detector_type, confidence และกล่องในระบบพิกัดเดียวกัน Mapping JSON ต้องระบุชื่อคลาส detector ไปยังประเภทใน classifier อย่างชัดเจน ดู `examples/room_detector_class_mapping.example.json` เป็นตัวอย่าง; ตรวจว่า label ทางขวามีอยู่ใน checkpoint ที่ใช้ คลาส detector ที่ไม่มีใน mapping ถูกตัดและนับแยก ไม่ได้ถือเป็น false positive ของคลาสที่รองรับ

หากติดตั้ง optional dependency `ultralytics` แล้ว ใช้ API ต่อไปนี้โดยแทนที่ path ด้วยไฟล์จริง ต้องมี pretrained/fine-tuned classifier checkpoint สอง heads และ YOLO weights ที่ใช้จริง:

```python
from furniture.room_evaluation import export_room_detections, evaluate_room_domain

export_room_detections('PATH/annotations.csv', 'PATH/images', 'PATH/detector.pt',
                       'PATH/detections.csv', device='auto', confidence=0.05, split='test')
evaluate_room_domain('PATH/annotations.csv', 'PATH/detections.csv', 'PATH/images',
                     'PATH/classifier.pt', 'PATH/new-report-directory',
                     'PATH/detector_class_mapping.json', device='auto', confidence=0.25,
                     min_crop_size=16, iou_threshold=0.5, split='test')
```

Exporter เรียก detector หนึ่งครั้งต่อภาพและเขียน CSV พร้อม metadata/hash sidecar; หากมีผลจาก detector อื่นแล้วสามารถเตรียม CSV ใน schema เดียวกันและเรียก evaluator ตรงได้ การส่งออก detector ที่ confidence 0.05 หมายความว่าตัวอย่างต่ำกว่านี้ถูกกรองก่อน CSV; ค่าที่ใช้ประเมินภายหลังไม่ควรต่ำกว่าค่านี้ มิฉะนั้นผลจะไม่ครบ

`room_metrics.json` แสดง classifier metrics แยกจาก reference crops และจาก detector crops ที่จับคู่ได้เท่านั้น รวมจำนวน matches, misses, false positives, precision/recall ที่ IoU ตามเกณฑ์และรายการที่ถูกกรอง `room_predictions.csv` เก็บ true/predicted labels และ confidence ของทั้งสอง crop รายวัตถุ Matcher เลือกคู่ชนิดเดียวกันแบบ greedy ตาม IoU สูงก่อนและจับคู่ได้ครั้งเดียว ผลนี้ไม่ใช่ mAP และไม่รวมภาพที่ไม่มี annotated object ใน CSV

ต้องเก็บ detector version, weights, class coverage และ inference settings คงที่เมื่อเปรียบเทียบ classifier ภาพห้องที่ใช้ทดสอบต้องไม่มาจากกลุ่มฝึกหรือ validation และไม่ควรเลือกตัวอย่างเพื่อทำให้ผลดูดี ตั้ง protocol ก่อนเปิด test คุณภาพคำแนะนำเรื่องสไตล์ยังต้องประเมินแยกกับคน ผล reference/detector crop ไม่ยืนยัน recommendation และไม่แสดงว่าระบบตรวจจับเฟอร์นิเจอร์ได้ทุกประเภท
