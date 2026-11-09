# ข้อมูลใหม่ การฝึกใหม่ และ augmentation

## CSV ที่รับเข้า
ตัวอย่างนี้สาธิต schema เท่านั้น ไฟล์รูปไม่ได้แนบมา:
```csv
filename,category,design_style,product_id,split
chair_001_a.jpg,chair,modern,catalogA-chair001,train
chair_001_b.jpg,chair,modern,catalogA-chair001,train
table_021.jpg,table,rustic,catalogA-table021,val
```
- column mapping อยู่ source.columns; สามารถเปลี่ยนชื่อคอลัมน์จริงโดยไม่แก้ Python
- product_id คือกลุ่มภาพสินค้าชิ้นเดียวกัน/ภาพที่เกี่ยวข้อง ใช้ชื่อที่ไม่ชนข้ามแหล่ง
- split มี train/val/test; ถ้าให้ระบบแบ่ง ตั้ง columns.split เป็น null
- ถ้ายังไม่มี group column ตั้ง columns.group_id เป็น null ได้ แต่ต้องยอมรับว่ายังป้องกันได้เฉพาะ exact duplicate และต้อง audit source/near duplicates เพิ่ม
- ทุกภาพต้องมี type และ style; ไม่มี weak-label guessing หรืออ่าน prompt มาเป็น label อัตโนมัติ

การนำเข้าเริ่มต้นหยุดเมื่อพบ labels ว่าง หากตั้งใจใช้เฉพาะแถวที่มี labels ครบ ให้เลือก `--exclude-incomplete-labels` กับ `prepare-data` อย่างชัดเจน และตั้ง dataset id/root ใหม่สำหรับ snapshot นั้น ตัวเลือกนี้ข้ามเฉพาะค่าว่างหรือ null ของ type/style หลังตรวจว่าภาพและ split ถูกต้อง; key ที่ไม่มี, ชนิด label ผิด, ภาพเสีย และ source test ที่ถูกระบุเป็น train/val ยังคงทำให้ import ล้มเหลว

ผล exclusion อยู่ใน `excluded_samples.json`, `dataset_metadata.json` และ `audit.json` โดยบันทึก source split, image ID, labels ที่ขาด และจำนวนแยกตาม split/label `audit-data` แสดงข้อมูลนี้ด้วย อย่าซ่อมหรือเขียนทับ snapshot ที่ import ไม่สำเร็จ ให้เก็บ `import_errors.json` เป็นหลักฐานและใช้ root ใหม่

## อัปเดต v1 → v2
1. สำเนา labels เดิมรวมข้อมูลใหม่ (เก็บไฟล์ raw เดิม)
2. ตั้ง id/root ใหม่ เช่น furniture-v2 / data/furniture-v2
3. ตั้ง split.previous_root: data/furniture-v1
4. prepare-data จะรักษา split ของภาพเดิมตาม pixel hash และ group_id ที่ระบุ
5. ถ้าภาพใหม่เชื่อมกลุ่ม train กับ test เดิม โปรแกรมหยุดให้ review ไม่ย้ายเงียบ ๆ
6. ดู counts และการกระจาย labels; rare joint label groups อาจอยู่ train อย่างเดียว
7. เทรน run ใหม่และประเมินบน holdout ที่ตรึงไว้ รวมภาพห้องจริงภายนอกเมื่อพร้อม

อัตรา 70/15/15 เป็นเป้าหมายระดับกลุ่ม อาจไม่ได้จำนวนภาพตามสัดส่วนเป๊ะ การรักษา groups/previous assignments สำคัญกว่า
ภาพที่ visually similar แต่ hash ต่างยังต้องตรวจ candidates และยืนยันความสัมพันธ์ก่อนจัดกลุ่ม
เมื่อแก้ label ภาพเดิม ให้บันทึก dataset version/เหตุผล อย่าแก้ manifest ของ run ที่ตีพิมพ์ผลแล้ว

## Audit และ near-duplicate review

`python -m furniture.cli audit-data --config configs/project.yaml` ตรวจภาพ/hash, labels, group และแสดงจำนวนต่อคลาส ตาราง type/style, คลาสที่ขาดใน holdout และ near-duplicate candidates ส่วน `prepare-data` เขียน `dataset_metadata.json` (source/revision, split hashes, mapping/hash และคลาสที่เพิ่มหรือลดเทียบ previous version) และ `near_duplicate_candidates.json` ให้ snapshot ใหม่ด้วย

Candidate ใช้ dHash 64 บิตระยะ Hamming ≤ 4; รายการมี image IDs, paths, split, group และ distance เพื่อเปิดภาพสองภาพเทียบกัน ระยะต่ำไม่ได้พิสูจน์ว่าเป็นสินค้าเดียวกัน เช่น ภาพสีพื้นและพื้นหลังขาวอาจชนกันได้ ต้องตรวจด้วยคน และสร้าง `group_id` ร่วมเฉพาะภาพที่ยืนยันว่าเกี่ยวข้องก่อน prepare snapshot ใหม่ หาก group เชื่อม split เก่าต่างกัน ระบบหยุดให้แก้ข้อมูล นโยบายจำกัดที่ 250,000 คู่ที่ตรวจหรือ 2,000 candidates: เมื่อ `truncated: true` รายงานไม่ครอบคลุมทั้งชุด ห้ามใช้เป็นหลักฐานว่าไม่มี leakage และควรใช้เครื่องมือทบทวนเพิ่มเติม ส่วน revision และสิทธิ์ในไฟล์ต้นทางต้องตรวจด้วยตนเอง

## ตรวจแหล่ง Hugging Face

จาก [dataset card ของ filnow/furniture-synthetic-dataset](https://huggingface.co/datasets/filnow/furniture-synthetic-dataset) ที่เปิดตรวจเมื่อ 2 พฤศจิกายน 2569 ผู้จัดทำระบุ Apache-2.0, 9,000 ภาพ train สังเคราะห์, 1,000 ภาพ test จริง และ test labels จาก Qwen2 VL ซึ่งอาจคลาดเคลื่อน; คอลัมน์ type และ style แสดงใน viewer ข้อมูลเหล่านี้เป็นคำอธิบายจากผู้จัดทำ ไม่ใช่จำนวนที่โค้ดนี้ audit แล้ว ค่า source `prompt` อาจมี label จึงไม่ถูกส่งเป็น feature ไปยัง classifier

ตั้ง revision เป็น commit SHA ที่ตรวจจริงก่อนดาวน์โหลด ห้ามใช้ placeholder ในการ prepare เมื่อ source ของ Hugging Face มี split `test` ระบบตรึงแถวเหล่านั้นไว้ test เสมอและปฏิเสธการ override เป็น train/val (นโยบายเปลี่ยนจากเวอร์ชันก่อนที่สุ่มรวมทุก source split) ส่วน source train ยังถูกแบ่งตาม group และอาจมีแถว synthetic ใน holdout ด้วย ดังนั้นต้องใช้ `source_split` แยกวิเคราะห์ synthetic/real และห้ามถือ test นี้เป็น room-domain evaluation โดยอัตโนมัติ

ข้อมูล preview ที่แนบเดิมเป็นเพียง subset ของ source train ไม่ใช่หลักฐานว่าตรวจครบทั้งชุด ต้องตรวจ labels, ตัวอย่างและ license ของ revision ที่เลือกก่อนเริ่ม protocol จริง

## เริ่ม weights ใหม่หรือใช้ของเดิม
- imagenet: เหมาะสำหรับ E1–E3 และเปรียบเทียบ dataset versions โดยใช้ initialization เดียวกัน
- scratch: smoke หรือ experiment ที่ตั้งใจทดสอบการฝึกใหม่ทั้งหมด
- checkpoint + full: labels/mapping ต้องเหมือนเดิม โหลด backbone และ heads แล้วเริ่ม optimizer ใหม่
- checkpoint + backbone: ใช้ feature extractor เดิมและสร้าง heads ใหม่ รองรับชุดคลาสใหม่; ต้องฝึกทั้งสอง heads ใหม่

ใช้ `configs/models/finetune_new_data.yaml` เป็นตัวอย่าง แล้วชี้ project.model ไปที่ไฟล์นั้น
checkpoint ต้องมี data membership ของเวอร์ชัน 0.2 เพื่อเช็ก leakage; v0.1 weights ไม่มียืนยันส่วนนี้
การย้าย data role ระหว่าง train กับ holdout จะถูกปฏิเสธ แม้ใช้ backbone scope
การฝึกด้วยข้อมูลใหม่อย่างเดียวอาจทำให้ผลชุดเดิมลดลง (forgetting); ถ้าต้องรักษาของเดิมให้ใช้ข้อมูลฝึกเดิมรวมข้อมูลใหม่อย่างเหมาะสม และวัด holdout ทั้งโดเมนเดิม/ใหม่
warm-start ไม่ใช่ resume; `resume --run-dir RUN_DIR` คืน model/optimizer/scheduler/scaler ของ checkpoint รุ่นใหม่และกำหนด random seed ใหม่ตาม epoch ทั้งตอนฝึกต่อเนื่องและตอน resume ใช้ได้เมื่อ code/config/data/PyTorch version เดิมและไม่มีการเปลี่ยน hardware protocol; GPU บาง operations อาจไม่ bitwise deterministic

## ใช้ augmentation อย่างเป็นธรรม
- แบ่งข้อมูลก่อน; augmented variants อยู่เฉพาะการโหลด train online
- การเพิ่มภาพ 1 เป็นหลาย variants ไม่ได้เพิ่มจำนวนต้นฉบับ independent samples
- validation/test/inference ใช้ deterministic preprocessing จาก checkpoint
- E3 เปลี่ยนเพียง horizontal flip; ต้องทดลองแยกหากจะเพิ่ม rotation/color jitter
- preset furniture_light เป็นข้อเสนอเริ่มต้น ไม่ใช่ค่าที่พิสูจน์แล้วว่าเพิ่มคะแนน
- สำหรับ detection ให้ใช้ config ของ YOLO ซึ่งปรับ boxes ตาม geometry ห้ามใช้ image-only transforms กับภาพแล้วลืม labels

## หลักฐานข้อมูลที่ควรเก็บ
แหล่งข้อมูลและrevision, จำนวนภาพ/กลุ่ม/คลาสแต่ละ split, ภาพตัวอย่าง, labels ที่ไม่ตรงภาพ, excluded samples,
exact/near duplicate review, group definition, split seed, snapshot hashes และช่องว่าง product image → room crop
