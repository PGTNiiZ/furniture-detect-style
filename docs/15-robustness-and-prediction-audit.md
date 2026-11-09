# Validation robustness และตรวจ predictions

หลัง 44 training runs และการเลือก checkpoint ตาม validation ที่ตรึงไว้ `scripts/post_training_audit.py` รอ suite เดิมจบ แล้วใช้ controlled runs ของ backbone ที่เลือกสำหรับ C3/C5/C6 และ C4/C7/C8. โปรแกรมสร้าง occlusion manifest จาก validation เดิมที่ seed 42 ครั้งเดียว และใช้ manifest เดียวกันกับทุก checkpoint ประเมิน clean, light และ strong; ไม่ใช้ผลนี้เปลี่ยนกฎเลือก checkpoint

รายงาน `artifacts/model-only-release/robustness-report/metrics.csv` แยก type/style, accuracy, macro F1 และส่วนต่างจาก clean สำหรับแต่ละ condition/level พร้อม `easy-contrasts.csv` และ `hard-contrasts.csv`. Manifest ถูกบันทึกแยกและทุกผลผูกกับ checkpoint, config, split, mapping และ hashes ของ bundle. การบังเป็นสัดส่วนของภาพหลัง resize/pad ไม่ใช่สัดส่วนพื้นที่วัตถุ: ไม่มี object mask จึงอ้าง object occlusion percentage ไม่ได้

คง none/easy/hard และ CoarseDropout off/light/strong ตาม C1–C8; ไม่เพิ่ม Mosaic ใน classifier ที่มีหนึ่ง category และหนึ่ง style ต่อภาพ หากทดลอง augmentation ฝั่ง detector ในอนาคตต้องเป็น protocol แยก

ก่อนรายงาน test จะตรวจ prediction IDs ไม่ซ้ำและตรง manifest ทุกแถว, reference labels/image paths, labels ใน mapping และคำนวณ accuracy, precision, recall, macro/weighted F1 และ confusion matrix จาก predictions เปรียบเทียบกับผลที่บันทึก สำหรับ external source ตรวจ missing/known/unseen, eligible subsets, coverage, joint denominator และรายการ missing/unseen แยก. Synthetic และ external ไม่รวมเป็นคะแนนเดียวกัน

ขั้นเหล่านี้ใช้ attempt/bundle ของ `scripts/release_evidence.py`: การขัดจังหวะไม่เขียนทับผลเดิม; attempt ค้างเก็บใน `_attempts/` และ events ใน `recovery-events/`. หาก hash, checkpoint, split หรือ controls ไม่ตรง ระบบปฏิเสธ reuse แทนการแก้ status ด้วยมือ. Mock tests แยกจากการรันข้อมูลจริง; ณ การเขียนยังไม่มีผล robustness หรือ held-out test ของ selected checkpoint
