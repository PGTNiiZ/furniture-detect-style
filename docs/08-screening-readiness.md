# ความพร้อมก่อน screening

**อัปเดต:** ประเด็น snapshot และ external protocol ได้ดำเนินการต่อใน `docs/10-dataset-protocol-v2.md`; ข้อความด้านล่างเก็บสถานะก่อนสร้าง v2 เพื่อใช้ตรวจประวัติ ไม่ใช่สถานะปัจจุบัน

ฐานโค้ด: เปรียบเทียบ SHA-256 ของไฟล์ Python/YAML ใน `src/furniture`, `configs` และ `tests` จำนวน 77 ไฟล์กับ `furniture-project2-research-20261102.zip` แล้วตรงกันก่อนแก้รอบนี้

## หลักฐานแยกประเภท

- **Source inspection:** อ่านครบ 10,000 แถว, 9,999 มี type/style, 1 แถวไม่มีทั้งสอง labels; snapshot ที่เตรียมใหม่ยังไม่ผ่าน validation เพราะ source test มี labels นอก mapping ของ train ดู `docs/evidence/full-source-inspection.json`
- **Near-duplicate inspection:** ตรวจทุกคู่ของ source train ด้วย dHash threshold 4 พบ 1,031 candidates รวม 240 คู่ train/validation; ภาพคล้ายกันไม่เท่ากับยืนยันว่าเป็นสินค้าเดียวกัน CSV อยู่ที่ `docs/evidence/source-train-near-audit.csv`
- **Automated tests:** `pytest -q --basetemp .pip-temp/pcomplete-final` ผ่าน 96 รายการ; tests จุดที่แก้ได้แก่ `tests/test_robustness.py`, `tests/test_recovery.py`, `tests/test_suite_retry.py`, `tests/test_cli_workflows.py` และ regression path ยาวใน `tests/test_training_controls.py`
- **Pretrained verification:** โหลด weights enum และ forward บน CUDA ครบ 10 backbone ผ่าน; weights hashes และรายละเอียดอยู่ที่ `docs/evidence/pretrained-verification-10.json` ไม่ได้วัด accuracy หรือฝึกเต็ม
- **Training benchmark:** ResNet18 บน RTX 4060 Laptop GPU, batch 32, วัด 3 batches ต่อ frozen/unfrozen ใช้เฉพาะ train จาก snapshot ที่ยังไม่ผ่าน audit; `docs/evidence/training-benchmark-resnet18-unvalidated.json` ไม่ใช่ผลวิจัยหรือประมาณเวลาทั้ง suite
- **Clean installation:** ติดตั้ง package พร้อม data/detection/dev/reports ใน virtual environment ว่าง, `pip check`, artificial CPU smoke และ regression สามจุดจาก ZIP ที่แตกใหม่ 20 tests ผ่าน; รายละเอียดอยู่ `docs/evidence/clean-environment.json` environment ใหม่นี้เป็น CPU-only ส่วน CUDA ตรวจใน environment เดิม
- **Full screening / held-out test:** ยังไม่ได้เริ่ม screening และไม่ได้คำนวณ metrics บน test

## สิ่งที่ต้องแก้ก่อนเริ่ม

1. ตรวจ 240 คู่ train/validation ด้วยคน เลือกเฉพาะคู่ที่มีหลักฐานว่าเป็นภาพต้นฉบับหรือสินค้าชิ้นเดียวกัน แล้วสร้างกลุ่มรวมและ snapshot version ใหม่; dHash ไม่พิสูจน์ shared identity เอง
2. กำหนด protocol สำหรับ source test ที่มี unseen labels โดยไม่ใช้ test เลือกหรือปรับโมเดล และไม่สร้าง labels ใหม่จาก test เข้าสู่ train อย่างเงียบ ๆ
3. เตรียม snapshot ใหม่ให้ `prepare-data` และ `audit-data` ผ่าน โดยเก็บ exclusions, class mapping, source revision และ split hashes
4. วัด training benchmark สำหรับ backbone และ batch/device ที่จะใช้จริง จาก snapshot ที่ audit ผ่านก่อนประเมินเวลาและเริ่ม suite

หน้าตรวจคู่ภาพที่สร้างบนเครื่องอยู่ใน `artifacts/source-train-duplicate-review/index.html`; หน้านี้แสดงเฉพาะ train/validation และบันทึกคำตัดสินใน browser จนผู้ใช้ export CSV ไม่ได้ส่งข้อมูลไป server ไม่รวมภาพ dataset ใน ZIP
