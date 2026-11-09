# Suite manifest และการรันต่อ

`furniture.suite.run_suite` บันทึก manifest ก่อนเริ่มฝึก และบันทึก atomic หลังเปลี่ยนสถานะทุก attempt โดยมี suite ID, resolved config และ digest ของแต่ละ run, output paths, สถานะ, เวลาและ error ของแต่ละ attempt ป้องกัน process สองตัวใช้ manifest เดียวกันด้วย OS lock ซึ่งระบบปล่อยเมื่อ process จบหรือเครื่องดับ

ตัวอย่างผ่าน Python API:

```python
from furniture.suite import plan_suite, run_suite

plans = plan_suite('configs/project_controlled.yaml', 'screening')
summary = run_suite(plans, manifest_path='artifacts/runs/suites/screening.json', fail_fast=False)
print(summary['manifest'], summary['counts'])

# ต่อรายการ planned และ retry รายการ failed/interrupted
summary = run_suite(manifest_path='artifacts/runs/suites/screening.json', retry_failed=True)

# ตรวจและกู้ run ที่ยังแสดง running หลังเครื่องดับ; หากไม่มี checkpoint เริ่ม attempt ใหม่
summary = run_suite(manifest_path='artifacts/runs/suites/screening.json', recover_running=True, retry_failed=True)

# เมื่อ failed checkpoint ใช้ resume ไม่ได้: เริ่มใหม่โดยเก็บ attempt เดิม
summary = run_suite(manifest_path='artifacts/runs/suites/screening.json', restart_failed=True)
```

ค่าเริ่มต้น `fail_fast=False` บันทึก failure แล้วทำ run ถัดไป; `fail_fast=True` หยุดเมื่อเกิด exception โดย manifest เก็บ run ที่ยังไม่ได้เริ่มไว้ ไม่ retry failure อัตโนมัติเว้นแต่สั่ง `retry_failed=True`. หากมี `last.pt` จะพยายาม resume ด้วย provenance/config/data/environment guards ของ `resume_run`; ถ้าไม่มี checkpoint จะสร้าง attempt ใหม่ใน directory ใหม่และเก็บ failure เดิม การกู้ `running` ต้องสั่ง `recover_running=True` และผ่านการตรวจ process และ checkpoint ของ recovery workflow ก่อนเสมอ

ถ้า `last.pt` เสียหรือ resume ไม่ผ่าน ต้องใช้ `restart_failed=True` เพื่อสร้าง attempt ใหม่อย่างชัดเจน โดยไม่ลบหรือเขียนทับ attempt เดิม flag นี้ไม่ข้ามการตรวจ owner ของ run ที่ยังมีสถานะ running กรณี running โดยไม่มี checkpoint ต้องระบุทั้ง `recover_running=True` และ `retry_failed=True` หรือ `restart_failed=True` เพื่อให้ตรวจ owner/process/provenance และบันทึก abandonment event ก่อนสร้าง directory ใหม่ กรณีไม่มี status/owner ให้หยุดเพื่อตรวจด้วยคน

Manifest version 2 ตรึง dataset config, root และ SHA-256 ของไฟล์ split train/val/test ตอนเริ่ม และตรวจซ้ำก่อน run ทุกรายการและเมื่อกลับมา continue หากไฟล์เปลี่ยนจะหยุด suite ก่อนฝึกเพิ่มเติม; การคำนวณ hash ไม่ประเมินหรือเลือกโมเดลด้วยผล test. Manifest version 1 ที่ไม่มี hashes ไม่สามารถ continue ได้อย่างปลอดภัย ให้เก็บไว้ตรวจสอบและสร้าง suite ใหม่หลังตรึง dataset

เมื่อกลับมาทำต่อ run ที่ completed จะไม่ฝึกซ้ำ: suite ตรวจ status, protocol marker, config/provenance digest, validation JSON และ SHA-256 ของ checkpoint ก่อนยอมรับเป็น completed หากหลักฐานไม่ตรงจะบันทึก error แทนการเขียนทับ run เดิม การแก้ไข config ใน manifest ทำให้ digest ไม่ตรงและถูกปฏิเสธ หากต้องเปลี่ยน protocol ให้สร้าง suite ใหม่เพื่อรักษาการเปรียบเทียบและหลักฐาน run เดิม

Manifest ไม่ยืนยันว่า dataset ผ่าน audit หรือฝึกได้ตรงเวลา ต้องตรวจข้อมูล, pretrained weights และ benchmark ตาม protocol ก่อนเริ่ม full screening ผล test ไม่ใช้เลือกโมเดล

คำสั่ง PowerShell ที่เทียบกับ API ด้านบน (แทน `PATH_TO_SUITE` ด้วย manifest ที่สร้างจริง):

```powershell
.venv\Scripts\python.exe -m furniture.cli suite --continue-manifest PATH_TO_SUITE --retry-failed
.venv\Scripts\python.exe -m furniture.cli suite --continue-manifest PATH_TO_SUITE --recover-running --retry-failed
.venv\Scripts\python.exe -m furniture.cli suite --continue-manifest PATH_TO_SUITE --restart-failed
```

CLI คืน exit code ไม่เป็นศูนย์หาก suite ยังมีรายการ failed, interrupted, running หรือ planned แม้จะตั้งให้ดำเนินการรายการถัดไปต่อแล้ว ตรวจ `counts` และ error ของแต่ละ attempt ใน manifest ประกอบ
