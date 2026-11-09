# การประเมินความทนต่อการบดบัง

`furniture.robustness.evaluate_robustness` เป็น evaluator แยกจาก augmentation ที่ใช้ฝึก ใช้ภาพ split เดิมหลัง resize/pad เป็นสี่เหลี่ยมและสร้าง clean, light, strong โดยบัง **ทุกภาพ** ใน light/strong ทั้งสองระดับใช้จำนวนและขนาดช่องตาม protocol แต่ไม่สุ่มว่าจะบังหรือไม่ จากนั้น normalize ตามค่า ImageNet เดียวกับ classifier

Manifest เก็บ sample ID, pixel hash, split hash, seed, ขนาดภาพ, สี RGB `(127,127,127)` และตำแหน่งช่องในพิกัดภาพหลัง resize/pad พิกัด `x1,y1,x2,y2` มีขอบขวาและล่างแบบไม่รวมตำแหน่งนั้น แต่ละ sample ใช้ seed ที่คำนวณจาก seed หลัก, sample ID และระดับ จึงไม่ขึ้นกับลำดับใน CSV

ส่ง manifest ที่สร้างครั้งแรกให้ evaluator ของโมเดลอื่นเพื่อยืนยันว่าทั้งสองเห็นภาพบังแบบเดียวกัน ระบบจะปฏิเสธหาก split, image size, seed, hashes หรือพิกัดไม่ตรง Manifest พร้อม checkpoint ที่มี class mapping และ split hashes เป็นข้อมูลจำเป็นสำหรับการตรวจย้อนหลัง

ผลใน output directory ใหม่ประกอบด้วย `occlusion_manifest.json`, `robustness.json` และ `predictions_clean.csv`, `predictions_light.csv`, `predictions_strong.csv` โดยรายงาน accuracy, macro precision/recall/F1, weighted F1, confusion matrix, class support และคะแนนที่ลดลงจาก clean แยกแต่ละ task `confidence` เป็น softmax score ไม่ใช่ความน่าจะเป็นที่ผ่าน calibration

API สำหรับเรียกจาก Python:

```python
from furniture.robustness import evaluate_robustness, compare_robustness

first = evaluate_robustness('data/furniture-v1', 'RUN_A/best.pt', 'artifacts/robust-a', split='val', device='cpu')
second = evaluate_robustness('data/furniture-v1', 'RUN_B/best.pt', 'artifacts/robust-b', device='cpu',
                             split='val', manifest='artifacts/robust-a/occlusion_manifest.json')
compare_robustness(['artifacts/robust-a', 'artifacts/robust-b'], 'artifacts/robust-comparison.csv',
                   vary=['backbone'])
```

ผลเป็นภาพแปลงจาก split เดิม ไม่ใช่ข้อมูล test ชุดอิสระ ตำแหน่ง/ขนาดช่องวัดบนภาพหลัง resize/pad ไม่สามารถบอกสัดส่วนวัตถุที่ถูกบังจริงหากไม่มี object masks การเปรียบเทียบตรวจ split hashes, class mapping, manifest, controls และสถานะ completed พร้อม protocol marker หลังแก้ P0 หากเป็น checkpoint เก่าหรือไม่มีสถานะยังประเมินแยกได้ แต่ไม่รวมในตารางเปรียบเทียบ ต้องตรวจแกนการทดลองก่อนสรุปผลเสมอ

ค่าเริ่มต้นของ evaluator คือ `val` เพื่อไม่เปิด test ระหว่างเลือกโมเดล จะใช้ `split='test'` เฉพาะเมื่อเลือกโมเดลและตรึง protocol แล้ว ผลที่เปรียบเทียบต้องใช้ `best.pt` ที่เลือกจาก validation; evaluator ตรวจ SHA-256 กับ `checkpoint.sha256`, ตรวจ epoch กับ `summary.json`, ตรวจ config กับ provenance และ settings ใน checkpoint รวมถึงตรวจ hash ของ source snapshot ก่อนบันทึก evidence สำหรับ comparison checkpoint แบบ standalone หรือ `last.pt` ยังประเมินแยกได้ แต่ไม่มี evidence ที่ใช้จัดตารางเปรียบเทียบ

`compare_robustness` กำหนดให้ controls ทุกตัวตรงกันโดยปริยาย ให้ระบุ `vary` เฉพาะแกนที่ protocol อนุญาต: `backbone`, `weights`, `augmentation`, `occlusion`, `training_mode`, `tasks`, `seed`, `device` การใช้ `backbone` อนุญาต weights คนละ enum ข้าม backbone แต่หากเป็น backbone เดียวกันและ weights ต่างกัน ต้องระบุ `weights` เพิ่ม การระบุ `augmentation` ไม่อนุญาตให้ occlusion ต่าง เว้นแต่ระบุ `occlusion` ด้วย

ไม่ว่าเลือกแกนใด ระบบยังบังคับ identity/revision ของข้อมูล, split hashes และ occlusion manifest ตรงกัน โดยไม่บังคับ path ของสำเนา dataset/source ให้เหมือนกัน; เมื่อ `tasks` ต่าง จะตรวจ mapping ของ tasks ที่มีร่วมกันและรายงานแยก task ไม่มีการจัดอันดับเปรียบเทียบค่าเฉลี่ยต่างชุด tasks ผล `robustness.json` รุ่นเก่าที่ไม่ได้บันทึก controls ครบต้องรัน evaluator ใหม่จาก checkpoint ก่อนเปรียบเทียบ และการเทียบ validation ไม่ใช้ test เพื่อเลือกโมเดล

Device control ตรวจชื่อที่ run เลือก เช่น `cuda` หรือ `cpu`; ยังไม่ตรวจว่า GPU รุ่น/driver และ package versions ของทุก run ตรงกัน ผู้ทดลองต้องตรวจ `environment.json` และ `environment.txt` เมื่อ protocol กำหนดให้ใช้ environment เดียวกัน
