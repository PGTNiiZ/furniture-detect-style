# Dataset protocol v2: controlled screening และ external reference

อ้างอิง `filnow/furniture-synthetic-dataset` revision `3582bc71ff1eec35e3283c899049c8a44c024d30`; snapshot ใหม่อยู่ `data/furniture-v2-conservative` และรักษา intake/หลักฐานเดิมไว้ ตารางและ hashes ที่ส่งพร้อมโค้ดอยู่ `docs/evidence/furniture-v2-dataset-audit.json` ภาพข้อมูลจริงไม่รวมใน ZIP

| Manifest | แหล่ง | จำนวน | หน้าที่ |
|---|---|---:|---|
| `train.csv` | source train สังเคราะห์ | 5,956 | ฝึกและสร้าง mapping เท่านั้น |
| `validation.csv` | source train สังเคราะห์ | 1,295 | เลือกโมเดลและ protocol |
| `synthetic_test.csv` | source train สังเคราะห์ | 1,127 | held-out synthetic test หลังตรึง protocol |
| `real_source_test.csv` | source test ภาพถ่าย | 1,000 | external reference หลังเลือกโมเดล ไม่ใช้เลือกโมเดลหรือ threshold |

`val.csv` และ `test.csv` เป็นสำเนา byte-identical ของ `validation.csv` และ `synthetic_test.csv` ตามลำดับเพื่อรองรับโค้ดเดิม; `test.csv` **ไม่รวม real-source test**. เก็บ source ID, source split, pixel hash และ dataset revision โดยไม่เพิ่ม real-source rows ใน train. Source test แถว `test-0000175` ขาดทั้ง type/style ถูกกู้ภาพและเก็บเป็น missing reference พร้อมบันทึก exclusion เดิม ไม่สร้าง labels ทดแทน

## Quarantine ที่ใช้ในรอบนี้

ก่อนใช้ได้เสนอทางเลือกไว้: กักทุกภาพใน connected component ของกราฟ dHash candidates ที่เชื่อมข้าม training splits โดยไม่ถือว่าทุกคู่เป็นภาพเดียวกัน กราฟใช้ threshold Hamming ≤4 และรวม edges เดิมกับ exact pixel/group เพื่อไม่ให้แตกกลุ่ม กัก train 374 จาก 6,330, validation 141 จาก 1,436 และ synthetic test 107 จาก 1,234 รวม 622 จาก 9,000 ภาพสังเคราะห์ ไม่มี type/style class หายจาก split เดิม แต่จำนวนแต่ละคลาสเปลี่ยน; ดู `class_counts`, `quarantine.csv` และ impact report

ตรวจ dHash ซ้ำแบบ exhaustive บน 8,378 ภาพที่คงไว้ ครบ 35,091,253 คู่ พบ candidates อยู่ภายใน split 123 คู่ และข้าม split 0 คู่ `docs/evidence/furniture-v2-near-audit.json` Group/exact hashes ข้าม training splits และ exact/group ระหว่าง training กับ external reference เป็นศูนย์ตาม audit. dHash ≤4 ไม่ครอบคลุมภาพที่คล้ายกันทุกชนิด และ source ไม่มี product/group IDs ที่ยืนยันได้ การผ่านนี้จึงเป็น **conservative audit ตาม threshold** ไม่ใช่หลักฐานว่าไม่มี leakage ทุกชนิดหรือว่ามีคนตรวจแล้ว

หน้า `artifacts/source-train-cross-split-review/index.html` พร้อมภาพตัวอย่างแสดงครบ 494 candidates เดิม แบ่ง train/validation 240, train/synthetic test 196, validation/synthetic test 58 พร้อม decision same/related/different/uncertain. ยังไม่มี human decisions; quarantine ทั้งหมดคงเป็น unresolved evidence และสามารถใช้ผล review สร้าง snapshot version ใหม่ โดย `same`/`related` รวมกลุ่มแบบ transitive และตรวจ group disjoint ซ้ำ ไม่แก้ snapshot นี้

## External protocol ก่อน prediction

Protocol ถูกสร้างจาก train และ real-source manifests ก่อน inference ที่ `docs/evidence/furniture-v2-external-protocol.json`: exact label matching ไม่ canonicalize และไม่ตั้ง unknown-rejection threshold. Source annotations: type known 995, missing 1, unseen 4; style known 863, missing 1, unseen 136; รู้จักทั้งสองงาน 859 จาก 1,000. การประเมินอนาคตต้องรายงาน metric แยก eligible subset ต่อ task พร้อม coverage และ joint correctness เฉพาะ 859 ที่รู้จักทั้งสอง task พร้อม denominator; unseen/missing แยกไว้ ไม่แปลว่าโมเดลตรวจจับ unknown ได้

ตาม [dataset card](https://huggingface.co/datasets/filnow/furniture-synthetic-dataset) ผู้เผยแพร่ระบุว่า test labels มาจาก Qwen2 VL 72B และอาจไม่สอดคล้องกัน; card ไม่ระบุว่ามี human/expert verification ผลเปรียบเทียบจึงเป็นผลเทียบ source annotations ไม่ใช่ expert ground truth. ยังไม่รัน external predictions หรือ metrics และไม่ใช้ external test เลือกโมเดลหรือ threshold

## คำสั่งตรวจ

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_snapshot.py tests/test_external_evaluation.py tests/test_full_duplicate_review.py
.venv\Scripts\python.exe scripts/audit_near_duplicates.py --data data/furniture-v2-conservative --output artifacts/NEW_AUDIT_PREFIX --threshold 4
```

Pretrained backward และ AdamW step ผ่านทั้ง 10 backbones บน CUDA โดยใช้ train batch 2 จาก snapshot ที่ audit แล้ว; ผลจริงอยู่ `docs/evidence/furniture-v2-pretrained-steps.json` การทดสอบนี้ไม่วัด accuracy และไม่ได้ยืนยันว่า batch 32 จะพอดี GPU ต้องดู training benchmark แยกก่อน screening
