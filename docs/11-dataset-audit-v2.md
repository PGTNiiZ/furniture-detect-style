# รายงาน audit ชุดข้อมูล v2

Snapshot `data/furniture-v2-conservative` ผ่าน **conservative training audit ตามเกณฑ์ที่กำหนด** โดยกักภาพที่เกี่ยวข้องกับ candidates ข้าม split ไว้ ยังไม่มีผู้ตรวจตัดสินภาพทั้ง 494 คู่ และการผ่าน audit ไม่รับประกันว่าไม่มีความสัมพันธ์ของภาพที่ dHash ตรวจไม่พบ ดู [dataset protocol](10-dataset-protocol-v2.md) สำหรับข้อตกลงฉบับเต็ม และ [metadata กับผล audit](evidence/furniture-v2-dataset-audit.json) สำหรับ counts รายคลาส, mapping, hashes และสถานะการตรวจ

## Snapshot และผลกระทบ

แหล่งข้อมูล `filnow/furniture-synthetic-dataset` revision `3582bc71ff1eec35e3283c899049c8a44c024d30`; manifest เก็บ source ID และ source split และไม่เขียนทับ snapshot เดิม

| Manifest | แหล่งข้อมูล | ก่อนกัก | กัก | คงไว้ | SHA-256 ของ manifest |
|---|---|---:|---:|---:|---|
| `train.csv` | source train | 6,330 | 374 | 5,956 | `0060336ce705d814bf3b447a05d4773325c0ee13cce452fe919af1ef0adfdbac` |
| `validation.csv` | source train | 1,436 | 141 | 1,295 | `e378d884101342d80a497e2bb8b4ac749852a330a3fa198d93f964b31267e88d` |
| `synthetic_test.csv` | source train | 1,234 | 107 | 1,127 | `bc83a11fc2edaa4492fcf10af76b44ae0e39244d911899cd9b3a7742b1122bd9` |
| `real_source_test.csv` | source test | 1,000 | — | 1,000 | `6fcfd767737783fafa0113993252a7d473e44c998d5e3c9ade521811182da701` |

กักรวม 622 จาก 9,000 ภาพ source train ไม่มี type หรือ style class ที่มีอยู่เดิมหายจาก split ใด แต่จำนวนรายคลาสเปลี่ยน ดู [ตารางผลกระทบรายคลาส](../artifacts/source-train-cross-split-review/quarantine-proposal-impact.csv) และ `class_counts` ใน metadata; [รายการภาพที่กัก](../data/furniture-v2-conservative/quarantine.csv) เก็บ ID และเหตุผลไว้ `val.csv` และ `test.csv` เป็นสำเนา byte-identical ของ validation และ synthetic test ตามลำดับ ไม่รวม real-source test

Class mapping สร้างจาก train เท่านั้น มี type 4 และ style 37 คลาส; validation และ synthetic test ไม่มี label ที่ไม่อยู่ใน mapping ส่วน real-source test เก็บ missing และ unseen เป็นข้อมูลอ้างอิงโดยไม่สร้าง training class index. ประวัติ exclusion อยู่ใน [source_exclusions.json](../data/furniture-v2-conservative/source_exclusions.json): source test `test-0000175` เดิมถูกคัดออกเพราะขาดทั้ง type และ style และถูกกู้กลับมาเป็น external reference ที่ยังคง missing label ทั้งสองรายการ

## Duplicate และ group audit

การสแกน source train เดิมด้วย 64-bit dHash, Hamming distance ≤4 ตรวจครบ 40,495,500 คู่ พบ candidates ทั้งหมด 1,031 คู่ ในจำนวนนี้ข้าม split 494 คู่: train/validation 240, train/synthetic test 196 และ validation/synthetic test 58. dHash เป็นเครื่องมือค้นหาคู่ที่ควรตรวจ ไม่ใช่คำตัดสินว่าเป็นภาพเดียวกันหรือเป็นสินค้าชิ้นเดียวกัน

นโยบายที่ใช้กับ snapshot นี้กักทั้ง connected component ที่แตะ candidate ข้าม split โดยสร้างกราฟจาก candidate edges และ group/exact links เดิม; การเชื่อม component เป็นมาตรการกัก ไม่ใช่การยืนยัน identity และอาจกักภาพที่ไม่เกี่ยวข้องกัน. [หลักฐาน candidate](../data/furniture-v2-conservative/candidate_evidence.csv), [group audit](../data/furniture-v2-conservative/group_audit.csv), [รายการ near-duplicate](../data/furniture-v2-conservative/near_duplicate_review.csv) และ [candidate audit](../data/furniture-v2-conservative/candidate_audit.json) เก็บไว้แยกจาก snapshot เดิม

หลังสร้าง snapshot สแกนภาพ source train ที่คงไว้ 8,378 ภาพอีกครั้งแบบ exhaustive รวม 35,091,253 คู่ พบ dHash candidates ภายใน split 123 คู่ และข้าม split 0 คู่; รายงานไม่ถูกตัดทอน ดู [ผลสแกนซ้ำ](evidence/furniture-v2-near-audit.json). Group และ exact pixel overlap ข้าม training splits เป็น 0; exact/group overlap ระหว่าง training splits กับ real-source reference เป็น 0 ตาม metadata. การสแกน dHash ซ้ำนี้ไม่รวม real-source test และไม่มี product/group ID จากแหล่งข้อมูลที่ยืนยันความเป็นสินค้าเดียวกัน จึงยังอาจมีความสัมพันธ์ที่เกณฑ์นี้ไม่พบ

## รายการพร้อมให้คนตรวจ

เปิด [หน้าตรวจภาพ 494 คู่](../artifacts/source-train-cross-split-review/index.html) ใน browser: แต่ละคู่แสดง source IDs, splits, labels, ระยะ dHash และภาพตัวอย่าง เปรียบเทียบภาพต้นฉบับตาม ID เมื่อภาพย่อไม่เพียงพอ; รายการเริ่มต้นอยู่ใน [CSV ที่ยังไม่ตัดสิน](../artifacts/source-train-cross-split-review/candidates-unreviewed.csv) โดย **ไม่มีการลงความเห็นแทนผู้ตรวจ**

เลือก `same` เมื่อมีหลักฐานว่าเป็นภาพต้นฉบับหรือวัตถุชิ้นเดียวกัน, `related` เมื่อมีหลักฐานว่าเป็นสินค้า/ต้นกำเนิดเดียวกัน เช่น คนละมุม, `different` เมื่อเป็นคนละรายการ และ `uncertain` เมื่อหลักฐานไม่พอ. `unreviewed` ยังไม่ใช่คำตัดสิน; ทั้ง `uncertain` และ `unreviewed` ต้องตรวจต่อหรือคงนโยบาย quarantine. หน้าเว็บเก็บสถานะใน browser จึงควร export CSV เป็นระยะและเก็บไฟล์ที่ export เป็นหลักฐาน ดู [คู่มือ review](../artifacts/source-train-cross-split-review/README.md)

หากมีผล review ภายหลัง ให้ใช้ผลเพื่อสร้าง **snapshot ใหม่** รวม `same`/`related` แบบ transitive จัดการทั้งกลุ่มโดยไม่แยกสมาชิกทีละภาพ และตรวจ group/exact/near-duplicate อีกครั้ง โดยเก็บ snapshot และหลักฐานรอบนี้ไว้

## External annotation และ protocol

Protocol ถูกกำหนดก่อนคำนวณ predictions: [external protocol](09-external-evaluation-protocol.md), [protocol ที่ตรึงกับ hashes](evidence/furniture-v2-external-protocol.json). จาก real-source 1,000 แถว ผลตรวจ labels เป็นดังนี้:

| งาน | Missing | Known | Unseen | Coverage ที่ eligible |
|---|---:|---:|---:|---:|
| type | 1 | 995 | 4 | 99.5% |
| style | 1 | 863 | 136 | 86.3% |

มี 859 ภาพที่ known ทั้งสองงาน (85.9%); joint correctness ในอนาคตใช้ 859 เป็น denominator ส่วนแต่ละงานใช้เฉพาะ known subset ของงานนั้น พร้อมรายงาน denominator และ coverage. แยก missing/unseen ออก ไม่อ้างว่าโมเดลรู้จัก unknown; matching ใช้ชื่อ label ตรงตัว ไม่มี canonicalization หรือ unknown-rejection threshold. หากจะออกแบบ threshold ต้องใช้ development data แยก และห้ามใช้ real-source test เลือกโมเดลหรือ threshold

[Dataset card](https://huggingface.co/datasets/filnow/furniture-synthetic-dataset) ระบุว่า source test annotations สร้างด้วย Qwen2 VL 72B และอาจไม่สอดคล้องกัน โดยไม่ได้ระบุว่าผ่าน human/expert verification; ผลในอนาคตจึงต้องระบุว่าเทียบกับ source annotations ไม่ใช่ expert ground truth. ยังไม่มี external predictions หรือ metrics และไม่รวมคะแนน synthetic test กับ real-source test เป็นคะแนนหลักเดียวกัน

## สถานะการทดลอง

Pretrained backward และ AdamW optimizer step ผ่าน 10 backbones บน CUDA ด้วย batch 2 จาก training snapshot นี้ ดู [หลักฐาน](evidence/furniture-v2-pretrained-steps.json); นี่เป็นการตรวจการทำงาน ไม่ใช่ผลประสิทธิภาพงานวิจัยหรือหลักฐานว่า batch 32 จะพอดีหน่วยความจำ. Benchmark ตาม batch ในแผนและสถานะ screening ให้ดูหลักฐานรอบล่าสุดแยกต่างหาก ไม่ตีความผลตรวจนี้เป็นคะแนน screening
