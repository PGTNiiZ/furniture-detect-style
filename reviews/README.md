# ผลที่คนตรวจ (ไฟล์เล็ก ๆ ที่เข้า git ได้)

ใส่ CSV ที่ export จากหน้า review หรือที่กรอกเองในโฟลเดอร์ตามงาน ตั้งชื่อไฟล์ให้บอกว่าใครทำอะไร:

| โฟลเดอร์ | งาน | ไฟล์ต้นทาง (ไม่อยู่ใน git) | ตั้งชื่อไฟล์ |
|---|---|---|---|
| `near-duplicate/` | A1–A3 ตรวจภาพซ้ำ | `artifacts/source-train-cross-split-review/candidates-unreviewed.csv` | `A1-<ชื่อ>.csv` |
| `label-samples/` | D ตรวจ label ตัวอย่าง | `artifacts/zip-label-review-v1/samples.csv` | `D-<ชื่อ>.csv` |
| `room/` | C ภาพห้อง/กรอบ | `data/room-independent-review-v1/` | `C-<ชื่อ>.csv` |

กฎ

- ตัดสินด้วยคนเท่านั้น ห้ามให้ AI กรอกแทน
- ห้ามแก้ไฟล์ต้นทาง ให้คัดลอกมากรอกใหม่
- เมื่อเปิด PR ให้บอกจำนวนแถวที่ตรวจเสร็จ (เช่น 120/240) ไว้ใน PR
