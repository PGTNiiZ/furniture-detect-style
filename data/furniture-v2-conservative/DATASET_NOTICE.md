# furniture-v2-conservative — ที่มาและเงื่อนไข

ชุดนี้เป็นงานดัดแปลง (แบ่ง split ใหม่ กัก quarantine และทำ manifest) จาก dataset สาธารณะ

| รายการ | ค่า |
|---|---|
| ต้นทาง | [filnow/furniture-synthetic-dataset](https://huggingface.co/datasets/filnow/furniture-synthetic-dataset) บน Hugging Face |
| Revision ที่ตรึง | `3582bc71ff1eec35e3283c899049c8a44c024d30` |
| สิทธิ์ใช้งาน | Apache-2.0 ตามที่ผู้เผยแพร่ระบุใน dataset card (ตรวจเมื่อ 2026-11-02) ข้อความสัญญาอนุญาตเต็ม: <https://www.apache.org/licenses/LICENSE-2.0> |
| ภาพ train/validation/synthetic test | สร้างด้วย Stable Diffusion (ตามที่ผู้เผยแพร่ระบุ) |
| ภาพ real-source test | ภาพถ่ายจริง 1,000 ภาพ label ติดด้วย Qwen2 VL ผู้เผยแพร่เตือนว่าอาจผิดและ **ไม่ใช่ label จากผู้เชี่ยวชาญ** |

## สิ่งที่เราเปลี่ยน

- แบ่ง train 5,956 / validation 1,295 / synthetic test 1,127 จากส่วน train ต้นทาง และเก็บ test ต้นทาง 1,000 ภาพเป็น real-source test (ใช้อ้างอิงอย่างเดียว ไม่ใช้ฝึกหรือเลือกโมเดล)
- กันภาพที่น่าสงสัยว่าซ้ำข้าม split ออก 622 ภาพ (`quarantine.csv`) ด้วยกราฟของ dHash ไม่ได้ยืนยันว่าเป็นภาพเดียวกัน
- ไม่แก้ pixel ของภาพ ไม่แก้ label ต้นทาง (แถว `test-0000175` ที่ขาด label ถูกเก็บเป็นข้อมูลอ้างอิงที่ไม่มี label)

## ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | ใน git | หมายเหตุ |
|---|---|---|
| `train.csv`, `validation.csv`, `synthetic_test.csv`, `real_source_test.csv` | ใช่ | manifest ของแต่ละ split (`val.csv`/`test.csv` เป็นสำเนาเหมือนกันทุก byte ของ validation/synthetic_test) |
| `quarantine.csv`, `group_audit.csv`, `candidate_evidence.csv`, `near_duplicate_review.csv`, `*.json` | ใช่ | หลักฐานการตรวจและการกัน |
| `images/*.png` (9,378 ภาพ, 1.4 GB) | **ไม่** | ใหญ่เกินจะเก็บใน git ดาวน์โหลดจาก Release ดูด้านล่าง |

## วิธีได้ภาพมาใช้

ดาวน์โหลด `furniture-v2-conservative-dataset.zip` จากหน้า [Releases](https://github.com/PGTNiiZ/furniture-detect-style/releases) ของ repo นี้ แล้วแตกไฟล์ลงที่ `data/` ให้ได้ `data/furniture-v2-conservative/images/*.png` ตรวจความถูกต้องด้วย SHA-256 ที่ระบุใน Release และตรวจ `pixel_hash` ของแต่ละแถวใน manifest ได้ด้วย `audit-data`

ถ้าไม่มี Release ให้ดึงต้นฉบับจาก Hugging Face ที่ revision ข้างบนเอง (ต้องแปลงเป็นโครงสร้างนี้ด้วย `prepare-data`; การจัด split และ quarantine ต้องใช้ manifest ในโฟลเดอร์นี้ ไม่สร้างซ้ำอัตโนมัติ)
