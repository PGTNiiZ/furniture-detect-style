# วิธีทำงานร่วมกัน (อ่านอันนี้ก่อนทำอะไร)

## หลักการสั้น ๆ

| Branch | ใช้ทำอะไร | ใครแตะได้ |
|---|---|---|
| `main` | เวอร์ชันที่นิ่งและส่งอาจารย์ได้ (เหมือน production) | **เจ้าของ repo เท่านั้นที่ merge ได้** ห้าม push ตรง |
| `develop` | รวมงานของทุกคน (ทดสอบรวม) | ทุกคน merge ได้ผ่าน Pull Request ห้าม push ตรง |
| `feature/...` `data/...` `exp/...` `docs/...` | งานของแต่ละคน | ของคนนั้นคนเดียว |

เส้นทางของงาน: `develop` → แตก branch ของตัวเอง → ทำงาน → เปิด Pull Request เข้า `develop` → เมื่อ develop นิ่งแล้ว เจ้าของ repo เปิด PR จาก `develop` เข้า `main`

## ขั้นตอนทีละข้อ

```bash
# ครั้งแรก
git clone <URL ของ repo>
cd furniture-project2

# ทุกครั้งที่เริ่มงานใหม่
git checkout develop
git pull
git checkout -b data/A1-review-train-val   # ตั้งชื่อตามตารางด้านล่าง

# ทำงาน แล้ว commit บ่อย ๆ
git add <ไฟล์ที่แก้>
git commit -m "A1: ตรวจคู่ภาพ 1-120 เสร็จ"
git push -u origin data/A1-review-train-val
```

จากนั้นไปที่หน้า repo บน GitHub กด **Compare & pull request** แล้วเลือก base เป็น `develop` (ไม่ใช่ `main`) ใส่ชื่อ PR ตามแม่แบบ

## ตั้งชื่อ branch

| ประเภทงาน | รูปแบบ | ตัวอย่าง |
|---|---|---|
| ตรวจ/แก้ข้อมูล, label | `data/<รหัสงาน>-<อธิบายสั้น>` | `data/A1-review-train-val` |
| แก้โค้ด/เพิ่มฟีเจอร์ | `feature/<อธิบายสั้น>` | `feature/style-merge-config` |
| การทดลองเทรน | `exp/<ชื่อ>-<ย่อ>` | `exp/nong-vit-c6` |
| เอกสาร/รายงาน | `docs/<อธิบายสั้น>` | `docs/report-draft` |
| แก้บั๊ก | `fix/<อธิบายสั้น>` | `fix/loader-path` |

รหัสงาน A–F ดูได้ในเอกสารคู่มือทีม (ลิงก์อยู่ใน README)

## สิ่งที่ห้ามอัปโหลดเข้า git

- ภาพ dataset (`data/`), checkpoint (`*.pt`), ผลเทรนทั้งโฟลเดอร์ (`artifacts/`), ไฟล์ `.zip` : ใหญ่เกิน และ `.gitignore` กันไว้แล้ว ให้ส่งผ่าน Google Drive แทน แล้วโพสต์ลิงก์กับ SHA-256 ใน PR
- รหัสผ่าน/token ใด ๆ (`.env` ถูกกันไว้ แต่ตรวจก่อน commit ทุกครั้ง)
- ผลตัดสินจากคนที่เป็นไฟล์เล็ก (CSV) ใส่ใน `reviews/` ได้ ดู `reviews/README.md`

## ก่อนเปิด PR

1. รัน `python -m pytest -q` ถ้าแก้โค้ดใน `src/` หรือ `tests/`
2. เขียนใน PR ว่า "ทำอะไร, ผลเป็นอย่างไร, ตัวเลขมาจากไฟล์ไหน" ตามแม่แบบที่ขึ้นมาให้
3. ถ้าเป็นการเทรน ต้องบอกด้วยว่าเลือกโมเดลจาก validation เท่านั้น (ห้ามใช้ test เลือก) ดู `docs/22-training-and-model-guide.md`

## สำหรับเจ้าของ repo: ตั้งค่า GitHub ครั้งเดียว

ไปที่ Settings → Branches → Add branch ruleset (หรือ Add classic branch protection rule)

**`main`**
- Require a pull request before merging, เปิด "Require approvals" = 1
- Restrict who can push / Restrict updates ให้เหลือเฉพาะเจ้าของ (กันคนอื่น merge เข้า main)
- Block force pushes และ Restrict deletions

**`develop`**
- Require a pull request before merging (ไม่ต้อง approve ถ้าอยากให้เร็ว หรือ 1 ถ้าอยากให้เพื่อนช่วยดูกัน)
- Block force pushes

หลัง workflow `CI` ผ่านครั้งแรกบน GitHub แล้ว ค่อยเปิด "Require status checks to pass" และเลือก `test` ไม่ควรเปิดก่อนเห็นว่าผ่านจริง เพราะ CI ที่เขียนไว้ยังไม่เคยรันบน GitHub

เชิญเพื่อนที่ Settings → Collaborators (สิทธิ์ Write) เพื่อให้แตก branch และเปิด PR ได้
