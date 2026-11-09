# คู่มือการเทรนและการเลือกโมเดล (สำหรับเพื่อนในทีม)

เขียนวันที่ 2026-11-09 จากไฟล์ config และผลใน `artifacts/` ตัวเลขทุกตัวมาจาก `artifacts/model-only-release/validation-report/` และ `screening-selection.json`

## 1. เทรนไปแล้วแค่ไหน และข้อมูลใหญ่แค่ไหน

เทรนครบทุกขั้นตามแผนแล้ว แต่ข้อมูล **ไม่ได้ใหญ่**: train 5,956 ภาพ, validation 1,295 ภาพ (ภาพสังเคราะห์ 4 ประเภท 37 สไตล์) ไม่ได้เทรนโมเดลจากศูนย์ แต่ใช้ **transfer learning**: เริ่มจากน้ำหนัก ImageNet ที่ torchvision เตรียมไว้ แล้ว fine-tune ด้วยข้อมูลของเรา

| ขั้น | ทำอะไร | จำนวน run |
|---|---|---:|
| Screening | 10 backbones, เงื่อนไข C3, seed 42, วัดบน validation | 10 |
| Controlled | 3 backbones ที่ดีสุดจาก screening × เงื่อนไข C1–C8 | 24 |
| Single-task | DenseNet121 เทรน type อย่างเดียว / style อย่างเดียว เทียบกับเทรนคู่ | 2 |
| Repeated seeds | DenseNet121 เงื่อนไข C1, C2, C3, C5 เพิ่ม seed 123 และ 2026 | 8 |

เลือก checkpoint สุดท้ายจาก validation ก่อนเปิด test ทั้งสองชุด

## 2. ทำไมถึงเลือก DenseNet121 และทำไมไม่ใช่ตัวอื่น

backbone ที่ลองครบ 10 ตัว ได้แก่ resnet18, resnet50, densenet121, efficientnet_b0, efficientnet_b2, mobilenet_v3_large, regnet_y_400mf, convnext_tiny, **swin_t** และ **vit_b_16** (ตัวหลังสองตัวคือโมเดลแบบ Transformer ที่เพื่อนอาจจำชื่อว่า ViT/Swin) ไม่ได้ลอง LeNet เพราะเป็นโมเดลเล็กรุ่นเก่าที่ไม่มีน้ำหนักฝึกมาก่อน

ผลคือ **ConvNeXt-tiny, ResNet50 และ DenseNet121 ได้คะแนนใกล้กันมาก** (คะแนน = ค่าเฉลี่ย macro-F1 ของ type กับ style บน validation)

| ขั้น | convnext_tiny | resnet50 | densenet121 |
|---|---:|---:|---:|
| Screening (C3) | 0.6136 | 0.6092 | 0.6086 |
| Controlled: ดีสุดของ backbone นั้น | 0.6157 (C5) | 0.6116 (C4) | **0.6177 (C6)** |

DenseNet121 + C6 ชนะที่ขั้น controlled ด้วยคะแนนนำ ConvNeXt-tiny แค่ **0.002** ซึ่งเล็กกว่าความแกว่งระหว่าง seed (ส่วนเบี่ยงเบนมาตรฐานของ style macro-F1 ระหว่าง seed ของ DenseNet121 อยู่ที่ 0.003–0.012) เพราะฉะนั้น พูดได้เพียงว่า **DenseNet121 ได้คะแนนสูงสุดตามกติกาที่ตั้งไว้ ไม่ได้พิสูจน์ว่าดีกว่า ConvNeXt-tiny หรือ ResNet50** ข้อจำกัดที่ต้องเขียนในรายงาน

- seed ซ้ำทำเฉพาะ DenseNet121 ส่วน ConvNeXt-tiny กับ ResNet50 ยังมี seed เดียว (42) การเทียบข้าม backbone จึงยังไม่ควบคุม seed
- type ได้ ~0.99 ในทุก backbone ดังนั้นคะแนนรวมถูกตัดสินด้วย style ที่ต่ำและแกว่ง (~0.22–0.24) ทำให้อันดับขึ้นกับสัญญาณรบกวนมาก
- screening ใช้ run สั้น อาจเข้าข้างโมเดลที่ลู่เข้าเร็ว (บันทึกไว้ใน `screening-selection.json`)

## 3. backbone ถูกใช้ยังไง

```text
ภาพ 224×224 → backbone (ร่วม) → head type  → 4 คลาส
                              → head style → 37 คลาส
loss = 1.0 × CE(type) + 1.0 × CE(style)
```

- รูปทั้งหมด: แปลงเป็น RGB, ย่อโดยรักษาสัดส่วนแล้วเติมขอบสีขาวให้เป็นสี่เหลี่ยมจัตุรัส, normalize แบบ ImageNet ชุดเดียวกันทุก backbone (เพื่อเทียบกันได้ ไม่ใช่สูตรทางการของแต่ละ weight)
- การตั้งค่าเทรน (`configs/training/controlled.yaml`): batch 32, สูงสุด 30 epoch, early stopping เมื่อไม่ดีขึ้น 7 epoch, optimizer AdamW, weight decay 1e-4, ลด learning rate ครึ่งหนึ่งเมื่อ validation ไม่ดีขึ้นต่อเนื่อง 2 epoch, เลือก checkpoint จากค่าเฉลี่ย macro-F1 ของ type และ style
- **2 epoch แรกแช่แข็ง backbone** เทรนเฉพาะ head (learning rate 1e-3) จากนั้นปลดแช่แข็งแล้ว fine-tune ทั้งโมเดล (backbone 1e-4, head 1e-3)
- เปลี่ยน backbone ได้โดยแก้บรรทัด `model:` ใน project config ให้ชี้ `configs/models/<ชื่อ>.yaml`

### เงื่อนไข C1–C8 คืออะไร

| รหัส | augmentation | บังบางส่วนของภาพ (occlusion) | ปลดแช่แข็ง backbone |
|---|---|---|---|
| C1 | ไม่มี | ไม่มี | ไม่เคย (เทรนแค่ head) |
| C2 | ไม่มี | ไม่มี | หลัง 2 epoch |
| C3 | easy | ไม่มี | หลัง 2 epoch |
| C4 | hard | ไม่มี | หลัง 2 epoch |
| C5 | easy | light (โอกาส 25%) | หลัง 2 epoch |
| C6 | easy | strong (โอกาส 50%) | หลัง 2 epoch |
| C7 | hard | light | หลัง 2 epoch |
| C8 | hard | strong | หลัง 2 epoch |

ผลที่เห็นได้ชัด: C1 (ไม่ fine-tune backbone) แย่ที่สุดในทุก backbone ส่วน C2–C8 ต่างกันเล็กน้อย

## 4. ใครอยากเทรนเพิ่ม ควรทำอะไร

**ไม่ต้องเทรนซ้ำสิ่งที่เสร็จแล้ว** และห้ามรัน `suite` กับ output เดิมเพราะจะทำให้ผลที่ตรึงไว้ปนกัน ถ้าจะเทรนให้ทำเป็นการทดลองใหม่ที่มีเหตุผลรองรับ เช่น

| ไอเดีย | ตอบคำถามอะไร | ข้อควรระวัง |
|---|---|---|
| รัน ConvNeXt-tiny และ ResNet50 ด้วย seed 123, 2026 (เงื่อนไข C5/C6) | DenseNet121 ดีกว่าจริงไหม หรือแค่ seed | ใช้ config/split เดิมเป๊ะ |
| เทรนซ้ำหลังตัดสินงาน B (รวมชื่อสไตล์) หรือหลังตรวจภาพซ้ำงาน A | สไตล์ดีขึ้นไหมเมื่อ label สะอาดขึ้น | เปลี่ยน dataset ต้องสร้าง snapshot ใหม่ ขอให้เจ้าของ repo ทำก่อน |
| ViT / Swin เทรนนานขึ้นหรือ learning rate ต่ำลง | Transformer ถูกเข้าข้างหรือเสียเปรียบเพราะ run สั้นไหม | บันทึกค่าที่เปลี่ยนใน PR |

### วิธีรัน (PowerShell, ที่ root ของโปรเจกต์)

```powershell
# 1) คัดลอก project config แล้วเปลี่ยน backbone
copy configs\project_v2_conservative.yaml configs\project_exp_<ชื่อ>.yaml
#    แก้บรรทัด  model: models/resnet18.yaml  เป็น  model: models/vit_b_16.yaml

# 2) ตรวจเครื่องและ config
.venv\Scripts\python.exe -m furniture.cli check-device
.venv\Scripts\python.exe -m furniture.cli check-config --config configs\project_exp_<ชื่อ>.yaml --experiment C6

# 3) เทรน เก็บผลในโฟลเดอร์ใหม่ที่ไม่ซ้ำของเดิม
.venv\Scripts\python.exe -m furniture.cli train --config configs\project_exp_<ชื่อ>.yaml --experiment C6 --run-dir artifacts\runs\friends\<ชื่อ>-vit_b_16-C6
```

เครื่องที่ใช้เทรนก่อนหน้า: RTX 4060 Laptop (ฝึกได้ราว 85–97 ภาพ/วินาทีสำหรับ ResNet18 วัดแบบสั้น ๆ อย่าใช้เป็นค่าประมาณเวลาของโมเดลอื่น)

### กติกาของการทดลอง

1. เลือกโมเดลจาก **validation** เท่านั้น ห้ามดูคะแนน test แล้วกลับมาเลือก
2. ไม่แก้ไฟล์ split/manifest ใน `data/furniture-v2-conservative` (มี hash ตรึงไว้)
3. ไม่แก้ run เดิมใน `artifacts/runs/furniture-v2-conservative`
4. ผลและ checkpoint ขนาดใหญ่ส่งผ่าน Drive พร้อม SHA-256 อย่า commit เข้า git (ดู `CONTRIBUTING.md`)
5. ในรายงานให้เขียนตรง ๆ ว่า style ยังต่ำ ดูตารางผลในคู่มือทีม

> หมายเหตุ: ชุดคำสั่งในหัวข้อ "วิธีรัน" เขียนจาก `src/furniture/cli.py` และ README ยังไม่ได้ทดลองรันตามลำดับบนเครื่องใหม่ ถ้าติดปัญหาให้แจ้งเจ้าของ repo
