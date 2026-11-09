# หลักฐานตรวจแหล่งข้อมูลและ preview

ตรวจเมื่อ 2 พฤศจิกายน 2569 จาก [dataset card](https://huggingface.co/datasets/filnow/furniture-synthetic-dataset): ผู้เผยแพร่ระบุ Apache-2.0, จำนวนทั้งหมด 10,000 ภาพ แบ่ง source train 9,000 ภาพสร้างด้วย Stable Diffusion และ source test 1,000 ภาพถ่ายจริงพร้อม annotations โดย Qwen2 VL; card เตือนว่าป้ายของ test อาจผิดและมีเพียง 4 ประเภท furniture ข้อมูลเป็นคำกล่าวของผู้เผยแพร่ ไม่ใช่ผลการนับจากการดาวน์โหลดทั้งชุด ค่า style เป็น annotation ไม่ใช่การยืนยันจากผู้เชี่ยวชาญ

ตัวอย่าง source train ใน viewer มีคอลัมน์ `image`, `type`, `style` และ `prompt` ซึ่งอาจมี label อยู่ในข้อความ; โค้ด classifier อ่านเฉพาะภาพและ labels จาก manifest ไม่ใช้ prompt เป็น input

ตรวจ local `data/furniture-preview-diverse/train.csv` กับภาพทั้ง 45 ไฟล์ที่มีจริง พบเปิดอ่านได้ 45/45, label type/style ว่าง 0, exact pixel-hash duplicate groups 0; ประเภท bed 15, sofa 10, chair 10, table 10; มี 26 style labels ตัวอย่าง `minimal`/`minimalist` และ `classic`/`classical` ต้องให้ผู้เชี่ยวชาญตรวจความหมายก่อนรวมชื่อ ห้ามรวมเองจากความคล้ายของชื่อ

ผล 45 ภาพนี้เป็นเพียงตัวอย่าง train ที่คัดไว้ ไม่ใช่การ audit 10,000 ภาพ ไม่ใช่ชุดแบ่ง train/validation/test ที่ใช้วัด model และไม่ยืนยันว่า labels ตรงภาพทุกแถว ตรวจ `git ls-remote https://huggingface.co/datasets/filnow/furniture-synthetic-dataset HEAD` ได้ commit `3582bc71ff1eec35e3283c899049c8a44c024d30` ตรงกับ README ของ preview และตรึงใน dataset config แล้ว ไม่ได้ตรวจทุกภาพจาก revision นี้ การตรวจภาพห้องจริงแยกต่างหากยังไม่ทำ

ข้อควรระวัง: loader ปัจจุบันตรึง source `test` ของ Hugging Face ให้อยู่ test เพื่อป้องกันการฝึกจาก holdout เดิม แต่ในบาง label group อาจมี source train ถูกจัดเข้า test เพิ่ม; ต้องกรองด้วย `source_split` ถ้าจะรายงานเฉพาะภาพจริงตาม source card และตรวจคุณภาพ annotation ด้วยคนก่อนอ้างผล

ตรวจ streaming ของ revision นี้ได้ `train` และ `test`; แถว train แรกมี `image`, `type`, `style` และ `prompt` รวมกับ attributes อื่น ภาพแรก 448×448 การตรวจหนึ่งแถวไม่ใช่การตรวจครบทั้ง dataset บน Windows เส้นทาง cache ยาวอาจทำให้เกิด WinError 206; ตั้ง `$env:HF_HOME` เป็น directory ที่เขียนได้และ path สั้นก่อนเรียก `prepare-data`

Smoke test 10 ภาพจาก preview ของ source train (ทั้งหมดเป็นภาพสังเคราะห์) แบ่งเป็น train 4 / validation 3 / test 3 เพื่อเช็คระบบเท่านั้น; กลุ่มสินค้าต้นทางยังไม่ทราบและไม่ได้ใช้ test เพื่อสรุปผล โมเดล ResNet18 ImageNet1K V1 ฝึก 1 epoch บน CUDA และเขียน checkpoint/validation สำเร็จที่ `artifacts/source-preview-smoke-final` จึงยืนยันเส้นทางเทคนิคเฉพาะกรณีนี้ ไม่ใช่ผลประเมินงานวิจัยหรือ benchmark สำหรับประมาณ suite ทั้งชุด

วัด forward บน synthetic tensors ขนาด 224, batch 1, warmup 5, 30 repetitions พร้อม CUDA synchronize ในไฟล์ `benchmark-v2.json`: RTX 4060 Laptop GPU วัดเฉลี่ยประมาณ 4.50 ms ต่อ batch โดยไม่รวม preprocessing และการอ่านภาพ อีกครั้งขณะเครื่องมีงานอื่นวัดประมาณ 11.58 ms จึงไม่ถือเป็นค่าคงที่หรือใช้ประมาณเวลาฝึก 44 runs ต้อง benchmark บน dataset, batch และ hardware ที่จะใช้จริงก่อนวางตาราง

## การตรวจ source เต็มรอบแรก

วันที่ 2 พฤศจิกายน 2569 คำสั่ง `prepare-data --config configs/project_controlled.yaml` อ่าน source ตาม revision ที่ตรึงไว้ แล้วหยุดก่อนสร้าง split เพราะพบ `test-0000175` มี type/style อย่างน้อยหนึ่งช่องว่าง; เก็บหลักฐานไว้ที่ `data/furniture-v1/import_errors.json` และภาพที่นำเข้าได้ 9,999 ภาพใน root เดิม Root นี้เป็น snapshot ที่เตรียมไม่สำเร็จ ห้ามใช้ฝึก

การนำเข้ารอบใหม่ใช้ `configs/project_full_audit.yaml` และ `--exclude-incomplete-labels` อย่างชัดเจนเพื่อสร้าง `data/furniture-v1-complete` โดยเก็บ `excluded_samples.json` และจำนวนใน audit; error แบบอื่นยังทำให้ import หยุด นี่คือการเตรียม complete-label subset ไม่ใช่การเติมหรือเดา labels ที่หายไป แถว source test ยังคงเป็น holdout และไม่ใช้คะแนน test เลือกโมเดล

ตรวจ pretrained weights ที่กำหนดแบบ enum ของ torchvision ทั้ง 10 backbone โดยดาวน์โหลด ตรวจ SHA-256 และรัน forward บน CUDA ข้อมูลอยู่ใน `docs/evidence/pretrained-verification-10.json` การตรวจนี้ยืนยันการโหลด weights และ output shape/finite ของ controlled input ไม่ใช่การฝึกหรือวัด accuracy

การนำเข้ารอบสองอ่านครบ 10,000 แถวและบันทึกแถวขาดทั้งสอง labels 1 แถว แต่ `validate_splits` หยุดก่อนสร้าง `audit.json`/`dataset_metadata.json`: source test มี 4 แถวที่ type ไม่อยู่ใน train และ 136 แถวที่ style ไม่อยู่ใน train (40 ชื่อ style ที่ไม่รู้จัก) จึงยังไม่ใช่ snapshot ที่ใช้ฝึกระบบหลักได้ รายงานการตรวจ manifests และข้อจำกัดอยู่ที่ `docs/evidence/full-source-inspection.json` โดยไม่คำนวณ prediction หรือ metrics บน test

Near-duplicate candidate scan บน source train ถูกจำกัดที่ 250,000 comparisons จึงไม่ครอบคลุมทุกคู่ และไม่มี product/room group จากต้นทาง; ผลพบ 72 candidate pairs ก่อนถูกตัด ผล cross-split ที่เป็นศูนย์ในส่วนที่ตรวจไม่ได้พิสูจน์ว่าไม่มี leakage ต้องทบทวนหรือใช้วิธีที่ครอบคลุมก่อน screening อย่าแก้หรือรวม labels ที่ไม่รู้จักใน source test โดยอัตโนมัติ และห้ามใช้ข้อมูล test เลือกโมเดล

ภายหลังตรวจ dHash แบบ exhaustive เฉพาะ 9,000 แถว source train ครบ 40,495,500 คู่ ที่ Hamming distance ≤4 พบ 1,031 candidate pairs รวม 240 คู่ข้าม train/validation, 196 คู่ train/test และ 58 คู่ validation/test ภายใน source train; ไม่พบคู่ exact pixel ในกลุ่ม candidates ผลอยู่ที่ `docs/evidence/source-train-near-audit.json` และ CSV คู่เหล่านี้เป็นเพียงภาพคล้ายกัน ยังต้องตรวจด้วยคนว่ามาจากวัตถุเดียวกันหรือไม่ แยกจาก source test ภาพจริงที่ไม่ได้เข้าการตรวจนี้ และไม่ควรเริ่ม screening ก่อนแก้ปัญหา group leakage ที่อาจมี

วัด throughput ฝึก ResNet18 บน CUDA โดยใช้เฉพาะ train จาก snapshot ที่ยังไม่ผ่าน audit, batch 32, warmup 1 และวัด 3 batches ต่อ phase ได้ประมาณ 85.6 ภาพ/วินาทีเมื่อ frozen และ 96.8 ภาพ/วินาทีเมื่อ unfrozen; ผลอยู่ใน `docs/evidence/training-benchmark-resnet18-unvalidated.json` ตัวอย่างสั้นและผลได้รับผลจาก cache, warmup และ I/O จึงไม่ควรตีความว่าการ unfreeze เร็วกว่า หรือใช้เป็นประมาณเวลาสำหรับทุก backbone; ไม่สร้าง checkpoint หรือประเมิน test
