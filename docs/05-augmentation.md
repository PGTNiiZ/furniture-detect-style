# Augmentation และ occlusion

ชุด controlled ใหม่ใช้ `albumentations==2.0.8` (แพ็กเกจ Albumentations รุ่น MIT) ส่วน E1–E3 เดิมใช้ torchvision ตามเดิม `configs/augmentation/{none,easy,hard}_{off,light,strong}.yaml` แต่ละไฟล์ระบุค่าทั้งหมด ส่วน `configs/experiments/C1.yaml` ถึง `C8.yaml` override ค่าเหล่านั้นแบบ explicit จึงตรวจค่าที่ใช้จริงได้ด้วย `check-config` และปรับได้ใน experiment config

ใน schema ใหม่ `preset: none` หมายถึงความน่าจะเป็นของ augmentation ทุกชนิดเป็นศูนย์ ส่วน `occlusion.preset: off` ต้องมี `probability: 0`; ไม่ใช้ `enabled` ที่ขัดกับชื่อ preset ค่าของ `params` และ `occlusion` เป็นค่าที่นำไปสร้าง transform จริง หากแก้ค่าจาก preset มาตรฐาน ควรตั้ง `preset: custom` เพื่อไม่ให้ชื่อชวนเข้าใจผิด ตรวจ unknown keys, probability, ranges และค่าขนาด kernel ก่อนรัน

ลำดับ: แปลงภาพเป็น RGB → resize/pad รักษาสัดส่วนโดยเติมขาว → horizontal flip → affine เดียว → brightness/contrast → Gaussian blur → occlusion → tensor → normalize ด้วย ImageNet mean/std การแปลง affine ใช้ OpenCV linear interpolation และขอบขาว; brightness/contrast ของ Albumentations เป็นการสุ่มค่าในช่วง ±limit โดย brightness คูณค่าสูงสุดของชนิดภาพ (255 เมื่อเป็น uint8) และ contrast เป็นการปรับ affine ของค่าพิกเซล จึงไม่รับประกันภาพเหมือน torchvision ColorJitter

Occlusion ใช้ CoarseDropout เติมค่า RGB 127 ก่อน normalize และระบุจำนวนช่องกับช่วงความสูง/กว้างเป็นสัดส่วนของด้านภาพ ไม่ใช่พื้นที่วัตถุ ค่า fill เปลี่ยนจากสีขาวของ prototype ก่อนหน้า จึงเป็น protocol ใหม่ ไม่ควรเปรียบเทียบว่าเป็น run เงื่อนไขเดียวกัน

Preview และ training ใช้ `pil_transforms` เดียวกัน โดยเลือกตัวอย่าง preview จาก train เท่านั้น เก็บ sample IDs, seed และ config สำหรับแต่ละภาพ; ไม่ใช้ test เพื่อปรับ augmentation. Validation/test ใช้ preprocessing แบบ deterministic ไม่มี augmentation หรือ occlusion แบบสุ่ม

**สำคัญ:** prototype ก่อนแก้ P0 ส่ง dictionary ผ่าน expression ที่แปลงเป็น boolean ทำให้ training ไม่ใช้ preset ที่ระบุจริง อย่านำผลจากโค้ดรุ่นนั้นไปสรุปเรื่อง none/easy/hard หรือ occlusion โดยอาศัย config ที่บันทึกไว้เพียงอย่างเดียว ตรวจ snapshot โค้ดของแต่ละ run ก่อนใช้งาน Config prototype ที่มีทั้ง `enabled: false` และ `preset` จะถูกปฏิเสธพร้อมข้อความให้ migrate ไปใช้ไฟล์ config ใหม่ ส่วน E1–E3 ไม่มี preset และยังคง semantics เดิม รวมทั้ง `enabled: false` ที่ปิด legacy transforms ทั้งหมด
