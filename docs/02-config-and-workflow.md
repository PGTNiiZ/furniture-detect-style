# คู่มือ config และการเรียกแต่ละส่วน

Config file คือไฟล์ตั้งค่า YAML; ใช้กำหนดค่าที่เปลี่ยนบ่อยโดยไม่แก้ logic ใน Python

## ลำดับการอ่าน
`configs/project.yaml` → อ่านไฟล์ dataset/model/training/augmentation/inference → นำ overrides ของ experiment มาทับเฉพาะค่าที่ระบุ → validate → บันทึก resolved config ใน run
ถ้าพิมพ์ชื่อ key ผิดหรือ probability เกิน 0..1 จะหยุดก่อนเริ่มเทรน
Paths ของไฟล์ config ย่อยสัมพันธ์กับโฟลเดอร์ project.yaml; paths ของข้อมูล/โมเดล/output สัมพันธ์กับ project_root
CLI paths ที่ผู้ใช้ระบุ เช่น --image/--checkpoint/--run-dir สัมพันธ์กับ terminal ปัจจุบัน หรือใช้ absolute path ได้

## แก้ตรงไหน
| ต้องการ | ไฟล์/ค่า |
|---|---|
| เปลี่ยนชุดข้อมูล | project.yaml → dataset |
| เปลี่ยนชื่อคอลัมน์ CSV | datasets/local_csv.yaml → source.columns |
| เพิ่มข้อมูลรอบใหม่ | dataset.id/root ใหม่ + split.previous_root |
| เปลี่ยน input resolution | models/resnet18.yaml → image_size |
| เปลี่ยน batch/epochs/LR | training/default.yaml |
| เปลี่ยนความสำคัญของสองงาน | training.loss_weights.type และ style; ทั้งคู่ต้อง >0 |
| เปิด/ปิดหรือปรับ augmentation | augmentation file ที่ project อ้างอิง หรือ experiment override |
| เทรนต่อด้วย weights เดิม | model.initialization: checkpoint, checkpoint path, checkpoint_scope |
| ตั้งค่าความมั่นใจ/target score | inference/default.yaml; จูนด้วย validation เท่านั้น |
| เพิ่มการทดลอง | คัดลอก experiment YAML เปลี่ยน id/description/overrides แล้วเพิ่มใน project.experiments |

มี model configs สำหรับ 10 torchvision backbones และระบุ weights enum ในแต่ละ config ดูสถานะทดสอบและ preprocessing protocol ใน `docs/04-implementation-status.md`; ViT-B/16 ต้องใช้ 224×224

## ตัวอย่าง augmentation
```yaml
enabled: true
horizontal_flip_p: 0.5
rotation_degrees: 5.0
brightness: 0.1
contrast: 0.1
saturation: 0.0
hue: 0.0
```
ภาพถูก pad เป็นสี่เหลี่ยมก่อน random transforms; rotation อาจตัดมุมของวัตถุ จึงต้องดู preview และ error cases
สี/วัสดุอาจเป็นหลักฐานของสไตล์ การเปลี่ยนสีมากอาจทำลาย label; ถ้าอนาคตมี color task ให้ปิด color-changing transforms ที่ขัดกับ label
นี่คือ augmentation ของภาพเดิม ไม่ใช่การสร้างสินค้าหรือ styles ใหม่ ไม่เพิ่ม independent test samples

## แต่ละคำสั่งเรียกอะไร
| คำสั่ง | Module | Input → Output |
|---|---|---|
| check-config | config.py | YAML ย่อย → merged config ที่ผ่าน schema validation |
| prepare-data | prepare.py → data.py | HF/CSV → images, manifests, audit |
| audit-data | prepare.py → data.py | manifests/images → counts, labels, hashes, group checks |
| preview-augmentation | augmentation.py | ภาพ + aug config → PNG original/variants |
| train | pipeline.py → train.py → model.py | config + data → checkpoint, history, validation |
| experiments | pipeline.py | config + E1/E2/E3 → runs + comparison CSV |
| compare | pipeline.py | run folders → controlled validation comparison |
| evaluate-run | train.py | best checkpoint + saved config → test metrics/predictions/confusion |
| room | room.py → recommend.py | room + target → annotated image + proposed actions |
| train-detector | detection.py | box dataset + detector config → Ultralytics run |

## เหตุใดเป็น Multi-task
ResNet18 สร้าง shared feature vector z จากภาพหนึ่งชิ้น จากนั้นส่งเข้า type head และ style head
`total_loss = type_weight × CE(type_logits, type_label) + style_weight × CE(style_logits, style_label)`
Gradient จากทั้งสองงานอัปเดต shared backbone ใน E2/E3 เมื่อ unfreeze; มี task losses และ metrics แยก
E1 แช่แข็ง backbone เป็น control จึงยังไม่มีหลักฐาน shared representation learning จากงาน furniture; ใช้ E2/E3 ที่ฝึกร่วมจริงเป็นหลักฐานตาม rubric
YOLO เป็นโมดูล detection อีกตัว การนำหลายโมเดลมาต่อกันอย่างเดียวไม่ใช่หลักฐาน Multi-task

## การเปรียบเทียบ
E1/E2/E3 เปลี่ยน freeze policy และ augmentation อย่างมีเป้าหมาย; runner ตรวจ dataset/model/seed/budget และ controls สำคัญเท่ากัน
summary ใช้ validation mean(type Macro-F1, style Macro-F1) ค่าเริ่มต้น; สามารถเลือก style_macro_f1 แต่ต้องกำหนดก่อนเทรนและคงที่ในชุดเปรียบเทียบ
การเปลี่ยน model/pretraining/checkpoint ถือเป็นชุดเปรียบเทียบใหม่; compare จะปฏิเสธเพื่อลดการสรุปว่าต่างกันเพราะ augmentation ทั้งที่เปลี่ยนหลายอย่าง

## การแก้ไขต่อเนื่อง
แก้ source configs หรือโค้ดได้ตามปกติ → check-config → pytest → train เป็น run ใหม่
run เดิมเก็บ config, source snapshot, environment และ hashes ไว้; ไม่แก้ history/metrics เก่าให้ตรงกับค่าปัจจุบัน
ไม่ใช่ online learning หรือบริการ retrain อัตโนมัติ ทุกการฝึกเริ่มเมื่อเรียกคำสั่ง

## Troubleshooting
- schema key ไม่รู้จัก: ตรวจสะกดตามไฟล์ตัวอย่าง ไม่ละเลย warning
- Dataset snapshot exists: เปลี่ยน id/root เป็น v2 หรือใช้ชุดเดิมเทรน ไม่เตรียมซ้ำ
- Group bridges different splits: ภาพหรือ product group เดียวกันมี train/test ปน ต้องแก้ต้นทาง
- Unseen label in val/test: ต้องมีหลักฐานเรียนรู้คลาสนั้นใน train หรือเปลี่ยน protocol เป็น open-set แยก ไม่เติม test เข้า train หลังเห็นผล
- Full checkpoint label mapping mismatch: ใช้ backbone scope และฝึก heads ใหม่ หรือเริ่มใหม่จาก ImageNet
- Warm-start leakage: split ใหม่ย้ายภาพฝึกเก่าไป test หรือย้าย holdout เก่ามาฝึก; รักษาบทบาทเดิม
- CUDA memory: ลด batch ใน config; ถ้าเปลี่ยนระหว่าง E1–E3 ต้องปรับเท่ากันและเริ่มชุดเปรียบเทียบใหม่
