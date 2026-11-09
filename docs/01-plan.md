# แผน Project 2 ตาม rubric — ส่งและนำเสนอ 14 พฤศจิกายน 2569

## เป้าหมายและขอบเขต
Input: ภาพห้องและ target style ที่ผู้ใช้เลือก
Output: boxes และประเภทเฟอร์นิเจอร์, style scores ต่อชิ้น, keep/review/replace candidates และภาพชิ้นทดแทนประเภทเดียวกัน

ใช้ YOLOv8 ตรวจวัตถุและ ResNet18 ที่เรียนรู้ type/style ร่วมกันผ่าน shared backbone + สอง heads
การประเมิน type/style แยกกันตาม rubric; Detection เป็นโมดูลในระบบที่วัดผลแยก
Recommendation รุ่นแรกเป็น heuristic ต้องประเมินโดยคนก่อนอ้างว่าเข้าใจความเข้ากัน

## Mapping คะแนน 15
| เกณฑ์ | เต็ม | หลักฐานที่ต้องส่ง |
|---|---:|---|
| ความยาก Multi-task | 4 | โมเดลฝึกร่วม >=2 งาน, shared parameters ได้รับ gradient จากสอง losses และผลแยก type/style |
| ปัญหาและข้อมูล | 2 | วัตถุประสงค์ แหล่ง/สิทธิ์ข้อมูล counts label audit, groups/splits และหลักฐานกัน leakage |
| โมเดลและทดลอง | 2 | โค้ดทำงาน configs อธิบายได้ เปรียบเทียบอย่างน้อย E1/E2/E3 อย่างเป็นธรรม |
| ประเมิน/วิเคราะห์ | 2 | test แยก, metrics ที่เหมาะสม, ตัวอย่างถูก/ผิด ข้อจำกัดและตารางผล |
| สื่อ | 2 | สไลด์อ่านง่าย สม่ำเสมอ ภาพกราฟตารางมี label |
| นำเสนอ/ตอบ | 3 | ลำดับชัด อยู่ในเวลาที่อาจารย์ให้ สมาชิกอธิบายหน้าที่และเหตุผลได้ |

คะแนนความยากเลือกสูงสุดประเภทเดียว ไม่บวก Multi-task 4 กับ Detection 2
โค้ดและ smoke tests เป็นหลักฐานความพร้อมของระบบ ไม่ใช่ผลประเมิน furniture และไม่รับประกันคะแนน
E1 frozen control ไม่ได้ปรับ shared representation; E2/E3 ที่มีการอัปเดต backbone จากสอง tasks เป็นหลักฐานที่ต้องใช้
กลุ่ม 6–8 คน; ยังไม่มีเวลาพูดที่ยืนยันใน rubric จึงไม่ยกเวลา 10 นาทีจาก Project 1 มาเป็นข้อบังคับ

## ข้อมูล 4 ส่วน
A Product images/type/style: ฝึก shared classifier (เริ่มจาก HF ที่ผู้ใช้ให้)
B Room images + boxes/types: ประเมิน/ฝึก detector เพิ่ม
C Room crops + human-reviewed style: external domain evaluation ของ classifier
D query/target/candidate/rater/relevance: ประเมินคำแนะนำโดยคน

A ไม่มี boxes/pair labels ที่ตรวจพบใน schema ที่อ่านก่อนหน้า จึงไม่ทดแทน B/C/D
ยังไม่มี verified counts ของ dataset เต็ม ให้คำนวณหลังดาวน์โหลด revision จริง
prompt ที่ระบุ type/style ใช้เป็นข้อมูลที่มาเท่านั้น ไม่ป้อนเป็น input classifier

## การทดลองที่ implement แล้ว
| ID | วิธี | สิ่งที่คงที่ | สิ่งที่ทดสอบ |
|---|---|---|---|
| E1 | Frozen backbone, two heads, no augmentation | dataset/split/seed/backbone/size/batch/max epochs/metric | คุณภาพ pretrained features |
| E2 | Freeze 2 epochs แล้ว fine-tune สองงาน | เหมือน E1; เปลี่ยน freeze policy | ประโยชน์จากการปรับ shared features |
| E3 | E2 + horizontal flip p=.5 | เหมือน E2; เปลี่ยน augmentation อย่างเดียว | generalization จาก flip |

ใช้ validation mean Macro-F1 ของสองงานเลือก checkpoint; ค่า test ไม่ใช้เลือก model/threshold
ทดลองทุกแบบด้วย image size 224, batch32, max20 epochs, seed42 เป็นค่าเริ่มต้น ไม่ใช่ผลที่พิสูจน์แล้ว
E1 freeze_epochs: all เพื่อรองรับการเปลี่ยน epochs ใน training config
อธิบาย actual epochs/early stopping และต้นทุนฝึกด้วย; หากมีเวลาให้รัน 3 seeds เป็นชุดเปรียบเทียบแยก
Type-only/style-only และ ViT อยู่ใน implementation ปัจจุบันแต่ยังต้องฝึกและประเมินจริง; segmentation-before-style และ learned compatibility ยังเป็น optional ที่ไม่มี implementation

## Metrics
Type และ style: Precision, Recall, Macro-F1, per-class report, Confusion Matrix; Accuracy ประกอบ
Detection: mAP50 และ mAP50–95, Precision/Recall พร้อมระบุ IoU; ตรวจ missed objects ไม่ประเมินเฉพาะ crop ที่หาเจอ
Recommendations: human precision@K/nDCG@K เมื่อกำหนด relevance scale, gallery และ rater protocol แล้ว
รายงาน style uncertainty, mixed-style cases และ synthetic-product → real-room domain gap

## ผลที่ต้องมีในรายงาน
1. ตารางจำนวนภาพ/กลุ่ม/คลาสแต่ละ split และตัวอย่างภาพ
2. รูป architecture ที่อธิบาย shared features และ task-specific heads
3. กราฟ learning curves ของ train/validation และ losses แยกงาน
4. ตาราง E1/E2/E3 ทั้งสองงาน พร้อม model settings/best epochs
5. Confusion matrices และภาพที่ถูก/ผิด พร้อมเหตุผล
6. ตัวอย่างภาพห้อง boxes และคำแนะนำ โดยระบุว่าคำแนะนำส่วนไหนเป็น heuristic
7. ข้อจำกัดและวิธีป้องกัน leakage; แยกผล synthetic กับ real-domain
8. วิธีรันโค้ด config requirements และหน้าที่สมาชิก

## Roadmap
| วัน | ผลงาน |
|---|---|
| 2–4 พ.ย. | ตรวจ labels/groups, เตรียมข้อมูล ตรึง split และรัน smoke |
| 5–8 พ.ย. | E1/E2/E3, validation comparison, เลือก checkpoint |
| 9–10 พ.ย. | Room pipeline, detector evaluation/fine-tune หากมี boxes, human review |
| 11–12 พ.ย. | Test ตาม protocol ที่ตรึง, error analysis, สไลด์ |
| 13 พ.ย. | ซ้อมและรันจาก clean environment |
| 14 พ.ย. | ส่งและนำเสนอ |

## แบ่งงานทีม 6–8 คน
1 ข้อมูล/provenance/splits; 2 shared classifier/config; 3 augmentation/experiments;
4 detector/room crops; 5 metrics/error analysis; 6 matching/demo;
สมาชิก7–8 (ถ้ามี) ช่วย annotation quality, slides และ reproducibility
ทุกคนต้องอธิบาย pipeline รวม และแยกความหมาย Multi-task/Multilabel/หลายโมเดลได้

## เชื่อมเนื้อหาสไลด์ (หน้า PDF จริง)
| ไฟล์ | หน้า | ใช้งาน |
|---|---|---|
| 8(1).pdf | 1–2 | Pretrained CNN, transfer learning, augmentation |
| 9(1).pdf | 4–7, 19–21 | Multiclass/multilabel, multi-task และ negative transfer |
| 10(1).pdf | 1–4, 72–73 | PyTorch classification, Dataset/DataLoader |
| 11.pdf | 1–4 | R-CNN family, YOLOv8 detection |
| 12.pdf | 1–4 | Segmentation เป็นงานต่อยอด |
| 13.pdf | 1–3 | Video temporal models เป็นงานต่อยอดเมื่อมี video scope |
| 14.pdf | 1, 3–4, 8, 11–14 | ViT, patches, embeddings เป็นการทดลองเสริม |
| 14.1.pdf | 1–3 | เลือก metrics ให้ตรงประเภทงาน |

## Technical references
- https://docs.pytorch.org/vision/stable/transforms.html
- https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
- https://docs.ultralytics.com/modes/train/
- https://docs.ultralytics.com/models/yolov8/
- https://huggingface.co/datasets/filnow/furniture-synthetic-dataset
Rubric และวันส่งอ้างจากข้อความที่ผู้ใช้ให้ในแชท ไม่ได้สมมติจากเอกสาร Project 1
