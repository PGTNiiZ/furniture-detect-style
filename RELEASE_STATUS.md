# สถานะ model-only release

Classifier research complete: การทดลองตาม protocol เสร็จและเลือก checkpoint ด้วย validation ก่อนประเมิน test.
Room inference: มีตัวอย่าง detector baseline จริง; สถานะตรวจแพ็กสุดท้ายอยู่ใน final-delivery.json.
Room end-to-end evaluation: ยังไม่ครบจนมี annotations อิสระและรายงานประเมินจริง.

- Backbone: `densenet121`, condition: `C6`
- Checkpoint บนเครื่อง: `D:\Download\Download from Chrome\furniture-project2-code-and-plan\furniture-project2\artifacts\runs\furniture-v2-conservative\controlled-densenet121-C6-42-type-style\20261103T003152Z-98889263\best.pt`
- SHA-256: `24e413586f98537fff9f56eb66c71a10c30f11154d3eaa58904ce5e50dc78328`
- ZIP มี selected checkpoint ที่ `release/checkpoint/best.pt`; ตรวจ SHA-256 ก่อนใช้
- Selection และ split hashes: `artifacts/model-only-release/selection-frozen.json`
- Controlled, single-task และ repeated validation: `artifacts/model-only-release/validation-report/`
- Robustness validation clean/light/strong: `artifacts/model-only-release/robustness-report/`
- Synthetic test, confusion matrices และ error examples: `artifacts/model-only-release/synthetic-test-report/`
- Real-source test: `artifacts/model-only-release/external-reference/result/metrics.json` แยกจาก synthetic test
- Source annotations ไม่ใช่ expert ground truth; confidence เป็น uncalibrated softmax
- 494 near-duplicate candidates ยังไม่มี human decisions; v2 ใช้ conservative quarantine ตาม protocol
- Detector เป็น optional preprocessing แยกจาก classifier และไม่มีการฝึก detector ใหม่
- Bonn/IKEA/Target เป็น future work ไม่ได้ผสมใน suite นี้

Synthetic test type macro-F1: 0.996466; style macro-F1: 0.200232.
ดู metrics, counts, confusion matrices และตัวอย่างผิดในไฟล์รายงาน ไม่รวมคะแนนสอง test เข้าด้วยกัน.

บันทึกเมื่อ 2026-11-04T00:51:53.494948+00:00 หลังตรวจ completion evidence และ checksum
