# สถานะการตรวจและเริ่ม screening ของ snapshot v2

> **อัปเดต 2026-11-09:** เอกสารนี้เป็นบันทึกของเวลานั้น ปัจจุบัน screening, controlled, ablation และ repeated seeds เสร็จครบแล้ว และได้เลือก DenseNet121 + C6 ดูสถานะล่าสุดที่ [docs/04](04-implementation-status.md) และ [RELEASE_STATUS.md](../RELEASE_STATUS.md)


ข้อมูลในไฟล์นี้เป็นสถานะขณะจัดชุดส่งมอบ โปรดดู suite manifest และ run directory ใน workspace สำหรับสถานะล่าสุด

- **Source inspection:** ใช้ ZIP `furniture-project2-research-followup-verified.zip` เป็นฐาน และตรวจเส้นทาง augmentation ใน `train.py`; protocol และข้อจำกัดของข้อมูลอยู่ใน [dataset protocol](10-dataset-protocol-v2.md) และ [audit](11-dataset-audit-v2.md)
- **Automated tests:** ชุดเต็มผ่าน 113 กรณี ดู [ผล pytest](evidence/furniture-v2-pytest.txt) รวม regression tests ของ robustness comparison, recovery, suite retry, snapshot และ external protocol
- **Installation:** ติดตั้งแพ็กเกจฉบับปัจจุบันใหม่โดยไม่ลง dependencies ซ้ำใน environment แยกที่สร้างไว้ก่อนหน้า แล้ว `pip check` ผ่าน; tests ด้าน snapshot และ external evaluation ผ่าน 13 กรณี ดู [install log](evidence/furniture-v2-clean-install.txt) และ [test log](evidence/furniture-v2-clean-tests.txt). นี่เป็นการติดตั้งซ้ำใน environment แยกเดิม ไม่ใช่ fresh environment ใหม่; หลักฐานการสร้าง environment เดิมอยู่ใน [clean-environment.json](evidence/clean-environment.json)
- **Pretrained verification:** 10 backbones ผ่าน forward, backward และ AdamW optimizer step บน CUDA โดยใช้ภาพ train จาก snapshot นี้ batch 2 ดู [ผลตรวจ](evidence/furniture-v2-pretrained-steps.json); ไม่ใช่คะแนนวิจัย
- **Training benchmark:** ทั้ง 10 backbones ผ่านการฝึกระยะสั้นแบบ frozen และ unfrozen ที่ batch 32 บน RTX 4060 Laptop GPU โดยใช้ภาพฝึกชุดเดียวกันและ seed 42 ดู [สรุป](evidence/furniture-v2-training-benchmark-summary.json) และ JSON รายโมเดลใน `artifacts/furniture-v2-training-benchmarks-matched/`. ประมาณเวลาจากตัวอย่างสั้น ๆ ไม่รวม validation, startup และ I/O จึงไม่ใช่เวลาจริงของทั้ง suite หรือคะแนนวิจัย; logs ใน `artifacts/furniture-v2-training-benchmarks/` เป็นการสำรวจรอบก่อนที่ใช้ภาพต่างกัน ห้ามใช้เทียบโมเดล
- **Screening:** suite 10 backbones เปิดด้วย `configs/project_v2_conservative.yaml` โดยใช้ train/validation เท่านั้นเพื่อฝึกและเลือกโมเดล เก็บแผนและสถานะใน `artifacts/runs/suites/furniture-v2-screening.json`; [launch record](evidence/furniture-v2-screening-launch.json) ระบุ process และ log. Manifest ใน ZIP เป็นสำเนา ณ เวลาแพ็ก ไม่อัปเดตตาม process ใน workspace และ ZIP ไม่มี checkpoints
- **Held-out evaluation:** ยังไม่มีการคำนวณ predictions หรือ metrics บน synthetic test หรือ real-source test เพื่อเลือกโมเดล ไม่มีการรวมคะแนนสองชุด. External label coverage เป็นการตรวจ annotations ก่อน inference ตาม [protocol](09-external-evaluation-protocol.md)

หาก process หยุด ให้ตรวจ manifest, stdout/stderr และ run status ก่อนใช้คำสั่ง continuation ของ suite; อย่าแก้ status เอง. ตัวอย่างหลังตรวจสาเหตุแล้ว:

```powershell
.venv\Scripts\python.exe -m furniture.cli suite --continue-manifest artifacts/runs/suites/furniture-v2-screening.json --retry-failed --recover-running
```

การ recovery ตรวจ process, checkpoint และ provenance; ดูเอกสาร recovery เดิมและเก็บ logs ก่อนดำเนินการ. ไม่ควรรัน continuation ซ้อนกับ suite ที่ยังทำงานอยู่
