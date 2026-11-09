# Protocol สำหรับ model-only release

Protocol นี้ตรึงก่อนผล controlled, single-task, repeated-seed และ held-out test ของ snapshot v2 จะพร้อม ใช้ [dataset protocol v2](10-dataset-protocol-v2.md) และไม่เปลี่ยน manifests, quarantine, architecture pool หรือ screening suite ที่ทำงานอยู่

1. ให้ screening เดิมจบทั้ง 10 backbones และตรวจหลักฐาน completed/source/checkpoint ทุก run เลือก 3 backbones ด้วย validation mean macro-F1 ของ type/style ตาม `select_screening`; หากคะแนนเท่ากันใช้ inference latency ภายใต้เงื่อนไขเดียวกัน แล้วใช้ชื่อ backbone เป็นลำดับสุดท้าย
2. ฝึก controlled C1–C8 บนแต่ละ backbone ที่ผ่าน screening ด้วย config เดิม seed 42 และ frozen/fine-tune, augmentation, occlusion ตาม config แต่ละ condition ไม่ใช้ held-out test เลือก condition
3. จาก controlled เลือก checkpoint ที่มี **shared backbone update อย่างน้อยหนึ่ง epoch** และมีคะแนน validation mean macro-F1 ของ type/style สูงสุดใน C2–C8; หากเท่ากันเรียงชื่อ backbone แล้ว condition จากน้อยไปมาก บันทึก run, checkpoint SHA-256, split hashes และกฎนี้ก่อนเปิด test. C1 เป็น frozen control และไม่เป็น candidate สำหรับโมเดลหลักที่ต้องอัปเดต backbone ร่วม
4. สำหรับ backbone ที่เลือก ฝึก single-task type-only และ style-only ด้วย C3 และเปรียบเทียบกับ controlled C3 แบบ joint บน validation; ผล ablation ไม่ใช้เปลี่ยน checkpoint ที่เลือก เก็บ mapping ของแต่ละงานแยกกัน
5. ฝึก repeated seeds 123 และ 2026 สำหรับ C1, C2, C3, C5 ของ backbone ที่เลือก เพื่อรายงานความแปรปรวนเมื่อมีผลครบ seed 42 อ้างอิงจาก controlled; ไม่ใช้ผล seed ซ้ำเปลี่ยนการเลือก checkpoint
6. เมื่อตรึง selection แล้ว ประเมิน synthetic test ของ checkpoint ที่เลือกหนึ่งตัว รายงาน type/style แยก, confusion matrices และตัวอย่างผิด; external real-source test แยกโดยใช้ protocol ที่ตรึงไว้ก่อนหน้า ไม่รวมคะแนนสองชุด และไม่ใช้ปรับ label mapping, threshold หรือโมเดล

Single-task/controlled/repeated เป็นการฝึกจริงแยก run และต้องรายงาน actual epochs, source/config/data provenance และ failure ที่เกิดขึ้น อย่าเติมคะแนนจาก smoke หรือแสดงว่างานใดเสร็จหากไม่มี evidence. ทุกชุดใช้ persistent suite manifest; หาก run fail ให้เก็บหลักฐานและหยุดขั้นพึ่งพาไว้จน recovery ผ่านการตรวจ process/checkpoint/provenance
