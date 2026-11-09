# กู้คืน run ที่เครื่องดับหรือ process ถูกยุติ

Run รุ่นใหม่นี้บันทึกชื่อเครื่อง, PID และ session ID ใน `status.json` พร้อมถือ OS lock ตลอดเวลาที่เขียน run ถ้าเครื่องดับ OS จะปล่อย lock แม้ status ยังเป็น `running` PID ที่ถูกนำกลับมาใช้ซ้ำจะทำให้ระบบปฏิเสธการกู้คืนไว้ก่อนเพื่อความปลอดภัย

สำหรับ run ที่มี `last.pt` หลังจบอย่างน้อยหนึ่ง epoch และ status ค้าง `running` ใช้:

```powershell
.venv\Scripts\python.exe -m furniture.cli resume --run-dir PATH_TO_RUN --recover-running
```

ก่อนกู้คืน ระบบต้องได้ exclusive lock, ตรวจว่าชื่อเครื่องตรงและ PID เจ้าของเดิมหยุดแล้ว, ตรวจ source snapshot และ resolved config กับ provenance, ตรวจความสมบูรณ์ของ `last.pt`/`best.pt`, dataset/mapping/split hashes, เวอร์ชันแพ็กเกจ, PyTorch และ device เดิม จากนั้นบันทึก `recovery-events.jsonl` พร้อม checkpoint hash, epoch, owner เดิมและ owner ใหม่ แล้วจึง resume ถ้าขั้นตอนใดพิสูจน์ไม่ได้จะหยุดและไม่เปลี่ยน status

หาก checkpoint จบ epoch สุดท้ายแล้ว แต่เครื่องดับก่อนสร้าง validation หรือ status completed ระบบจะคำนวณ validation ในโฟลเดอร์ staging แล้วเก็บผลที่ยังไม่มี หากไฟล์เดิมมีเนื้อหาขัดแย้ง จะหยุดและเก็บ staging ไว้ให้ตรวจ โดยไม่เขียนทับหลักฐานเดิม

ถ้า suite ค้าง `running` ก่อนมี `last.pt` สามารถใช้ `suite --continue-manifest PATH_TO_SUITE --recover-running --retry-failed` ระบบตรวจ lock, host/PID, config/source/provenance เทียบ suite manifest และบันทึก event ก่อนทำเครื่องหมาย attempt เดิมว่า interrupted แล้วสร้าง attempt ใหม่โดยรักษาไฟล์เดิม การเริ่มใหม่ต้องสั่งทั้งสอง flags เพราะใช้ทรัพยากรฝึกใหม่ หากไม่มี status/owner/provenance หรือยืนยันความตรงกันไม่ได้ ระบบหยุดเฉพาะ attempt นั้นและไม่เริ่มใหม่อัตโนมัติ

Run รุ่นเก่าที่ค้าง `running` แต่ไม่มี owner metadata ไม่สามารถพิสูจน์ว่า process เดิมหยุดแล้ว จึงถูกปฏิเสธอัตโนมัติ ไม่ควรแก้ `status.json` เองเพื่อข้ามการตรวจ รันที่อยู่คนละเครื่อง, PID ยังทำงาน, ตรวจ PID ไม่ได้, checkpoint ไม่ครบ, หรือ provenance แตกต่างก็ถูกปฏิเสธเช่นกัน ต้องตรวจสาเหตุและหากจำเป็นเริ่ม run ใหม่โดยรักษาข้อมูลเดิมไว้

`resume --run-dir PATH_TO_RUN` ยังคงใช้กับสถานะ `failed`/`interrupted` และตรวจ checkpoint/provenance เช่นเดียวกัน Recovery คืน optimizer/scheduler/scaler และใช้ seed ราย epoch เดิม แต่ GPU operations บางชนิดอาจไม่ให้ผล bitwise deterministic ทุก environment
