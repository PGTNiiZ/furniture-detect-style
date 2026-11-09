# Model registry และ preprocessing

โมเดลทั้ง 10 ใช้ shared feature vector และ heads แยกตาม tasks ที่เลือก การระบุ weights ด้านล่างเป็น enum แบบ explicit ของ torchvision; config แต่ละโมเดลอยู่ใน `configs/models/` โดย `initialization: imagenet` จึงดาวน์โหลด/โหลด pretrained weights และ `scratch` ใช้ random initialization

| Config | เหตุผลที่เปรียบเทียบ | Feature dimension | Weights identifier | Official resize → crop |
|---|---|---:|---|---|
| resnet18 | baseline CNN ขนาดเล็ก | 512 | ResNet18_Weights.IMAGENET1K_V1 | 256 → 224 |
| resnet50 | เปรียบเทียบ ResNet ที่ลึกขึ้น | 2048 | ResNet50_Weights.IMAGENET1K_V2 | 232 → 224 |
| densenet121 | dense feature reuse | 1024 | DenseNet121_Weights.IMAGENET1K_V1 | 256 → 224 |
| efficientnet_b0 | ตัวเลือกเน้นสมดุลทรัพยากร | 1280 | EfficientNet_B0_Weights.IMAGENET1K_V1 | 256 → 224 |
| efficientnet_b2 | เปรียบเทียบขนาดใน EfficientNet | 1408 | EfficientNet_B2_Weights.IMAGENET1K_V1 | 288 → 288 |
| mobilenet_v3_large | ตัวเลือกโมเดลสำหรับต้นทุน inference ต่ำ | 1280 | MobileNet_V3_Large_Weights.IMAGENET1K_V2 | 232 → 224 |
| regnet_y_400mf | CNN family อีกแบบ | 440 | RegNet_Y_400MF_Weights.IMAGENET1K_V2 | 232 → 224 |
| convnext_tiny | CNN สมัยใหม่ | 768 | ConvNeXt_Tiny_Weights.IMAGENET1K_V1 | 236 → 224 |
| swin_t | hierarchical vision transformer | 768 | Swin_T_Weights.IMAGENET1K_V1 | 232 → 224 |
| vit_b_16 | patch-based vision transformer | 768 | ViT_B_16_Weights.IMAGENET1K_V1 | 256 → 224 |

ข้อมูลตารางอ่านจาก torchvision 0.22.1+cu128 ใน environment ที่ตรวจ ค่า official resize/crop หมายถึง recipe ของ weights ไม่ใช่ขั้นตอนที่ project ใช้จริง และไม่รับประกันว่าโมเดลที่ชื่อ tiny จะใช้ทรัพยากรน้อยที่สุด ต้องวัดจริง

Controlled protocol ของโปรเจกต์ใช้ RGB (รวม EXIF orientation), resize รักษา aspect ratio และ pad สีขาวให้เป็น 224×224 ด้วย PIL bicubic ตามด้วย normalization mean `[0.485, 0.456, 0.406]` และ std `[0.229, 0.224, 0.225]` สำหรับทุกโมเดล มีค่า normalization ตรงกับ weights enums เหล่านี้ แต่ขั้น resize/pad แตกต่างจาก official center-crop recipes รวมถึง EfficientNet-B2 ที่ official crop 288 ความแตกต่างอาจส่งผลต่อคุณภาพ ควรระบุในรายงาน การเปลี่ยนไปใช้ official recipe ต้องถือเป็นอีก protocol และไม่เปลี่ยนเงียบ ๆ

Layer ถูกแทนด้วย `Identity` ตามโครงสร้างของแต่ละ architecture (`fc`, `classifier`, `head` หรือ `heads.head`) จากนั้นเชื่อม shared feature vector เข้า heads; `freeze_backbone` ตรึง gradients ของ backbone และ training loop กำหนด backbone eval mode เพื่อไม่อัปเดต BatchNorm เมื่อ frozen checkpoint บันทึก config, weights identifier และ preprocessing metadata

การทดสอบ random initialization ครอบคลุม output shape และ checkpoint round-trip ของแต่ละ backbone; ทดลองโหลด pretrained ResNet18 และฝึก smoke บน CUDA แล้ว ยังไม่ได้ตรวจ pretrained download, accuracy, latency หรือ GPU training ของอีกเก้าโมเดล
