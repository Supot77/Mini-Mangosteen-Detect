# ผลสรุปโมเดลปัจจุบัน

ผลนี้ยึด artifact ที่ใช้งานปัจจุบันจาก run `20260917_001333` ซึ่งมีหลักฐานครบใน `current_model/results/20260917_001333/` และถูกชี้โดย `current_model/model/metadata.json`

| รายการ | ค่า |
|---|---|
| โมเดลที่ใช้งาน | Separable CNN 64 (depthwise-separable CNN) |
| Parameters | 61,059 / 100,000 (ผ่านเกณฑ์) |
| Input | 96 × 96 × 3 RGB |
| Output | 3 classes: `overripe`, `ripe`, `unripe` |
| Validation accuracy | **91.18% (31/34)** |
| Float32 original-crop robustness accuracy | **94.74% (36/38)** |
| Full INT8 original-crop robustness accuracy | **94.74% (36/38)** |
| Deployment-matched crop check | 100.00% (38/38) |
| INT8 model size | 102,912 bytes (100.5 KiB) |
| Dataset | deployment-matched v4: train 214, validation 34, test 38 |
| TFLite SHA-256 | `0A21E9C4FB086115829022174630B1B8523A933F95495AB5A88CCA86E64AD929` |
| ไฟล์ TFLite ปัจจุบัน | `current_model/model/separable_cnn_64_int8.tflite` |
| ไฟล์ C array ที่ใช้ build | `src/mangosteen_model_data.cc` |

โมเดลนี้ใช้ภาพจากต้นฉบับในโครงการและไม่ได้เพิ่มภาพจากภายนอก ชุด validation/test 34/38 ภาพเป็นภาพที่ไม่ผ่าน augmentation สำหรับผลหลัก ใช้ชุด original crop ที่ snapshot ไว้ใน `current_model/evaluation/original_crop_test/` ส่วน matched-crop เป็น engineering check ของ preprocessing ใหม่

## ขอบเขตของผลความแม่นยำ

| Input/evaluation path | Result | Interpretation |
|---|---:|---|
| Original 96×96 crop, Float32 | 94.74% (36/38) | **Primary offline evaluation** |
| Original 96×96 crop, Full INT8 | 94.74% (36/38) | **Primary offline evaluation; no quantization drop** |
| Deployment-matched fruit crop, Float32/Full INT8 | 100.00% (38/38) | Secondary engineering check |
| Simulated RGB565 full-frame input | 55.26% (21/38) | Domain-shift diagnostic; not the matched-crop test result |
| Real OV2640 board accuracy | Not measured | Requires a ground-truth live-camera test |

The 94.74% result is the primary offline evaluation under the original crop style. The 100.00% result is retained as a deployment-matched preprocessing check, not as verified real-camera accuracy. The test set contains only two `overripe` images, so both results should be read with that limitation.

## Primary original-crop confusion matrix

```text
              predicted
true          overripe ripe unripe
overripe            2     0      0
ripe                1    16      1
unripe              0     0     18
```

The original-crop result has balanced accuracy **96.30%** and macro F1 **0.905**. The single `ripe` to `overripe` error and single `ripe` to `unripe` error account for the 36/38 result.

## สถานะการนำขึ้นบอร์ด

- โมเดลถูกแปลงเป็น C-array และ build สำเร็จสำหรับ LilyGO T-SIMCAM ESP32-S3
- อัปโหลดผ่าน `COM9` สำเร็จ และ PlatformIO ยืนยัน `Hash of data verified`
- model bytes ใน firmware ตรงกับ current TFLite artifact ตาม SHA-256 ที่บันทึกไว้ใน flash log
- firmware ล่าสุดใช้ preprocessing แบบ full-frame: `RGB565 240x240 -> RGB888 -> 96x96 -> INT8`
- มี Camera Config panel สำหรับปรับ OV2640 แบบ runtime โดยไม่เปลี่ยน model input path
- ยังไม่มีหลักฐาน prediction สดที่มี ground truth จากกล้อง OV2640 หลังการแฟลช จึงยังไม่สรุป live-camera accuracy หรือ live latency

รายละเอียดอยู่ที่:

- `../../deployment/FLASH_LOG_20260917.md`
- `../../deployment/FLASH_LOG_FULL_FRAME_BASELINE_20260917.md`
- `../../deployment/FLASH_LOG_CAMERA_CONFIG_20260917.md`
- `../../deployment/DOMAIN_SHIFT_DIAGNOSIS_20260917.md`
