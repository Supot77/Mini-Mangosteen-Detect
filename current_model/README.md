# Current model snapshot

โฟลเดอร์นี้คือ snapshot ของโมเดลที่ใช้งานปัจจุบันจาก run `20260917_001333`

## Canonical files

- `model/separable_cnn_64_int8.tflite` — Full INT8 model สำหรับ deploy
- `model/separable_cnn_64.keras` — Keras training artifact
- `model/metadata.json` — model contract และแหล่งที่มา
- `results/20260917_001333/` — training history, metrics, classification reports, confusion matrix และ logs
- `dataset_deployment_matched_v4/` — dataset ที่ใช้ใน run นี้: train 214, validation 34, test 38
- `evaluation/original_crop_test/` — held-out original-crop test set 38 ภาพสำหรับผลหลัก 94.74%
- `figures/` — training/validation curves, original-crop INT8 confusion matrix และ architecture
- `deployment/` — deployment mirror ของ TFLite และ C-array
- `report/` — รายงานที่สอดคล้องกับ artifact ปัจจุบัน

## Verified result

โมเดลมี 61,059 parameters, input `96 × 96 × 3` RGB และ TFLite ขนาด 102,912 bytes (100.5 KiB)

- Validation: 91.18% (31/34)
- Original-crop robustness test: **94.74% (36/38)** ทั้ง Float32 และ INT8
- Balanced accuracy: **96.30%** · Macro F1: **0.905** บน original-crop test เดียวกัน
- Confusion matrix: `overripe 2/2`, `ripe 16/18`, `unripe 18/18`
- Deployment-matched crop check: 100.00% (38/38) ทั้ง Float32 และ INT8

ค่า 94.74% ใช้เป็นผลหลักในการนำเสนอ เพราะเป็น primary offline test ด้วย crop แบบเดิมบนภาพที่กันไว้จากการ train ส่วนค่า 100.00% เป็น secondary engineering check ของ matched crop และยังไม่ใช่ live-camera accuracy จาก OV2640 จริง

## Deploy manually

```bash
python convert_tflite_to_c.py current_model/model/separable_cnn_64_int8.tflite --out-dir src
pio run -t upload
```

ไฟล์ C-array ที่ firmware ใช้งานจริงยังคงอยู่ที่ `src/mangosteen_model_data.{h,cc}` ส่วนสำเนาเดิมและรายงานเก่าถูกเก็บไว้ใน `provenance/` เพื่อการตรวจสอบย้อนหลัง
