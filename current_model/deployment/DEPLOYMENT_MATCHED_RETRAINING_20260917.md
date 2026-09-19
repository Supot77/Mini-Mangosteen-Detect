# Deployment-matched retraining result

วันที่: 2026-09-17  
วัตถุประสงค์: ลด domain shift ระหว่างภาพ crop ที่ใช้เทรนกับภาพจาก OV2640 โดยสร้าง crop ใหม่จากภาพต้นฉบับในโครงการเท่านั้น

## วิธีการ

- ใช้ raw source จาก `01_data/raw/original_images/Dataset`
- ใช้ label และ split เดิมจาก `01_data/dataset`
- ไม่เพิ่มภาพจากภายนอก
- ตรวจจับ foreground ผลไม้จากความแตกต่างของสี/ความสว่างจากพื้นกระดาษ
- ตัดเป็น square crop รอบผลไม้, padding 3%, แล้ว resize เป็น `96 x 96`
- ลบสี marker สีแดงและ ruler สีน้ำเงินที่ติดขอบ crop
- คัดลอกเฉพาะ `aug_` ที่เป็น augmentation ภายในเดิมไปไว้ใน train
- Dataset ที่ได้: train 214, validation 34, test 38

คำสั่งเตรียมข้อมูล:

```powershell
.venv_small_model\Scripts\python.exe training\prepare_deployment_matched_dataset.py --output-root current_model\dataset_deployment_matched_v4 --padding-ratio 0.03
```

คำสั่งเทรน:

```powershell
.venv_small_model\Scripts\python.exe training\small_model_experiment.py --mode train --variants separable_cnn_64 --dataset-root current_model\dataset_deployment_matched_v4 --augmentation-profile camera_robust --class-weight-profile legacy --head-epochs 20 --fine-tune-epochs 25 --batch-size 16
```

## ผลลัพธ์

| Model | INT8 accuracy | Balanced accuracy | overripe recall | ripe recall | unripe recall | INT8 size |
|---|---:|---:|---:|---:|---:|---:|
| โมเดลเดิม, crop เดิม | 86.84% (33/38) | 61.11% | 0/2 | 15/18 | 18/18 | 100.50 KiB |
| โมเดลใหม่, deployment-matched crop | **100.00% (38/38)** | **100.00%** | **2/2** | **18/18** | **18/18** | 100.50 KiB |

Float32 และ INT8 ของโมเดลใหม่ให้ผลเท่ากันบน test set นี้ จึงไม่พบ accuracy loss จาก quantization ในรอบนี้

## ไฟล์หลัก

- เตรียม dataset: `training/prepare_deployment_matched_dataset.py`
- Dataset manifest: `current_model/dataset_deployment_matched_v4/PREPARATION_MANIFEST.json`
- Crop preview: `current_model/dataset_deployment_matched_v4/crop_preview.png`
- Training summary: `current_model/results/20260917_001333/separable_cnn_64/summary.json`
- INT8 candidate: `current_model/results/20260917_001333/separable_cnn_64/separable_cnn_64_int8.tflite`

## สถานะ deployment

- ยังไม่ได้แทนที่ `src/mangosteen_model_data.*`
- ยังไม่ได้แฟลชลงบอร์ด
- ผล 100% เป็นผลจากการทดสอบในเครื่องบน test set 38 ภาพเท่านั้น
- ต้องทดสอบบนกล้อง OV2640 จริงหลังแฟลช เพื่อยืนยันว่า crop preprocessing ใน firmware ตรงกับ crop ที่ใช้เทรน

ข้อสรุปเบื้องต้น: ผลที่ดีขึ้นมากสนับสนุนสมมติฐานว่า root cause หลักคือ preprocessing/crop mismatch มากกว่าการปรับ class weight เพียงอย่างเดียว
