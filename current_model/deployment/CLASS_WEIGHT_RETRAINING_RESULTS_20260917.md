# Class-weight retraining results

วันที่: 2026-09-17  
วัตถุประสงค์: ทดลองปรับน้ำหนักคลาสบนชุดภาพเดิม เพื่อดูว่าจะลดปัญหาโมเดลบนบอร์ดสับสนผลไม้ `unripe` โดยไม่เพิ่มภาพจากภายนอกหรือไม่

## ขอบเขตการทดลอง

- ใช้ชุดภาพเดิมที่ผ่านการตัดเป็นภาพ `96 x 96` แล้วเท่านั้น
- ไม่เพิ่มภาพจากภายนอก
- แบ่งชุดข้อมูลเดิม: train 215 ภาพ, validation 35 ภาพ, test 38 ภาพ
- สถาปัตยกรรมเดิม: Separable CNN 64
- จำนวนพารามิเตอร์: 61,059 (ต่ำกว่าเกณฑ์ 100,000)
- augmentation: `camera_robust`
- head training 20 epochs + fine-tuning 25 epochs
- batch size 16
- ตรวจทั้ง Float32 และ INT8

คำสั่งที่ใช้:

```powershell
.venv_small_model\Scripts\python.exe training\small_model_experiment.py --mode train --variants separable_cnn_64 --augmentation-profile camera_robust --class-weight-profile uniform --head-epochs 20 --fine-tune-epochs 25 --batch-size 16
.venv_small_model\Scripts\python.exe training\small_model_experiment.py --mode train --variants separable_cnn_64 --augmentation-profile camera_robust --class-weight-profile unripe_guard --head-epochs 20 --fine-tune-epochs 25 --batch-size 16
```

## ผลการทดลอง

| Run | Class-weight profile | INT8 accuracy | Balanced accuracy | overripe recall | ripe recall | unripe recall | Model size |
|---|---|---:|---:|---:|---:|---:|---:|
| `20260916_235808` | `uniform` | 86.84% (33/38) | 61.11% | 0/2 (0%) | 15/18 (83.33%) | 18/18 (100%) | 102,912 bytes / 100.50 KiB |
| `20260916_235945` | `unripe_guard` | 86.84% (33/38) | 61.11% | 0/2 (0%) | 15/18 (83.33%) | 18/18 (100%) | 102,912 bytes / 100.50 KiB |

Float32 ให้ผลเท่ากับ INT8 ทั้งสองรอบ จึงไม่พบความคลาดเคลื่อนจาก quantization ในการทดสอบชุดนี้

## ไฟล์ผลลัพธ์

- [uniform summary](D:/MiniProject%20mangosteen/training/Mangosteen_EdgeAI/04_results/experiments/20260916_235808/separable_cnn_64/summary.json)
- [uniform INT8 model](D:/MiniProject%20mangosteen/training/Mangosteen_EdgeAI/04_results/experiments/20260916_235808/separable_cnn_64/separable_cnn_64_int8.tflite)
- [unripe_guard summary](D:/MiniProject%20mangosteen/training/Mangosteen_EdgeAI/04_results/experiments/20260916_235945/separable_cnn_64/summary.json)
- [unripe_guard INT8 model](D:/MiniProject%20mangosteen/training/Mangosteen_EdgeAI/04_results/experiments/20260916_235945/separable_cnn_64/separable_cnn_64_int8.tflite)

## ข้อสรุป

ยังไม่เลือกโมเดลใดไปแทนโมเดลที่ใช้งานบนบอร์ด และยังไม่ได้แฟลชโมเดลรอบนี้ เนื่องจากทั้งสองรอบยังทำนาย `overripe` ในชุดทดสอบไม่ได้เลย แม้จะรักษา recall ของ `unripe` ได้ 100%.

ผลนี้ชี้ว่าการเพิ่ม class weight อย่างเดียวไม่แก้ปัญหาหลัก ปัญหาน่าจะอยู่ที่ความแตกต่างระหว่างภาพ crop `96 x 96` ในชุดฝึกกับภาพจาก OV2640 จริง รวมถึงข้อมูล `overripe` ที่มีน้อยมากในชุด train/test.

สถานะบอร์ดยังคงเป็น firmware diagnostic ที่ใช้ center-crop ROI และโมเดลเดิม ไม่ใช่โมเดลจากการทดลองสองรอบนี้
