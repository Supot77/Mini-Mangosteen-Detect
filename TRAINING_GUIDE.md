# Training guide: Separable CNN 64

โมเดลที่ใช้งานปัจจุบันคือ **Separable CNN 64** ซึ่งมี 61,059 parameters จึงผ่านเกณฑ์ไม่เกิน 100,000 parameters มี validation accuracy **91.18% (31/34)** และมี primary original-crop offline accuracy **94.74% (36/38)** ทั้ง Float32 และ Full INT8 พร้อม balanced accuracy **96.30%** และ macro F1 **0.905**

ค่า 94.74% เป็นผลจากการใช้ crop แบบเดิมกับภาพ test ที่กันไว้และไม่ผ่าน augmentation ส่วน deployment-matched crop check ได้ 100.00% (38/38) ทั้ง Float32 และ Full INT8 ซึ่งเป็นเพียง secondary preprocessing check ผลทั้งสองแบบยังไม่ใช่ตัวเลขยืนยัน accuracy จากภาพกล้อง OV2640 บนบอร์ดจริง

## Local

ติดตั้ง dependencies ของงานฝึกโมเดล แล้วรันจาก project root:

```powershell
python -m pip install tensorflow scikit-learn matplotlib pillow
python training/small_model_experiment.py --mode full
```

ผลลัพธ์ใหม่จะถูกบันทึกใน:

```text
training/Mangosteen_EdgeAI/04_results/experiments/<timestamp>/
```

ทุก run จะมี `run.log`, `events.jsonl`, `summary.json`, model summary, training history, classification report และ TFLite artifact

## Local training panel

ถ้าต้องการนั่งกดบนเครื่องเอง ให้ดับเบิลคลิก [training/start_local_training_panel.bat](training/start_local_training_panel.bat) แล้วกดตามลำดับ:

1. **แยก Camera-Matched Dataset** — ใช้ภาพต้นฉบับในโปรเจกต์และ label/split เดิม โดยจำลอง `RGB565 240x240 -> 96x96`
2. **เริ่มเทรน Separable CNN 64** — ใช้ dataset ที่เพิ่งสร้างด้วย `raw_camera_robust` augmentation

หน้าต่างจะแสดง log แบบต่อเนื่อง และสร้าง output แบบ timestamped โดยไม่เขียนทับ dataset เดิมหรือใช้ภาพจากภายนอก

## Google Colab

เปิดไฟล์ [training/Mangosteen_Training_SeparableCNN64.ipynb](training/Mangosteen_Training_SeparableCNN64.ipynb) (หรือ [Mangosteen_Training_SeparableCNN64.ipynb](Mangosteen_Training_SeparableCNN64.ipynb)) บน [Google Colab](https://colab.research.google.com/) โดยมีปุ่ม **Open in Colab** พร้อมรันได้ทันที

ไฟล์ Notebook บรรจุขั้นตอนการเทรน **Separable CNN 64** ฉบับสมบูรณ์:
1. **Clone & Setup Environment** — ดึง Repository จาก GitHub พร้อมตรวจสอบ GPU
2. **Dataset Inspection & Loading** — โหลดชุดข้อมูลและแสดงรูปตัวอย่าง
3. **Build Separable CNN 64** — โครงสร้าง Depthwise-Separable CNN 61,059 parameters (< 100k)
4. **Data Augmentation & Balance** — In-Memory Oversampling และ Camera-Robust Augmentation
5. **Two-Stage Training** — Head Training (lr=1e-3) + Fine-Tuning (lr=3.5e-5)
6. **Evaluation & Visualization** — กราฟการฝึกสอน, Classification Report, Confusion Matrix
7. **Full INT8 Quantization** — แปลงเป็น TFLite INT8 (~100.5 KiB) และทดสอบความแม่นยำเทียบ Float32
8. **Export C/C++ Array** — แปลงโมเดลเป็น `mangosteen_model_data.{h,cc}` สำหรับ ESP32-S3
9. **Download Artifacts** — รวมไฟล์และกดดาวน์โหลดลงเครื่องเพื่อนำไป Build ใน PlatformIO ได้ทันที

## การเลือกไฟล์ไป deploy

หลังตรวจผลแล้ว ให้ใช้ไฟล์:

```text
current_model/model/separable_cnn_64_int8.tflite
```

หรือคัดลอกไปที่:

```text
current_model/model/separable_cnn_64_int8.tflite
```

จากนั้น `python setup.py` จะใช้โมเดลนี้เป็นค่าเริ่มต้น แปลงเป็น `src/mangosteen_model_data.{h,cc}` และเรียก PlatformIO สำหรับ build/upload

## สรุปผลล่าสุด

ดูตารางและขอบเขตหลักฐานได้ที่ [RESULT_SUMMARY.md](current_model/results/20260917_001333/RESULT_SUMMARY.md)

โมเดล MobileNetV2 และคู่มือเดิมถูกเก็บไว้ที่ `archive/mobilenet_v2_legacy/` เพื่ออ้างอิงย้อนหลังเท่านั้น
