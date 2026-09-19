# 🧠 พื้นที่การเทรนและทดลองโมเดล AI (Training Workspace)

พื้นที่นี้รวบรวมโค้ดเตรียมข้อมูล (Data Preparation), การทำ Augmentation, สคริปต์การทดลองเทรนโมเดลภายใต้ข้อจำกัด Edge AI, และการแปลงโมเดลเป็น Full INT8 Quantization สำหรับไมโครคอนโทรลเลอร์ ESP32-S3

---

## 📁 รายการสคริปต์และเครื่องมือในโฟลเดอร์นี้

| ไฟล์ / สคริปต์ | หน้าที่และการทำงาน |
|---|---|
| **`small_model_experiment.py`** | สคริปต์หลักสำหรับเทรนและทดสอบโมเดล **Separable CNN** (คุมพารามิเตอร์ไม่เกิน 100,000 parameters), ทำ Full INT8 Quantization, และบันทึกผลลัพธ์เป็น log/json |
| **`local_training_panel.py`** | หน้าต่าง GUI ควบคุมการเตรียมข้อมูลและกดเริ่มเทรนบนเครื่องคอมพิวเตอร์ แสดงสถานะและ Log แบบเรียลไทม์ |
| **`start_local_training_panel.bat`** | ไฟล์สำหรับดับเบิลคลิกเปิดหน้าต่าง GUI `local_training_panel.py` บน Windows แบบ 1-Click |
| **`prepare_deployment_matched_dataset.py`** | ตรวจจับ foreground ผลมังคุดจากภาพต้นฉบับ สร้าง square fruit crop พร้อม padding แล้วปรับขนาดเป็น `96x96` เพื่อวัดผล deployment-matched |
| **`prepare_camera_matched_dataset.py`** | จัดระเบียบและจับคู่ภาพจากกล้อง OV2640 ให้ตรงกับสเปก Preprocessing ของเฟิร์มแวร์ |
| **`prepare_raw_clean_dataset.py`** | สคริปต์ตรวจเช็คความถูกต้องของชุดภาพต้นฉบับ ลบภาพชำรุดหรือไม่สมบูรณ์ |
| **`augment_and_save_dataset.py`** | สคริปต์เพิ่มจำนวนภาพฝึกสอน (Data Augmentation) เช่น พลิกภาพ หมุนภาพ ปรับแสง โดยเฉพาะคลาสที่มีตัวอย่างน้อย |
| **`generate_latest_figures.py`** | สคริปต์สร้างแผนภาพ Confusion Matrix, Architecture SVG และ Training Curves จากผลการรันล่าสุด |
| **`Mangosteen_Training_SeparableCNN64.ipynb`** | สมุดโน้ต Jupyter Notebook สำหรับเปิดรันบน **Google Colab** ครบวงจรตั้งแต่เทรนจนถึงแปลงเป็น C-Array และดาวน์โหลด |
| **`colab_small_model_experiment_code.txt`** | ซอร์สโค้ดฉบับสำรองสำหรับ Copy ไปวางและรันบน Google Colab |
| **`Mangosteen_EdgeAI/`** | คลังเก็บ Dataset, ข้อมูลโมเดล และผลการทดลองแยกตาม timestamp (`01_data`, `03_models`, `04_results`, `05_deployment`, `06_docs`) |

---

## 🚀 วิธีการใช้งาน

### วิธีที่ 1: รันผ่านหน้าต่างควบคุม Local Training Panel (สะดวกที่สุดบนเครื่องคอม)
1. ดับเบิลคลิกที่ไฟล์ **`start_local_training_panel.bat`**
2. ในหน้าต่างที่เปิดขึ้นมา ให้กดปุ่ม **`แยก Camera-Matched Dataset`** เพื่อเตรียมชุดข้อมูล
3. กดปุ่ม **`เริ่มเทรน Separable CNN 64`** หน้าต่างจะแสดงผลการเทรนทีละ epoch จนจบ พร้อมแปลงเป็น TFLite INT8 อัตโนมัติ

### วิธีที่ 2: รันผ่าน Terminal / Command Line
```bash
# รันการทดลองเทรนโมเดลเต็มรูปแบบ
python training/small_model_experiment.py --mode full

# หรือรันเฉพาะขั้นตอนประเมินผล
python training/small_model_experiment.py --mode eval
```

ผลลัพธ์จะถูกบันทึกอัตโนมัติลงใน `training/Mangosteen_EdgeAI/04_results/experiments/<timestamp>/` ซึ่งประกอบด้วย:
- `run.log` — บันทึกข้อความการเทรนทั้งหมด
- `events.jsonl` — ข้อมูลสถิติเชิงโครงสร้าง
- `summary.json` — ค่าความแม่นยำ (Accuracy, F1-score, Confusion Matrix)
- `separable_cnn_64_int8.tflite` — ไฟล์โมเดลที่พร้อมนำไป deploy

### วิธีที่ 3: รันบน Google Colab (ผ่านคลาวด์ GPU ฟรี)
1. อัปโหลดหรือเปิดไฟล์ **`training/Mangosteen_Training_SeparableCNN64.ipynb`** บน [Google Colab](https://colab.research.google.com/)
2. สลับ Runtime เป็น GPU (`Runtime` -> `Change runtime type` -> `T4 GPU`)
3. รันเซลล์ตามลำดับจากบนลงล่าง เซลล์จะเทรนโมเดล, แสดงกราฟ, ทำ Quantization INT8, แปลงเป็นโค้ดภาษา C และดาวน์โหลดไฟล์ `.zip` ให้อัตโนมัติ

