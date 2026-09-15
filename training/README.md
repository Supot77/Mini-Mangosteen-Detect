# Training workspace

โฟลเดอร์นี้รวมไฟล์สำหรับเตรียมข้อมูล เทรนโมเดล และเก็บผลการทดลอง โดยไม่ปะปนกับไฟล์ firmware ที่ใช้ flash ลง ESP32-S3

## โครงสร้าง

- `train_edge_model.py` — เทรน MobileNetV2 Alpha 0.50, ทำ Full INT8 quantization และเขียน model array ไปยัง `../src/`
- `augment_and_save_dataset.py` — สร้างภาพ augmentation ลงใน dataset
- `colab_notebook.ipynb` — notebook สำหรับรันบน Google Colab
- `colab_notebook_code.txt` — text export ของ notebook
- `Mangosteen_EdgeAI/` — dataset, model artifacts, experiment results และ report

รัน local training จาก project root ได้ด้วย:

```text
python training/train_edge_model.py
```

หรือจัดการ augmentation ด้วย:

```text
python training/augment_and_save_dataset.py
```
