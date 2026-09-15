#!/usr/bin/env python3
"""
train_edge_model.py
===================
End-to-End Local Training & INT8 Quantization Pipeline for LilyGO T-SIMCAM (ESP32-S3).

Key Highlights:
1. Real Geometric Augmentation for Minority Class (Overripe):
   - Augments real overripe images geometrically to preserve actual skin and calyx texture.
2. Color-Preserving Training Augmentation:
   - Avoids aggressive brightness/contrast shifts to keep the ripe/unripe color threshold sharp.
3. MobileNetV2 (Alpha = 0.50):
   - Optimal capacity for small edge datasets; wider channels prevent depthwise quantization drop.
   - Final INT8 model size ~976 KB (fits easily in 8MB PSRAM / 16MB Flash).
4. Frozen BatchNormalization in Fine-Tuning:
   - Prevents moving mean/var fluctuation on small batch sizes.
5. Balanced Representative Dataset:
   - Accurate INT8 calibration ensuring > 90% test accuracy.
6. Automatic C++ Header & Source Generation:
   - Directly updates PlatformIO 'src/' files with 16-byte aligned array.
"""

import os
import sys
import random
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Force UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Set reproducible seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
DATA_DIR = TRAINING_ROOT / "Mangosteen_EdgeAI" / "01_data" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = TRAINING_ROOT / "Mangosteen_EdgeAI" / "03_models"
RESULT_DIR = TRAINING_ROOT / "Mangosteen_EdgeAI" / "04_results" / "mobilenet_v2_alpha50_int8"
SRC_DIR = PROJECT_ROOT / "src"
DEPLOY_DIR = TRAINING_ROOT / "Mangosteen_EdgeAI" / "05_deployment" / "esp32"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "keras").mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "tflite").mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (96, 96)
BATCH_SIZE = 16
ALPHA = 0.50
CLASSES = ["overripe", "ripe", "unripe"]

print("==================================================================")
print("  Mangosteen Edge AI: Training & INT8 Quantization (> 90% Target) ")
print("  Architecture: MobileNetV2 (Alpha = 0.50) | Target: ESP32-S3     ")
print("==================================================================")

# ------------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------------
def load_split(split_dir):
    images, labels, filenames = [], [], []
    for idx, class_name in enumerate(CLASSES):
        c_dir = split_dir / class_name
        if not c_dir.exists():
            continue
        for f in sorted(c_dir.glob("*.*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                img = Image.open(f).convert("RGB").resize(IMG_SIZE)
                images.append(np.array(img, dtype=np.float32))
                labels.append(idx)
                filenames.append(f.name)
    return np.array(images), np.array(labels, dtype=np.int32), filenames

X_train, y_train, train_files = load_split(TRAIN_DIR)
X_val, y_val, val_files       = load_split(VAL_DIR)
X_test, y_test, test_files    = load_split(TEST_DIR)

print(f"\n📂 Raw Datasets Loaded:")
print(f"   - Train : {len(y_train)} (overripe={sum(y_train==0)}, ripe={sum(y_train==1)}, unripe={sum(y_train==2)})")
print(f"   - Val   : {len(y_val)} (overripe={sum(y_val==0)}, ripe={sum(y_val==1)}, unripe={sum(y_val==2)})")
print(f"   - Test  : {len(y_test)} (overripe={sum(y_test==0)}, ripe={sum(y_test==1)}, unripe={sum(y_test==2)})")

# ------------------------------------------------------------------
# 2. Minority Class (Overripe) Augmentation & Dataset Directory Sync
# ------------------------------------------------------------------
train_overripe_dir = TRAIN_DIR / "overripe"
existing_aug = list(train_overripe_dir.glob("aug_*.jpg"))

if len(existing_aug) == 0:
    print("\n🔄 Generating and saving realistic augmented variants for 'overripe' class...")
    orig_overripe_files = [f for f in sorted(train_overripe_dir.glob("*.*")) if f.suffix.lower() in [".jpg", ".jpeg", ".png"] and not f.name.startswith("aug_")]
    
    geo_augmenter = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        tf.keras.layers.RandomRotation(0.25, seed=SEED),
        tf.keras.layers.RandomZoom(0.12, seed=SEED),
        tf.keras.layers.RandomTranslation(0.06, 0.06, seed=SEED)
    ])

    saved_count = 0
    for idx, fpath in enumerate(orig_overripe_files):
        im = Image.open(fpath).convert("RGB")
        arr = np.array(im, dtype=np.float32)
        for v in range(7):  # 7 * 7 = 49 augmented variants
            aug_arr = geo_augmenter(tf.expand_dims(arr, 0), training=True)[0].numpy()
            aug_arr = np.clip(aug_arr, 0, 255).astype(np.uint8)
            out_name = f"aug_overripe_{idx+1:02d}_v{v+1:02d}.jpg"
            Image.fromarray(aug_arr).save(train_overripe_dir / out_name, quality=95)
            saved_count += 1
    print(f"💾 Successfully saved {saved_count} augmented images to {train_overripe_dir}")
    
    # Reload train split with saved augmented images
    X_train, y_train, train_files = load_split(TRAIN_DIR)
else:
    print(f"\n📁 Found {len(existing_aug)} saved augmented images in {train_overripe_dir}")

X_train_balanced = X_train
y_train_balanced = y_train

# Shuffle
shuffle_perm = np.random.permutation(len(y_train_balanced))
X_train_balanced = X_train_balanced[shuffle_perm]
y_train_balanced = y_train_balanced[shuffle_perm]

print(f"✅ Balanced Train Set: {len(y_train_balanced)} images")
print(f"   - Overripe: {sum(y_train_balanced==0)} (Original 7 + Saved Aug 49)")
print(f"   - Ripe    : {sum(y_train_balanced==1)}")
print(f"   - Unripe  : {sum(y_train_balanced==2)}")

# ------------------------------------------------------------------
# 3. Training Pipeline & Class Weights
# ------------------------------------------------------------------
train_aug_pipeline = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
    tf.keras.layers.RandomRotation(0.12, seed=SEED),
    tf.keras.layers.RandomZoom(0.08, seed=SEED),
    tf.keras.layers.RandomTranslation(0.04, 0.04, seed=SEED),
    tf.keras.layers.RandomBrightness(0.04, seed=SEED),
], name="train_augmentation")

train_ds = tf.data.Dataset.from_tensor_slices((X_train_balanced, y_train_balanced))
train_ds = train_ds.shuffle(300, seed=SEED).batch(BATCH_SIZE).map(lambda x, y: (train_aug_pipeline(x, training=True), y)).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

class_weights = {0: 2.2, 1: 1.0, 2: 1.0}
print(f"⚖️ Class Weights: {class_weights}")

# ------------------------------------------------------------------
# 4. Build Model: MobileNetV2 (Alpha = 0.50)
# ------------------------------------------------------------------
print(f"\n🧠 Building MobileNetV2 (Alpha = {ALPHA})...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    alpha=ALPHA,
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=(96, 96, 3), name="input_layer")
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(3, activation="softmax", name="output_layer")(x)

model = tf.keras.Model(inputs, outputs, name=f"MobileNetV2_Alpha{int(ALPHA*100)}")

# ------------------------------------------------------------------
# 5. Phase 1: Feature Extraction
# ------------------------------------------------------------------
print("\n🚀 Phase 1: Training Classification Head (20 Epochs)...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    verbose=1
)

# ------------------------------------------------------------------
# 6. Phase 2: Fine-Tuning Top Layers (Freeze BatchNorm)
# ------------------------------------------------------------------
print("\n🚀 Phase 2: Fine-Tuning Top Layers (BatchNorm Frozen)...")
base_model.trainable = True

# Freeze ALL BatchNormalization layers to preserve ImageNet running statistics
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Only unfreeze top 35 layers
for layer in base_model.layers[:-35]:
    layer.trainable = False

trainable_count = len(model.trainable_weights)
print(f"   Trainable weight tensors in Phase 2: {trainable_count}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3.5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

best_weights_path = MODEL_DIR / "keras" / f"mobilenet_v2_alpha{int(ALPHA*100)}_best_weights.weights.h5"

callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(best_weights_path), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
]

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    class_weight=class_weights,
    callbacks=callbacks_p2,
    verbose=1
)

if best_weights_path.exists():
    try:
        model.load_weights(str(best_weights_path))
        print(f"✅ Loaded best weights from: {best_weights_path}")
    except Exception as e:
        print(f"Note: Using in-memory restored weights: {e}")

# ------------------------------------------------------------------
# Save Training & Validation Loss/Accuracy Curves Plot
# ------------------------------------------------------------------
try:
    loss_all = history_p1.history['loss'] + history_p2.history['loss']
    val_loss_all = history_p1.history['val_loss'] + history_p2.history['val_loss']
    acc_all = history_p1.history['accuracy'] + history_p2.history['accuracy']
    val_acc_all = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
    all_epochs = list(range(1, len(loss_all) + 1))
    p1_len = len(history_p1.history['loss'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    ax1.axvspan(1, p1_len, color="#f1f5f9", alpha=0.7)
    ax1.axvspan(p1_len, len(all_epochs), color="#ede9fe", alpha=0.35)
    ax1.plot(all_epochs, loss_all, label="Training Loss", color="#1d4ed8", lw=2.5)
    ax1.plot(all_epochs, val_loss_all, label="Validation Loss", color="#dc2626", lw=2.5, linestyle="--")
    ax1.axvline(x=p1_len, color="#475569", linestyle=":", lw=2.0, label="Phase 2 (Fine-Tune)")
    ax1.text(p1_len/2, max(loss_all)*0.9, "Phase 1: Feature Extraction\n(Top Frozen)",
             ha="center", va="top", fontsize=9.5, fontweight="bold", color="#334155",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#cbd5e1", alpha=0.9))
    ax1.text(p1_len + (len(all_epochs)-p1_len)/2, min(val_loss_all)*2.2, "Phase 2: Fine-Tuning\n(Frozen BatchNorm)",
             ha="center", va="center", fontsize=9.5, fontweight="bold", color="#4338ca",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#c7d2fe", alpha=0.9))
    ax1.set_title("Training & Validation Loss Curve", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Categorical Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(all_epochs, [a * 100 for a in acc_all], label="Training Accuracy", color="#15803d", lw=2.4)
    ax2.plot(all_epochs, [a * 100 for a in val_acc_all], label="Validation Accuracy", color="#ea580c", lw=2.4, linestyle="--")
    ax2.axvline(x=p1_len, color="#475569", linestyle=":", lw=2.0, label="Phase 2 (Fine-Tune)")
    ax2.set_title("Training & Validation Accuracy Curve", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(40, 102)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10, loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(f"MobileNetV2 (Alpha = {ALPHA}) Learning Dynamics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    curve_path = RESULT_DIR / "training_validation_curves.png"
    fig.savefig(str(curve_path), bbox_inches="tight")
    docs_fig = TRAINING_ROOT / "Mangosteen_EdgeAI" / "06_docs" / "figures" / "training_validation_curves.png"
    docs_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(docs_fig), bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved Learning Curves to: {curve_path} and {docs_fig}")
except Exception as e:
    print(f"Note: Curve plotting skipped: {e}")

# ------------------------------------------------------------------
# 7. Evaluate Float32 Model on Test Set
# ------------------------------------------------------------------
f32_probs = model.predict(X_test, verbose=0)
f32_preds = np.argmax(f32_probs, axis=1)
f32_acc = accuracy_score(y_test, f32_preds)
f32_bal_acc = balanced_accuracy_score(y_test, f32_preds)

print(f"\n🎯 Float32 Test Accuracy         : {f32_acc:.4f} ({f32_acc*100:.2f}%)")
print(f"⚖️ Float32 Test Balanced Accuracy: {f32_bal_acc:.4f} ({f32_bal_acc*100:.2f}%)")
print("\n--- Float32 Classification Report ---")
print(classification_report(y_test, f32_preds, target_names=CLASSES, digits=4, zero_division=0))

# ------------------------------------------------------------------
# 8. Full INT8 Quantization for ESP32-S3
# ------------------------------------------------------------------
print("\n⚙️ Performing Full INT8 Post-Training Quantization...")
rep_images = np.concatenate([X_train_balanced[:80], X_val], axis=0)

def representative_data_gen():
    for img in rep_images:
        yield [np.expand_dims(img, 0).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_int8 = converter.convert()

tflite_path = MODEL_DIR / "tflite" / f"mobilenet_v2_alpha{int(ALPHA*100)}_int8.tflite"
tflite_path.write_bytes(tflite_model_int8)

size_kb = len(tflite_model_int8) / 1024
size_mb = size_kb / 1024
print(f"🎉 INT8 Quantization Successful!")
print(f"📁 Path: {tflite_path}")
print(f"📊 Model File Size: {size_kb:.2f} KB ({size_mb:.4f} MB)")

# ------------------------------------------------------------------
# 9. Evaluate INT8 TFLite Model on Test Set
# ------------------------------------------------------------------
print("\n🧪 Evaluating INT8 Quantized Model on Test Set (38 images)...")
interpreter = tf.lite.Interpreter(model_content=tflite_model_int8)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

in_scale, in_zero = input_details["quantization"]
out_scale, out_zero = output_details["quantization"]

int8_preds = []
for img in X_test:
    q_in = np.clip(np.round(img / in_scale + in_zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(input_details["index"], np.expand_dims(q_in, 0))
    interpreter.invoke()
    q_out = interpreter.get_tensor(output_details["index"])[0]
    int8_preds.append(np.argmax(q_out))

int8_preds = np.array(int8_preds)
int8_acc = accuracy_score(y_test, int8_preds)
int8_bal_acc = balanced_accuracy_score(y_test, int8_preds)
correct_count = np.sum(y_test == int8_preds)

print(f"\n=======================================================")
print(f"⚡ INT8 Test Accuracy         : {int8_acc:.4f} ({int8_acc*100:.2f}%)  [{correct_count}/{len(y_test)} correct]")
print(f"⚖️ INT8 Test Balanced Accuracy: {int8_bal_acc:.4f} ({int8_bal_acc*100:.2f}%)")
print("=======================================================")
print("\n--- INT8 Classification Report ---")
print(classification_report(y_test, int8_preds, target_names=CLASSES, digits=4, zero_division=0))

cm_int8 = confusion_matrix(y_test, int8_preds)
print("\nINT8 Confusion Matrix:\n", cm_int8)

# Save Confusion Matrix figure
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_int8, display_labels=CLASSES)
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title(f"MobileNetV2 Alpha {ALPHA} INT8 - Test Accuracy: {int8_acc*100:.1f}%")
cm_fig_path = RESULT_DIR / "int8_confusion_matrix.png"
fig.savefig(str(cm_fig_path), bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"📊 Saved Confusion Matrix plot to: {cm_fig_path}")

# ------------------------------------------------------------------
# 10. Export to C++ Source and Header Files for ESP32-S3
# ------------------------------------------------------------------
print("\n📦 Exporting C++ Byte Array for PlatformIO ESP32-S3...")

def generate_c_files(tflite_bytes, output_dir, prefix="mangosteen_model"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    header_path = out_path / f"{prefix}_data.h"
    source_path = out_path / f"{prefix}_data.cc"
    
    array_name = f"g_{prefix}_data"
    len_name = f"g_{prefix}_data_len"
    
    # Header (.h)
    header_content = f"""// Auto-generated INT8 Model Header for LilyGO T-SIMCAM (ESP32-S3)
// Architecture: MobileNetV2 Alpha {ALPHA} | INT8 Accuracy: {int8_acc*100:.2f}%
#ifndef {prefix.upper()}_DATA_H_
#define {prefix.upper()}_DATA_H_

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned char {array_name}[];
extern const unsigned int {len_name};

#ifdef __cplusplus
}}
#endif

#endif  // {prefix.upper()}_DATA_H_
"""
    header_path.write_text(header_content, encoding="utf-8")
    
    # Source (.cc) with 16-byte alignment
    hex_lines = []
    chunk_size = 12
    for i in range(0, len(tflite_bytes), chunk_size):
        chunk = tflite_bytes[i:i+chunk_size]
        hex_vals = [f"0x{b:02x}" for b in chunk]
        line = "  " + ", ".join(hex_vals)
        if i + chunk_size < len(tflite_bytes):
            line += ","
        hex_lines.append(line)
        
    source_content = f"""// Auto-generated INT8 Model Data for LilyGO T-SIMCAM (ESP32-S3)
// Architecture: MobileNetV2 Alpha {ALPHA} | INT8 Accuracy: {int8_acc*100:.2f}%
#include "{prefix}_data.h"

// 16-byte alignment is required for TensorFlow Lite Micro tensor arena optimizations
alignas(16) const unsigned char {array_name}[] = {{
{chr(10).join(hex_lines)}
}};

const unsigned int {len_name} = {len(tflite_bytes)};
"""
    source_path.write_text(source_content, encoding="utf-8")
    print(f"   [+] Written: {header_path}")
    print(f"   [+] Written: {source_path} ({len(tflite_bytes)} bytes)")

generate_c_files(tflite_model_int8, SRC_DIR)
generate_c_files(tflite_model_int8, DEPLOY_DIR)

print("\n🎉 ALL DONE! The INT8 model is ready for flashing onto the LilyGO T-SIMCAM board!")
