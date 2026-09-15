#!/usr/bin/env python3
"""
augment_and_save_dataset.py
===========================
Utility to generate and save augmented images directly into the dataset directories:
- Expands minority class ('overripe') from 7 to 56 samples to balance the dataset on disk.
- Applies realistic geometric transformations (flip, rotation, zoom, translation) that preserve
  true pericarp color and calyx characteristics.
- Supports cleaning up augmented images or augmenting all classes.

Usage:
    python training/augment_and_save_dataset.py            # Augments overripe to balance dataset
    python training/augment_and_save_dataset.py --clean    # Removes generated 'aug_*' images
    python training/augment_and_save_dataset.py --factor 7 # Number of variants per image
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_TRAIN = PROJECT_ROOT / "Mangosteen_EdgeAI" / "01_data" / "dataset" / "train"

def get_original_images(folder: Path):
    if not folder.exists():
        return []
    valid_exts = {".jpg", ".jpeg", ".png"}
    return [f for f in sorted(folder.glob("*.*")) if f.suffix.lower() in valid_exts and not f.name.startswith("aug_")]

def clean_augmented(folder: Path):
    aug_files = list(folder.glob("aug_*.*"))
    for f in aug_files:
        try:
            f.unlink()
        except Exception as e:
            print(f"Error removing {f.name}: {e}")
    return len(aug_files)

def augment_image_pil(img: Image.Image, seed: int):
    rng = np.random.default_rng(seed)
    # 1. Flip
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 2. Rotation (-20 to +20 degrees) with bicubic resampling
    angle = rng.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    # 3. Random zoom / crop (0.92 to 1.08)
    zoom = rng.uniform(0.92, 1.08)
    w, h = img.size
    new_w, new_h = int(w * zoom), int(h * zoom)
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    # Crop or pad back to original (w, h)
    if zoom > 1.0:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        img = img_resized.crop((left, top, left + w, top + h))
    else:
        pad_img = Image.new("RGB", (w, h), (0, 0, 0))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        pad_img.paste(img_resized, (left, top))
        img = pad_img

    return img

def main():
    parser = argparse.ArgumentParser(description="Augment and save images to dataset directory")
    parser.add_argument("--clean", action="store_true", help="Remove all generated aug_* images")
    parser.add_argument("--target-class", default="overripe", choices=["overripe", "ripe", "unripe", "all"], help="Class to augment")
    parser.add_argument("--factor", type=int, default=7, help="Augmentation multiplier per original image")
    args = parser.parse_args()

    classes_to_process = ["overripe", "ripe", "unripe"] if args.target_class == "all" else [args.target_class]

    print("==========================================================")
    print("  Mangosteen Dataset Augmentation & Storage Manager      ")
    print("==========================================================")

    if args.clean:
        total_removed = 0
        for c in ["overripe", "ripe", "unripe"]:
            c_dir = DATASET_TRAIN / c
            removed = clean_augmented(c_dir)
            total_removed += removed
            print(f"🗑️  Removed {removed} augmented images from '{c}'")
        print(f"\n✅ Clean complete. Total removed: {total_removed} files.")
        return

    for c in classes_to_process:
        c_dir = DATASET_TRAIN / c
        if not c_dir.exists():
            print(f"⚠️  Directory not found: {c_dir}")
            continue

        cleaned = clean_augmented(c_dir)
        if cleaned > 0:
            print(f"🧹 Cleaned {cleaned} previous augmented images from '{c}'")

        originals = get_original_images(c_dir)
        print(f"\n📂 Processing Class '{c}': Found {len(originals)} original images")

        saved_count = 0
        for idx, orig_path in enumerate(originals):
            try:
                base_img = Image.open(orig_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open {orig_path.name}: {e}")
                continue

            for v in range(args.factor):
                seed = (idx + 1) * 1000 + v + 42
                aug_img = augment_image_pil(base_img, seed=seed)
                out_name = f"aug_{c}_{idx+1:02d}_v{v+1:02d}.jpg"
                out_path = c_dir / out_name
                aug_img.save(out_path, quality=95)
                saved_count += 1

        total_files = len(list(c_dir.glob("*.*")))
        print(f"✅ Saved {saved_count} new augmented images.")
        print(f"📊 Total images in '{c}' folder: {total_files} (Original: {len(originals)}, Augmented: {saved_count})")

    print("\n----------------------------------------------------------")
    print("📁 Summary of Dataset Train directory:")
    for c in ["overripe", "ripe", "unripe"]:
        c_dir = DATASET_TRAIN / c
        if c_dir.exists():
            cnt = len(list(c_dir.glob("*.*")))
            orig_cnt = len(get_original_images(c_dir))
            aug_cnt = cnt - orig_cnt
            print(f"   - {c:<10}: {cnt:>3} images (Original: {orig_cnt}, Aug: {aug_cnt})")
    print("----------------------------------------------------------")

if __name__ == "__main__":
    main()
