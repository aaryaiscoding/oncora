"""
split_data.py
Splits the Training/ dataset into 3 hospital partitions.
Run this ONCE before starting federated learning.
Each hospital gets a different, non-overlapping subset of images.
"""

import os
import shutil
import random

# ---- Config ----
SOURCE_DIR = "../datasets/Training"
OUTPUT_DIR = "../datasets/federated"
NUM_HOSPITALS = 3
SEED = 42
random.seed(SEED)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def split_dataset():
    # Create hospital folders
    for i in range(1, NUM_HOSPITALS + 1):
        for cls in CLASSES:
            os.makedirs(f"{OUTPUT_DIR}/hospital_{i}/{cls}", exist_ok=True)

    for cls in CLASSES:
        class_path = os.path.join(SOURCE_DIR, cls)
        images = os.listdir(class_path)
        random.shuffle(images)

        # Split images evenly across hospitals
        splits = [images[i::NUM_HOSPITALS] for i in range(NUM_HOSPITALS)]

        for hospital_idx, hospital_images in enumerate(splits):
            dest_dir = f"{OUTPUT_DIR}/hospital_{hospital_idx + 1}/{cls}"
            for img in hospital_images:
                shutil.copy(
                    os.path.join(class_path, img),
                    os.path.join(dest_dir, img)
                )
            print(f"Hospital {hospital_idx + 1} | {cls}: {len(hospital_images)} images")


if __name__ == "__main__":
    print("Splitting dataset into hospital partitions...\n")
    split_dataset()
    print("\nDone. Federated dataset structure created at datasets/federated/")
