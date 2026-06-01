import os
import shutil
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
DATA = BASE / "data"

TRAIN = DATA / "train"
VAL = DATA / "val"
TEST = DATA / "test"

SPLIT = (0.7, 0.15, 0.15)

def make_dirs(classes):
    for d in [TRAIN, VAL, TEST]:
        d.mkdir(parents=True, exist_ok=True)
        for c in classes:
            (d / c).mkdir(parents=True, exist_ok=True)

def run_split():
    if not RAW.exists():
        print("❌ RAW folder missing:", RAW)
        return

    classes = [c.name for c in RAW.iterdir() if c.is_dir()]
    print("Detected classes:", classes)

    make_dirs(classes)

    for cls in classes:
        src = RAW / cls
        images = [img for img in src.iterdir() if img.is_file()]
        random.shuffle(images)

        total = len(images)
        n_train = int(total * SPLIT[0])
        n_val = int(total * SPLIT[1])

        train_files = images[:n_train]
        val_files = images[n_train:n_train+n_val]
        test_files = images[n_train+n_val:]

        for img in train_files:
            shutil.move(str(img), str(TRAIN / cls / img.name))
        for img in val_files:
            shutil.move(str(img), str(VAL / cls / img.name))
        for img in test_files:
            shutil.move(str(img), str(TEST / cls / img.name))

        print(f"✔ {cls}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    print("\n🎉 Brain Tumor dataset split complete!")

if __name__ == "__main__":
    run_split()
