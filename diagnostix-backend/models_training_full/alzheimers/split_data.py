import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "raw")
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

SPLIT_RATIO = (0.7, 0.15, 0.15)  # train/val/test

classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

def ensure_dirs():
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(split, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(split, cls), exist_ok=True)

def split_and_move():
    for cls in classes:
        print(f"Processing class: {cls}")

        src_folder = os.path.join(RAW_DIR, cls)
        if not os.path.exists(src_folder):
            print(f"❌ Missing folder: {src_folder}")
            continue

        images = os.listdir(src_folder)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * SPLIT_RATIO[0])
        val_end = train_end + int(total * SPLIT_RATIO[1])

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        # Move images
        for img in train_imgs:
            shutil.move(os.path.join(src_folder, img),
                        os.path.join(TRAIN_DIR, cls, img))

        for img in val_imgs:
            shutil.move(os.path.join(src_folder, img),
                        os.path.join(VAL_DIR, cls, img))

        for img in test_imgs:
            shutil.move(os.path.join(src_folder, img),
                        os.path.join(TEST_DIR, cls, img))

        print(f"✔ Done {cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    print("\n🎉 Splitting complete!")

if __name__ == "__main__":
    ensure_dirs()
    split_and_move()
