import os
import subprocess
import sys

BASE = os.path.dirname(__file__)

scripts = [
    "train_pneumonia.py",
    "train_alzheimers.py",
    "train_brain_tumor.py",
    "train_heart_disease.py",
    "train_diabetes.py"
]

print("🚀 Starting training for all models...\n")

for script in scripts:
    path = os.path.join(BASE, script)

    print(f"🔵 Running: {script}")
    result = subprocess.call([sys.executable, path])

    if result != 0:
        print(f"❌ Error while running {script}. Stopping.")
        break

    print(f"✅ Completed: {script}\n")

print("🎉 All done! Model training pipeline finished.")
