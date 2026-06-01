# train_brain_tumor.py
"""
Optimized Brain Tumor Classification Training Script
----------------------------------------------------
✓ Uses pretrained ResNet18 (FAST + HIGH ACCURACY)
✓ Trains 5x–10x faster on CPU
✓ Saves state_dict safely -> models/image/brain_tumor.pt
✓ Supports 4 classes:
      - glioma_tumor
      - meningioma_tumor
      - pituitary_tumor
      - no_tumor
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from utils_image import get_dataloaders

# -------------------------------------
# CLI / DEFAULT CONFIG
# -------------------------------------
parser = argparse.ArgumentParser(description="Train Optimized Brain Tumor Classifier")

parser.add_argument(
    "--data-dir",
    default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models_training_full",
        "brain_tumor",
        "data"
    )
)

parser.add_argument(
    "--out",
    default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "image",
        "brain_tumor.pt"
    )
)

# Key optimizations here:
parser.add_argument("--epochs", type=int, default=3)        # Faster training
parser.add_argument("--batch-size", type=int, default=8)     # CPU-friendly
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num-workers", type=int, default=0)    # Windows CPU must use 0

args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------------------
# LOAD DATA
# -------------------------------------
train_loader, val_loader, classes = get_dataloaders(
    args.data_dir,
    input_size=224,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)

print("Classes detected:", classes)
num_classes = len(classes)

if num_classes != 4:
    print("⚠️ Warning: Expected 4 classes for brain tumor dataset.")

# -------------------------------------
# MODEL — RESNET18 WITH PRETRAINED WEIGHTS
# -------------------------------------
print("Loading pretrained ResNet18...")

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

best_acc = 0.0
start = time.time()

# -------------------------------------
# TRAINING LOOP
# -------------------------------------
for epoch in range(args.epochs):
    print(f"\n🔥 Epoch {epoch+1}/{args.epochs}")
    model.train()

    running_loss = 0
    running_corrects = 0
    running_total = 0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        _, pred = out.max(1)
        running_corrects += pred.eq(y).sum().item()
        running_total += y.size(0)

    train_loss = running_loss / running_total
    train_acc = running_corrects / running_total

    # ----- VALIDATION -----
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss_sum = 0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            out = model(X)
            loss = criterion(out, y)

            _, pred = out.max(1)
            val_correct += pred.eq(y).sum().item()
            val_total += y.size(0)
            val_loss_sum += loss.item() * X.size(0)

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # ----- SAVE BEST MODEL -----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), args.out)
        print(f"💾 Saved BEST model → {args.out} (val_acc={best_acc:.4f})")

end = time.time()
print(f"\n⏳ Training finished in {int((end-start)//60)}m {int((end-start)%60)}s.")
print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")
