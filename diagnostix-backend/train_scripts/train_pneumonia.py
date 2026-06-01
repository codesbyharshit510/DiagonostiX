# train_pneumonia.py
"""
Train a ResNet18 model for Pneumonia detection and save the state_dict (safe).
Saves: models/image/pneumonia.pt (state_dict)
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils_image import get_dataloaders

# -------------------------
# Config / CLI
# -------------------------
parser = argparse.ArgumentParser(description="Train Pneumonia ResNet18 (state_dict)")
parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                      "models_training_full", "pneumonia", "data"),
                    help="Path to pneumonia data (expects train/ and val/)")
parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                  "models", "image", "pneumonia.pt"),
                    help="Output state_dict path")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--batch-size", type=int, default=16, help="Batch size (lower on CPU)")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers (0 for Windows/CPU)")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Dataloaders
# -------------------------
train_loader, val_loader, classes = get_dataloaders(args.data_dir,
                                                   input_size=224,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers)
print("Classes detected:", classes)
num_classes = len(classes)
if num_classes != 2:
    print("Warning: pneumonia dataset expected 2 classes (NORMAL, PNEUMONIA). Detected:", num_classes)

# -------------------------
# Model
# -------------------------
# Build a fresh ResNet18 architecture (no pretrained weights for state_dict approach)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# -------------------------
# Training loop
# -------------------------
best_acc = 0.0
start_time = time.time()

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        running_total += inputs.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_corrects / running_total

    # Validation
    model.eval()
    val_running_corrects = 0
    val_running_total = 0
    val_running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += (preds == labels).sum().item()
            val_running_total += labels.size(0)
            val_running_loss += loss.item() * inputs.size(0)

    val_loss = val_running_loss / val_running_total if val_running_total > 0 else 0.0
    val_acc = val_running_corrects / val_running_total if val_running_total > 0 else 0.0

    print(f"Epoch [{epoch+1}/{args.epochs}] train_loss: {epoch_loss:.4f} train_acc: {epoch_acc:.4f} "
          f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

    # Save best state_dict
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), args.out)
        print(f"Saved best model state_dict to {args.out} (val_acc={best_acc:.4f})")

elapsed = time.time() - start_time
print(f"Training finished in {elapsed//60:.0f}m {elapsed%60:.0f}s. Best val_acc: {best_acc:.4f}")
