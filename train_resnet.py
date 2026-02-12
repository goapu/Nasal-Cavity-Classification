import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
import numpy as np
import random

from custom_dataset import BlockLevelDataset
from utils import (
    save_predictions, plot_learning_curve,
    plot_confusion_matrix, print_classification_report
)
from focal_loss import FocalLoss

# ------------------ Setup ------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all()

# ------------------ Paths ------------------
BASE_PATH = "/Volumes/T9/EndonasalAR/Splitted Images/Augmented Images"
TRAIN_DIR = os.path.join(BASE_PATH, "Training")
VAL_DIR = os.path.join(BASE_PATH, "Validation")
TEST_DIR = os.path.join(BASE_PATH, "Testing")
OUTPUT_DIR = "/Users/gmac/Desktop/ThesisCode/ResnetLSTM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ Config ------------------
BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 5
LR = 1e-3    # Start higher, reduce for fine-tuning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ------------------ Transforms ------------------
train_tfms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(0.3, 0.3, p=0.8),
    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=20, p=0.8),
    A.HueSaturationValue(p=0.2),
    A.GaussianBlur(p=0.2),
    A.CLAHE(p=0.2),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2()
])
val_tfms = A.Compose([
    A.Resize(224, 224),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ------------------ Data ------------------
train_ds = BlockLevelDataset(TRAIN_DIR, transform=train_tfms)
val_ds   = BlockLevelDataset(VAL_DIR, transform=val_tfms)
test_ds  = BlockLevelDataset(TEST_DIR, transform=val_tfms)
classes = ["anterior", "middle", "posterior"]

train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_ld  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Entropy-inspired class weights for Focal Loss ---
label_counts = Counter([label for _, label in train_ds])
print("Class distribution in training set:", label_counts)
print("Classes:", classes)

total_samples = sum(label_counts.values())
class_freqs = [label_counts.get(i, 0) / total_samples for i in range(len(classes))]
class_freqs = [max(f, 1e-6) for f in class_freqs]  # Avoid log(0)
entropy_weights = torch.tensor([-np.log(f) for f in class_freqs], dtype=torch.float32).to(DEVICE)
print("Class frequencies:", class_freqs)
print("Entropy-inspired weights (for focal loss alpha):", entropy_weights)

# Use FocalLoss with these weights
criterion = FocalLoss(alpha=entropy_weights, gamma=2, reduction='mean')

# ------------------ Model: Pretrained ResNet50 ------------------
def build_model(num_classes):
    base = models.resnet50(pretrained=True)
    for param in base.parameters():
        param.requires_grad = False  # Freeze feature extractor
    in_features = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return base

model = build_model(num_classes=len(classes)).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

# ------------------ Training ------------------
best_acc = 0
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    tl, tc = 0.0, 0
    for xb, yb in tqdm(train_ld, desc=f"[Train E{epoch}]"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if yb.dtype != torch.long:
            yb = yb.long()
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        tl += loss.item() * xb.size(0)
        tc += (out.argmax(1) == yb).sum().item()
    train_loss = tl / len(train_ds)
    train_acc = tc / len(train_ds)

    model.eval()
    vl, vc = 0.0, 0
    with torch.no_grad():
        for xb, yb in tqdm(val_ld, desc="[Validation]"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if yb.dtype != torch.long:
                yb = yb.long()
            out = model(xb)
            loss = criterion(out, yb)
            vl += loss.item() * xb.size(0)
            vc += (out.argmax(1) == yb).sum().item()
    val_loss = vl / len(val_ds)
    val_acc = vc / len(val_ds)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch}: TL={train_loss:.4f}, TA={train_acc:.4f}, VL={val_loss:.4f}, VA={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, os.path.join(OUTPUT_DIR, "best_model.pth"))
        print("✅ Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping.")
            break

# ------------------ Fine-tune entire model if plateaued ------------------
print("🔄 Unfreezing ResNet layers for fine-tuning...")
for param in model.parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    tl, tc = 0.0, 0
    for xb, yb in tqdm(train_ld, desc=f"[Finetune E{epoch}]"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if yb.dtype != torch.long:
            yb = yb.long()
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        tl += loss.item() * xb.size(0)
        tc += (out.argmax(1) == yb).sum().item()
    train_loss = tl / len(train_ds)
    train_acc = tc / len(train_ds)

    model.eval()
    vl, vc = 0.0, 0
    with torch.no_grad():
        for xb, yb in tqdm(val_ld, desc="[Validation]"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if yb.dtype != torch.long:
                yb = yb.long()
            out = model(xb)
            loss = criterion(out, yb)
            vl += loss.item() * xb.size(0)
            vc += (out.argmax(1) == yb).sum().item()
    val_loss = vl / len(val_ds)
    val_acc = vc / len(val_ds)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Finetune Epoch {epoch}: TL={train_loss:.4f}, TA={train_acc:.4f}, VL={val_loss:.4f}, VA={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, os.path.join(OUTPUT_DIR, "best_model.pth"))
        print("✅ Best finetuned model saved.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping (finetuning).")
            break

# ------------------ Evaluation ------------------
checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

save_predictions(val_ld, model, DEVICE, classes, os.path.join(OUTPUT_DIR, "val_results.csv"))
save_predictions(test_ld, model, DEVICE, classes, os.path.join(OUTPUT_DIR, "test_results.csv"))

def get_preds(loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().argmax(1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred

yt_val, yp_val = get_preds(val_ld)
yt_test, yp_test = get_preds(test_ld)

plot_confusion_matrix(yt_val, yp_val, classes, os.path.join(OUTPUT_DIR, "val_confusion_matrix.png"))
plot_confusion_matrix(yt_test, yp_test, classes, os.path.join(OUTPUT_DIR, "test_confusion_matrix.png"))
print_classification_report(yt_val, yp_val, classes, "Validation Classification Report")
print_classification_report(yt_test, yp_test, classes, "Test Classification Report")
plot_learning_curve(history)

print("✅ Training complete.")
