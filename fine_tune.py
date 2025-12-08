import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

# ===================== CONFIG =====================
DATA_DIR = r"C:\Users\deeks\OneDrive\Desktop\data_split"
OUTPUT_DIR = "outputs"
BATCH_SIZE = 16
LR = 1e-5
IMG_SIZE = 384
NUM_CLASSES = 7
EPOCHS_FINE = 10   # 🔁 increased for stronger fine-tuning

CLASS_NAMES = [
    "No_DR", "Mild_NPDR", "Moderate_NPDR",
    "Severe_NPDR", "Very_Severe_NPDR", "Proliferative_DR", "Advanced_PDR"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== DATA AUGMENTATION =====================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_tf)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

# ===================== MODEL LOADING =====================
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

best_model_path = os.path.join(OUTPUT_DIR, "best.pt")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)
print("✅ Loaded model from:", best_model_path)

# ===================== UNFREEZE + SETUP =====================
for param in model.features.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FINE)

# -------- Optional Focal Loss (better for hard classes) --------
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.softmax(logits, 1).gather(1, targets.unsqueeze(1)).squeeze()
        focal = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal

criterion = FocalLoss().to(device)

# ===================== FINE-TUNING LOOP =====================
print("\n🔧 Starting fine-tuning with cosine LR + focal loss...\n")
best_acc = 0

for epoch in range(1, EPOCHS_FINE + 1):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch}/{EPOCHS_FINE}"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    scheduler.step()
    train_acc = correct / total
    print(f"Train Accuracy: {train_acc:.4f}")

    # ===================== VALIDATION =====================
    model.eval()
    preds, tgts = [], []
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # 🔁 Test-Time Augmentation (original + flipped)
            logits_orig = model(imgs)
            logits_flip = model(torch.flip(imgs, dims=[3]))
            logits = (logits_orig + logits_flip) / 2

            p = logits.argmax(1)
            preds.extend(p.cpu().numpy())
            tgts.extend(labels.cpu().numpy())
            v_correct += (p == labels).sum().item()
            v_total += imgs.size(0)

    val_acc = v_correct / v_total
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(classification_report(tgts, preds, target_names=CLASS_NAMES, zero_division=0))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "finetuned_92.pt"))
        print("✅ Fine-tuned model saved!")

print("🎯 Fine-tuning Complete. Best val accuracy:", best_acc)