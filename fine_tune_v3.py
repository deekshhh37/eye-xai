import os, cv2, torch, numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR

# ===================== CONFIG =====================
DATA_DIR   = r"C:\Users\deeks\OneDrive\Desktop\data_split"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE   = 384
NUM_CLASSES = 7
BATCH_SIZE  = 16
EPOCHS      = 12
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "No_DR","Mild_NPDR","Moderate_NPDR",
    "Severe_NPDR","Very_Severe_NPDR","Proliferative_DR","Advanced_PDR"
]

# ===================== CLAHE ENHANCEMENT =====================
def apply_clahe(img):
    img_yuv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

class CLAHETransform:
    def __call__(self, img):
        return apply_clahe(img)

# ===================== DATA AUGMENTATION =====================
train_tf = transforms.Compose([
    CLAHETransform(),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(0.15,0.15,0.15,0.05),   # 🔧 slightly higher brightness/contrast
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,"train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR,"val"),   transform=val_tf)
train_loader = DataLoader(train_ds,BATCH_SIZE,shuffle=True,num_workers=0)
val_loader   = DataLoader(val_ds,BATCH_SIZE,shuffle=False,num_workers=0)

# ===================== MODEL =====================
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

best_model_path = os.path.join(OUTPUT_DIR,"best.pt")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path,map_location=DEVICE))
model = model.to(DEVICE)

# Freeze BN layers for stability
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        m.requires_grad_(False)

# 🔧 Slightly increased learning rates
optimizer = torch.optim.AdamW([
    {"params": model.features.parameters(), "lr": 5e-6},
    {"params": model.classifier.parameters(),  "lr": 2e-5}
], weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ===================== FOCAL LOSS =====================
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma)*ce
        return loss.mean()

criterion = FocalLoss(gamma=1.0)

# ===================== TRAINING LOOP =====================
best_acc = 0.0
patience, patience_ctr = 6, 0
train_losses, val_accuracies = [], []

for epoch in range(1, EPOCHS+1):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()*imgs.size(0)
        correct  += (logits.argmax(1)==labels).sum().item()
        total    += imgs.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    scheduler.step()
    train_acc = correct/total
    train_losses.append(loss_sum/total)
    print(f"Train Accuracy: {train_acc:.4f}")

    # ---------- VALIDATION ----------
    model.eval()
    preds, tgts = [], []
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            pred = logits.argmax(1)
            preds.extend(pred.cpu().numpy())
            tgts.extend(labels.cpu().numpy())
            v_correct += (pred==labels).sum().item()
            v_total   += imgs.size(0)
    val_acc = v_correct/v_total
    val_accuracies.append(val_acc)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(classification_report(tgts, preds, target_names=CLASS_NAMES, zero_division=0))

    if val_acc > best_acc:
        best_acc = val_acc
        patience_ctr = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR,"finetuned_v3.pt"))
        print("✅ Saved new best model!")
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print("⏹️ Early stopping: no improvement.")
            break

print(f"🎯 Fine-tuning Complete — Best Validation Accuracy: {best_acc:.4f}")

# ===================== TRAINING CURVES =====================
plt.figure(figsize=(8,5))
plt.plot(range(1,len(train_losses)+1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1,len(val_accuracies)+1), val_accuracies, label="Val Accuracy", marker="s")
plt.title("Fine-tuning Progress (MobileNetV3-Small)")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"training_curve_v3.png"))
plt.show()
