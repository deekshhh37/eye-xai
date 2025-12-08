import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

DATA_DIR = r"C:\Users\deeks\OneDrive\Desktop\data_split"
OUTPUT_DIR = "outputs"
EPOCHS = 10
BATCH_SIZE = 16
LR = 3e-4
IMG_SIZE = 384
NUM_CLASSES = 7
CLASS_NAMES = ["No_DR","Mild_NPDR","Moderate_NPDR","Severe_NPDR",
               "Very_Severe_NPDR","Proliferative_DR","Advanced_PDR"]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1,0.1,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # ✅ start with workers=0 on Windows; you can try 2 later
    num_workers = 0
    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=val_tf)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=num_workers)

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model = model.to(device)

    counts = np.zeros(NUM_CLASSES)
    for _, y in train_loader:
        for i in y: counts[i] += 1
    weights = torch.tensor(1.0/(counts+1e-6), dtype=torch.float32)
    weights = (weights/weights.sum())*NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total = correct = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += (logits.argmax(1)==labels).sum().item()
            total += imgs.size(0)
        print(f"Train Accuracy: {correct/total:.4f}")

        # validation
        model.eval()
        preds, tgts = [], []
        v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                p = model(imgs).argmax(1)
                preds.extend(p.cpu().numpy()); tgts.extend(labels.cpu().numpy())
                v_correct += (p==labels).sum().item(); v_total += imgs.size(0)
        val_acc = v_correct/v_total
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(classification_report(tgts, preds, target_names=CLASS_NAMES))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pt"))
            print("✅ Best model saved")

    print("🎯 Training Complete. Best val accuracy:", best_acc)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # needed for Windows executables / spawn
    main()
