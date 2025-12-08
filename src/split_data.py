import os, shutil, random

# --- Paths ---
SOURCE_DIR = r"C:\Users\deeks\OneDrive\Desktop\Dataset from fundus images for the study of diabetic retinopathy_V03"   # path after renaming
OUTPUT_DIR = r"C:\Users\deeks\OneDrive\Desktop\data_split"   # new folder for train/val
TRAIN_RATIO = 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

for cls in os.listdir(SOURCE_DIR):
    src_cls = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(src_cls):
        continue

    imgs = [f for f in os.listdir(src_cls) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(imgs)
    split_idx = int(len(imgs) * TRAIN_RATIO)

    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]

    for subset, files in [("train", train_imgs), ("val", val_imgs)]:
        dst_cls = os.path.join(OUTPUT_DIR, subset, cls)
        os.makedirs(dst_cls, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(src_cls, f), os.path.join(dst_cls, f))

print("✅ Split complete! Data saved to:", OUTPUT_DIR)