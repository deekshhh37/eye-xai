import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ===================== PATHS =====================
MODEL_PATH = r"C:\Users\deeks\OneDrive\Desktop\ML CBP\outputs\best.pt"
IMG_PATH = r"C:\Valid Images\1192.jpg"   # change if needed
OUTPUT_PATH = r"C:\Users\deeks\OneDrive\Desktop\ML CBP\outputs\gradcam_side_by_side.jpg"

# ===================== SETUP =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "No_DR","Mild_NPDR","Moderate_NPDR",
    "Severe_NPDR","Very_Severe_NPDR","Proliferative_DR","Advanced_PDR"
]
IMG_SIZE = 384

# ===================== MODEL =====================
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# ===================== IMAGE LOAD =====================
img_cv = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
input_tensor = tf(img_pil).unsqueeze(0).to(device)

# ===================== GRAD-CAM HOOKS =====================
target_layer = model.features[-1]
gradients = None
activations = None

def save_gradients(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def save_activations(module, input, output):
    global activations
    activations = output

target_layer.register_forward_hook(save_activations)
target_layer.register_backward_hook(save_gradients)

# ===================== FORWARD + BACKWARD =====================
output = model(input_tensor)
prob = torch.nn.functional.softmax(output, dim=1)
confidence, pred_class = torch.max(prob, dim=1)
pred_class_idx = pred_class.item()
pred_label = CLASS_NAMES[pred_class_idx]
print(f"Predicted class: {pred_label} | Confidence: {confidence.item():.2f}")

model.zero_grad()
one_hot = torch.zeros_like(output)
one_hot[0][pred_class_idx] = 1
output.backward(gradient=one_hot, retain_graph=True)

# ===================== BUILD GRAD-CAM =====================
grads = gradients.cpu().data.numpy()[0]
acts = activations.cpu().data.numpy()[0]
weights = np.mean(grads, axis=(1, 2))
cam = np.sum(weights[:, None, None] * acts, axis=0)
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
cam = cam / cam.max()

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)

# ===================== OVERLAY TEXT =====================
label_text = f"{pred_label} ({confidence.item():.2f})"
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(overlay, label_text, (20, 50), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(overlay, label_text, (20, 50), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

# ===================== COMBINE IMAGES SIDE-BY-SIDE =====================
img_resized = cv2.resize(img_cv, (overlay.shape[1], overlay.shape[0]))
heatmap_resized = cv2.resize(heatmap, (overlay.shape[1], overlay.shape[0]))

comparison = np.hstack([img_resized, overlay, heatmap_resized])

# ===================== SAVE RESULT =====================
cv2.imwrite(OUTPUT_PATH, comparison)
print(f"✅ Side-by-side Grad-CAM saved at: {OUTPUT_PATH}")