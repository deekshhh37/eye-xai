# eye-xai

**Lightweight and Explainable Deep Learning for Diabetic Retinopathy Detection**

`eye-xai` is a deep learning project for classifying **retinal fundus images** into Diabetic Retinopathy (DR) severity levels. It uses **MobileNetV3-Small** for efficiency and integrates **Grad-CAM** to provide visual explanations of predictions. The goal is to deliver **accurate, lightweight, and interpretable AI models** for reliable DR screening in healthcare.

---

## 🚀 Features

* **Classification:** Predicts 5 severity levels of DR (No DR → Proliferative DR).
* **Lightweight Model:** MobileNetV3-Small backbone with reduced parameters.
* **Explainability:** Grad-CAM heatmaps to highlight key retinal regions.
* **Performance:** Evaluated with Accuracy, Sensitivity, Specificity, and F1-score.
* **Imbalance Handling:** Uses augmentation and class weighting to improve early DR detection.

---

## 📁 Datasets

This project is trained and tested on publicly available **retinal fundus image datasets**:

* **APTOS 2019 Blindness Detection** – [Link](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
* **EyePACS Dataset** – [Link](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
* **IDRiD (Indian Diabetic Retinopathy Dataset)** – [Link](https://idrid.grand-challenge.org/)
* **Zenodo DR Fundus Dataset (757 images)** – [Link](https://zenodo.org/records/4647952)

> ⚠️ Datasets are not included in this repo. Download separately and place inside the `data/` folder.

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/eye-xai.git
cd eye-xai
pip install -r requirements.txt
```

---

## 🔧 Training

Example (APTOS dataset):

```bash
python src/train.py --config configs/aptos.yaml
```

---

## 🔎 Inference & Explainability

Run inference with Grad-CAM visualization:

```bash
python src/infer.py --checkpoint outputs/checkpoints/best.pt --image path/to/fundus.jpg --cam
```

---

## 📊 Results

* Classification metrics: Accuracy, Sensitivity, Specificity, F1-score
* Outputs: Model checkpoints, Grad-CAM visual heatmaps, logs

---

## 📚 Reference

* Base Paper: [A Lightweight Diabetic Retinopathy Detection Model Using a Deep-Learning Technique](https://www.mdpi.com/2075-4418/13/19/3120) (*Diagnostics, MDPI, 2023*)

