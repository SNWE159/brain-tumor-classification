# 🧠 NeuroScan AI — Brain Tumor Classification

<div align="center">

![NeuroScan AI](https://img.shields.io/badge/NeuroScan-AI-blue?style=for-the-badge&logo=brain&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Models-yellow?style=for-the-badge)

**A deep learning ensemble system for brain tumor classification from MRI scans.**  
Trained on Google Colab · Deployed on Streamlit · Models hosted on Hugging Face

[🚀 **Live Demo**](#) &nbsp;|&nbsp; [🤗 **Hugging Face Repo**](#) &nbsp;|&nbsp; [📂 **Dataset**](#) &nbsp;|&nbsp; [📦 **Models**](#)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Hugging Face Repository](#-hugging-face-repository)
- [Dataset](#-dataset)
- [Disclaimer](#-disclaimer)

---

## 🔬 Overview

NeuroScan AI classifies brain MRI scans into **4 categories** using a soft-voting ensemble of four CNN models:

| Class | Description | Severity |
|---|---|---|
| 🔴 Glioma | Tumor arising from glial cells | High |
| 🟠 Meningioma | Tumor on brain/spinal cord membranes | Moderate |
| 🟢 No Tumor | No evidence of tumor detected | None |
| 🔵 Pituitary | Tumor in the pituitary gland | Moderate |

**How it works:**
1. Upload a brain MRI scan (JPG/PNG)
2. All 4 models independently analyze the image
3. Their probability outputs are averaged (soft-voting ensemble)
4. Final prediction is shown with confidence score and per-model breakdown

---

## 🚀 Live Demo

> View the deployed Streamlit application on Hugging Face Spaces:

**🌐 Live App:** `[ https://huggingface.co/spaces/Siya-34/brain-tumor-classification-app ]`

---

## 🤗 Hugging Face Repository

All trained model files are publicly hosted on Hugging Face Hub for direct use.

| Field | Value |
|---|---|
| **Repository ID** | `Siya-34/brain-tumor-classification` |
| **Repository Link** | `[ https://huggingface.co/Siya-34/brain-tumor-classification ]` |
| **Visibility** | Public |

### Model Files in Repository

| File | Architecture |
|---|---|
| `effnet_best.keras` | EfficientNetB0 |
| `resnet50_brain_tumor_final.keras` | ResNet-50 |
| `vgg16_brain_tumor_final.keras` | VGG-16 |
| `mobilenetv2_best.keras` | MobileNetV2 |

### Load a model directly

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

path = hf_hub_download(
    repo_id="Siya-34/brain-tumor-classification",
    filename="effnet_best.keras"
)
model = tf.keras.models.load_model(path)
```

---

## 📂 Dataset

The model was trained on a publicly available brain MRI dataset with 4 classes.

| Resource | Link |
|---|---|
| 📂 Preprocessed Dataset (Google Drive) | `[ https://drive.google.com/drive/folders/1erUoJmlzVJz5ne5aMTjegcxCRZfvILKh?usp=sharing ]` |


---

## ⚠️ Disclaimer

> This project is intended for **research and educational purposes only**.  
> It is **not** a substitute for professional medical diagnosis, advice, or treatment.  
> Always consult a qualified healthcare professional for medical decisions.

---

<div align="center">

Made with ❤️ using TensorFlow · Streamlit · Hugging Face

</div>
