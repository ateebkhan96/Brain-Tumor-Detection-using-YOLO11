# 🧠 AI-Assisted Brain Tumor Detection in MRI Scans

![YOLOv11](https://img.shields.io/badge/YOLOv11-ultralytics-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)

An AI-powered clinical support tool for detecting brain tumors in axial MRI scans using the latest **YOLOv11** object detection model. Designed to assist radiologists by quickly highlighting potential tumor regions for review.

---

## 📌 Key Features

- ✅ **Model:** Ultralytics **YOLOv11** (anchor-free, fast, and accurate)
- 🧠 **Dataset:** [Kaggle LGG MRI Brain Tumor Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (3,929 MRI slices)
- 📈 **Test Performance:**
  - **Precision**: 92.7%
  - **Recall**: 91.9%
  - **mAP@0.5**: 95.1%
- 🛠️ **Pipeline:**
  - Segmentation mask ➞ Bounding box conversion (YOLO format)
  - Albumentations-powered data augmentation
  - Training with YOLOv11 via PyTorch + Ultralytics API
- 🌐 **Deployment:** Streamlit-based Web App for clinical usability

---

## 📂 Dataset

- **Source:** [Kaggle LGG MRI Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Size:** 3,929 axial T1-weighted MRI images from 110 patients
- **Resolution:** 256×256 pixels
- **Annotations:** Binary segmentation masks (converted to bounding boxes)

---

## ⚙️ Model Training Pipeline

1. **Preprocessing:**
   - Mask-to-bounding box conversion
   - Normalized YOLOv11 annotations
   - Patient-level Train/Val/Test split

2. **Augmentations (via Albumentations):**
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - HSV shifts, flips, rotations, blurs

3. **Training Details:**
   - Model: YOLOv11 (Ultralytics)
   - Epochs: 100 with early stopping
   - Batch Size: 128
   - Optimizer: AdamW
   - Hardware: Google Colab (Tesla T4 GPU)
   - Image Size: 640×640

---

## 📊 Evaluation Metrics

| Metric        | Validation Set | Test Set |
|--------------|---------------|----------|
| Precision    | 94.6%         | 92.7%    |
| Recall       | 86.7%         | 91.9%    |
| mAP@0.5      | 94.4%         | 95.1%    |
| mAP@0.5:0.95 | 71.4%         | 69.8%    |

- 📉 Visualized PR curves, confidence tradeoffs, and loss trends
- 🔍 High model sensitivity ensured by using a **low confidence threshold (0.25)** to avoid false negatives

---

## 💻 Web Application (Streamlit)

### 🎯 Live Demo:

[🔗 Open App](https://brain-tumor-det.streamlit.app/)

### 💻 Local Setup:

```bash
git clone https://github.com/ateebkhan96/Brain-Tumor-Detection-using-YOLO11.git
cd Brain-Tumor-Detection-using-YOLO11

# Create virtual environment
python -m venv brain-tumor
source brain-tumor/bin/activate   # On Linux/Mac
# OR
brain-tumor\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
## 🧪 Features

- Upload `.jpg`, `.png`, `.bmp`, `.tiff` MRI scans (max 200MB)
- Real-time tumor detection with bounding boxes
- Adjustable **Confidence** and **IoU** thresholds
- Export annotated images
- FPS & inference time indicators

---

## 🤝 Radiologist User Journey

1. Launch web app (locally or via demo)
2. Upload axial MRI scan
3. View original + annotated image side-by-side
4. Adjust thresholds for clinical sensitivity
5. Download annotated image for records or reports

---

## ⚠️ Limitations

- Currently supports only **Low-Grade Glioma (LGG)** tumors
- Uses 2D slice-based detection (not full 3D volumetric context)
- No feedback loop yet for continuous model improvement
- Bounding box predictions, not pixel-perfect segmentation

---

## 🚀 Future Enhancements

- 3D volumetric tumor detection with hybrid models (YOLO + U-Net)
- Integration of T2, FLAIR modalities for richer context
- Explainability via Grad-CAM / attention maps
- Radiologist feedback mechanism for retraining
- Clinical fairness audits and safety testing

---
