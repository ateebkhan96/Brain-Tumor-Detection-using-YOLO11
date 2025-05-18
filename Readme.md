# 🧠 AI-Assisted Brain Tumor Detection in MRI Scans

![YOLOv11](https://img.shields.io/badge/YOLOv11-ultralytics-blue) 
![Python](https://img.shields.io/badge/Python-3.8%2B-green)

The notebook trains YOLO11 Model to detect brain tumors in MRI scans, designed to assist radiologists with preliminary screening.

## 📌 Key Features
- **Model**: YOLOv11 (Ultralytics)
- **Dataset**: [Kaggle LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (3,929 slices)
- **High Accuracy**: 95.1% mAP@0.5 on test set
- **Efficient Pipeline**: 
  - Segmentation mask → YOLO bounding box conversion
  - Albumentations-powered data augmentation
  - Training notebook
 
