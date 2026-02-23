# 🌾 Crop Disease Image Segmentation using Detectron2 (Mask R-CNN)

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Detectron2](https://img.shields.io/badge/Detectron2-InstanceSegmentation-green)
![Computer Vision](https://img.shields.io/badge/Domain-Agriculture--AI-yellow)

---

## 📌 Project Overview

This project implements **instance segmentation for crop leaf diseases** using **Mask R-CNN with Detectron2**.  

The objective is to automatically detect and segment diseased regions on crop leaves and classify them into specific disease categories.

The model is trained on a **COCO-format annotated dataset** and fine-tuned using a pretrained Mask R-CNN backbone.

---

## 🎯 Problem Statement

Crop diseases significantly reduce agricultural productivity. Early and precise detection is essential for:

- Yield protection
- Targeted pesticide use
- Sustainable farming
- Automated monitoring systems

This project focuses on **pixel-level segmentation**, rather than simple classification, enabling:

- Localization of disease regions
- Multi-instance detection
- Better interpretability

---

## 🧠 Model Architecture

We use:

**Mask R-CNN with ResNet-50 FPN backbone**

Configuration:
```
COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
```

Model framework:
- Backbone: ResNet-50
- Feature Pyramid Network (FPN)
- Region Proposal Network (RPN)
- ROI Heads for classification + mask prediction

Pretrained weights are fine-tuned on crop disease dataset.

---

## 🌿 Disease Classes

The model predicts the following 7 categories:

1. Healthy  
2. Leaf Blast  
3. Brown Spot  
4. Bacterial Blight  
5. Tungro  
6. Sheath Blight  
7. Hispa  

---

## 📂 Dataset Structure

Dataset is in **COCO format**:

```
crop_disease_data/
│
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│
├── valid/
│   ├── _annotations.coco.json
│   ├── imageX.jpg
```

Dataset registration:

```python
register_coco_instances("trains", {}, trains_json, trains_img)
register_coco_instances("value", {}, value_json, value_img)
```

---

## ⚙ Configuration Details

```python
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
cfg.MODEL.WEIGHTS = "model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"
```

Key Parameters:

- Custom number of classes
- Confidence threshold: 0.5
- CPU inference (can switch to CUDA for GPU)
- Custom trained weights

---

## 🔍 Inference Pipeline

1️⃣ Load configuration  
2️⃣ Load trained weights  
3️⃣ Read image using OpenCV  
4️⃣ Perform prediction  
5️⃣ Extract predicted class IDs  
6️⃣ Map to disease names  
7️⃣ Visualize segmentation masks  

Prediction snippet:

```python
outputs = predictor(im)
pred_classes = outputs["instances"].pred_classes.cpu().numpy()
```

Visualization using Detectron2 Visualizer:

```python
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("crop_train"))
```

---

## 📊 Output Example

For each detected instance:

```
Predicted Disease: Leaf Blast
Predicted Disease: Brown Spot
```

Model also overlays segmentation masks and bounding boxes on the image.

---

## 🛠 Tech Stack

- Python
- PyTorch
- Detectron2
- OpenCV
- Matplotlib
- COCO Dataset Format

---

## 🚀 Installation

### 1️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install PyTorch

Visit:
https://pytorch.org/get-started/locally/

### 3️⃣ Install Detectron2

```bash
pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

(Adjust CUDA version if needed)

### 4️⃣ Install other dependencies

```bash
pip install opencv-python matplotlib
```

---

## ▶ How to Run

1. Place trained weights file (`model_final.pth`)
2. Set correct dataset paths
3. Run inference script:

```bash
python inference.py
```

---

## 📈 Research Relevance

This project demonstrates:

- Instance segmentation expertise
- COCO dataset handling
- Detectron2 configuration
- Transfer learning
- Agricultural AI application
- Model deployment readiness

It aligns with research themes in:

- Precision Agriculture
- Computer Vision
- Deep Learning
- Automated Disease Monitoring

---

## 🔬 Future Improvements

- GPU acceleration
- mAP evaluation metrics
- Model quantization for edge deployment
- Real-time camera integration
- Attention-based backbone comparison
- Semi-supervised training

---

## 📁 Project Structure

```
crop-disease-image-segmentation/
│
├── train.py
├── inference.py
├── model_final.pth
├── crop_disease_data/
├── requirements.txt
└── README.md
```

---

## 👤 Author

Shubham Chaudhary
B.Tech – Artificial Intelligence & Machine Learning  
BIT Mesra  

---
