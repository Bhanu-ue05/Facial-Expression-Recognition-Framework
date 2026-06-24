# Face Expression Recognition Framework
### Facial Emotion Classification Using CNN and Transfer Learning on RAF-DB

**Author:** Bhanu Prasad Mudraboina  
**Course:** Machine Learning — Masters in Data Science  
**Professor:** Raja Hashmi Ali  
**Phase:** 2 — Proposal and Code Implementation  

---

## Project Overview

This project implements a complete deep learning pipeline for **facial expression recognition** using the **Real-world Affective Faces Database (RAF-DB)**. It trains and compares three models:

| Model | Type | Strategy |
|---|---|---|
| Custom CNN | From scratch | 4-block VGG-style CNN |
| ResNet50 | Transfer Learning | ImageNet weights + fine-tuning |
| MobileNetV2 | Transfer Learning | ImageNet weights + fine-tuning |

**Target: 7 emotion classes** — Neutral, Happiness, Surprise, Sadness, Anger, Disgust, Fear

---

## Dataset

- **Source:** [RAF-DB on Kaggle](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)
- **Total images:** 15,339 (12,271 train / 3,068 test)
- **Classes:** 7 basic emotion categories
- **Image type:** RGB JPEG, resized to 224×224

---

## Repository Structure

```
├── raf_db_facial_expression_recognition.ipynb   # Main Kaggle Notebook
├── figures/                                      # Output visualisations
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── custom_cnn_curves.png
│   ├── resnet50_curves.png
│   ├── mobilenetv2_curves.png
│   ├── cm_custom_cnn.png
│   ├── cm_resnet50.png
│   ├── cm_mobilenetv2.png
│   ├── roc_custom_cnn.png
│   ├── roc_resnet50.png
│   ├── roc_mobilenetv2.png
│   ├── gradcam_visualisations.png
│   ├── model_comparison_bar.png
│   └── model_comparison.csv
├── Phase2_Proposal_Bhanu_Prasad_Mudraboina.docx # Full proposal document
└── README.md
```

---

## How to Run

### On Kaggle (Recommended)

1. Go to [RAF-DB Dataset](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset) and add it to your notebook.
2. Upload `raf_db_facial_expression_recognition.ipynb` as a new Kaggle Notebook.
3. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2).
4. Run all cells top to bottom.

### Locally

```bash
pip install tensorflow scikit-learn matplotlib seaborn opencv-python pandas numpy

# Update DATA_ROOT in the notebook to point to your local RAF-DB folder:
DATA_ROOT = '/path/to/DATASET/'

jupyter notebook raf_db_facial_expression_recognition.ipynb
```

---

## Implementation Highlights

- **Data Augmentation:** rotation, flipping, zoom, brightness, shear
- **Custom CNN:** 4 convolutional blocks (32→64→128→256 filters), BatchNorm, Dropout, GAP
- **Transfer Learning:** two-phase strategy — frozen base → selective fine-tuning
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
- **Evaluation:** Confusion matrix, classification report, ROC-AUC curves
- **Interpretability:** Grad-CAM visualisations using tf.GradientTape
- **Error Analysis:** grid of misclassified examples

---

## Key References

- Li & Deng (2020). Deep Facial Expression Recognition: A Survey. *IEEE TAFFC.*
- Li & Deng (2017). Reliable Crowdsourcing and Deep Locality-Preserving Learning. *CVPR.*
- He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR.*
- Selvaraju et al. (2017). Grad-CAM. *ICCV.*
