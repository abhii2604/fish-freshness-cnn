# Fish Freshness Assessment Using Convolutional Neural Networks

> Machine Learning Course — Phase 2: Proposal & Code Implementation

---

## Project Overview

This project implements an automated fish freshness classification system using Convolutional Neural Networks (CNNs). The system classifies fish images as **Fresh** or **Non-Fresh** by learning visual features such as eye clarity, skin texture, and color saturation.

Four models are trained and compared:
- **Custom CNN** — designed from scratch
- **VGG16** — fine-tuned from ImageNet weights
- **ResNet50** — fine-tuned from ImageNet weights
- **MobileNetV2** — fine-tuned from ImageNet weights

---

## Repository Structure

```
fish-freshness-cnn/
│
├── fish_freshness_cnn.py       # Complete implementation (also usable as Kaggle Notebook)
├── Fish_Freshness_CNN_Proposal.docx  # Phase 2 proposal document
├── README.md                   # This file
│
├── outputs/                    # Generated figures and saved models (after running)
│   ├── fig01_class_distribution.png
│   ├── fig02_sample_images.png
│   ├── fig03_augmentation_examples.png
│   ├── fig_history_custom_cnn.png
│   ├── fig_history_vgg16.png
│   ├── fig_history_resnet50.png
│   ├── fig_history_mobilenetv2.png
│   ├── fig_cm_*.png                  # Confusion matrices
│   ├── fig_roc_curves.png
│   ├── fig_model_comparison.png
│   ├── fig_gradcam_*.png             # Grad-CAM saliency maps
│   ├── fig_error_analysis_*.png      # Error analysis
│   └── table_model_comparison.csv
│
└── requirements.txt
```

---

## Dataset

| Property | Details |
|---|---|
| **Primary Dataset** | [Large-Scale Fish Dataset — Kaggle](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) |
| **Freshness Labels** | [Fishy Dataset — Kaggle](https://www.kaggle.com/datasets/tariqsays/fishy-dataset) |
| **Classes** | Fresh / Non-Fresh |
| **Total Images** | ~9,000 |
| **Image Size** | Resized to 224×224 RGB |
| **Split** | 70% train / 15% val / 15% test (stratified) |

The dataset should be organized as:
```
dataset/
  fresh/          ← fresh fish images
  non_fresh/      ← spoiled/non-fresh fish images
```

---

## Setup & Installation

### Requirements

```bash
pip install tensorflow>=2.12 numpy pandas matplotlib seaborn scikit-learn opencv-python Pillow
```

Or install from file:
```bash
pip install -r requirements.txt
```

### Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fish-freshness-cnn.git
cd fish-freshness-cnn

# 2. Download the dataset from Kaggle and organize as described above

# 3. Edit DATASET_DIR in fish_freshness_cnn.py to point to your dataset folder

# 4. Run the script
python fish_freshness_cnn.py
```

> **Note:** If no dataset is found, a synthetic demo dataset is automatically generated so the pipeline can be tested end-to-end.

### Running on Kaggle Notebook

1. Open a new Kaggle Notebook
2. Add the dataset as an input (`+ Add Data`)
3. Upload `fish_freshness_cnn.py` or paste it into a code cell
4. Set `DATASET_DIR = Path("/kaggle/input/fishy-dataset")` (or your dataset path)
5. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
6. Run all cells

---

## Methodology

```
Dataset Loading
      ↓
Exploratory Data Analysis (class distribution, sample images)
      ↓
Train / Val / Test Split (70/15/15, stratified)
      ↓
Preprocessing (resize 224×224, normalize [0,1])
      ↓
Data Augmentation (flip, rotate, brightness, contrast, zoom)
      ↓
┌─────────────────────────────────┐
│  Custom CNN (trained from scratch) │
└─────────────────────────────────┘
      +
┌──────────────────────────────────────────────────────────────┐
│  Transfer Learning: VGG16 / ResNet50 / MobileNetV2           │
│  Phase 1: Freeze base → train head                           │
│  Phase 2: Unfreeze top layers → fine-tune with low LR        │
└──────────────────────────────────────────────────────────────┘
      ↓
Evaluation: Accuracy / Precision / Recall / F1 / AUC-ROC
      ↓
Confusion Matrix + Classification Report
      ↓
ROC Curves (all models)
      ↓
Grad-CAM Saliency Maps
      ↓
Error Analysis
      ↓
Model Comparison Table
```

---

## Results (Expected)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Custom CNN | ~82% | ~0.81 | ~0.80 | ~0.81 | ~0.88 |
| VGG16 | ~91% | ~0.90 | ~0.91 | ~0.90 | ~0.96 |
| ResNet50 | ~93% | ~0.92 | ~0.93 | ~0.92 | ~0.97 |
| MobileNetV2 | ~89% | ~0.88 | ~0.89 | ~0.88 | ~0.95 |

*Actual results will depend on the dataset version and hardware.*

---

## Key Features

- ✅ Full preprocessing pipeline with data augmentation
- ✅ Custom 4-block CNN architecture with BatchNorm and Dropout
- ✅ Two-phase transfer learning for VGG16, ResNet50, MobileNetV2
- ✅ Confusion matrix (counts + normalized)
- ✅ Training accuracy/loss curves per model
- ✅ ROC curves for all models on one plot
- ✅ Grad-CAM saliency visualizations
- ✅ Error analysis (most confident wrong predictions)
- ✅ CSV comparison table of all metrics
- ✅ All figures auto-saved to `outputs/`
- ✅ Synthetic demo dataset for pipeline testing without real data

---

## Author

**[Your Name]**  
Machine Learning Course — Phase 2 Submission  
Date: June 2026

---

## References

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
3. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks. *ICLR 2015*.
4. Howard, A. G., et al. (2017). MobileNets. *arXiv:1704.04861*.
5. Selvaraju, R. R., et al. (2017). Grad-CAM. *ICCV 2017*.
