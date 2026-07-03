# Fish Freshness Assessment Using CNNs

A CNN-based pipeline for classifying fish images as **Fresh** or **Non-Fresh**,
built on top of the [Large-Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset).
Since that dataset only contains fresh fish, "Non-Fresh" examples are synthesized
by applying randomized color-shift, noise, and contrast/sharpness degradation to
fresh images.

The notebook trains and compares four models:
- A custom 4-block CNN
- VGG16 (transfer learning, 2-phase fine-tuning)
- ResNet50 (transfer learning, 2-phase fine-tuning)
- MobileNetV2 (transfer learning, 2-phase fine-tuning)

It also includes EDA, augmentation previews, ROC/confusion-matrix plots,
Grad-CAM visualizations, and error analysis.

## Contents
- `fish_freshness_cnn.ipynb` — main notebook (run this)
- `fish_freshness_cnn.py` — same content as a plain script, in [jupytext "light" format](https://jupytext.readthedocs.io/), kept in sync with the notebook for clean diffs
- `requirements.txt` — Python dependencies

## Usage
Designed to run on **Kaggle** (GPU T4 x2 recommended) with the
[Large-Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)
attached as an input. Update `FISH_BASE` near the top of the notebook if your
dataset path differs.

To run locally or on Colab instead:
1. Download the dataset and update `FISH_BASE` to point to it.
2. `pip install -r requirements.txt`
3. Open and run `fish_freshness_cnn.ipynb`.

## Keeping the .py and .ipynb in sync
This repo uses jupytext so the notebook stays diffable in git. After editing
either file, resync with:
```bash
pip install jupytext
jupytext --sync fish_freshness_cnn.ipynb
```

## Outputs
Running the notebook creates an `outputs/` directory with figures, the
model comparison CSV, and saved `.keras` model files (git-ignored by default
since trained models can be large).
