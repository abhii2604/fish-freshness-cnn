# =============================================================================
# Fish Freshness Assessment Using Convolutional Neural Networks
# Machine Learning Course — Phase 2 Code Implementation
# =============================================================================
# This script is structured as a top-to-bottom runnable pipeline.
# Run on Kaggle Notebook (GPU T4 x2 recommended) or Google Colab.
# Dataset: https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset
#          https://www.kaggle.com/datasets/tariqsays/fishy-dataset
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Imports and Configuration
# ─────────────────────────────────────────────────────────────────────────────

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import train_test_split

import cv2
from PIL import Image

warnings.filterwarnings("ignore")

# ── Reproducibility seeds ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ── Global configuration ───────────────────────────────────────────────────
IMG_SIZE      = 224          # Input size for CNN and all transfer learning models
BATCH_SIZE    = 32
EPOCHS        = 50           # Max epochs; early stopping will likely trigger before
TL_EPOCHS     = 30           # Transfer learning fine-tuning epochs
LEARNING_RATE = 1e-3
TL_LR         = 1e-4         # Lower LR for fine-tuning transfer models
DROPOUT_RATE  = 0.5
NUM_CLASSES   = 2            # Fresh / Non-Fresh
CLASS_NAMES   = ["Fresh", "Non-Fresh"]

# ── Output directory for saving figures ───────────────────────────────────
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Fish Freshness Assessment — CNN Implementation")
print(f"TensorFlow version : {tf.__version__}")
print(f"GPU available      : {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"Output directory   : {OUTPUT_DIR.resolve()}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────
# Expected directory structure (either layout works):
#
# Layout A (Kaggle fish freshness dataset):
#   /dataset/
#     fresh/      ← fresh fish images
#     non_fresh/  ← spoiled fish images
#
# Layout B (Large-scale fish dataset with GT folders):
#   /dataset/
#     NA Gilt-Head Bream/     ← each species subfolder contains images
#     ...
#   In this case, freshness labels come from a separate CSV.
#
# We support Layout A directly below. For Layout B, set USE_CSV_LABELS = True.
# ─────────────────────────────────────────────────────────────────────────────

# ── Adjust these paths to match your Kaggle / local setup ─────────────────
DATASET_DIR    = Path("/kaggle/input/fishy-dataset")   # Kaggle path
USE_CSV_LABELS = False                                  # Set True for Layout B

# ── If dataset not found, build a synthetic demo dataset for testing ───────
def build_demo_dataset(base_path: Path, n_per_class: int = 60):
    """
    Create a minimal synthetic dataset so the script runs end-to-end
    without downloading anything. Images are random noise patches.
    Remove this function when using a real dataset.
    """
    print("[INFO] Building synthetic demo dataset for testing...")
    for cls in CLASS_NAMES:
        cls_dir = base_path / cls.lower().replace("-", "_")
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            # Add slight class-specific tint so the model has a signal to learn
            if cls == "Fresh":
                arr[:, :, 1] = np.clip(arr[:, :, 1].astype(int) + 40, 0, 255)
            else:
                arr[:, :, 0] = np.clip(arr[:, :, 0].astype(int) + 40, 0, 255)
            img = Image.fromarray(arr)
            img.save(cls_dir / f"{cls.lower()}_{i:04d}.jpg")
    print(f"[INFO] Demo dataset created at: {base_path}")


# ── Build file-path dataframe ──────────────────────────────────────────────
def load_dataset_paths(dataset_dir: Path) -> pd.DataFrame:
    """
    Walk dataset_dir and return a DataFrame with columns:
        filepath (str), label (str), label_idx (int)
    Expects subfolders named 'fresh' and 'non_fresh' (or 'non-fresh').
    """
    records = []
    label_map = {}
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir():
            continue
        name_lower = folder.name.lower()
        if "fresh" in name_lower and ("non" in name_lower or "not" in name_lower or "bad" in name_lower):
            label = "Non-Fresh"
        elif "fresh" in name_lower:
            label = "Fresh"
        else:
            print(f"  [SKIP] Unrecognized folder: {folder.name}")
            continue

        label_map[label] = label_map.get(label, len(label_map))
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for fpath in folder.rglob("*"):
            if fpath.suffix.lower() in exts:
                records.append({"filepath": str(fpath), "label": label})

    df = pd.DataFrame(records)
    df["label_idx"] = df["label"].map({"Fresh": 0, "Non-Fresh": 1})
    return df


# ── Load or build dataset ──────────────────────────────────────────────────
if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
    DEMO_DIR = Path("./demo_dataset")
    build_demo_dataset(DEMO_DIR, n_per_class=80)
    DATASET_DIR = DEMO_DIR

df = load_dataset_paths(DATASET_DIR)
print(f"\n[Dataset] Total images found: {len(df)}")
print(df["label"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Exploratory Data Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, save_path: Path):
    """Bar chart of class distribution."""
    counts = df["label"].value_counts()
    colors = ["#2E75B6", "#E04E3A"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{val}\n({val / len(df) * 100:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")

plot_class_distribution(df, OUTPUT_DIR / "fig01_class_distribution.png")


def show_sample_images(df: pd.DataFrame, save_path: Path, n_per_class: int = 4):
    """Display sample images for each class."""
    fig, axes = plt.subplots(2, n_per_class, figsize=(n_per_class * 3, 6))
    for row_idx, cls in enumerate(CLASS_NAMES):
        samples = df[df["label"] == cls].sample(min(n_per_class, len(df[df["label"] == cls])), random_state=SEED)
        for col_idx, (_, row) in enumerate(samples.iterrows()):
            img = Image.open(row["filepath"]).resize((IMG_SIZE, IMG_SIZE))
            axes[row_idx][col_idx].imshow(img)
            axes[row_idx][col_idx].axis("off")
            if col_idx == 0:
                axes[row_idx][col_idx].set_ylabel(cls, fontsize=12, fontweight="bold")
    plt.suptitle("Sample Images per Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")

show_sample_images(df, OUTPUT_DIR / "fig02_sample_images.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Train / Validation / Test Split
# ─────────────────────────────────────────────────────────────────────────────

# Stratified split: 70% train, 15% validation, 15% test
df_train, df_temp = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label"])
df_val,   df_test = train_test_split(df_temp, test_size=0.50, random_state=SEED, stratify=df_temp["label"])

print(f"\n[Split] Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    print(f"  {split_name}: {split_df['label'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Data Preprocessing and Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(filepath: str, augment: bool = False) -> np.ndarray:
    """
    Load and preprocess a single image.
    - Resize to IMG_SIZE x IMG_SIZE
    - Normalize pixel values to [0, 1]
    """
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def augment_image(img: tf.Tensor) -> tf.Tensor:
    """Apply random augmentation to a single image tensor."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
    # Random rotation via tfa (if available) or manual crop
    img = tf.image.random_crop(img, [int(IMG_SIZE * 0.9), int(IMG_SIZE * 0.9), 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def make_dataset(df_subset: pd.DataFrame, augment: bool = False,
                 batch_size: int = BATCH_SIZE, shuffle: bool = False) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a filepath/label DataFrame.
    Applies preprocessing and optional augmentation.
    """
    filepaths = df_subset["filepath"].values
    labels    = df_subset["label_idx"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df_subset), seed=SEED)

    def load_and_preprocess(fp, label):
        img = preprocess_image(fp)
        if augment:
            img = augment_image(img)
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# Build all three datasets
train_ds = make_dataset(df_train, augment=True,  shuffle=True)
val_ds   = make_dataset(df_val,   augment=False, shuffle=False)
test_ds  = make_dataset(df_test,  augment=False, shuffle=False)

print(f"\n[Dataset] Batches — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# Visualize augmented images
def show_augmented_samples(df_subset: pd.DataFrame, save_path: Path, n: int = 8):
    """Show original vs augmented image pairs."""
    samples = df_subset.sample(n // 2, random_state=SEED)
    fig, axes = plt.subplots(2, n // 2, figsize=(n // 2 * 3, 6))
    for i, (_, row) in enumerate(samples.iterrows()):
        img = preprocess_image(row["filepath"])
        img_aug = augment_image(img)
        axes[0][i].imshow(img.numpy())
        axes[0][i].set_title(row["label"], fontsize=9)
        axes[0][i].axis("off")
        axes[1][i].imshow(img_aug.numpy())
        axes[1][i].set_title("Augmented", fontsize=9)
        axes[1][i].axis("off")
    axes[0][0].set_ylabel("Original", fontsize=10, fontweight="bold")
    axes[1][0].set_ylabel("Augmented", fontsize=10, fontweight="bold")
    plt.suptitle("Data Augmentation Examples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")

show_augmented_samples(df_train, OUTPUT_DIR / "fig03_augmentation_examples.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Custom CNN Architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=1) -> keras.Model:
    """
    Custom CNN with 4 convolutional blocks, BatchNorm, and Dropout.
    Designed specifically for binary image classification.

    Architecture:
        Input → [Conv → BN → ReLU → MaxPool] x4 → GAP → Dense → Dropout → Output
    """
    inputs = keras.Input(shape=input_shape, name="input")

    # ── Block 1: 32 filters ─────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="block1_conv")(inputs)
    x = layers.BatchNormalization(name="block1_bn")(x)
    x = layers.Activation("relu", name="block1_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="block1_pool")(x)

    # ── Block 2: 64 filters ─────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="block2_conv")(x)
    x = layers.BatchNormalization(name="block2_bn")(x)
    x = layers.Activation("relu", name="block2_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="block2_pool")(x)

    # ── Block 3: 128 filters ────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="block3_conv")(x)
    x = layers.BatchNormalization(name="block3_bn")(x)
    x = layers.Activation("relu", name="block3_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="block3_pool")(x)

    # ── Block 4: 256 filters ────────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="block4_conv")(x)
    x = layers.BatchNormalization(name="block4_bn")(x)
    x = layers.Activation("relu", name="block4_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="block4_pool")(x)

    # ── Global Average Pooling instead of Flatten (reduces overfitting) ─────
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Dense classifier ────────────────────────────────────────────────────
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="dense1")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout1")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="dense2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)

    # ── Output layer: Sigmoid for binary classification ──────────────────────
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="CustomCNN")
    return model


custom_cnn = build_custom_cnn()
custom_cnn.summary()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Training Utilities (Callbacks, Compile, History Plotting)
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(model_name: str, patience: int = 8) -> list:
    """Return a standard set of training callbacks."""
    return [
        callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / f"{model_name}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
    ]


def compile_model(model: keras.Model, lr: float = LEARNING_RATE) -> keras.Model:
    """Compile a binary classification model with standard settings."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )
    return model


def plot_training_history(history, model_name: str, save_dir: Path):
    """Plot accuracy and loss curves for a training history object."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Accuracy ────────────────────────────────────────────────────────────
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy", color="#2E75B6", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   color="#E04E3A", linewidth=2)
    axes[0].set_title(f"{model_name} — Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Loss ────────────────────────────────────────────────────────────────
    axes[1].plot(history.history["loss"],     label="Train Loss", color="#2E75B6", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#E04E3A", linewidth=2)
    axes[1].set_title(f"{model_name} — Loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = save_dir / f"fig_history_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"  Saved: {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Train Custom CNN
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Training Custom CNN...")
print("=" * 60)

custom_cnn = compile_model(custom_cnn, lr=LEARNING_RATE)

history_custom = custom_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=get_callbacks("CustomCNN"),
    verbose=1
)

plot_training_history(history_custom, "Custom CNN", OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Transfer Learning Models
# ─────────────────────────────────────────────────────────────────────────────

def build_transfer_model(base_arch: str, input_shape=(IMG_SIZE, IMG_SIZE, 3)) -> keras.Model:
    """
    Build a transfer learning model by attaching a custom classification head
    to a pre-trained base (VGG16, ResNet50, or MobileNetV2).

    Strategy:
        Phase 1 — Freeze base, train only the new head (fast convergence)
        Phase 2 — Unfreeze last N layers, fine-tune with low LR
    """
    arch_map = {
        "vgg16":      (VGG16,      vgg_preprocess,     -6),   # unfreeze last 2 blocks
        "resnet50":   (ResNet50,   resnet_preprocess,  -15),  # unfreeze last residual block
        "mobilenetv2":(MobileNetV2, mobilenet_preprocess, -20),
    }
    BaseClass, preprocess_fn, finetune_layer = arch_map[base_arch.lower()]

    # ── Load pre-trained base without top classifier ─────────────────────────
    base = BaseClass(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base.trainable = False  # Freeze all base layers initially

    # ── Add preprocessing + custom head ──────────────────────────────────────
    inputs = keras.Input(shape=input_shape, name="input")
    # Note: preprocess_fn scales to model's expected range (e.g., [-1, 1] for MobileNet)
    x = keras.layers.Lambda(lambda img: preprocess_fn(img * 255.0),
                            name="preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name=f"TL_{base_arch.upper()}")
    model._base_model = base
    model._finetune_layer = finetune_layer
    model._preprocess_fn = preprocess_fn
    return model


def finetune_model(model: keras.Model) -> keras.Model:
    """Unfreeze the top layers of the base model for fine-tuning."""
    base = model._base_model
    base.trainable = True
    # Freeze all layers except the last N
    finetune_from = model._finetune_layer
    for layer in base.layers[:finetune_from]:
        layer.trainable = False
    print(f"  Fine-tuning: {sum(not l.trainable for l in base.layers)} frozen, "
          f"{sum(l.trainable for l in base.layers)} trainable layers in base")
    return model


def train_transfer_model(arch_name: str) -> tuple:
    """Full two-phase training for a transfer learning model."""
    print(f"\n{'=' * 60}")
    print(f"Transfer Learning: {arch_name.upper()}")
    print(f"{'=' * 60}")

    model = build_transfer_model(arch_name)
    model = compile_model(model, lr=LEARNING_RATE)

    # Phase 1: Train head only
    print(f"\n[Phase 1] Training classification head (base frozen)...")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=get_callbacks(f"TL_{arch_name}_phase1", patience=5),
        verbose=1
    )

    # Phase 2: Unfreeze and fine-tune
    print(f"\n[Phase 2] Fine-tuning unfrozen base layers...")
    model = finetune_model(model)
    model = compile_model(model, lr=TL_LR)

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TL_EPOCHS,
        callbacks=get_callbacks(f"TL_{arch_name}_best", patience=6),
        verbose=1
    )

    # Merge histories for plotting
    combined_history_dict = {}
    for key in history_phase1.history:
        combined_history_dict[key] = (
            history_phase1.history[key] + history_phase2.history.get(key, [])
        )

    class CombinedHistory:
        def __init__(self, d): self.history = d
    combined_history = CombinedHistory(combined_history_dict)

    plot_training_history(combined_history, arch_name.upper(), OUTPUT_DIR)
    return model, combined_history


# Train all three transfer learning models
model_vgg16,       history_vgg16       = train_transfer_model("vgg16")
model_resnet50,    history_resnet50    = train_transfer_model("resnet50")
model_mobilenetv2, history_mobilenetv2 = train_transfer_model("mobilenetv2")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model: keras.Model, test_dataset: tf.data.Dataset,
                   model_name: str) -> dict:
    """
    Evaluate a trained model on the test set.
    Returns a dict with accuracy, precision, recall, F1, AUC.
    """
    y_true, y_pred_prob = [], []
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0).flatten()
        y_pred_prob.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

    y_true      = np.array(y_true).astype(int)
    y_pred_prob = np.array(y_pred_prob)
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    results = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc":       round(roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5, 4),
        "y_true":    y_true,
        "y_pred":    y_pred,
        "y_pred_prob": y_pred_prob
    }
    return results


# Evaluate all models
print("\n[Evaluating all models on test set...]")
results_custom   = evaluate_model(custom_cnn,        test_ds, "Custom CNN")
results_vgg16    = evaluate_model(model_vgg16,        test_ds, "VGG16")
results_resnet50 = evaluate_model(model_resnet50,     test_ds, "ResNet50")
results_mobilenet= evaluate_model(model_mobilenetv2,  test_ds, "MobileNetV2")

all_results = [results_custom, results_vgg16, results_resnet50, results_mobilenet]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str, class_names: list, save_dir: Path):
    """Plot and save a normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(axes, [cm, cm_norm], ["Counts", "Normalized"]):
        fmt = "d" if title == "Counts" else ".2f"
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="white")
        ax.set_title(f"{model_name} — Confusion Matrix ({title})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    plt.tight_layout()
    fname = save_dir / f"fig_cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved: {fname}")


for res in all_results:
    plot_confusion_matrix(res["y_true"], res["y_pred"], res["model"], CLASS_NAMES, OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: Classification Report
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Classification Reports")
print("=" * 60)

for res in all_results:
    print(f"\n--- {res['model']} ---")
    print(classification_report(res["y_true"], res["y_pred"], target_names=CLASS_NAMES))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: Model Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

comparison_df = pd.DataFrame([{
    "Model":     r["model"],
    "Accuracy":  r["accuracy"],
    "Precision": r["precision"],
    "Recall":    r["recall"],
    "F1-Score":  r["f1"],
    "AUC-ROC":   r["auc"],
} for r in all_results])

print("\n" + "=" * 60)
print("Model Comparison Table")
print("=" * 60)
print(comparison_df.to_string(index=False))
comparison_df.to_csv(OUTPUT_DIR / "table_model_comparison.csv", index=False)

# Visual comparison bar chart
def plot_model_comparison(df: pd.DataFrame, save_path: Path):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    x = np.arange(len(df))
    width = 0.15
    colors = ["#2E75B6", "#E04E3A", "#2ECC71", "#F39C12", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, df[metric], width, label=metric, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")

plot_model_comparison(comparison_df, OUTPUT_DIR / "fig_model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: ROC Curves (All Models on One Plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(all_results: list, save_path: Path):
    """Plot ROC curves for all models on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2E75B6", "#E04E3A", "#2ECC71", "#F39C12"]

    for res, color in zip(all_results, colors):
        if len(np.unique(res["y_true"])) > 1:
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_pred_prob"])
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{res['model']} (AUC = {res['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")

plot_roc_curves(all_results, OUTPUT_DIR / "fig_roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14: Grad-CAM Visualization
# ─────────────────────────────────────────────────────────────────────────────

def get_gradcam_heatmap(model: keras.Model, img_array: np.ndarray,
                         last_conv_layer_name: str) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a given image.
    Grad-CAM uses the gradient of the predicted class score with respect to
    the last convolutional layer to highlight discriminative regions.

    Reference: Selvaraju et al. (2017), ICCV.
    """
    # Build a model that outputs both the last conv layer activations and predictions
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array[np.newaxis], training=False)
        # For binary classification, the loss is the prediction itself
        loss = predictions[:, 0]

    # Gradient of the class score w.r.t. the conv layer outputs
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients over spatial dimensions → importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by their importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray,
                     alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    original_rgb = (original_img * 255).astype(np.uint8)
    overlaid = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid


def plot_gradcam_grid(model: keras.Model, df_test: pd.DataFrame,
                       last_conv_name: str, model_name: str,
                       save_path: Path, n: int = 6):
    """
    Plot a grid of original images with Grad-CAM overlays for
    correctly and incorrectly classified examples.
    """
    samples = df_test.sample(min(n, len(df_test)), random_state=SEED)
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))

    for col_idx, (_, row) in enumerate(samples.iterrows()):
        img = preprocess_image(row["filepath"]).numpy()
        heatmap = get_gradcam_heatmap(model, img, last_conv_name)
        overlaid = overlay_gradcam(img, heatmap)
        pred_prob = model.predict(img[np.newaxis], verbose=0)[0, 0]
        pred_label = CLASS_NAMES[int(pred_prob >= 0.5)]
        true_label = row["label"]
        color = "green" if pred_label == true_label else "red"

        axes[0][col_idx].imshow(img)
        axes[0][col_idx].set_title(f"True: {true_label}", fontsize=8)
        axes[0][col_idx].axis("off")

        axes[1][col_idx].imshow(overlaid)
        axes[1][col_idx].set_title(f"Pred: {pred_label}\n({pred_prob:.2f})",
                                    fontsize=8, color=color)
        axes[1][col_idx].axis("off")

    axes[0][0].set_ylabel("Original", fontsize=9, fontweight="bold")
    axes[1][0].set_ylabel("Grad-CAM", fontsize=9, fontweight="bold")
    plt.suptitle(f"Grad-CAM Visualizations — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")


# Apply Grad-CAM to Custom CNN (last conv layer name: "block4_conv")
# and best transfer model (ResNet50 last conv: "conv5_block3_out")
try:
    plot_gradcam_grid(custom_cnn, df_test, "block4_conv",
                      "Custom CNN", OUTPUT_DIR / "fig_gradcam_custom_cnn.png")
except Exception as e:
    print(f"[WARN] Grad-CAM failed for Custom CNN: {e}")

try:
    plot_gradcam_grid(model_resnet50, df_test, "conv5_block3_out",
                      "ResNet50", OUTPUT_DIR / "fig_gradcam_resnet50.png")
except Exception as e:
    print(f"[WARN] Grad-CAM failed for ResNet50: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15: Error Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_analysis(model: keras.Model, df_test: pd.DataFrame,
                          model_name: str, save_path: Path, n: int = 8):
    """
    Display the model's most confident wrong predictions.
    Helps identify systematic failure modes.
    """
    errors = []
    for _, row in df_test.iterrows():
        img = preprocess_image(row["filepath"]).numpy()
        pred_prob = float(model.predict(img[np.newaxis], verbose=0)[0, 0])
        pred_label = CLASS_NAMES[int(pred_prob >= 0.5)]
        true_label = row["label"]
        if pred_label != true_label:
            confidence = pred_prob if pred_prob >= 0.5 else 1 - pred_prob
            errors.append({
                "filepath": row["filepath"],
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "pred_prob": pred_prob
            })

    if not errors:
        print(f"  No errors found for {model_name}.")
        return

    errors = sorted(errors, key=lambda x: x["confidence"], reverse=True)[:n]
    ncols = min(n, 4)
    nrows = (len(errors) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))
    axes = np.array(axes).reshape(-1) if nrows * ncols > 1 else [axes]

    for i, err in enumerate(errors):
        img = preprocess_image(err["filepath"]).numpy()
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {err['true_label']}\nPred: {err['pred_label']} ({err['confidence']:.2f})",
            fontsize=8, color="red"
        )
        axes[i].axis("off")

    for j in range(len(errors), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Error Analysis — {model_name} (Most Confident Mistakes)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")
    print(f"  Total errors: {len(errors)} of {len(df_test)}")

plot_error_analysis(custom_cnn, df_test, "Custom CNN",
                    OUTPUT_DIR / "fig_error_analysis_custom_cnn.png")
plot_error_analysis(model_resnet50, df_test, "ResNet50",
                    OUTPUT_DIR / "fig_error_analysis_resnet50.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16: Save Models and Final Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Saving models...]")
custom_cnn.save(OUTPUT_DIR / "model_custom_cnn.keras")
model_vgg16.save(OUTPUT_DIR / "model_vgg16.keras")
model_resnet50.save(OUTPUT_DIR / "model_resnet50.keras")
model_mobilenetv2.save(OUTPUT_DIR / "model_mobilenetv2.keras")
print("  All models saved.")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.loc[comparison_df["F1-Score"].idxmax(), "Model"]
best_f1 = comparison_df["F1-Score"].max()
print(f"\n  Best model by F1-Score: {best_model_name} (F1 = {best_f1:.4f})")
print(f"\n  All output figures saved to: {OUTPUT_DIR.resolve()}")
print("=" * 60)
print("Done!")
