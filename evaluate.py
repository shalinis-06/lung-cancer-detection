import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)

# ========================================================
# SETTINGS & DIRECTORY CONFIGURATION
# ========================================================

DATASET_PATH = r"C:\Users\User\Downloads\lung_cancer_dataset"

TEST_DIR = os.path.join(DATASET_PATH, "test")

MODEL_PATH = "models/lung_cancer_cnn.keras"
RESULTS_DIR = "results"

IMG_SIZE = (256, 256)
BATCH_SIZE = 32

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================================================
# LOAD TEST DATASET
# ========================================================

print("==================================================")
print("Loading test dataset for comprehensive evaluation...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)

class_names = test_dataset.class_names

print(f"Detected Evaluation Classes: {class_names}")
print("==================================================")

# ========================================================
# LOAD TRAINED CNN MODEL
# ========================================================

print("\nLoading trained Lung Cancer CNN model weights...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model architecture and weights loaded successfully.")

except Exception as e:
    raise RuntimeError(
        f"Failed to load model from {MODEL_PATH}. Error: {e}"
    )

# ========================================================
# EXTRACT TRUE LABELS
# ========================================================

print("\nExtracting ground truth labels from test set...")

y_true = []

for images, labels in test_dataset:
    y_true.extend(labels.numpy().flatten())

y_true = np.array(y_true).astype(int)

# ========================================================
# GENERATE PREDICTIONS
# ========================================================

print("Generating inferences on test data...")

y_probability = model.predict(
    test_dataset,
    verbose=1
)

y_probability = y_probability.flatten()

# Convert probabilities to binary predictions
# using a 0.5 threshold

y_pred = (y_probability >= 0.5).astype(int)

# ========================================================
# CALCULATE CLINICAL METRICS
# ========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

# Confusion Matrix
cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

# Specificity (True Negative Rate)
specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

# ROC-AUC
roc_auc = roc_auc_score(
    y_true,
    y_probability
)

# ========================================================
# PRINT RESULTS TO CONSOLE
# ========================================================

print("\n==================================================")
print("TEST SET EVALUATION RESULTS")
print("==================================================")

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")

print("==================================================")

print("\nConfusion Matrix:")
print(cm)

print("\nDetailed Classification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Benign/Normal",
            "Malignant/Cancer"
        ]
    )
)

# ========================================================
# EXPORT METRICS TO JSON
# ========================================================

metrics_dict = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "specificity": float(specificity),
    "roc_auc": float(roc_auc)
}

metrics_filepath = os.path.join(
    RESULTS_DIR,
    "metrics.json"
)

with open(metrics_filepath, "w") as file:
    json.dump(
        metrics_dict,
        file,
        indent=4
    )

print(
    f"\nMetrics serialized to: {metrics_filepath}"
)

# ========================================================
# PLOT CONFUSION MATRIX
# ========================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=[
        "NORMAL",
        "LUNG CANCER"
    ],
    yticklabels=[
        "NORMAL",
        "LUNG CANCER"
    ],
    annot_kws={
        "size": 16
    }
)

plt.xlabel(
    "Predicted Condition",
    fontsize=12
)

plt.ylabel(
    "Actual Condition",
    fontsize=12
)

plt.title(
    "Lung Cancer Detection Confusion Matrix",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()

# ========================================================
# PLOT ROC CURVE
# ========================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_probability
)

plt.figure(figsize=(8, 7))

plt.plot(
    fpr,
    tpr,
    color="darkred",
    lw=2,
    label=f"ROC Curve (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    color="navy",
    lw=2,
    linestyle="--",
    label="Random Guess"
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel(
    "False Positive Rate",
    fontsize=12
)

plt.ylabel(
    "True Positive Rate",
    fontsize=12
)

plt.title(
    "ROC Curve - Lung Cancer Detection",
    fontsize=14,
    fontweight="bold"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

roc_path = os.path.join(
    RESULTS_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300
)

plt.close()

# ========================================================
# SCRIPT COMPLETION
# ========================================================

print("\n==================================================")
print("EVALUATION COMPLETELY FINISHED")
print("==================================================")

print("Artifacts generated in results/:")
print(" - metrics.json")
print(" - confusion_matrix.png")
print(" - roc_curve.png")

print("==================================================")
