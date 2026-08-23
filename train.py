import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ========================================================
# SETTINGS & HYPERPARAMETERS
# ========================================================

# LOAD DATASET
DATASET_PATH = r"C:\Users\User\Downloads\lung_cancer_dataset"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")

# Image parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Output directories
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lung_cancer_cnn.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# ========================================================
# CHECK DATASET DIRECTORIES
# ========================================================

print("==================================================")
print("Checking dataset directories for Lung Cancer Detection...")
print(f"Train folder: {TRAIN_DIR}")
print(f"Validation folder: {VAL_DIR}")
print("==================================================")

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"CRITICAL ERROR: Training folder not found at {TRAIN_DIR}"
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"CRITICAL ERROR: Validation folder not found at {VAL_DIR}"
    )

# ========================================================
# LOAD DATASET
# ========================================================

print("\nLoading training lung scan images...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=True,
    seed=123
)

print("\nLoading validation lung scan images...")

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)

print("\nDetected Classes for Lung Cancer Analysis:")
class_names = train_dataset.class_names
print(class_names)

# ========================================================
# PERFORMANCE OPTIMIZATION
# ========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

val_dataset = val_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

# ========================================================
# DATA AUGMENTATION PIPELINE
# ========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1)
], name="lung_scan_augmentation")

# ========================================================
# BUILD DEEP CNN MODEL ARCHITECTURE
# ========================================================

def build_lung_cancer_model(input_shape):
    """
    Constructs a custom deep Convolutional Neural Network
    optimized for extracting complex pulmonary nodules
    and features from lung scans.
    """

    inputs = layers.Input(shape=input_shape)

    # Apply Data Augmentation and Normalization
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    # Block 1
    x = layers.Conv2D(
        64, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        64, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        128, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        256, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.3)(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.4)(x)

    # Fully Connected / Classification Head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        512, activation="relu"
    )(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(
        256, activation="relu"
    )(x)
    x = layers.Dropout(0.5)(x)

    # Binary output:
    # Normal (0) vs. Malignant/Lung Cancer (1)
    outputs = layers.Dense(
        1, activation="sigmoid"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="LungCancerCNN"
    )

    return model


model = build_lung_cancer_model(
    (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
)

# ========================================================
# DISPLAY MODEL ARCHITECTURE
# ========================================================

print("\n==================================================")
print("MODEL ARCHITECTURE SUMMARY")
print("==================================================")

model.summary()

# ========================================================
# COMPILE MODEL
# ========================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision")
    ]
)

# ========================================================
# DEFINE CALLBACKS
# ========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ========================================================
# TRAIN THE MODEL
# ========================================================

print("\n==================================================")
print("STARTING LUNG CANCER CNN TRAINING
