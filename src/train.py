"""
Two-Stage CNN Model Training using MobileNetV2 Transfer Learning
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Requirements:
- MobileNetV2 backbone pre-trained on ImageNet.
- Classification Head: GlobalAveragePooling2D -> Dense(256, ReLU) -> Dropout(0.3) -> Dense(num_classes, Softmax).
- Two-stage training:
  Stage 1: Frozen MobileNetV2 base, Adam lr=1e-3.
  Stage 2: Fine-tune top layers with lower learning rate lr=1e-5.
- Categorical crossentropy loss.
- EarlyStopping and ModelCheckpoint.
- Outputs saved to models/crop_disease_model.keras and history logged.
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.preprocess import get_data_generators

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

def build_model(num_classes: int = config.NUM_CLASSES):
    """
    Constructs the MobileNetV2 transfer learning model architecture.
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )
    
    # Freeze base model layers initially
    base_model.trainable = False

    inputs = keras.Input(shape=config.INPUT_SHAPE)
    # Forward through MobileNetV2 base
    x = base_model(inputs, training=False)
    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.3, name="dropout_0.3")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CropLeaf_MobileNetV2")
    return model, base_model

def train_model():
    """
    Executes two-stage training workflow with early stopping and checkpointing.
    """
    print("=" * 70)
    print("AcenAAr Smart Spray Robot - Crop Leaf Disease Detection Model Training")
    print("=" * 70)

    # 1. Load Data Generators
    print("\n[Step 1/4] Preparing Data Generators...")
    train_gen, val_gen, test_gen = get_data_generators(batch_size=config.BATCH_SIZE)
    print(f"Loaded {train_gen.samples} training, {val_gen.samples} validation, {test_gen.samples} test images.")

    # 2. Build Model
    print("\n[Step 2/4] Initializing MobileNetV2 Architecture...")
    model, base_model = build_model(num_classes=config.NUM_CLASSES)
    model.summary()

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(config.KERAS_MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    # 3. Stage 1: Train Top Classification Head (Frozen Base)
    print("\n" + "=" * 50)
    print(f"STAGE 1: Training Classification Head ({config.STAGE1_EPOCHS} epochs, lr={config.STAGE1_LR})")
    print("=" * 50)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.STAGE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_stage1 = time.time()
    history_stage1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.STAGE1_EPOCHS,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    stage1_duration = time.time() - start_stage1
    print(f"Stage 1 completed in {stage1_duration:.1f}s")

    # 4. Stage 2: Fine-Tuning Selected Top Layers
    print("\n" + "=" * 50)
    print(f"STAGE 2: Fine-Tuning MobileNetV2 Top Layers ({config.STAGE2_EPOCHS} epochs, lr={config.STAGE2_LR})")
    print("=" * 50)

    base_model.trainable = True
    # Freeze the first 100 layers and fine-tune from layer 100 onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"Number of layers in base model: {len(base_model.layers)}")
    print(f"Frozen layers: {fine_tune_at} | Trainable fine-tuning layers: {len(base_model.layers) - fine_tune_at}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.STAGE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_stage2 = time.time()
    initial_epoch = len(history_stage1.history["accuracy"])
    total_epochs = initial_epoch + config.STAGE2_EPOCHS

    history_stage2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    stage2_duration = time.time() - start_stage2
    print(f"Stage 2 completed in {stage2_duration:.1f}s")

    # Merge Histories
    combined_history = {
        "accuracy": history_stage1.history["accuracy"] + history_stage2.history["accuracy"],
        "val_accuracy": history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"],
        "loss": history_stage1.history["loss"] + history_stage2.history["loss"],
        "val_loss": history_stage1.history["val_loss"] + history_stage2.history["val_loss"]
    }

    history_path = config.MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=4)
    print(f"\nTraining history successfully saved to: {history_path}")

    # Ensure best model is saved
    if not config.KERAS_MODEL_PATH.exists():
        model.save(str(config.KERAS_MODEL_PATH))
    print(f"Best Keras model saved at: {config.KERAS_MODEL_PATH}")

    return model, combined_history

if __name__ == "__main__":
    train_model()
