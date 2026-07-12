"""
Train the LSTM energy-consumption forecasting model.

Usage (from the project root, with venv activated):
    python src/train.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from config import (
    MODEL_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    LOSS_PLOT_PATH,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
)
from data_preprocessing import get_train_val_test_sequences
from model import build_lstm_model


def set_seeds(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_history(history):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved training loss plot to {LOSS_PLOT_PATH}")


def main():
    set_seeds()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, target_idx = (
        get_train_val_test_sequences()
    )

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history)
    print(f"Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
