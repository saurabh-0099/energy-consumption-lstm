"""
Evaluate the trained LSTM model on the held-out test set.

Usage:
    python src/evaluate.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
import joblib

from config import (
    MODEL_PATH,
    SCALER_PATH,
    PRED_PLOT_PATH,
    METRICS_PATH,
    OUTPUT_DIR,
    FEATURE_COLS,
    TARGET_COL,
)
from data_preprocessing import get_train_val_test_sequences


def inverse_transform_target(scaled_values, scaler, target_idx, n_features):
    """
    MinMaxScaler was fit on all features together, so to invert just the
    target column we build a dummy array with the same number of columns,
    place our values in the target column, inverse-transform, then slice.
    """
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, target_idx] = scaled_values.flatten()
    inverted = scaler.inverse_transform(dummy)
    return inverted[:, target_idx]


def plot_predictions(y_true, y_pred, n_points=500):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:n_points], label="Actual", linewidth=1.5)
    plt.plot(y_pred[:n_points], label="Predicted", linewidth=1.5, alpha=0.8)
    plt.title(f"Predicted vs Actual {TARGET_COL} (first {n_points} test hours)")
    plt.xlabel("Time step (hours)")
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PRED_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved prediction plot to {PRED_PLOT_PATH}")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python src/train.py` first."
        )

    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    (_, _), (_, _), (X_test, y_test), _, target_idx = get_train_val_test_sequences()

    y_pred_scaled = model.predict(X_test).flatten()

    n_features = len(FEATURE_COLS)
    y_test_actual = inverse_transform_target(y_test, scaler, target_idx, n_features)
    y_pred_actual = inverse_transform_target(y_pred_scaled, scaler, target_idx, n_features)

    rmse = float(np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)))
    mae = float(mean_absolute_error(y_test_actual, y_pred_actual))
    r2 = float(r2_score(y_test_actual, y_pred_actual))
    mape = float(
        np.mean(np.abs((y_test_actual - y_pred_actual) / np.clip(y_test_actual, 1e-3, None))) * 100
    )

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_percent": mape}
    print("Test set performance:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    plot_predictions(y_test_actual, y_pred_actual)


if __name__ == "__main__":
    main()
