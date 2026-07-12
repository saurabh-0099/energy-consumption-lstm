"""
Use the trained model to forecast the next hour's energy consumption
given the most recent LOOKBACK hours of readings.

Usage:
    python src/predict.py
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

from config import (
    MODEL_PATH,
    SCALER_PATH,
    PROCESSED_DATA_PATH,
    FEATURE_COLS,
    TARGET_COL,
    LOOKBACK,
)


def predict_next_hour():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(PROCESSED_DATA_PATH, index_col="Datetime", parse_dates=True)
    df = df[FEATURE_COLS]

    last_window = df.values[-LOOKBACK:]
    scaled_window = scaler.transform(last_window)
    X = np.expand_dims(scaled_window, axis=0)  # shape (1, LOOKBACK, n_features)

    pred_scaled = model.predict(X).flatten()[0]

    target_idx = FEATURE_COLS.index(TARGET_COL)
    dummy = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, target_idx] = pred_scaled
    pred_actual = scaler.inverse_transform(dummy)[0, target_idx]

    last_timestamp = df.index[-1]
    print(f"Last known reading: {last_timestamp}")
    print(f"Predicted {TARGET_COL} for next hour: {pred_actual:.4f} kW")
    return pred_actual


if __name__ == "__main__":
    predict_next_hour()
