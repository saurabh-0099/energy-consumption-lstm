"""
Data loading & preprocessing for the UCI "Individual Household Electric
Power Consumption" dataset.

Dataset link (download manually, see README):
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Raw file: household_power_consumption.txt
Columns : Date;Time;Global_active_power;Global_reactive_power;Voltage;
          Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
Missing values are marked with '?' in the raw file.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    FEATURE_COLS,
    TARGET_COL,
    LOOKBACK,
    FORECAST_HORIZON,
    TEST_SPLIT,
    VAL_SPLIT,
    SCALER_PATH,
)


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw semicolon-separated UCI txt file into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Download 'household_power_consumption.txt' from the UCI repository "
            "and place it inside the 'data/' folder. See README.md for the link."
        )

    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    # Combine Date + Time into a single datetime index
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("Datetime").sort_index()

    # Numeric columns come in as object dtype because of the '?' markers
    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_and_resample(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """
    Handle missing values and resample the minute-level readings to an
    hourly mean (smooths noise and keeps the sequence length manageable).
    """
    # Forward/backward fill short gaps, then interpolate anything left
    df = df.interpolate(method="time", limit_direction="both")

    hourly = df.resample(freq).mean()
    hourly = hourly.dropna()

    return hourly


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical/calendar features useful for an energy-usage model."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    return df


def build_processed_dataset() -> pd.DataFrame:
    """Full pipeline: load -> clean -> resample -> feature engineer -> save."""
    raw = load_raw_data()
    hourly = clean_and_resample(raw)
    featured = add_time_features(hourly)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    featured.to_csv(PROCESSED_DATA_PATH)
    print(f"Processed dataset saved to {PROCESSED_DATA_PATH} "
          f"({len(featured)} hourly rows).")
    return featured


def create_sequences(data: np.ndarray, lookback: int, horizon: int, target_idx: int):
    """
    Turn a 2D array (timesteps x features) into supervised LSTM sequences.

    X[i] = data[i : i+lookback]                 -> shape (lookback, n_features)
    y[i] = data[i+lookback+horizon-1, target_idx] -> scalar (next-step target)
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback + horizon - 1, target_idx])
    return np.array(X), np.array(y)


def get_train_val_test_sequences():
    """
    Loads the processed CSV (building it first if needed), scales features,
    and returns train/val/test sequence tensors ready for the LSTM.
    """
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col="Datetime", parse_dates=True)
    else:
        df = build_processed_dataset()

    df = df[FEATURE_COLS]
    target_idx = FEATURE_COLS.index(TARGET_COL)

    n = len(df)
    test_start = int(n * (1 - TEST_SPLIT))
    train_val_df = df.iloc[:test_start]
    test_df = df.iloc[test_start:]

    val_start = int(len(train_val_df) * (1 - VAL_SPLIT))
    train_df = train_val_df.iloc[:val_start]
    val_df = train_val_df.iloc[val_start:]

    # Fit scaler ONLY on training data to avoid leakage
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    X_train, y_train = create_sequences(train_scaled, LOOKBACK, FORECAST_HORIZON, target_idx)
    X_val, y_val = create_sequences(val_scaled, LOOKBACK, FORECAST_HORIZON, target_idx)
    X_test, y_test = create_sequences(test_scaled, LOOKBACK, FORECAST_HORIZON, target_idx)

    print(f"Train sequences: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, target_idx


if __name__ == "__main__":
    build_processed_dataset()
