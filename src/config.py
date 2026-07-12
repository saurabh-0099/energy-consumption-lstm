"""
Central configuration for the Sustainable Energy Consumption Prediction project.
Change hyperparameters here — every other script imports from this file.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "household_power_consumption.txt")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "hourly_power_consumption.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "lstm_energy_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")

LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_loss.png")
PRED_PLOT_PATH = os.path.join(OUTPUT_DIR, "predictions_vs_actual.png")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# Data / feature engineering
# ---------------------------------------------------------------------------
# Target column we are forecasting
TARGET_COL = "Global_active_power"

# Extra engineered time features fed into the model along with the target
FEATURE_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
    "dayofweek",
    "month",
]

# How many past hourly time steps the LSTM looks at to predict the next hour
LOOKBACK = 24          # 24 hours of history
FORECAST_HORIZON = 1   # predict 1 hour ahead

# Fraction of the (chronologically ordered) dataset used for testing
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1  # taken from the training portion

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
DENSE_UNITS = 16

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 6

RANDOM_SEED = 42
