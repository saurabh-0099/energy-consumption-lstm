"""
LSTM model architecture for time-series energy consumption forecasting.
"""

from tensorflow import keras
from tensorflow.keras import layers

from config import (
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DROPOUT_RATE,
    DENSE_UNITS,
    LEARNING_RATE,
)


def build_lstm_model(input_shape):
    """
    input_shape: (lookback, n_features)
    Returns a compiled Keras model that outputs a single scalar
    (next-hour Global_active_power).
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(LSTM_UNITS_1, return_sequences=True),
            layers.Dropout(DROPOUT_RATE),
            layers.LSTM(LSTM_UNITS_2, return_sequences=False),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(DENSE_UNITS, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    return model


if __name__ == "__main__":
    m = build_lstm_model((24, 10))
    m.summary()
