"""
lstm_forecaster.py
------------------
LSTM-based climate feature forecaster.

Trains on a time-series of climate data and predicts the next
step's features: temperature, humidity, wind speed, and pressure.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


CLIMATE_FEATURES = ["meantemp", "humidity", "wind_speed", "meanpressure"]


def load_climate_data(filepath: str) -> pd.DataFrame:
    """Load and sort climate CSV by date."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df[CLIMATE_FEATURES].ffill()


def create_sequences(data: np.ndarray, seq_length: int = 5):
    """Convert a 2-D array into (X, y) sequence pairs."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape) -> Sequential:
    """Construct the LSTM architecture."""
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(4),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_forecaster(
    filepath: str,
    seq_length: int = 5,
    test_split: float = 0.2,
    epochs: int = 10,
    batch_size: int = 32,
):
    """
    Full pipeline: load → scale → sequence → train → predict.

    Returns
    -------
    predictions_rescaled : np.ndarray  shape (n_test, 4)
    scaler               : MinMaxScaler (fitted)
    history              : Keras History object
    """
    df = load_climate_data(filepath)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = create_sequences(scaled, seq_length)
    split = int((1 - test_split) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)

    return predictions_rescaled, scaler, history
