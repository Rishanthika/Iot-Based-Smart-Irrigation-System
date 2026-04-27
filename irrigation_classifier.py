"""
irrigation_classifier.py
------------------------
Neural network classifier for irrigation need prediction.

Fuses IoT soil/field features with LSTM-derived weather forecasts,
then classifies irrigation need as Low (0), Medium (1), or High (2).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


SENSOR_FEATURES = [
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
    "Soil_Moisture",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
]

CATEGORICAL_FEATURES = ["Mulching_Used", "Region", "Soil_Type", "Crop_Type"]

FORECAST_FEATURES = [
    "forecast_temp",
    "forecast_humidity",
    "forecast_wind",
    "forecast_pressure",
]


def fuse_data(
    irrigation_df: pd.DataFrame, forecast_array: np.ndarray
) -> pd.DataFrame:
    """
    Merge irrigation sensor data with LSTM forecast features.

    Parameters
    ----------
    irrigation_df   : Raw irrigation CSV as a DataFrame.
    forecast_array  : shape (n, 4) — temp, humidity, wind, pressure.
    """
    min_len = min(len(irrigation_df), len(forecast_array))
    df = irrigation_df.iloc[:min_len].copy()
    df["forecast_temp"] = forecast_array[:min_len, 0]
    df["forecast_humidity"] = forecast_array[:min_len, 1]
    df["forecast_wind"] = forecast_array[:min_len, 2]
    df["forecast_pressure"] = forecast_array[:min_len, 3]
    return df


def prepare_features(df: pd.DataFrame):
    """
    Encode labels and build feature matrix X, target vector y.

    Returns
    -------
    X : pd.DataFrame  (one-hot encoded)
    y : pd.Series     (integer-encoded irrigation need)
    le : LabelEncoder
    """
    le = LabelEncoder()
    df = df.copy()
    df["Irrigation_Need"] = le.fit_transform(df["Irrigation_Need"])

    feature_cols = SENSOR_FEATURES + FORECAST_FEATURES
    X = df[feature_cols].join(df[CATEGORICAL_FEATURES])
    X = pd.get_dummies(X, drop_first=True)
    y = df["Irrigation_Need"]
    return X, y, le


def build_classifier(n_features: int) -> Sequential:
    """Construct the dense classification network."""
    model = Sequential(
        [
            Input(shape=(n_features,)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    epochs: int = 30,
    batch_size: int = 16,
    random_state: int = 42,
):
    """
    Train the irrigation classifier with class-weight balancing.

    Returns
    -------
    model   : trained Keras model
    history : Keras History object
    X_test, y_test : held-out evaluation data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = dict(enumerate(class_weights))

    model = build_classifier(n_features=X_train.shape[1])
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        verbose=1,
    )

    return model, history, X_test, y_test
