"""
Full multivariate, multi‑step LSTM pipeline — **from raw dataframe to a saved model**
===================================================================================
* Builds engineered time‑based features (hour, day‑of‑week, holidays, cyclical encodings …)
* Normalises all inputs with a single `MinMaxScaler`
* Creates 72‑step windows (3 days) that predict the next 24 hours simultaneously
* Splits into train/validation/test, trains with early‑stopping, prints metrics
* Saves the trained network to `lstm_chargers.keras`

---
⚠️ **Replace the _data‑loading stub_ with your own CSV/SQL/Parquet reader.**  
The script expects a `DatetimeIndex` at hourly resolution and a numeric
column called **“Chargers”** (target).
"""

from __future__ import annotations

import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------
TARGET_COL   = "Chargers"   # column to forecast
LOOK_BACK    = 72           # 3 days of history (72×1 h)
N_FORECAST   = 24           # predict next 24 hours
EPOCHS       = 80
BATCH_SIZE   = 256
PATIENCE     = 8            # early‑stopping patience
MODEL_PATH   = pathlib.Path("./models/chargers/lstm_chargers.keras")
START_DATE   = "2022-09-11"
END_DATE     = "2025-02-19"

# --------------------------------------------------------------------------------------
# 0. LOAD YOUR DATA — replace this stub -------------------------------------------------
# --------------------------------------------------------------------------------------
# Dummy sine‑wave example so the script runs. 
df = pd.read_csv('./hourly_charging_data.csv').set_index("Hour")

# Replace with: df = pd.read_csv(...).set_index("timestamp")
df = pd.DataFrame({
    "Hour": pd.date_range(START_DATE, END_DATE, freq="H"),
    TARGET_COL: 2000 + 200*np.sin(np.arange(0, (pd.Timestamp(END_DATE)-pd.Timestamp(START_DATE)).seconds/3600 + 1)/24)
}).set_index("Hour")

df = df.loc[START_DATE:END_DATE].copy()

# --------------------------------------------------------------------------------------
# 1. FEATURE ENGINEERING ----------------------------------------------------------------
# --------------------------------------------------------------------------------------
be_holidays = set(holidays.country_holidays("BE", years=[2022, 2023, 2024, 2025]).keys())

# basic time signals
df["hour"]        = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"]       = df.index.month

# categorical flags
df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
df["is_festive"]      = df.index.to_series().apply(lambda d: int(d.date() in be_holidays))
df["working_hour"]    = df["hour"].between(8, 18).astype(int)

# seasonal / peak flags
df["is_summer"]       = df["month"].isin([6, 7, 8]).astype(int)
df["is_winter"]       = df["month"].isin([12, 1, 2]).astype(int)
df["is_morning_peak"] = df["hour"].between(7, 9).astype(int)
df["is_evening_peak"] = df["hour"].between(17, 20).astype(int)

# cyclical encodings
df["hour_sin"]        = np.sin(2*np.pi*df["hour"] / 24)
df["hour_cos"]        = np.cos(2*np.pi*df["hour"] / 24)
df["dow_sin"]         = np.sin(2*np.pi*df["day_of_week"] / 7)
df["dow_cos"]         = np.cos(2*np.pi*df["day_of_week"] / 7)

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_festive", "working_hour",
    "is_summer", "is_winter", "is_morning_peak", "is_evening_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos"
]

# --------------------------------------------------------------------------------------
# 2. NORMALISATION ----------------------------------------------------------------------
# --------------------------------------------------------------------------------------
scaler = MinMaxScaler()
df[[TARGET_COL] + FEATURE_COLS] = scaler.fit_transform(df[[TARGET_COL] + FEATURE_COLS])

# --------------------------------------------------------------------------------------
# 3. CREATE SEQUENCES -------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def create_sequences(dataframe: pd.DataFrame,
                     look_back: int,
                     n_forecast: int,
                     target: str,
                     features: list[str]):
    X, y = [], []
    arr = dataframe[[target] + features].values
    for i in range(look_back, len(arr) - n_forecast + 1):
        X.append(arr[i-look_back:i])                # (look_back, n_features)
        y.append(arr[i:i+n_forecast, 0])            # (n_forecast,)
    return np.array(X), np.array(y)

X, y = create_sequences(df, LOOK_BACK, N_FORECAST, TARGET_COL, FEATURE_COLS)
print("X shape:", X.shape, "y shape:", y.shape)

# --------------------------------------------------------------------------------------
# 4. TRAIN / VAL / TEST SPLIT -----------------------------------------------------------
# --------------------------------------------------------------------------------------
train_size = int(len(X) * 0.8)
val_size   = int(len(X) * 0.1)

X_train, y_train = X[:train_size], y[:train_size]
X_val,   y_val   = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test,  y_test  = X[train_size+val_size:], y[train_size+val_size:]

N_FEATURES = X.shape[2]

# --------------------------------------------------------------------------------------
# 5. BUILD LSTM -------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def build_lstm(look_back: int, n_features: int, n_forecast: int):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(n_forecast)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

model = build_lstm(LOOK_BACK, N_FEATURES, N_FORECAST)
model.summary()

# --------------------------------------------------------------------------------------
# 6. TRAIN ------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
callbacks = [tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=2,
)

# --------------------------------------------------------------------------------------
# 7. METRICS ON TEST --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def evaluate(name: str, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

y_pred = model.predict(X_test)
evaluate("TEST", y_test, y_pred)

# --------------------------------------------------------------------------------------
# 8. SAVE MODEL -------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
model.save(MODEL_PATH)
print(f"Saved model → {MODEL_PATH.resolve()}")
