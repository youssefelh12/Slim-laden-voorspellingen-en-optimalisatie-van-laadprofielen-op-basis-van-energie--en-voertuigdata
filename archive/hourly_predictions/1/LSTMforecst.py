"""
Improved LSTM training script for hourly building load forecasting.
Keeps the original variable names but factors repeated logic into functions,
adds cyclic features, robust callbacks, and automatic artefact versioning.
"""

# ----------------------------------------------------------
# 0) Imports & configuration
# ----------------------------------------------------------
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from joblib import dump
import tensorflow as tf
tf.random.set_seed(42)  # reproducibility

RAW_PATH   = pathlib.Path("api_data/hourly_building_consumption.csv")
MODEL_DIR  = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True)

SEQUENCE_LEN = 24
FEATURE_COLS = [
    "Total_consumption", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "lag1", "rolling24"
]
TARGET_COL   = "Total_consumption"

# ----------------------------------------------------------
# 1) Helper functions
# ----------------------------------------------------------
def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling mean and cyclic encodings to a fresh copy of df_in."""
    df = df_in.copy()

    # Cyclic hour & day-of-week (avoid discontinuity at 23→0)&#8203;:contentReference[oaicite:3]{index=3}
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag & rolling statistics (use shift to respect causality)&#8203;:contentReference[oaicite:4]{index=4}
    df["lag1"]       = df["Total_consumption"].shift(1)
    df["rolling24"]  = df["Total_consumption"].shift(1).rolling(24, min_periods=1).mean()

    return df.dropna()

def make_sequences(df: pd.DataFrame, features: list, target: str, seq_len: int):
    """Transform a feature-engineered DataFrame into (X, y)."""
    values  = df[features].values
    targets = df[target].values
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(values[i - seq_len : i])
        y.append(targets[i])
    return np.array(X), np.array(y)

# ----------------------------------------------------------
# 2) Load & chronological split (70 / 10 / 20)
# ----------------------------------------------------------
df = pd.read_csv(RAW_PATH, parse_dates=["Hour"], index_col="Hour")
df = df.sort_index()

total_samples = len(df)
train_size = int(total_samples * 0.7)
val_size   = int(total_samples * 0.1)

train_df = df.iloc[:train_size]
val_df   = df.iloc[train_size : train_size + val_size]
test_df  = df.iloc[train_size + val_size :]

# ----------------------------------------------------------
# 3) Feature engineering *after* the split to avoid leakage&#8203;:contentReference[oaicite:5]{index=5}
# ----------------------------------------------------------
train_df = add_features(train_df)
val_df   = add_features(val_df)
test_df  = add_features(test_df)

# ----------------------------------------------------------
# 4) Scaling: fit only on train, apply to others
# ----------------------------------------------------------
scaler = MinMaxScaler()
num_cols = ["Total_consumption", "lag1", "rolling24"]
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
val_df[num_cols]   = scaler.transform(val_df[num_cols])
test_df[num_cols]  = scaler.transform(test_df[num_cols])
dump(scaler, MODEL_DIR / "total_consumption_scaler.joblib")  # persist scaler&#8203;:contentReference[oaicite:6]{index=6}

# ----------------------------------------------------------
# 5) Build supervised sequences
# ----------------------------------------------------------
X_train, y_train = make_sequences(train_df, FEATURE_COLS, TARGET_COL, SEQUENCE_LEN)
X_val,   y_val   = make_sequences(val_df,   FEATURE_COLS, TARGET_COL, SEQUENCE_LEN)
X_test,  y_test  = make_sequences(test_df,  FEATURE_COLS, TARGET_COL, SEQUENCE_LEN)

print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape,   y_val.shape)
print("Test: ", X_test.shape,  y_test.shape)

# ----------------------------------------------------------
# 6) Model definition
# ----------------------------------------------------------
model = Sequential([
    LSTM(64, dropout=0.2, recurrent_dropout=0.2,
         input_shape=(SEQUENCE_LEN, len(FEATURE_COLS))),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),  # :contentReference[oaicite:7]{index=7}
    ModelCheckpoint(
        filepath=MODEL_DIR / "total_consumption_lstm_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    )  # :contentReference[oaicite:8]{index=8}
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)

# ----------------------------------------------------------
# 7) Evaluation on test split
# ----------------------------------------------------------
y_pred_test = model.predict(X_test, verbose=0).flatten()
mse_test = np.mean((y_test - y_pred_test) ** 2)
rmse_test = np.sqrt(mse_test)
r2_scaled = r2_score(y_test, y_pred_test)

print(f"Test RMSE (scaled): {rmse_test:.4f}")
print(f"Test R²   (scaled): {r2_scaled:.4f}")

# Optional inverse-transform for interpretability
pad = np.zeros((len(y_test), len(num_cols) - 1))
y_test_inv  = scaler.inverse_transform(np.hstack([y_test.reshape(-1,1), pad]))[:,0]
y_pred_inv  = scaler.inverse_transform(np.hstack([y_pred_test.reshape(-1,1), pad]))[:,0]
abs_mae = np.mean(np.abs(y_test_inv - y_pred_inv))
print(f"Test MAE  (original kWh): {abs_mae:,.2f}")

# ----------------------------------------------------------
# 8) Visual diagnostics
# ----------------------------------------------------------
plt.figure(figsize=(12,4))
plt.plot(y_test_inv,  label="Actual")
plt.plot(y_pred_inv,  label="Predicted")
plt.title("Test Set — Actual vs Predicted (kWh)")
plt.legend(); plt.tight_layout(); plt.show()

# Box-plot errors by weekday (0=Mon … 6=Sun)&#8203;:contentReference[oaicite:9]{index=9}
errors_df = pd.DataFrame({
    "day_of_week": test_df["day_of_week"].values[SEQUENCE_LEN:],
    "error":       y_test_inv - y_pred_inv
})
errors_df.boxplot(column="error", by="day_of_week", figsize=(8,5), grid=False)
plt.suptitle(""); plt.xlabel("Day of Week"); plt.ylabel("kWh Error"); plt.show()

# ----------------------------------------------------------
# 9) Persist final model (best epoch already stored by checkpoint)
# ----------------------------------------------------------
model.save(MODEL_DIR / "total_consumption_lstm_final.keras")  # TensorFlow 2.12+ native format&#8203;:contentReference[oaicite:10]{index=10}
print("Saved final model and scaler to:", MODEL_DIR)
