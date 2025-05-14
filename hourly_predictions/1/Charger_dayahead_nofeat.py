import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────────── CONFIG ──────────────────────────
MODEL_PATH  = Path("./models/chargers/lstm_chargers_nofeat.keras")
SCALER_PATH = Path("./models/chargers/charger_scaler_nofeat.joblib")
CSV_PATH     = "./3days_charging_data.csv"

DATE_COL     = "Date"
TARGET_COL   = "Chargers"
LOOK_BACK    = 48    # 48-hour history window
FORECAST_HR  = 24    # 24-hour forecast
HIST_HRS     = 72    # plot last 72 hours

# ────────────────────── DATA LOADING ───────────────────────
def load_data(path):
    df = pd.read_csv(path, parse_dates=[DATE_COL], index_col=DATE_COL)
    df = df.sort_index()
    return df

# ────────────────────── SCALING HELPERS ────────────────────
def inverse_scale(vec, scaler):
    """Invert MinMaxScaler on the target column."""
    tmin, tmax = scaler.data_min_[0], scaler.data_max_[0]
    return vec * (tmax - tmin) + tmin

# ────────────────── FORECASTING FUNCTION ──────────────────
def forecast_next_24h(model, scaler, df):
    # 1) take last LOOK_BACK hours of the target only
    window = df[[TARGET_COL]].tail(LOOK_BACK).values
    # 2) scale and reshape to (1, LOOK_BACK, 1)
    X = scaler.transform(window).reshape(1, LOOK_BACK, 1)
    # 3) predict normalized values
    preds_norm = model.predict(X, verbose=0)[0]
    # 4) inverse-scale back to original units
    preds = inverse_scale(preds_norm, scaler)
    # 5) build timestamp index for the forecast period
    idx = pd.date_range(df.index[-1] + pd.Timedelta(hours=1),
                        periods=FORECAST_HR, freq="h")
    # 6) return a DataFrame of predictions
    return pd.DataFrame({ "Predicted_kWh": preds }, index=idx)

if __name__ == "__main__":
    # load model and scaler
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # debug: ensure scaler was trained on target only
    print("Scaler features:", scaler.feature_names_in_)

    # load recent data
    df_hist = load_data(CSV_PATH)

    # generate forecast
    forecast_df = forecast_next_24h(model, scaler, df_hist)

    # display results
    print("\nNext 24-hour forecast (kWh):")
    print(forecast_df.to_string())

    past = df_hist[TARGET_COL].iloc[-HIST_HRS:]
    print("\nLast 72 hours actual (kWh):")
    print(past.to_string())

    # plot history vs forecast
    plt.figure(figsize=(12, 4))
    plt.plot(past.index, past.values, label="Past 72 h")
    plt.plot(forecast_df.index, forecast_df["Predicted_kWh"], "-o", label="Forecast next 24 h")
    plt.title("Charger load – history vs forecast")
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()
    plt.show()
