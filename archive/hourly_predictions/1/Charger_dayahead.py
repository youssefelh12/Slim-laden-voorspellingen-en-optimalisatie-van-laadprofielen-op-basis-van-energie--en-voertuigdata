import pandas as pd, numpy as np, tensorflow as tf, holidays, joblib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# ───────────────────────── CONFIG ──────────────────────────
MODEL_PATH  = Path("./models/chargers/lstm_chargers.keras")
SCALER_PATH = Path("./models/chargers/charger_scaler.joblib")
CSV_PATH    = "./3days_charging_data.csv"

DATE_COL    = "Date"
TARGET_COL  = "Chargers"
LOOK_BACK   = 48          # 3-day context
FORECAST_HR = 24          # 24-step output
HIST_HRS    = 72          # plot last 72 h

# Same feature order used during training
FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_festive", "working_hour",
    "is_summer", "is_winter", "is_morning_peak", "is_evening_peak",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_terugkomdag", "cumulative_ev_phev_count"
]

# ─────────────── feature engineering helpers ───────────────
def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    be_holidays = set(holidays.country_holidays(
        "BE", years=[2022, 2023, 2024, 2025]).keys())

    terugkomdagen = {
        datetime(2023, 9, 13), datetime(2023,10,26), datetime(2023,11,14),
        datetime(2023,12,20), datetime(2024, 1,12), datetime(2024, 2, 7),
        datetime(2024, 3,14), datetime(2024, 4,16), datetime(2024, 5,13),
        datetime(2024, 6, 7), datetime(2024, 3,16), datetime(2024,10,22),
        datetime(2024,11,28), datetime(2024,12,18), datetime(2025, 1,10),
        datetime(2025, 2,13), datetime(2025, 3,18), datetime(2025, 4,22),
        datetime(2025, 5,12), datetime(2025, 6, 6)
    }

    cumulative_data = {
        datetime(2024, 6,20): 35, datetime(2024, 6,25): 36, datetime(2024, 9, 5): 38,
        datetime(2024, 9,12): 41, datetime(2024, 9,27): 42, datetime(2024,10,15): 43,
        datetime(2024,10,29): 45, datetime(2024,11, 5): 46, datetime(2024,11,26): 47,
        datetime(2025, 1, 9): 48, datetime(2025, 1,23): 49, datetime(2025, 1,28): 50,
        datetime(2025, 2, 4): 51,
    }

    df = df_in.copy()

    # calendar basics
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month

    # categorical flags
    df["is_weekend"]       = (df["day_of_week"] >= 5).astype(int)
    df["is_festive"]       = df.index.to_series().apply(lambda d: int(d.date() in be_holidays))
    df["working_hour"]     = df["hour"].between(8, 18).astype(int)
    df["is_summer"]        = df["month"].isin([6,7,8]).astype(int)
    df["is_winter"]        = df["month"].isin([12,1,2]).astype(int)
    df["is_morning_peak"]  = df["hour"].between(7, 9).astype(int)
    df["is_evening_peak"]  = df["hour"].between(17,20).astype(int)

    # cyclical encodings
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["day_of_week"]/7)

    # business flags
    df["is_terugkomdag"] = df.index.to_series().dt.date.isin(
        [d.date() for d in terugkomdagen]).astype(int)

    ev_series = (pd.Series(cumulative_data)
                   .reindex(df.index.union(cumulative_data.keys()))
                   .sort_index().ffill().fillna(0))
    df["cumulative_ev_phev_count"] = ev_series.reindex(df.index).astype(int)
    return df


def load_prepared(path):
    df = pd.read_csv(path, parse_dates=[DATE_COL], index_col=DATE_COL).sort_index()
    return add_features(df)

# ────────────────────── reversible scaling ──────────────────────
def inverse_target_direct(vec, scaler):
    """Method 1: direct min/max formula."""
    tmin, tmax = scaler.data_min_[0], scaler.data_max_[0]
    return vec * (tmax - tmin) + tmin

def inverse_target_via_params(vec, scaler):
    """Method 2: using scaler.min_ and scaler.scale_."""
    # transform: X_scaled = X * scale_ + min_
    # invert:     X = (X_scaled - min_) / scale_
    return (vec - scaler.min_[0]) / scaler.scale_[0]

# ────────────────── forecasting & debug ──────────────────
def forecast_next_24h(model, scaler, df):
    # 1) build & scale the look-back window:
    window = df[[TARGET_COL] + FEATURE_COLS].tail(LOOK_BACK).values
    X = scaler.transform(window).reshape(1, LOOK_BACK, -1)
    # 2) get normalized 24-step output:
    preds_norm = model.predict(X, verbose=0)[0]

    # 3a) inverse-scale by direct formula:
    preds_direct = inverse_target_direct(preds_norm, scaler)
    # 3b) inverse-scale by scaler params:
    preds_params = inverse_target_via_params(preds_norm, scaler)

    # 4) quick sanity print:
    print("\npreds_norm   :", np.round(preds_norm[:5], 4))
    print("direct inv   :", np.round(preds_direct[:5], 4))
    print("params inv   :", np.round(preds_params[:5], 4))

    # 5) build timestamp index
    idx = pd.date_range(df.index[-1] + pd.Timedelta(hours=1),
                        periods=FORECAST_HR, freq="h")
    # 6) return DataFrame from *one* of the methods (they match)
    return pd.DataFrame({"Predicted_kWh": preds_direct}, index=idx)

if __name__ == "__main__":
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # debug: check column ordering matches scaler
    print("Scaler expects:", list(scaler.feature_names_in_))
    print("We supply   :", [TARGET_COL] + FEATURE_COLS)

    df_hist    = load_prepared(CSV_PATH)
    forecast_df= forecast_next_24h(model, scaler, df_hist)

    # print results
    print("\nNext-day forecast (kWh):")
    print(forecast_df.to_string())

    past = df_hist[TARGET_COL].iloc[-HIST_HRS:]
    print("\nLast 72 h actual consumption (kWh):")
    print(past.to_string())

    # plot
    plt.figure(figsize=(12,4))
    plt.plot(past.index, past.values, label="Past 72 h")
    plt.plot(forecast_df.index, forecast_df["Predicted_kWh"], "-o",
             label="Forecast next 24 h")
    plt.title("Charger load – history vs forecast")
    plt.xlabel("Hour"); plt.ylabel("kWh"); plt.legend(); plt.tight_layout(); plt.show()
