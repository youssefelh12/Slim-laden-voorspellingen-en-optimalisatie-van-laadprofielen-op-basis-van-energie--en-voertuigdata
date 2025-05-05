import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
import matplotlib.pyplot as plt
from collections import deque

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
MODEL_PATH    = "models/total_consumption_lstm_final.keras"
SCALER_PATH   = "models/total_consumption_scaler.joblib"
CSV_PATH      = "api_data/hourly_building_consumption.csv"

DATE_COL      = "Hour"
TARGET_COL    = "Total_consumption"

SEQUENCE_LEN   = 24    # matches training SEQUENCE_LEN
FORECAST_HOURS = 24    # horizon to forecast
BLOCK_SIZE     = 8     # log progress every 8 hours
HIST_HOURS     = 48    # hours of past consumption to display

FEATURE_COLS = [
    "Total_consumption",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag1",
    "rolling24",
]

# Only these were scaled during training
SCALED_COLS    = ["Total_consumption", "lag1", "rolling24"]
SCALED_INDICES = [FEATURE_COLS.index(c) for c in SCALED_COLS]


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic encodings, lag, and rolling mean features."""
    df = df_in.copy()
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    # cyclic hour & day-of-week
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # lag & rolling stats
    df["lag1"]      = df[TARGET_COL].shift(1)
    df["rolling24"] = df[TARGET_COL].shift(1).rolling(24, min_periods=1).mean()

    return df.dropna()


def partial_transform(row: list, scaler) -> np.ndarray:
    """Scale only the subset of columns in SCALED_COLS."""
    arr = np.array(row, dtype=float)
    sub = arr[SCALED_INDICES].reshape(1, -1)
    arr[SCALED_INDICES] = scaler.transform(sub)[0]
    return arr


def partial_inverse_transform(row_scaled: np.ndarray, scaler) -> np.ndarray:
    """Inverse scale only the subset of columns in SCALED_COLS."""
    arr = np.array(row_scaled, dtype=float)
    sub = arr[SCALED_INDICES].reshape(1, -1)
    arr[SCALED_INDICES] = scaler.inverse_transform(sub)[0]
    return arr


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load CSV with a datetime index, add features, and drop NaNs."""
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL], index_col=DATE_COL)
    df = df.sort_index()
    df = add_features(df)
    return df


def forecast(model, scaler, data: pd.DataFrame) -> pd.DataFrame:
    """Run recursive forecasting and return a DataFrame of predictions."""
    if len(data) < SEQUENCE_LEN:
        raise ValueError(f"Need ≥{SEQUENCE_LEN} rows after feature engineering, got {len(data)}")

    # Seed history and sliding window
    seed    = data.iloc[-SEQUENCE_LEN:]
    history = deque(seed[TARGET_COL].tolist(), maxlen=SEQUENCE_LEN + FORECAST_HOURS)
    window  = deque(
        [partial_transform(r[FEATURE_COLS].tolist(), scaler) for _, r in seed.iterrows()],
        maxlen=SEQUENCE_LEN
    )
    last_ts = seed.index[-1]
    preds   = []

    for i in range(FORECAST_HOURS):
        next_ts = last_ts + pd.Timedelta(hours=1)

        # compute new features
        hr       = next_ts.hour
        dow      = next_ts.dayof_week if False else next_ts.dayofweek
        hour_sin = np.sin(2 * np.pi * hr / 24)
        hour_cos = np.cos(2 * np.pi * hr / 24)
        dow_sin  = np.sin(2 * np.pi * dow / 7)
        dow_cos  = np.cos(2 * np.pi * dow / 7)

        lag1    = history[-1]
        roll24  = float(np.mean(list(history)[-24:]))

        # build and scale feature vector
        raw    = [0.0, hour_sin, hour_cos, dow_sin, dow_cos, lag1, roll24]
        scaled = partial_transform(raw, scaler)

        # model input
        x_scaled = np.array(window).reshape(1, SEQUENCE_LEN, len(FEATURE_COLS))
        y_scaled = float(model.predict(x_scaled, verbose=0))

        # inverse-transform prediction
        scaled[0] = y_scaled
        y_orig    = partial_inverse_transform(scaled, scaler)[0]

        # update state
        history.append(y_orig)
        window.append(scaled)
        last_ts = next_ts
        preds.append((next_ts, y_orig))

        # log progress
        if (i + 1) % BLOCK_SIZE == 0 or (i + 1) == FORECAST_HOURS:
            print(f"Block complete → {i+1}/{FORECAST_HOURS} hours predicted")

    return pd.DataFrame(preds, columns=[DATE_COL, "Predicted_kWh"]).set_index(DATE_COL)


if __name__ == "__main__":
    # Load artefacts
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = load(SCALER_PATH)

    # Sanity-check feature count
    exp_feats = model.input_shape[-1]
    if exp_feats != len(FEATURE_COLS):
        raise ValueError(
            f"Model expects {exp_feats} features, but FEATURE_COLS has {len(FEATURE_COLS)}"
        )

    # Prepare data & forecast
    df           = load_and_prepare(CSV_PATH)
    forecast_df  = forecast(model, scaler, df)

    # Extract past two days of actual consumption
    past_consumption = df[TARGET_COL].iloc[-HIST_HOURS:]
    two_days_ago     = past_consumption.iloc[0:24].values
    one_day_ago      = past_consumption.iloc[24:48].values
    forecast_vals    = forecast_df['Predicted_kWh'].values

    # Use forecast timestamps for x-axis
    index = forecast_df.index

    # Plot historical and forecasted
    plt.figure(figsize=(12, 4))
    plt.plot(index, two_days_ago, '--o', label='2 days ago')
    plt.plot(index, one_day_ago, '-o', label='1 day ago')
    plt.plot(index, forecast_vals, '-x', label='Forecast Next 24h')
    plt.title('Comparison: Past Two Days vs. Forecast')
    plt.xlabel(DATE_COL)
    plt.ylabel('kWh')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

