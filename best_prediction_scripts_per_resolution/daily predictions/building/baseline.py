"""
baseline_naive.py
-----------------
Seasonal-naïve benchmarks for daily EV-charger load:

• “Yesterday”   – y[t-1]
• “Last-week”   – y[t-7]

Change `freq="H"` if you ever switch to hourly data (lags become 24 h and 168 h).
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ────────────────────────────────────────────────────────────────────────────────
# I/O HELPERS
# ────────────────────────────────────────────────────────────────────────────────
def create_results_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_consumption_data(file_path: str) -> pd.Series:
    """Reads CSV with a ‘Day’ column and returns the Total_consumption series."""
    df = pd.read_csv(file_path)
    df.set_index("Day", inplace=True)
    df.index = pd.to_datetime(df.index)
    return df["Total_consumption"]


# ────────────────────────────────────────────────────────────────────────────────
# METRICS
# ────────────────────────────────────────────────────────────────────────────────
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100


def calculate_smape(y_true, y_pred):
    return (
        100
        * np.mean(
            2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
        )
    )


# ────────────────────────────────────────────────────────────────────────────────
# BASELINE FORECASTS
# ────────────────────────────────────────────────────────────────────────────────
def seasonal_naive_forecasts(y: pd.Series, horizon: int = 1, freq: str = "D"):
    """
    Returns two Series aligned with y.index:
      • yesterday (lag-1)
      • last-week (lag-7)
    If freq == "H", lags switch to 24 h and 168 h automatically.
    """
    if freq == "H":
        lag_1 = 24 * horizon      # previous day (in hours)
        lag_7 = 24 * 7 * horizon  # same hour last week
    else:                         # daily data
        lag_1 = horizon
        lag_7 = 7 * horizon

    naive_yesterday = y.shift(lag_1)
    naive_lastweek = y.shift(lag_7)
    return naive_yesterday, naive_lastweek


def score_forecasts(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)          # ← nieuw
    return mae, rmse, mape, smape, r2


# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────
def main():
    results_dir = "results/naive_baselines"
    create_results_directory(results_dir)

    # 1. Load target series
    y = load_consumption_data("api_data/daily_cleaned_chargers.csv")

    # 2. Train/test split (walk-forward style)
    train_size = int(0.80 * len(y))
    test_idx = y.index[train_size:]
    y_test = y.loc[test_idx]

    # 3. Generate baseline forecasts
    naive_y, naive_w = seasonal_naive_forecasts(y, horizon=1, freq="D")
    naive_y = naive_y.loc[test_idx]
    naive_w = naive_w.loc[test_idx]

    # 4. Score
    yesterday_scores = score_forecasts(y_test, naive_y)
    lastweek_scores = score_forecasts(y_test, naive_w)

    print("=== Seasonal-naïve baselines ===")
    print(f"{'Model':<15} | {'MAE':>8} | {'RMSE':>8} | {'MAPE %':>8} | {'sMAPE %':>8} | {'R²':>6}")
    print("-" * 72)

    def fmt(s): return f"{s:8.3f}"

    print(f"{'Yesterday':<15} | {fmt(yesterday_scores[0])} | {fmt(yesterday_scores[1])} | "
        f"{yesterday_scores[2]:8.2f} | {yesterday_scores[3]:8.2f} | {yesterday_scores[4]:6.3f}")

    print(f"{'Last-week':<15} | {fmt(lastweek_scores[0])} | {fmt(lastweek_scores[1])} | "
        f"{lastweek_scores[2]:8.2f} | {lastweek_scores[3]:8.2f} | {lastweek_scores[4]:6.3f}")

    # 4b. Plot: actual vs. naïve forecasts
    import matplotlib.pyplot as plt

    # ── Plot 1: Actual vs. "Yesterday" ────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test,   label="Actual",     linewidth=2)
    plt.plot(y_test.index, naive_y,  label="Yesterday",  alpha=0.7)

    plt.title("Actual vs. Yesterday-baseline")
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()

    plot_path_y = os.path.join(results_dir, "baseline_yesterday.png")
    plt.savefig(plot_path_y, dpi=150)
    plt.show()  # verwijder bij headless runs
    print(f"Plot opgeslagen ➜ {plot_path_y}")

    # ── Plot 2: Actual vs. "Last-week" ────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test,   label="Actual",     linewidth=2)
    plt.plot(y_test.index, naive_w,  label="Last-week",  alpha=0.7)

    plt.title("Actual vs. Last-week-baseline")
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()

    plot_path_w = os.path.join(results_dir, "baseline_lastweek.png")
    plt.savefig(plot_path_w, dpi=150)
    plt.show()  # verwijder bij headless runs
    print(f"Plot opgeslagen ➜ {plot_path_w}")
        # 5. Save predictions for inspection
    pd.DataFrame(
        {
            "Actual_kWh": y_test,
            "Yesterday_kWh": naive_y,
            "LastWeek_kWh": naive_w,
        }
    ).to_csv(os.path.join(results_dir, "naive_baseline_predictions.csv"))


if __name__ == "__main__":
    main()
