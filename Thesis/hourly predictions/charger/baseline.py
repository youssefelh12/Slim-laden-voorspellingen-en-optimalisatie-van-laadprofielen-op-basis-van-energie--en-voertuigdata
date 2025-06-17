import numpy as np
import pandas as pd
import holidays
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────── config ──────────────────────────────
CHARGING_FILE    = "./charging_forecasts/Charging_data_hourly.csv"
CALENDAR_FILE    = "./layout1_full_calendar_2023-2025.csv"

# ───────────────────── 1  LOAD CHARGING DATA ─────────────────────
df = (pd.read_csv(CHARGING_FILE)
        .rename(columns={"Date":"ds","Chargers":"y"}))
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)
df["cal_date"] = df["ds"].dt.date

# ───────────────────── 2  LOAD WORKFORCE CALENDAR ────────────────
cal = (pd.read_csv(CALENDAR_FILE, parse_dates=["Datum"])
         .rename(columns={"Totaal_Vakantiedagen":"vacation_cnt",
                          "Totaal_Thuiswerkdagen":"homework_cnt"}))
cal["cal_date"] = cal["Datum"].dt.date
df = df.merge(cal[["cal_date","vacation_cnt","homework_cnt"]],
              on="cal_date", how="left")
df[["vacation_cnt","homework_cnt"]] = (
    df[["vacation_cnt","homework_cnt"]]
      .fillna(0).astype(int)
)

# ───────────────────── 3  TIME / HOLIDAY FLAGS ───────────────────
be_holidays = holidays.Belgium(years=df["ds"].dt.year.unique())
df["weekday"]      = df["ds"].dt.weekday
df["hour"]         = df["ds"].dt.hour
df["is_weekend"]   = (df["weekday"] >= 5).astype(int)
df["is_holiday"]   = df["ds"].dt.normalize().isin(be_holidays).astype(int)
# … eventueel je andere flags hier …

# ─────── 4  HISTORISCH WEEKGEMIDDEL ───────────────────────────────
# Baseline: gemiddelde per (weekday, hour) over ALLE data vóór de valutatieperiode
# (we kunnen dat ook per dag laten oplopen, maar voor baseline is één statisch patroon voldoende)
avg_map = (
    df.groupby(["weekday","hour"])["y"]
      .mean()
      .rename("avg_hourly_by_weekday")
      .reset_index()
)

# ───────────────────── 5  ROLLING 30-DAY BACK-TEST ───────────────
start_eval = df["ds"].max() - pd.Timedelta(days=30)
days_iter  = pd.date_range(start_eval.normalize(),
                           df["ds"].max().normalize(),
                           freq="D")

all_preds = []
for day in days_iter:
    day_start = pd.Timestamp(day)
    # testset voor deze dag
    day_df = df[(df["ds"] >= day_start) &
                (df["ds"] < day_start + pd.Timedelta(days=1))].copy()

    # merge de baseline (weekpattern)
    day_df = day_df.merge(avg_map, on=["weekday","hour"], how="left")

    # Baseline-voorspelling: gewoon het historische weekgemiddelde
    day_df["yhat"] = day_df["avg_hourly_by_weekday"]

    all_preds.append(day_df[["ds","y","yhat","weekday","hour"]])

test_bt = (pd.concat(all_preds)
             .sort_values("ds")
             .reset_index(drop=True))

# ───────────────────── 6  METRICS ────────────────────────────────
test_bt["abs_err"] = np.abs(test_bt["y"] - test_bt["yhat"])
test_bt["pct_err"] = 100 * test_bt["abs_err"] / (test_bt["y"] + 1e-10)

mae   = mean_absolute_error(test_bt["y"], test_bt["yhat"])
rmse  = np.sqrt(mean_squared_error(test_bt["y"], test_bt["yhat"]))
r2    = r2_score(test_bt["y"], test_bt["yhat"])
mape  = test_bt["pct_err"].mean()
smape = (
    2 * np.abs(test_bt["y"] - test_bt["yhat"]) /
    (np.abs(test_bt["y"]) + np.abs(test_bt["yhat"]) + 1e-10)
).mean() * 100

print("\n--- Baseline Back-test Metrics (Last 30 Days) ---")
print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.2f}")
print(f"MAPE  : {mape:.2f}%")
print(f"sMAPE : {smape:.2f}%")

# Actual vs Predicted
plt.figure(figsize=(12,5))
plt.plot(test_bt["ds"], test_bt["y"],    label='Actual',    alpha=0.6)
plt.plot(test_bt["ds"], test_bt["yhat"], label='Forecast',  alpha=0.8)
plt.title("Back-test: Actual vs Forecast (30-Day)")
plt.xlabel("Date"); plt.ylabel("Chargers")
plt.legend(); plt.tight_layout(); plt.show()