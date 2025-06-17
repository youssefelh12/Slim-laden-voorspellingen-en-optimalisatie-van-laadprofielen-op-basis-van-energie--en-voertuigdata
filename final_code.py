#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Workplace‑charging load forecasting
----------------------------------
• Model: XGBoost (tree‑based, handles NaNs natively)
• Features: calendar flags, terugkomdagen, cyclic encodings, people‑at‑office,
            168‑h lag, weekday×hour averages, weekday kWh/EV proxy,
            rolling 24 h & 168 h statistical moments
• Bid‑market rule: **training data only includes observations up to 12:00 (noon)
  on the day *before* the forecast date** so that nothing after the bidding gate
  leaks into the model.
• Two workflows:
  1. 30‑day rolling back‑test to gauge recent accuracy
  2. One‑day forecast for any `FORECAST_DATE`
     – If that date is *in* the data ➜ train on everything *≤ yesterday‑noon*
     – If that date is in the *future* ➜ train on all history *≤ yesterday‑noon*
'''

# ───────────────────────── imports ──────────────────────────
import numpy as np
import pandas as pd
import holidays
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────── constants ────────────────────────
CUTOFF_HOURS   = 12           # how many hours before 00:00 of forecast day
TOTAL_HEADCOUNT = 105         # full‑time employees
SESSION_KWH     = 9.5         # typical energy per charging session
RANDOM_STATE    = 42

# ────────────────────── user parameters ─────────────────────
FORECAST_DATE = "2025-04-21"  # ← change to any YYYY‑MM‑DD

# ───────────────────────── config ───────────────────────────
CHARGING_FILE   = "./charging_forecasts/Charging_data_hourly.csv"
CALENDAR_FILE   = "./hourly_predictions/layout1_full_calendar_2023-2025.csv"

# ═════════════ 1. LOAD HISTORICAL DATA ═════════════
df = (pd.read_csv(CHARGING_FILE)
        .rename(columns={"Date": "ds", "Chargers": "y"}))
df["ds"]       = pd.to_datetime(df["ds"])
df             = df.sort_values("ds").reset_index(drop=True)
df["cal_date"] = df["ds"].dt.date

# workforce vacation / home‑office counts
cal = (pd.read_csv(CALENDAR_FILE, parse_dates=["Datum"])
         .rename(columns={
             "Totaal_Vakantiedagen":  "vacation_cnt",
             "Totaal_Thuiswerkdagen": "homework_cnt"
         }))
cal["cal_date"] = cal["Datum"].dt.date
df = df.merge(cal[["cal_date", "vacation_cnt", "homework_cnt"]],
              on="cal_date", how="left")
df[["vacation_cnt", "homework_cnt"]] = (
    df[["vacation_cnt", "homework_cnt"]].fillna(0).astype(int))

# ═════════════ 2. CALENDAR / CYCLIC FLAGS ══════════
terugkomdagen = pd.to_datetime([
    "2023-09-13","2023-10-26","2023-11-14","2023-12-20",
    "2024-01-12","2024-02-07","2024-03-14","2024-04-16",
    "2024-05-13","2024-06-07","2024-10-22","2024-11-28",
    "2024-12-18","2025-01-10","2025-02-13","2025-03-18",
    "2025-04-22","2025-05-12","2025-06-06"
]).normalize()

years_needed = np.unique(df["ds"].dt.year.tolist() +
                         [pd.Timestamp(FORECAST_DATE).year])
be_holidays = holidays.Belgium(years=years_needed)

def add_time_flags(frame: pd.DataFrame) -> None:
    frame["weekday"]        = frame["ds"].dt.weekday
    frame["hour"]           = frame["ds"].dt.hour
    frame["is_weekend"]     = (frame["weekday"] >= 5).astype(int)
    frame["is_work_hour"]   = frame["hour"].between(8, 18).astype(int)
    frame["is_holiday"]     = frame["ds"].dt.normalize().isin(be_holidays).astype(int)
    frame["is_terugkomdag"] = frame["ds"].dt.normalize().isin(terugkomdagen).astype(int)
    frame["hour_sin"]       = np.sin(2*np.pi*frame["hour"]/24)
    frame["hour_cos"]       = np.cos(2*np.pi*frame["hour"]/24)

add_time_flags(df)

# ═════════════ 3. WORK‑AT‑OFFICE COUNTS ════════════
def compute_work_at_office(frame: pd.DataFrame) -> None:
    is_workday  = (frame["weekday"] < 5) & (frame["is_holiday"] == 0)
    is_workhour = frame["is_work_hour"] == 1
    frame["work_at_office"] = np.where(
        is_workday & is_workhour,
        (TOTAL_HEADCOUNT -
         frame.get("vacation_cnt", 0) -
         frame.get("homework_cnt", 0)).clip(lower=0),
        0
    )

compute_work_at_office(df)

# ═════════════ 4. ROLLING MOMENT FEATURES ══════════
for win in (24, 168):                # 1‑day and 1‑week windows
    r = df["y"].rolling(win, min_periods=1)
    df[f"m1_{win}h"] = r.mean()
    df[f"m2_{win}h"] = r.var()
    df[f"m3_{win}h"] = r.skew()
    df[f"m4_{win}h"] = r.kurt()

# ═════════════ 5. FEATURE LIST ═════════════════════
MOMENT_COLS = [f"m{i}_{w}h" for i in range(1, 5) for w in (24, 168)]
FEATURES = [
    "is_work_hour", "is_weekend", "is_holiday", "is_terugkomdag",
    "work_at_office", "lag_168", "avg_hourly_by_weekday",
    "avg_cars_weekday", "hour_sin", "hour_cos"
] + MOMENT_COLS

# ═════════════ 6. 30‑DAY ROLLING BACK‑TEST ═════════
start_eval = df["ds"].max() - pd.Timedelta(days=30)
days_iter  = pd.date_range(start_eval.normalize(),
                           df["ds"].max().normalize(), freq="D")

bt_rows = []
for target_day in days_iter:
    # training cutoff at yesterday‑noon
    cutoff = pd.Timestamp(target_day) - pd.Timedelta(hours=CUTOFF_HOURS)
    hist = df[df["ds"] <= cutoff].copy()

    # weekday‑hour mean map
    avg_map = (hist.groupby(["weekday","hour"])["y"].mean()
                    .rename("avg_hourly_by_weekday").reset_index())

    hist["lag_168"] = hist["y"].shift(168).fillna(hist["y"].mean())
    hist = hist.merge(avg_map, on=["weekday","hour"], how="left")
    hist["avg_cars_weekday"] = (
        hist.groupby("weekday")["y"].transform("mean") / SESSION_KWH
    )
    train = hist.dropna(subset=FEATURES+["y"])

    model = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        objective="reg:squarederror", random_state=RANDOM_STATE, verbosity=0)
    model.fit(train[FEATURES], train["y"])

    # the 24‑h period we want to predict (actual values known during back‑test)
    day_df = df[(df["ds"] >= target_day) & (df["ds"] < target_day + pd.Timedelta(days=1))].copy()
    day_df = day_df.merge(avg_map, on=["weekday","hour"], how="left")

    lag = hist[["ds","y"]].copy(); lag["ds"] += pd.Timedelta(days=7)
    day_df = day_df.merge(lag.rename(columns={"y":"lag_168"}), on="ds", how="left")
    day_df["lag_168"].fillna(day_df["avg_hourly_by_weekday"], inplace=True)

    cars_map = (hist.groupby("weekday")["y"].mean()/SESSION_KWH
               ).rename("avg_cars_weekday").reset_index()
    day_df = day_df.merge(cars_map, on="weekday", how="left")

    day_df["yhat"] = model.predict(day_df[FEATURES])
    bt_rows.append(day_df[["ds","y","yhat"]])

# back‑test metrics
backtest = pd.concat(bt_rows).reset_index(drop=True)
mae   = mean_absolute_error(backtest["y"], backtest["yhat"])
rmse  = np.sqrt(mean_squared_error(backtest["y"], backtest["yhat"]))
r2    = r2_score(backtest["y"], backtest["yhat"])
print("\n── Back‑test (last 30 days)")
print(f"MAE  {mae:.3f}   RMSE {rmse:.3f}   R² {r2:.3f}")

plt.figure(figsize=(6,5))
plt.scatter(backtest["y"], backtest["yhat"], alpha=0.3)
lims = [min(backtest[["y","yhat"]].min()), max(backtest[["y","yhat"]].max())]
plt.plot(lims, lims, 'k--')
plt.xlabel("Actual kW"); plt.ylabel("Predicted kW"); plt.title("Back‑test")
plt.tight_layout(); plt.show()

# ═════════════ 7. FINAL MODEL (NO LEAKAGE) ════════

def training_slice(data: pd.DataFrame, forecast_day: str) -> pd.DataFrame:
    """Return subset ≤ yesterday‑noon relative to forecast day."""
    cutoff = pd.Timestamp(forecast_day) - pd.Timedelta(hours=CUTOFF_HOURS)
    return data[data["ds"] <= cutoff]

train_subset = training_slice(df, FORECAST_DATE)

avg_map_sub = (train_subset.groupby(["weekday","hour"])["y"].mean()
               .rename("avg_hourly_by_weekday").reset_index())

train_subset = train_subset.copy()
train_subset["lag_168"] = train_subset["y"].shift(168).fillna(train_subset["y"].mean())
train_subset = train_subset.merge(avg_map_sub, on=["weekday","hour"], how="left")
train_subset["avg_cars_weekday"] = (
    train_subset.groupby("weekday")["y"].transform("mean") / SESSION_KWH
)
final_train = train_subset.dropna(subset=FEATURES+["y"])

final_xgb = xgb.XGBRegressor(
    n_estimators=800, learning_rate=0.05, max_depth=6,
    objective="reg:squarederror", random_state=RANDOM_STATE, verbosity=0)
final_xgb.fit(final_train[FEATURES], final_train["y"])

# ═════════════ 8. BUILD FORECAST FRAME ════════════

def build_forecast_frame(date_str: str) -> pd.DataFrame:
    start = pd.Timestamp(date_str)
    hrs   = pd.date_range(start, start+pd.Timedelta(hours=23), freq="H")
    fdf   = pd.DataFrame({"ds": hrs})
    fdf["cal_date"] = fdf["ds"].dt.date
    fdf = fdf.merge(cal[["cal_date","vacation_cnt","homework_cnt"]],
                    on="cal_date", how="left")
    fdf[["vacation_cnt","homework_cnt"]] = (
        fdf[["vacation_cnt","homework_cnt"]].fillna(0).astype(int))

    add_time_flags(fdf); compute_work_at_office(fdf)
    fdf = fdf.merge(avg_map_sub, on=["weekday","hour"], how="left")

    # lag‑168 from history subset
    hist_series = train_subset.groupby("ds")["y"].mean()      # unique index
    fdf["lag_168"] = hist_series.reindex(fdf["ds"]-pd.Timedelta(hours=168)).values
    fdf["lag_168"].fillna(hist_series.mean(), inplace=True)

    # weekday cars
    fdf = fdf.merge(
        (train_subset.groupby("weekday")["y"].mean()/SESSION_KWH
         ).rename("avg_cars_weekday").reset_index(),
        on="weekday", how="left"
    )

    # rolling moments computed from subset history
    for idx, ts in enumerate(fdf["ds"]):
        hist = hist_series[hist_series.index < ts]
        for win in (24, 168):
            tail = hist.tail(win)
            fdf.loc[idx, f"m1_{win}h"] = tail.mean()
            fdf.loc[idx, f"m2_{win}h"] = tail.var()
            fdf.loc[idx, f"m3_{win}h"] = tail.skew()
            fdf.loc[idx, f"m4_{win}h"] = tail.kurt()
    return fdf

forecast = build_forecast_frame(FORECAST_DATE)
forecast["yhat"] = final_xgb.predict(forecast[FEATURES])

# attach actuals if available
mask = df["cal_date"] == pd.to_datetime(FORECAST_DATE).date()
if mask.any():
    forecast = forecast.merge(df.loc[mask, ["ds","y"]], on="ds", how="left")

# ═════════════ 9. PLOT FORECAST ═══════════════════
plt.figure(figsize=(12,4))
plt.plot(forecast["ds"], forecast["yhat"], label="Predicted", color="tab:orange")
if "y" in forecast.columns:
    plt.plot(forecast["ds"], forecast["y"], label="Actual", color="tab:blue")
    plt.title(f"{FORECAST_DATE} – Actual vs Predicted ")
else:
    plt.title(f"{FORECAST_DATE} – Forecast (future date)")

plt.ylim(0, 100)

plt.xlabel("Hour"); plt.ylabel("kW"); plt.legend(); plt.tight_layout(); plt.show()
