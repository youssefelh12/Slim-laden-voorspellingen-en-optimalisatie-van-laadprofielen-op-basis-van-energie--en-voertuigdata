# ───────────────────────────── imports ─────────────────────────────
import numpy as np
import pandas as pd
import holidays
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────── config ──────────────────────────────
CHARGING_FILE = "./charging_forecasts/Charging_data_hourly.csv"      # hourly EV kWh
CALENDAR_FILE = "./layout1_full_calendar_2023-2025.csv"              # Layout-1 daily totals
TOTAL_HEADCOUNT = 105                                                # max office staff
SESSION_KWH = 9.5                                                    # avg session energy
RANDOM_SEED = 42

# ───────────────────── 1  LOAD CHARGING DATA ─────────────────────
df = pd.read_csv(CHARGING_FILE).rename(columns={"Date": "ds", "Chargers": "y"})
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)
df["cal_date"] = df["ds"].dt.date

# ───────────────────── 2  LOAD WORKFORCE CALENDAR ────────────────
cal = (pd.read_csv(CALENDAR_FILE, parse_dates=["Datum"])
         .rename(columns={
             "Totaal_Vakantiedagen":  "vacation_cnt",
             "Totaal_Thuiswerkdagen": "homework_cnt"
         }))
cal["cal_date"] = cal["Datum"].dt.date

df = df.merge(
    cal[["cal_date", "vacation_cnt", "homework_cnt"]],
    on="cal_date",
    how="left"
)
df[["vacation_cnt", "homework_cnt"]] = (
    df[["vacation_cnt", "homework_cnt"]].fillna(0).astype(int)
)

# ───────────────────── 3  TIME / HOLIDAY FLAGS ───────────────────
tkd = pd.to_datetime([
    "2023-09-13","2023-10-26","2023-11-14","2023-12-20",
    "2024-01-12","2024-02-07","2024-03-14","2024-04-16",
    "2024-05-13","2024-06-07","2024-10-22","2024-11-28",
    "2024-12-18","2025-01-10","2025-02-13","2025-03-18",
    "2025-04-22","2025-05-12","2025-06-06"
]).normalize()

be_holidays = holidays.Belgium(years=df["ds"].dt.year.unique())

df["weekday"]       = df["ds"].dt.weekday
df["hour"]          = df["ds"].dt.hour
df["is_weekend"]    = (df["weekday"] >= 5).astype(int)
df["is_work_hour"]  = df["hour"].between(8, 18).astype(int)
df["is_holiday"]    = df["ds"].dt.normalize().isin(be_holidays).astype(int)
df["is_terugkomdag"]= df["ds"].dt.normalize().isin(tkd).astype(int)

# ───────────────────── 4  work_at_office FEATURE ─────────────────
is_workday  = (df["weekday"] < 5) & (df["is_holiday"] == 0)
is_workhour = df["is_work_hour"] == 1

df["work_at_office"] = np.where(
    is_workday & is_workhour,
    (TOTAL_HEADCOUNT - df["vacation_cnt"] - df["homework_cnt"]).clip(lower=0),
    0
)

# ───────────────────── 5  LAG FEATURE (168 h) ────────────────────
df["lag_168"] = df["y"].shift(168)               # 7 × 24 h
df["lag_168"].fillna(df["y"].mean(), inplace=True)

# ───────────── 6  80 / 20 TIME-SERIES SPLIT ──────────────
tscv = TimeSeriesSplit(
    n_splits=5,
    test_size=int(0.20 * len(df)),
    gap=0
)
train_idx, val_idx = next(tscv.split(df))
train = df.iloc[train_idx].copy()
val   = df.iloc[val_idx].copy()

# ───────────── 7  TRAIN-DEPENDENT AVERAGES (NO LEAKAGE) ─────────
avg_hourly_map = (
    train.groupby(["weekday", "hour"])["y"]
         .mean()
         .rename("avg_hourly_by_weekday")
         .reset_index()
)
avg_cars_map = (
    train.groupby("weekday")["y"]
         .mean()
         .div(SESSION_KWH)
         .rename("avg_cars_weekday")
         .reset_index()
)

# merge into TRAIN
train = train.merge(avg_hourly_map, on=["weekday", "hour"], how="left")
train = train.merge(avg_cars_map,   on="weekday",          how="left")
train["lag_168"].fillna(train["avg_hourly_by_weekday"], inplace=True)

# merge into VAL  (must also use maps learned on TRAIN!)
val = val.merge(avg_hourly_map, on=["weekday", "hour"], how="left")
val = val.merge(avg_cars_map,   on="weekday",          how="left")
val["lag_168"].fillna(val["avg_hourly_by_weekday"], inplace=True)

FEATURES = [
    "is_work_hour","is_weekend","is_holiday","is_terugkomdag",
    "work_at_office",
    "lag_168","avg_hourly_by_weekday","avg_cars_weekday"
]

# ───────────── 8  TRAIN & PREDICT ──────────────────────────────
model = lgb.LGBMRegressor(
    objective="regression",
    alpha=0.55,
    n_estimators=800,
    learning_rate=0.05,
    random_state=42,
    force_col_wise=True          # <-- helps when RAM is tight
)
model.fit(train[FEATURES], train["y"])
val["yhat"] = model.predict(val[FEATURES])

# ───────────── 9  METRICS (unchanged) ─────────────────────────
mae  = mean_absolute_error(val["y"], val["yhat"])
rmse = np.sqrt(mean_squared_error(val["y"], val["yhat"]))
r2   = r2_score(val["y"], val["yhat"])

weekday_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
r2_by_wd = {
    weekday_map[d]: r2_score(val[val["weekday"]==d]["y"],
                             val[val["weekday"]==d]["yhat"])
    for d in val["weekday"].unique()
}

daily = (
    val.groupby(val["ds"].dt.date)
       .agg(y_sum=("y","sum"), yhat_sum=("yhat","sum"))
)
r2_daily = r2_score(daily["y_sum"], daily["yhat_sum"])

print("\n────── Validation Metrics (80 / 20) ──────")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")
print("R² by weekday:")
for wd, v in r2_by_wd.items():
    print(f"  {wd}: {v:.3f}")
print(f"R² (daily sums): {r2_daily:.3f}")


# ─────────────────── 10  OPTIONAL PLOTS ──────────────────────────
plt.figure(figsize=(12,5))
plt.plot(val["ds"], val["y"],    label="Actual",    alpha=0.6)
plt.plot(val["ds"], val["yhat"], label="Predicted", alpha=0.8)
plt.title("80 / 20 Hold-out: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Chargers")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.barh(FEATURES, model.feature_importances_)
plt.title("LightGBM Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
