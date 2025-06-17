#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workplace-charging load forecasting
‒ Baseline: XGBoost
‒ Additional model: Feed-Forward Back-Propagation Neural Network (FFBPNN)
‒ Dynamic site-limit feature (site_limit_kw + helpers)
"""

# ───────────────────────── imports ──────────────────────────
import numpy as np
import pandas as pd
import holidays
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)

# ───────────────────────── config ───────────────────────────
CHARGING_FILE   = "./charging_forecasts/Charging_data_hourly.csv"
CALENDAR_FILE   = "./hourly_predictions/layout1_full_calendar_2023-2025.csv"
TOTAL_HEADCOUNT = 105
SESSION_KWH     = 9.5
RANDOM_STATE    = 42

# ───────────── 0.  SITE-LIMIT TABLE ──────────────
SITE_LIMITS = [
    ("2022-06-01",  40),
    ("2022-11-01",  55),
    ("2023-05-01",  70),
    ("2023-07-01",  80),
    ("2023-10-01", 100),
    ("2023-12-01", 120),
    ("2024-02-01", 130),
    ("2025-01-01", 150),
    ("2025-03-01", 180),
]

site_limit_df = (
    pd.DataFrame(SITE_LIMITS, columns=["effective_from", "site_limit_kw"])
      .assign(effective_from=lambda d: pd.to_datetime(d.effective_from))
      .set_index("effective_from")
      .resample("1H")
      .ffill()
      .rename_axis("ds")
      .reset_index()
)

# ───────────── 1. LOAD CHARGING DATA ──────────────
df = (pd.read_csv(CHARGING_FILE)
        .rename(columns={"Date": "ds", "Chargers": "y"}))
df["ds"]      = pd.to_datetime(df["ds"])
df            = df.sort_values("ds").reset_index(drop=True)
df["cal_date"] = df["ds"].dt.date

# ───────────── 2. LOAD WORKFORCE CALENDAR ─────────
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

# ───────────── 3. TIME  &  HOLIDAY FLAGS ──────────
tkd = pd.to_datetime([
    "2023-09-13","2023-10-26","2023-11-14","2023-12-20",
    "2024-01-12","2024-02-07","2024-03-14","2024-04-16",
    "2024-05-13","2024-06-07","2024-10-22","2024-11-28",
    "2024-12-18","2025-01-10","2025-02-13","2025-03-18",
    "2025-04-22","2025-05-12","2025-06-06"
]).normalize()

be_holidays = holidays.Belgium(years=df["ds"].dt.year.unique())

df["weekday"]        = df["ds"].dt.weekday
df["hour"]           = df["ds"].dt.hour
df["is_weekend"]     = (df["weekday"] >= 5).astype(int)
df["is_work_hour"]   = df["hour"].between(8, 18).astype(int)
df["is_holiday"]     = df["ds"].dt.normalize().isin(be_holidays).astype(int)
df["is_terugkomdag"] = df["ds"].dt.normalize().isin(tkd).astype(int)

# ───────── 4. WORK-AT-OFFICE FEATURE ──────────────
is_workday  = (df["weekday"] < 5) & (df["is_holiday"] == 0)
is_workhour = df["is_work_hour"] == 1
df["work_at_office"] = np.where(
    is_workday & is_workhour,
    (TOTAL_HEADCOUNT - df["vacation_cnt"] - df["homework_cnt"]).clip(lower=0),
    0
)

# ───────── 5. MERGE SITE-LIMIT & HELPERS ──────────
df = df.merge(site_limit_df, on="ds", how="left")
df["site_limit_kw"].fillna(40, inplace=True)      # assume 40 kW before first record
df["headroom_kw"]     = df["site_limit_kw"] - df["y"].clip(upper=df["site_limit_kw"])
df["utilization_pct"] = df["y"] / df["site_limit_kw"]

# ───────── 6. STATISTICAL-MOMENT FEATURES ────────
for win in (24, 168):                       # 24 h and 7×24 h
    r = df["y"].rolling(win, min_periods=1)
    df[f"m1_mean_{win}h"]  = r.mean()
    df[f"m2_var_{win}h"]   = r.var()
    df[f"m3_skew_{win}h"]  = r.skew()
    df[f"m4_kurt_{win}h"]  = r.kurt()

# ───────── 7. CYCLIC SINE/COS ENCODINGS ───────────
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

# ──────── 8. ROLLING 30-DAY BACK-TEST ─────────────
start_eval = df["ds"].max() - pd.Timedelta(days=30)
days_iter  = pd.date_range(start_eval.normalize(),
                           df["ds"].max().normalize(), freq="D")

xgb_preds, nn_preds = [], []
scaler = StandardScaler()
imputer = SimpleImputer(strategy="mean")

FEATURES = [
    # calendar / categorical flags
    "is_work_hour", "is_weekend", "is_holiday", "is_terugkomdag",
    # engineered counts
    "work_at_office",
    # lags & averages (added later per split)
    "lag_168", "avg_hourly_by_weekday", "avg_cars_weekday",
    # cyclical encodings
    "hour_sin", "hour_cos",
    # site-limit aware
    "site_limit_kw", "headroom_kw", "utilization_pct",
    # statistical moments
    "m1_mean_24h",  "m1_mean_168h",
    "m2_var_24h",   "m2_var_168h",
    "m3_skew_24h",  "m3_skew_168h",
    "m4_kurt_24h",  "m4_kurt_168h"
]

for day in days_iter:
    day_start = pd.Timestamp(day)
    train = df[df["ds"] < day_start].copy()

    # baseline hourly mean by weekday-hour
    avg_map = (train.groupby(["weekday","hour"])["y"]
                     .mean()
                     .rename("avg_hourly_by_weekday")
                     .reset_index())

    # 7-day lag
    train["lag_168"] = train["y"].shift(168).fillna(train["y"].mean())
    train = train.merge(avg_map, on=["weekday","hour"], how="left")
    train["avg_cars_weekday"] = (train.groupby("weekday")["y"]
                                   .transform("mean") / SESSION_KWH)

    # ── XGB MODEL ──
    xgb_model = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        objective="reg:squarederror", random_state=RANDOM_STATE,
        verbosity=0)
    xgb_model.fit(train[FEATURES], train["y"])

    # ── PREP TEST DAY ──
    day_df = df[(df["ds"] >= day_start) & (df["ds"] < day_start + pd.Timedelta(days=1))].copy()
    day_df = day_df.merge(avg_map, on=["weekday","hour"], how="left")

    lag_df = train[["ds","y"]].copy()
    lag_df["ds"] += pd.Timedelta(days=7)
    lag_df.rename(columns={"y":"lag_168"}, inplace=True)
    day_df = day_df.merge(lag_df, on="ds", how="left")
    day_df["lag_168"] = day_df["lag_168"].fillna(day_df["avg_hourly_by_weekday"])

    cars_map = (train.groupby("weekday")["y"].mean() / SESSION_KWH
               ).rename("avg_cars_weekday").reset_index()
    day_df = day_df.merge(cars_map, on="weekday", how="left")

    # ── NN MODEL ──
    X_train = imputer.fit_transform(train[FEATURES])   # mean-impute
    X_test  = imputer.transform(day_df[FEATURES])

    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)

    nn = MLPRegressor(
        hidden_layer_sizes=(30,15),
        activation='tanh', solver='adam',
        learning_rate_init=1e-3,
        max_iter=1000,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    )
    nn.fit(X_train_std, train["y"])

    # ── STORE PREDICTIONS ──
    day_df["yhat_xgb"] = xgb_model.predict(day_df[FEATURES])
    day_df["yhat_nn"]  = nn.predict(X_test_std)
    day_df["yhat_blend"] = 0.6 * day_df["yhat_xgb"] + 0.4 * day_df["yhat_nn"]

    xgb_preds.append(day_df[["ds","y","yhat_xgb"]])
    nn_preds.append(day_df[["ds","yhat_nn","yhat_blend"]])

# ──────── 9. MERGE RESULTS & METRICS ──────────────
xgb_df = pd.concat(xgb_preds).reset_index(drop=True)
nn_df  = pd.concat(nn_preds).reset_index(drop=True)
test_bt = pd.concat([xgb_df, nn_df[["yhat_nn","yhat_blend"]]], axis=1)

def metrics(obs, pred):
    return (mean_absolute_error(obs, pred),
            np.sqrt(mean_squared_error(obs, pred)),
            r2_score(obs, pred))

mae_xgb,  rmse_xgb,  r2_xgb  = metrics(test_bt["y"], test_bt["yhat_xgb"])
mae_nn,   rmse_nn,   r2_nn   = metrics(test_bt["y"], test_bt["yhat_nn"])
mae_bl,   rmse_bl,   r2_bl   = metrics(test_bt["y"], test_bt["yhat_blend"])

print("\n──────── BACK-TEST (Last 30 Days) ────────")
print(f"XGB    – MAE {mae_xgb:.3f}  RMSE {rmse_xgb:.3f}  R² {r2_xgb:.3f}")
print(f"FFBPNN – MAE {mae_nn :.3f}  RMSE {rmse_nn :.3f}  R² {r2_nn :.3f}")
print(f"Blend  – MAE {mae_bl :.3f}  RMSE {rmse_bl :.3f}  R² {r2_bl :.3f}")

# ──────── 10. OPTIONAL PLOT ───────────────────────
plt.figure(figsize=(6,5))
plt.scatter(test_bt["y"], test_bt["yhat_blend"], alpha=0.3)
lims = [min(test_bt["y"].min(), test_bt["yhat_blend"].min()),
        max(test_bt["y"].max(), test_bt["yhat_blend"].max())]
plt.plot(lims, lims, 'k--', lw=1)
plt.xlabel("Actual kW"); plt.ylabel("Predicted kW (Blend)")
plt.title("Predicted vs Actual – Blend"); plt.tight_layout(); plt.show()
