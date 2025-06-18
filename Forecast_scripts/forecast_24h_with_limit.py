#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hourly workplace-charging – 30-day rolling back-test
• Dynamic ceiling   : site_limit_kw
• Peak anchor       : avg_peak_prev_weeks  (mean of prev 4 weeks’ peaks on same weekday)
• XGBoost Tweedie   : variance_power = 1.3
• Diagnostics       : heat-map, time-series P-vs-A, scatter, CDF, feature-importance,
                      rolling R², residual ACF
"""

# ───────────────────────── imports ──────────────────────────
import numpy as np, pandas as pd, holidays, xgboost as xgb, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf

# ───────────────────────── config ───────────────────────────
CHARGING_FILE   = "./Forecast_scripts/Charging_data_hourly.csv"
CALENDAR_FILE   = "./Forecast_scripts/layout1_full_calendar_2023-2025.csv"
TOTAL_HEADCOUNT = 105
SESSION_KWH     = 9.5
ANCHOR_WEEKS    = 4        # weeks averaged for anchor

# ───── 0. dynamic site-limit timeline ───────────────────────
SITE_LIMITS = [
    ("2022-06-01",  40), ("2022-11-01",  55), ("2023-05-01",  70),
    ("2023-07-01",  80), ("2023-10-01", 100), ("2023-12-01", 120),
    ("2024-02-01", 130), ("2025-01-01", 150), ("2025-03-01", 180),
]
site_limit_df = (pd.DataFrame(SITE_LIMITS, columns=["effective_from","site_limit_kw"])
                   .assign(effective_from=lambda d: pd.to_datetime(d.effective_from))
                   .set_index("effective_from")
                   .resample("1h").ffill()
                   .rename_axis("ds").reset_index())

# ── 1. load hourly data ─────────────────────────────────────
df = (pd.read_csv(CHARGING_FILE)
        .rename(columns={"Date":"ds","Chargers":"y"}))
df["ds"] = pd.to_datetime(df["ds"]); df = df.sort_values("ds").reset_index(drop=True)
df["cal_date"] = df["ds"].dt.date

# ── 2. workforce calendar ──────────────────────────────────
cal = (pd.read_csv(CALENDAR_FILE, parse_dates=["Datum"])
         .rename(columns={"Totaal_Vakantiedagen":"vacation_cnt",
                          "Totaal_Thuiswerkdagen":"homework_cnt"}))
cal["cal_date"] = cal["Datum"].dt.date
df = df.merge(cal[["cal_date","vacation_cnt","homework_cnt"]],
              on="cal_date", how="left")
df[["vacation_cnt","homework_cnt"]] = df[["vacation_cnt","homework_cnt"]].fillna(0)

# ── 3. time / holiday flags ─────────────────────────────────
tkd = pd.to_datetime([
    "2023-09-13","2023-10-26","2023-11-14","2023-12-20",
    "2024-01-12","2024-02-07","2024-03-14","2024-04-16",
    "2024-05-13","2024-06-07","2024-10-22","2024-11-28",
    "2024-12-18","2025-01-10","2025-02-13","2025-03-18",
    "2025-04-22","2025-05-12","2025-06-06"
]).normalize()

be_holidays = pd.to_datetime(list(
    holidays.Belgium(years=df["ds"].dt.year.unique()).keys()
))

df["weekday"]        = df["ds"].dt.weekday
df["hour"]           = df["ds"].dt.hour
df["is_weekend"]     = (df["weekday"]>=5).astype(int)
df["is_work_hour"]   = df["hour"].between(8,18).astype(int)
df["is_holiday"]     = df["ds"].dt.normalize().isin(be_holidays).astype(int)
df["is_terugkomdag"] = df["ds"].dt.normalize().isin(tkd).astype(int)
df["is_friday"]      = (df["weekday"]==4).astype(int)
df["is_pre_holiday"] = ((df["ds"].dt.normalize()+pd.Timedelta(days=1))
                         .isin(be_holidays)).astype(int)

# ── 4. work-at-office ───────────────────────────────────────
mask = ((df["weekday"]<5)&(df["is_holiday"]==0)&(df["is_work_hour"]==1))
df["work_at_office"] = np.where(
    mask, (TOTAL_HEADCOUNT-df["vacation_cnt"]-df["homework_cnt"]).clip(lower=0), 0)

# ── 5. merge ceiling & build anchor ─────────────────────────
df = df.merge(site_limit_df, on="ds", how="left")
df["site_limit_kw"] = df["site_limit_kw"].fillna(40)

daily_peak = (df.groupby(df["ds"].dt.normalize())["y"]
                .max().rename("daily_peak")
                .reset_index())
daily_peak["weekday"] = daily_peak["ds"].dt.weekday
daily_peak["avg_peak_prev_weeks"] = (
    daily_peak.groupby("weekday")["daily_peak"]
              .transform(lambda s: (s.shift(1)
                                    .rolling(ANCHOR_WEEKS,min_periods=1)
                                    .mean()))
)

anchor_dict = dict(zip(daily_peak["ds"].values,
                       daily_peak["avg_peak_prev_weeks"].values))
df["avg_peak_prev_weeks"] = df["ds"].dt.normalize().map(anchor_dict)
df["avg_peak_prev_weeks"] = df["avg_peak_prev_weeks"].fillna(
    df["y"].rolling(24*7, min_periods=1).mean())

# ── 6. 30-day rolling back-test ─────────────────────────────
start_eval = df["ds"].max() - pd.Timedelta(days=30)
days_iter  = pd.date_range(start_eval.normalize(), df["ds"].max().normalize(), freq="D")
all_preds  = []

for day_start in days_iter:
    train = df[df["ds"] < day_start].copy()

    avg_map = (train.groupby(["weekday","hour"])["y"]
                     .mean().rename("avg_hourly_by_weekday").reset_index())

    train["lag_168"] = train["y"].shift(168).fillna(train["y"].mean())
    train = train.merge(avg_map, on=["weekday","hour"], how="left")
    train["avg_cars_weekday"] = (
        train.groupby("weekday")["y"].transform("mean") / SESSION_KWH)

    FEATURES = [
        "is_work_hour","is_weekend","is_holiday","is_terugkomdag",
        "is_friday","is_pre_holiday",
        "work_at_office","lag_168",
        "avg_hourly_by_weekday","avg_cars_weekday",
        "site_limit_kw","avg_peak_prev_weeks"
    ]

    model = xgb.XGBRegressor(
        n_estimators=900, learning_rate=0.05, max_depth=6,
        objective="reg:tweedie", tweedie_variance_power=1.3,
        random_state=42, verbosity=0)
    model.fit(train[FEATURES], train["y"])

    day_df = df[(df["ds"]>=day_start)&(df["ds"]<day_start+pd.Timedelta(days=1))].copy()
    day_df = day_df.merge(avg_map, on=["weekday","hour"], how="left")

    lag_df = train[["ds","y"]].copy(); lag_df["ds"] += pd.Timedelta(days=7)
    lag_df.rename(columns={"y":"lag_168"}, inplace=True)
    day_df = day_df.merge(lag_df, on="ds", how="left")
    day_df["lag_168"].fillna(day_df["avg_hourly_by_weekday"], inplace=True)

    cars_map = (train.groupby("weekday")["y"].mean()/SESSION_KWH
               ).rename("avg_cars_weekday").reset_index()
    day_df = day_df.merge(cars_map, on="weekday", how="left")

    day_df["yhat"] = model.predict(day_df[FEATURES])
    all_preds.append(day_df[["ds","y","yhat","weekday","hour"]])

test_bt = pd.concat(all_preds).sort_values("ds").reset_index(drop=True)

# ── 7. metrics ──────────────────────────────────────────────
test_bt["abs_err"] = np.abs(test_bt["y"] - test_bt["yhat"])
mae  = mean_absolute_error(test_bt["y"], test_bt["yhat"])
rmse = np.sqrt(mean_squared_error(test_bt["y"], test_bt["yhat"]))
r2   = r2_score(test_bt["y"], test_bt["yhat"])
print(f"\n30-day back-test: MAE {mae:.2f}  RMSE {rmse:.2f}  R² {r2:.2f}")

# ── 8. plots ────────────────────────────────────────────────
# 8a error heat-map
heat = (test_bt.pivot_table(index='weekday', columns='hour',
                            values='abs_err', aggfunc='mean')
               .reindex(index=range(7), columns=range(24)))
plt.figure(figsize=(12,5))
plt.imshow(heat, aspect='auto', cmap='Reds', origin='lower')
plt.colorbar(label='Abs Error (kW)')
plt.xticks(range(24)); plt.yticks(range(7),
    ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.xlabel('Hour'); plt.ylabel('Weekday')
plt.title('Error Heat-map'); plt.tight_layout(); plt.show()

# 8b time-series predicted vs actual
plt.figure(figsize=(12,4))
plt.plot(test_bt['ds'], test_bt['y'],    label='Actual',    alpha=0.6)
plt.plot(test_bt['ds'], test_bt['yhat'], label='Predicted', alpha=0.8)
plt.legend(); plt.xlabel('Date'); plt.ylabel('kW')
plt.title('Actual vs Predicted – 30-day Back-test'); plt.tight_layout(); plt.show()


plt.title('MAE Cumulative Distribution'); plt.tight_layout(); plt.show()

# 8e feature importance
plt.figure(figsize=(7,5))
plt.barh(FEATURES, model.feature_importances_)
plt.xlabel('Importance'); plt.title('XGBoost Feature Importance')
plt.tight_layout(); plt.show()


# 8h  ── total absolute-error per day ──────────────────────────
daily_err = (test_bt
             .set_index('ds')['abs_err']
             .resample('1D')        # one row per day
             .sum())                # or .mean() for daily-MAE

plt.figure(figsize=(10,4))
plt.plot(daily_err.index, daily_err.values, marker='o')
plt.ylabel('Total Abs Error (kW)')
plt.xlabel('Date')
plt.title('Total Absolute Error per Day – 30-day Back-test')
plt.tight_layout(); plt.show()
