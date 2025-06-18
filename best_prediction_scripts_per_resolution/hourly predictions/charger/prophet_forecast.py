import numpy as np
import pandas as pd
import holidays
from prophet import Prophet
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────── config ──────────────────────────────
CHARGING_FILE    = "./charging_forecasts/Charging_data_hourly.csv"
CALENDAR_FILE    = "./layout1_full_calendar_2023-2025.csv"
TOTAL_HEADCOUNT = 105
SESSION_KWH      = 9.5

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

df = df.merge(cal[["cal_date", "vacation_cnt", "homework_cnt"]],
              on="cal_date", how="left")
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

# ───────────────────── 5  ROLLING 30-DAY BACK-TEST ───────────────
start_eval = df["ds"].max() - pd.Timedelta(days=30)
days_iter  = pd.date_range(start_eval.normalize(),
                           df["ds"].max().normalize(), freq="D")

all_preds = []
FEATURES = [
    "is_work_hour","is_weekend","is_holiday","is_terugkomdag",
    "work_at_office","lag_168","avg_hourly_by_weekday","avg_cars_weekday"
]

for day in days_iter:
    day_start = pd.Timestamp(day)
    train = df[df["ds"] < day_start].copy()

    # baseline hourly mean by weekday-hour
    avg_map = (train.groupby(["weekday","hour"])["y"]
                     .mean().rename("avg_hourly_by_weekday").reset_index())
    train = train.merge(avg_map, on=["weekday","hour"], how="left")

    # 7-day lag
    train["lag_168"] = train["y"].shift(168).fillna(train["y"].mean())

    # cars per weekday proxy
    train["avg_cars_weekday"] = train.groupby("weekday")["y"].transform("mean") / SESSION_KWH

    # prepare and fit Prophet
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    for reg in FEATURES:
        m.add_regressor(reg)
    train_prop = train[["ds","y"] + FEATURES]
    m.fit(train_prop)

    # build feature frame for the target day
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

    # forecast with Prophet
    future = day_df[["ds"] + FEATURES]
    fcst   = m.predict(future)
    day_df["yhat"] = fcst["yhat"].values

    all_preds.append(day_df[["ds","y","yhat","weekday"]])

test_bt = pd.concat(all_preds).sort_values("ds").reset_index(drop=True)

# ───────────────────── 6  ADDITIONAL METRICS ─────────────────────
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

print("\n--- Back-test Metrics (Last 30 Days) ---")
print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.2f}")
print(f"MAPE  : {mape:.2f}%")
print(f"sMAPE : {smape:.2f}%")

# ───────────────────── 7  PLOTS ────────────────────────────────────
# R² by weekday + daily sum
weekday_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri'}
r2_by_wd = {
    wd: r2_score(
        test_bt.loc[test_bt['weekday']==d,'y'],
        test_bt.loc[test_bt['weekday']==d,'yhat']
    )
    for d,wd in weekday_map.items()
}
daily = test_bt.groupby(test_bt["ds"].dt.date).agg(y_sum=('y','sum'), yhat_sum=('yhat','sum'))
r2_daily = r2_score(daily['y_sum'], daily['yhat_sum'])

labels = list(r2_by_wd.keys()) + ['All-hours','Daily-sum']
values = list(r2_by_wd.values()) + [r2, r2_daily]
plt.figure(figsize=(10,4))
bars = plt.bar(labels, values, color='steelblue')
plt.axhline(0, color='k', linewidth=0.7)
plt.ylabel("R²")
plt.title("R² by Weekday • All Hours • Daily Sum")
for bar,v in zip(bars, values):
    plt.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout(); plt.show()

# Actual vs Predicted
plt.figure(figsize=(12,5))
plt.plot(test_bt["ds"], test_bt["y"],    label='Actual',    alpha=0.6)
plt.plot(test_bt["ds"], test_bt["yhat"], label='Forecast',  alpha=0.8)
plt.title("Back-test: Actual vs Forecast (30-Day)")
plt.xlabel("Date"); plt.ylabel("Chargers")
plt.legend(); plt.tight_layout(); plt.show()
