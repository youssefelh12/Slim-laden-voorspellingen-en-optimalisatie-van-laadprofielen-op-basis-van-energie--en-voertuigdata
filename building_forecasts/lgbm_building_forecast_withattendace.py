# ───────────────────────────── imports ─────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────── config ──────────────────────────────
CONSUMPTION_FILE = "./building_forecasts/building_data.csv"          # hourly kWh
CALENDAR_FILE    = "./layout1_full_calendar_2023-2025.csv"      # Layout-1 daily totals
      # optional <date,headcount>

CUT_OFF          = pd.Timestamp("2025-05-21 23:59")   # ignore future rows
TOTAL_HEADCOUNT  = 105                               # fallback denominator
TARGET_COL       = "y"

# ───────────────────── 1. load consumption data ───────────────────
df = (pd.read_csv(CONSUMPTION_FILE)
        .rename(columns={"Date": "ds", "Consumption": TARGET_COL}))
df["ds"] = pd.to_datetime(df["ds"])
df = df[df["ds"] <= CUT_OFF].sort_values("ds").reset_index(drop=True)
df["cal_date"] = df["ds"].dt.date

# ────────────────── 2. merge Layout-1 calendar counts ─────────────
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



# ─────────────── 4. calendar & time flags ────────────────────────
tkd = pd.to_datetime([
    "2023-09-13","2023-10-26","2023-11-14","2023-12-20",
    "2024-01-12","2024-02-07","2024-03-14","2024-04-16",
    "2024-05-13","2024-06-07","2024-10-22","2024-11-28",
    "2024-12-18","2025-01-10","2025-02-13","2025-03-18",
    "2025-04-22","2025-05-12","2025-06-06"
]).normalize()
be_holidays = holidays.Belgium(years=df["ds"].dt.year.unique())

df["weekday"]      = df["ds"].dt.weekday
df["hour"]         = df["ds"].dt.hour
df["is_weekend"]   = (df["weekday"] >= 5).astype(int)
df["is_work_hour"] = df["hour"].between(8, 18).astype(int)
df["is_holiday"]   = df["ds"].dt.normalize().isin(be_holidays).astype(int)
df["is_terugkomdag"]= df["ds"].dt.normalize().isin(tkd).astype(int)

# ─────────────── 5. base-number occupancy feature ───────────────
#   – only counted on business weekdays & work-hours; 0 elsewhere
is_workday  = (df["weekday"] < 5) & (df["is_holiday"] == 0)
is_workhour = df["is_work_hour"] == 1

df["work_at_office"] = np.where(
    is_workday & is_workhour,
    (TOTAL_HEADCOUNT - df["vacation_cnt"] - df["homework_cnt"]).clip(lower=0),
    0
)
# ─────────────── 4. quick correlation matrix ─────────────────────
corr_cols = [
    TARGET_COL, "work_at_office"
]
corr = df[corr_cols].corr()

plt.figure(figsize=(6,4))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr)), corr.index)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# ─────────────── 6. 30-day rolling back-test ─────────────────────
start_eval = df["ds"].max() - pd.Timedelta(days=30)
days_iter  = pd.date_range(start_eval.normalize(),
                           df["ds"].max().normalize(), freq="D")

all_preds = []

FEATURES = [
    "is_weekend","is_work_hour","weekday","hour",
    "work_at_office","lag_168",
    "is_holiday","is_terugkomdag",
    "avg_hourly_by_weekday"
]

for day in days_iter:
    day_start = pd.Timestamp(day)
    train = df[df["ds"] < day_start].copy()

    # baseline hourly pattern by weekday-hour
    avg_map = (train.groupby(["weekday","hour"])[TARGET_COL]
                     .mean().rename("avg_hourly_by_weekday").reset_index())

    train["lag_168"] = train[TARGET_COL].shift(168).fillna(train[TARGET_COL].mean())
    train = train.merge(avg_map, on=["weekday","hour"], how="left")

    model = lgb.LGBMRegressor(
        objective="regression", alpha=0.55,
        n_estimators=800, learning_rate=0.05,
        random_state=42
    ).fit(train[FEATURES], train[TARGET_COL])

    # 24-row frame for the target day
    day_df = df[(df["ds"] >= day_start) & (df["ds"] < day_start + pd.Timedelta(days=1))].copy()
    day_df = day_df.merge(avg_map, on=["weekday","hour"], how="left")

    lag_df = train[["ds", TARGET_COL]].copy()
    lag_df["ds"] += pd.Timedelta(days=7)
    lag_df.rename(columns={TARGET_COL:"lag_168"}, inplace=True)
    day_df = day_df.merge(lag_df, on="ds", how="left")
    day_df["lag_168"] = day_df["lag_168"].fillna(day_df["avg_hourly_by_weekday"])

    day_df["yhat"] = model.predict(day_df[FEATURES])
    all_preds.append(day_df[["ds", TARGET_COL, "yhat", "weekday"]])

test_bt = pd.concat(all_preds).sort_values("ds").reset_index(drop=True)

# ─────────────── 7. evaluation ──────────────────────────────────
mae  = mean_absolute_error(test_bt[TARGET_COL], test_bt["yhat"])
rmse = np.sqrt(mean_squared_error(test_bt[TARGET_COL], test_bt["yhat"]))
r2   = r2_score(test_bt[TARGET_COL], test_bt["yhat"])

print("\n--- Back-test (last 30 days) ---")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.2f}")


weekday_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'sat',6:'sun'}
r2_by_wd = {weekday_map[d]:
            r2_score(test_bt.loc[test_bt['weekday']==d,'y'],
                     test_bt.loc[test_bt['weekday']==d,'yhat'])
            for d in range(7)}

daily = test_bt.groupby(test_bt['ds'].dt.date).agg(
            y_sum=('y','sum'), yhat_sum=('yhat','sum'))
r2_daily_sum = r2_score(daily['y_sum'], daily['yhat_sum'])

# Plot 1: R² by weekday + aggregates
labels = list(r2_by_wd.keys()) + ['All-hours','Daily-sum']
values = list(r2_by_wd.values()) + [r2, r2_daily_sum]

plt.figure(figsize=(10,4))
bars = plt.bar(labels, values)
plt.axhline(0, color='k', linewidth=0.7)
plt.ylabel("R²")
plt.title("R² by Weekday • All Hours • Daily Sum")
for bar,v in zip(bars,values):
    plt.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.2f}",
             ha='center', va='bottom', fontsize=8)
plt.tight_layout(); plt.show()

# Plot 2: 30-day actual vs predicted
plt.figure(figsize=(12,5))
plt.plot(test_bt['ds'], test_bt['y'],    label='Actual',    alpha=0.6)
plt.plot(test_bt['ds'], test_bt['yhat'], label='Predicted', alpha=0.8)
plt.title("Back-test: Actual vs Predicted (30-Day)")
plt.xlabel("Date"); plt.ylabel("kWh")
plt.legend(); plt.tight_layout(); plt.show()

# Plot 3: model feature importances
plt.figure(figsize=(6,4))
plt.barh(FEATURES, model.feature_importances_)
plt.title("LightGBM Feature Importances")
plt.xlabel("Importance")
plt.tight_layout(); plt.show()
