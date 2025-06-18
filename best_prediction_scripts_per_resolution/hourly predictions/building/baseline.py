import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────── 1  LOAD DATA ─────────────────────────
df = pd.read_csv("building_forecasts/building_data.csv")
df['ds'] = pd.to_datetime(df['Date'])
df = df.sort_values('ds').rename(columns={'Consumption': 'y'})  # hourly kWh

# ─────────────────────── 2  STATIC FLAGS ───────────────────────
tkd = pd.to_datetime([
    "2023-09-13","2023-10-26","2023-11-14","2023-12-20",
    "2024-01-12","2024-02-07","2024-03-14","2024-04-16",
    "2024-05-13","2024-06-07","2024-10-22","2024-11-28",
    "2024-12-18","2025-01-10","2025-02-13","2025-03-18",
    "2025-04-22","2025-05-12","2025-06-06",
]).normalize()
be_holidays = holidays.Belgium(years=df['ds'].dt.year.unique())

df['weekday']        = df['ds'].dt.weekday
df['hour']           = df['ds'].dt.hour
df['is_weekend']     = (df['weekday'] >= 5).astype(int)
df['is_work_hour']   = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
df['is_holiday']     = df['ds'].dt.normalize().isin(be_holidays).astype(int)
df['is_terugkomdag'] = df['ds'].dt.normalize().isin(tkd).astype(int)

# ─────── 3  HISTORISCH WEEKGEMIDDEL ────────────────────────────
# Bereken éénmalig het gemiddelde verbruik per weekday-hour over ALLE data
avg_map = (
    df.groupby(['weekday','hour'])['y']
      .mean()
      .rename('avg_hourly_by_weekday')
      .reset_index()
)

# ───────── 4  30-day rolling back-test ────────────────────────
start_eval = df['ds'].max() - pd.Timedelta(days=30)
days_iter  = pd.date_range(start_eval.normalize(),
                           df['ds'].max().normalize(),
                           freq='D')

all_preds = []
for day in days_iter:
    day_start = pd.Timestamp(day)
    day_end   = day_start + pd.Timedelta(days=1)

    # selecteer alleen de test‐uren voor deze dag
    day_df = df[(df['ds'] >= day_start) & (df['ds'] < day_end)].copy()

    # merge de baseline
    day_df = day_df.merge(avg_map, on=['weekday','hour'], how='left')

    # baseline-voorspelling
    day_df['yhat'] = day_df['avg_hourly_by_weekday']
    all_preds.append(day_df[['ds','y','yhat','weekday','hour']])

test_bt = pd.concat(all_preds).sort_values('ds').reset_index(drop=True)

# ─────────────────────── 5  EVALUATIE ──────────────────────────
test_bt['abs_err'] = np.abs(test_bt['y'] - test_bt['yhat'])
test_bt['pct_err'] = 100 * test_bt['abs_err'] / (test_bt['y'] + 1e-10)

mae   = mean_absolute_error(test_bt['y'], test_bt['yhat'])
rmse  = np.sqrt(mean_squared_error(test_bt['y'], test_bt['yhat']))
r2    = r2_score(test_bt['y'], test_bt['yhat'])
mape  = test_bt['pct_err'].mean()
smape = (
    2 * np.abs(test_bt['y'] - test_bt['yhat']) /
    (np.abs(test_bt['y']) + np.abs(test_bt['yhat']) + 1e-10)
).mean() * 100

print("\n--- Baseline Back-test Metrics (Laatste 30 Dagen) ---")
print(f"MAE    : {mae:.2f} kWh")
print(f"RMSE   : {rmse:.2f} kWh")
print(f"R²     : {r2:.3f}")
print(f"MAPE   : {mape:.2f}%")
print(f"sMAPE  : {smape:.2f}%")

# ──────────────────────
