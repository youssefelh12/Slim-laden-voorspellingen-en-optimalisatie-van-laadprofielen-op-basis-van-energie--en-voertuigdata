import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────── Config & Paths ───────────────────────
results_dir       = "results/daily_baseline_last_week"
consumption_file  = 'api_data/daily_cleaned_chargers.csv'
os.makedirs(results_dir, exist_ok=True)

# ─────────────────────── 1  LOAD DATA ─────────────────────────
df = pd.read_csv(consumption_file, parse_dates=['Day'])
df.set_index('Day', inplace=True)
df.sort_index(inplace=True)
df.rename(columns={'Total_consumption': 'y'}, inplace=True)

# ─────────────────────── 2  DEFINE EVAL PERIOD ────────────────
eval_days = 30
train_end = df.index.max() - pd.Timedelta(days=eval_days)
train_df  = df.loc[:train_end]
test_df   = df.loc[train_end + pd.Timedelta(days=1):]

# ─────────────────────── 3  BUILD LAST-WEEK LAG ──────────────
# Voor iedere datum de waarde van y van 7 dagen eerder
df['y_lag_7d'] = df['y'].shift(7)

# Voeg die kolom toe aan de testset
test_df = test_df.copy()
test_df['yhat'] = df['y_lag_7d'].loc[test_df.index]

# ─────────────────────── 4  EVALUATIE ─────────────────────────
test_df['abs_err'] = np.abs(test_df['y'] - test_df['yhat'])
test_df['pct_err'] = 100 * test_df['abs_err'] / (test_df['y'] + 1e-10)

mae   = mean_absolute_error(test_df['y'], test_df['yhat'])
rmse  = np.sqrt(mean_squared_error(test_df['y'], test_df['yhat']))
r2    = r2_score(test_df['y'], test_df['yhat'])
mape  = test_df['pct_err'].mean()
smape = (
    2 * np.abs(test_df['y'] - test_df['yhat']) /
    (np.abs(test_df['y']) + np.abs(test_df['yhat']) + 1e-10)
).mean() * 100

print("\n--- Baseline (Last Week) Back-test Metrics (Laatste 30 Dagen) ---")
print(f"MAE    : {mae:.2f} kWh")
print(f"RMSE   : {rmse:.2f} kWh")
print(f"R²     : {r2:.3f}")
print(f"MAPE   : {mape:.2f}%")
print(f"sMAPE  : {smape:.2f}%")

# ─────────────────────── 5  PLOT MAE PER WEEKDAY ─────────────
wd_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
test_df['weekday'] = test_df.index.dayofweek
mae_by_wd = test_df.groupby('weekday')['abs_err'].mean().reindex(range(7))

plt.figure(figsize=(8,4))
bars = plt.bar(wd_labels, mae_by_wd.values)
plt.ylabel("MAE (kWh)")
plt.title("Baseline (Last Week): Gem. Absolute Fout per Weekdag")
for bar, val in zip(bars, mae_by_wd.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             val, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "mae_per_weekday_last_week.png"))
plt.show()
