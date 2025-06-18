import os
import pandas as pd
import numpy as np
import holidays
import calendar
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load, Filter & Aggregate
# -------------------------------
df = pd.read_csv("api_data/monthly_api_data.csv")

# Keep only charger‐related measurements
df = df[df["Measurement"].isin(["Chargers", "Chargers achteraan"])]

# Parse Month & sort
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df.sort_values("Month", inplace=True)

# Sum into Total_charger_consumption (kWh), then convert to MWh
df = (
    df
    .groupby("Month")["Consumption"]
    .sum()
    .reset_index(name="Total_charger_consumption")
)
df["Total_charger_consumption"] /= 1000  # now in MWh

# -------------------------------
# 2. Safe Feature Engineering
# -------------------------------
be_holidays = holidays.BE()

def count_holidays_and_weekends(ts):
    year, month = ts.year, ts.month
    _, last_day = calendar.monthrange(year, month)
    days = pd.date_range(ts.replace(day=1), ts.replace(day=last_day), freq="D")
    return pd.Series({
        "holiday_count": sum(d in be_holidays for d in days),
        "weekend_count": sum(d.weekday() >= 5 for d in days)
    })

df[["holiday_count", "weekend_count"]] = df["Month"].apply(count_holidays_and_weekends)
df["month_number"] = df["Month"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month_number"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_number"] / 12)
df["trend_index"] = np.arange(len(df))
df["lag_1"]  = df["Total_charger_consumption"].shift(1)
df["lag_12"] = df["Total_charger_consumption"].shift(12)

# Drop rows where lags are not available
df.dropna(subset=["lag_1", "lag_12"], inplace=True)

# -------------------------------
# 3. Define target & exog
# -------------------------------
target = "Total_charger_consumption"
exog_features = [
    "holiday_count",
    "weekend_count",
    "month_sin",
    "month_cos",
    "trend_index",
    "lag_1",
    "lag_12",
]
y    = df[target]
X    = df[exog_features]

# -------------------------------
# 4. Train/Test Split (80/20)
# -------------------------------
split = int(0.8 * len(df))
X_train, X_test   = X.iloc[:split], X.iloc[split:]
y_train, y_test   = y.iloc[:split], y.iloc[split:]
months_test       = df["Month"].iloc[split:]

print(f"Training: {df['Month'].iloc[0].date()} → {df['Month'].iloc[split-1].date()}")
print(f"Testing:  {months_test.iloc[0].date()} → {months_test.iloc[-1].date()}")

# -------------------------------
# 5. Train Random Forest
# -------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# -------------------------------
# 6. Forecast & Evaluate
# -------------------------------
y_pred = rf.predict(X_test)

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

def calculate_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-10
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)

def calculate_mase(y_true, y_pred, y_train):
    d = np.abs(np.diff(y_train)).mean() if len(y_train) > 1 else 1e-10
    return np.mean(np.abs(y_true - y_pred)) / d

mae   = mean_absolute_error(y_test, y_pred)
mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)
nrmse = rmse / (y_test.max() - y_test.min())
mape_ = calculate_mape(y_test.values, y_pred)
smape_= calculate_smape(y_test.values, y_pred)
mase_ = calculate_mase(y_test.values, y_pred, y_train.values)
r2    = r2_score(y_test, y_pred)
n, p  = len(y_test), X_test.shape[1]
r2_adj= 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\n=== Random Forest Forecast Metrics (MWh) ===")
print(f"MAE:          {mae:.2f}")
print(f"MSE:          {mse:.2f}")
print(f"RMSE:         {rmse:.2f}")
print(f"NRMSE:        {nrmse:.4f}")
print(f"MAPE:         {mape_:.2f}%")
print(f"sMAPE:        {smape_:.2f}%")
print(f"MASE:         {mase_:.2f}")
print(f"R²:           {r2:.4f}")
print(f"Adjusted R²:  {r2_adj:.4f}")

# -------------------------------
# 7. Bar Plot Actual vs Forecast & Save
# -------------------------------
os.makedirs("results", exist_ok=True)
x     = np.arange(len(months_test))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, y_test,  width, label="Actual")
plt.bar(x + width/2, y_pred,  width, label="Forecast")

plt.xlabel("Month")
plt.ylabel("Charger Consumption (MWh)")
plt.title("Random Forest: Actual vs Forecast (Bar Plot)")
plt.xticks(x, [m.strftime("%Y-%m") for m in months_test], rotation=45)
plt.legend()
plt.grid(axis='y')
plt.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig("results/monthly_charger/random_forest_charger_forecast_MWh_bar.png")
plt.show()
