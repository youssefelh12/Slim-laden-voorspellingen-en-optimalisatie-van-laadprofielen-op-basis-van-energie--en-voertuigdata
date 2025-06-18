import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
# 2. Train/Test Split (80/20)
# -------------------------------
split = int(0.8 * len(df))
train_df = df.iloc[:split].copy()
test_df  = df.iloc[split:].copy()
months_test = test_df["Month"]

# -------------------------------
# 3. Prepare for Prophet
# -------------------------------
train_prophet = train_df.rename(columns={
    "Month": "ds",
    "Total_charger_consumption": "y"
})

# -------------------------------
# 4. Fit Prophet
# -------------------------------
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)
m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
m.add_country_holidays(country_name="BE")
m.fit(train_prophet)

# -------------------------------
# 5. Forecast
# -------------------------------
future = m.make_future_dataframe(periods=len(test_df), freq="MS")
forecast = m.predict(future)

# extract yhat for the test months in order
fcst_test = (
    forecast
    .set_index("ds")
    .loc[months_test, "yhat"]
    .values
)

# -------------------------------
# 6. Evaluate
# -------------------------------
y_true = test_df["Total_charger_consumption"].values
y_pred = fcst_test

mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
nrmse = rmse / (y_true.max() - y_true.min())
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
smape = np.mean(2 * np.abs(y_pred - y_true) /
                (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
r2   = r2_score(y_true, y_pred)
n, p = len(y_true), 0
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\n=== Prophet Forecast Metrics (MWh) ===")
print(f"MAE:         {mae:.2f}")
print(f"MSE:         {mse:.2f}")
print(f"RMSE:        {rmse:.2f}")
print(f"NRMSE:       {nrmse:.4f}")
print(f"MAPE:        {mape:.2f}%")
print(f"sMAPE:       {smape:.2f}%")
print(f"R²:          {r2:.4f}")
print(f"Adjusted R²: {r2_adj:.4f}")

# -------------------------------
# 7. Bar Plot Actual vs Forecast & Save
# -------------------------------
os.makedirs("results", exist_ok=True)

x = np.arange(len(months_test))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, y_true,  width, label="Actual")
plt.bar(x + width/2, y_pred,  width, label="Forecast")

plt.xlabel("Month")
plt.ylabel("Charger Consumption (MWh)")
plt.title("Prophet Forecast vs Actual (Bar Plot)")

# set x-ticks to formatted month strings
plt.xticks(x, [m.strftime("%Y-%m") for m in months_test], rotation=45)

plt.legend()
plt.grid(axis='y')

# disable scientific notation
plt.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig("./results/prophet_charger_forecast_MWh_bar.png")
plt.show()
