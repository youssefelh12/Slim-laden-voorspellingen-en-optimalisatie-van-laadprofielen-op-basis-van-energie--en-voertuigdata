import pandas as pd
import numpy as np
import holidays
import calendar
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Sum across all measurements into Total_charger_consumption
# -------------------------------
# 1. Load, Filter & Aggregate
# -------------------------------
df = pd.read_csv("api_data/monthly_api_data.csv")

# Keep only the two charger‐related measurement types
df = df[df["Measurement"].isin(["Chargers", "Chargers achteraan"])]

# Convert Month to datetime and sort
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df.sort_values("Month", inplace=True)

# Sum consumption of both charger types into your target
df = (
    df
    .groupby("Month")["Consumption"]
    .sum()
    .reset_index(name="Total_charger_consumption")
)


# -------------------------------
# 2. Safe Feature Engineering
# -------------------------------
# Holiday & weekend counts
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

# Cyclical month encoding
df["month_number"] = df["Month"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month_number"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_number"] / 12)

# Trend index
df["trend_index"] = np.arange(len(df))

# Pure lags of the target
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
exog = df[exog_features]

# -------------------------------
# 4. Train/Test Split (80/20)
# -------------------------------
split = int(0.8 * len(df))
y_train, y_test       = y.iloc[:split], y.iloc[split:]
exog_train, exog_test = exog.iloc[:split], exog.iloc[split:]
months_test           = df["Month"].iloc[split:]

print(f"Training: {df['Month'].iloc[0].date()} → {df['Month'].iloc[split-1].date()}")
print(f"Testing:  {months_test.iloc[0].date()} → {months_test.iloc[-1].date()}")

# -------------------------------
# 5. Fit Auto-ARIMA on train
# -------------------------------
auto_model = auto_arima(
    y_train,
    exogenous=exog_train,
    seasonal=False,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)
print("Selected ARIMA order:", auto_model.order)

# -------------------------------
# 6. Fit SARIMAX
# -------------------------------
model = SARIMAX(
    y_train,
    exog=exog_train,
    order=auto_model.order,
    enforce_stationarity=False,
    enforce_invertibility=True
)
model_fit = model.fit(disp=False)

# -------------------------------
# 7. Forecast & Evaluate
# -------------------------------
y_pred = model_fit.forecast(steps=len(y_test), exog=exog_test)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-10
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)

def mase(y_true, y_pred, y_train):
    d = np.abs(np.diff(y_train)).mean() if len(y_train) > 1 else 1e-10
    return np.mean(np.abs(y_true - y_pred)) / d

mae   = mean_absolute_error(y_test, y_pred)
mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)
nrmse = rmse / (y_test.max() - y_test.min())
mape_ = mape(y_test.values, y_pred)
smape_ = smape(y_test.values, y_pred)
mase_ = mase(y_test.values, y_pred, y_train.values)
r2    = r2_score(y_test, y_pred)
n, p  = len(y_test), exog_test.shape[1]
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"\n=== Forecast Metrics ===")
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
# 8. Plot Actual vs Forecast
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(months_test, y_test,  marker="o", label="Actual")
plt.plot(months_test, y_pred,  marker="x", label="Forecast")
plt.xlabel("Month")
plt.ylabel("Charger Consumption (kWh)")
plt.title("SARIMAX Forecast (Leakage-Safe Features)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
