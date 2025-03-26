import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
import calendar

# -------------------------------
# 1. Data Loading and Aggregation
# -------------------------------
df = pd.read_csv("api_data/monthly_api_data.csv")

# Convert Month to datetime and sort
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df.sort_values("Month", inplace=True)

# Group by Month & Measurement, then sum consumption across phases
df_grouped = df.groupby(["Month", "Measurement"])["Consumption"].sum().unstack(fill_value=0)

# Ensure required measurements exist
for col in ["Grid Organi lbc", "Chargers", "Chargers achteraan"]:
    if col not in df_grouped.columns:
        df_grouped[col] = 0

# Combine all measurements into a single Total_Consumption
df_grouped["Total_Consumption"] = (
    df_grouped["Grid Organi lbc"] +
    df_grouped["Chargers"] +
    df_grouped["Chargers achteraan"]
)

# Reset index so Month becomes a column
df_grouped.reset_index(inplace=True)

# -------------------------------
# 2. Basic Feature Engineering
# -------------------------------
be_holidays = holidays.BE()

def count_holidays_and_weekends(ts):
    """
    Given a Timestamp for a month, count holidays & weekend days in that month.
    """
    year = ts.year
    month = ts.month
    _, last_day = calendar.monthrange(year, month)
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = pd.Timestamp(year=year, month=month, day=last_day)
    days = pd.date_range(start_date, end_date, freq="D")

    holiday_count = sum(1 for day in days if day in be_holidays)
    weekend_count = sum(1 for day in days if day.weekday() >= 5)
    return pd.Series({"holiday_count": holiday_count, "weekend_count": weekend_count})

df_grouped[["holiday_count", "weekend_count"]] = df_grouped["Month"].apply(count_holidays_and_weekends)
df_grouped["month_number"] = df_grouped["Month"].dt.month
df_grouped["year"] = df_grouped["Month"].dt.year

# -------------------------------
# 3. Additional Features from Historical Data
# -------------------------------
# 3.1 Lag features
df_grouped["lag_1"] = df_grouped["Total_Consumption"].shift(1)
df_grouped["lag_12"] = df_grouped["Total_Consumption"].shift(12)

# 3.2 Rolling averages
df_grouped["rolling_mean_3"] = df_grouped["Total_Consumption"].rolling(window=3).mean()
df_grouped["rolling_mean_6"] = df_grouped["Total_Consumption"].rolling(window=6).mean()

# 3.3 Residual from 3-month rolling mean
df_grouped["rolling_resid_3"] = df_grouped["Total_Consumption"] - df_grouped["rolling_mean_3"]

# 3.4 Month-over-month percentage change
df_grouped["mom_change"] = df_grouped["Total_Consumption"].pct_change()

# 3.5 Cyclical encoding of month_number
df_grouped["month_sin"] = np.sin(2 * np.pi * df_grouped["month_number"] / 12)
df_grouped["month_cos"] = np.cos(2 * np.pi * df_grouped["month_number"] / 12)

# 3.6 Simple trend index (e.g., 0,1,2,... for each month)
df_grouped["trend_index"] = np.arange(len(df_grouped))

# 3.7 Example interaction features
df_grouped["weekend_month_interaction"] = df_grouped["weekend_count"] * df_grouped["month_number"]
df_grouped["holiday_lag1_interaction"] = df_grouped["holiday_count"] * df_grouped["lag_1"]

# Fill NaN from rolling/lags with 0
df_grouped.fillna(0, inplace=True)

# -------------------------------
# 4. Correlation Matrix (Optional)
# -------------------------------
all_features = [
    "Total_Consumption", "holiday_count", "weekend_count", "month_number", "year",
    "lag_1", "lag_12", "rolling_mean_3", "rolling_mean_6", "rolling_resid_3",
    "mom_change", "month_sin", "month_cos", "trend_index",
    "weekend_month_interaction", "holiday_lag1_interaction"
]
corr_matrix = df_grouped[all_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Features and Target")
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Define Target and Exogenous Variables
# -------------------------------
target = "Total_Consumption"

# Choose a subset or all features as exogenous regressors
exog_features = [
    "holiday_count", "weekend_count", "month_number", "year",
    "lag_1", "lag_12", "rolling_mean_3", "rolling_mean_6",
    "rolling_resid_3", "mom_change", "month_sin", "month_cos",
    "trend_index", "weekend_month_interaction", "holiday_lag1_interaction"
]

y = df_grouped[target]
exog = df_grouped[exog_features]

# -------------------------------
# 6. Train-Test Split (80/20)
# -------------------------------
split_index = int(0.80 * len(df_grouped))
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
exog_train, exog_test = exog.iloc[:split_index], exog.iloc[split_index:]

print(f"Training data from {df_grouped['Month'].iloc[0].date()} to {df_grouped['Month'].iloc[split_index-1].date()}")
print(f"Testing data from {df_grouped['Month'].iloc[split_index].date()} to {df_grouped['Month'].iloc[-1].date()}")

# -------------------------------
# 7. Random Forest Model Training
# -------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(exog_train, y_train)

# -------------------------------
# 8. Forecasting
# -------------------------------
y_pred = rf_model.predict(exog_test)

# -------------------------------
# 9. Metrics: MAE, MSE, MAPE, sMAPE, MASE
# -------------------------------
def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-10
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)

def calculate_mase(y_true, y_pred, y_train):
    """
    Mean Absolute Scaled Error using naive forecast (lag-1) on the training set.
    """
    # Naive forecast errors for training
    naive_errors = np.abs(np.diff(y_train))
    d = naive_errors.mean() if len(naive_errors) > 0 else 1e-10
    return np.mean(np.abs(y_true - y_pred)) / (d + 1e-10)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = calculate_mape(y_test.values, y_pred)
smape = calculate_smape(y_test.values, y_pred)
mase = calculate_mase(y_test.values, y_pred, y_train.values)

print(f"MAE:   {mae:.2f}")
print(f"MSE:   {mse:.2f}")
print(f"MAPE:  {mape:.2f}%")
print(f"sMAPE: {smape:.2f}%")
print(f"MASE:  {mase:.2f}")

# -------------------------------
# 10. Plotting Results
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df_grouped["Month"].iloc[split_index:], y_test, label="Actual", marker="o")
plt.plot(df_grouped["Month"].iloc[split_index:], y_pred, label="Forecast", marker="x")
plt.xlabel("Month")
plt.ylabel("Total Consumption")
plt.title("Random Forest Forecast with Extended Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
