import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import calendar
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create results directory if it doesn't exist
os.makedirs("results/forecast1day", exist_ok=True)

# ---------------------------
# Load Daily Data
# ---------------------------
# The CSV is expected to have columns: Day, Chargers, Chargers achteraan, Grid Organi lbc, Solar
df = pd.read_csv('api_data/daily_cleaned.csv')

# Set 'Day' as datetime index
df.set_index("Day", inplace=True)
df.index = pd.to_datetime(df.index)



# As the data is already daily, we directly copy it
df_daily = df.copy()

print("Dataset Information (Daily Data):")
print(f"Time range: {df_daily.index.min()} to {df_daily.index.max()}")
print(f"Total observations: {len(df_daily)}")
print(f"Missing values: {df_daily['Total_consumption'].isna().sum()}")

# ---------------------------
# Transform Target: Log Consumption
# ---------------------------
# Ensure all consumption values are positive by shifting if needed
shift_val = abs(df_daily["Total_consumption"].min()) + 1  
df_daily["log_consumption"] = np.log(df_daily["Total_consumption"] + shift_val)

# ---------------------------
# Feature Engineering for Daily Data
# ---------------------------
be_holidays = holidays.BE()  # Belgian holidays

# Basic time features
df_daily['day_of_week'] = df_daily.index.dayofweek
df_daily['month'] = df_daily.index.month

# Categorical features
df_daily['is_weekend'] = df_daily['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df_daily['is_festive'] = df_daily.index.to_series().apply(lambda x: 1 if x in be_holidays else 0)

# Seasonal features
df_daily['is_summer'] = df_daily.index.month.isin([6, 7, 8]).astype(int)
df_daily['is_winter'] = df_daily.index.month.isin([12, 1, 2]).astype(int)

# Cyclical features for day of week
df_daily['day_of_week_sin'] = np.sin(2 * np.pi * df_daily['day_of_week'] / 7)
df_daily['day_of_week_cos'] = np.cos(2 * np.pi * df_daily['day_of_week'] / 7)

# Lagged features (avoid data leakage)
df_daily['consumption_lag_1d'] = df_daily['Total_consumption'].shift(1)   # 1-day lag
df_daily['consumption_lag_7d'] = df_daily['Total_consumption'].shift(7)   # 7-day lag

# Additional lag features for monthly and yearly
df_daily['consumption_lag_30d'] = df_daily['Total_consumption'].shift(30)   # Approx. monthly lag
df_daily['consumption_lag_365d'] = df_daily['Total_consumption'].shift(365)   # Yearly lag

# Advanced lag features for additional cycles
df_daily['consumption_lag_14d'] = df_daily['Total_consumption'].shift(14)   # 14-day lag for biweekly patterns
df_daily['consumption_lag_21d'] = df_daily['Total_consumption'].shift(21)   # 21-day lag for extended cycles
df_daily['rolling_avg_3d'] = df_daily['Total_consumption'].rolling(window=3).mean()  # 3-day moving average
df_daily['rolling_std_3d'] = df_daily['Total_consumption'].rolling(window=3).std()   # 3-day rolling std

# Drop rows with NaN values resulting from lag features and rolling calculations
df_daily.dropna(inplace=True)

# ---------------------------
# Visualization: Correlation Heatmap (Daily Data)
# ---------------------------
plt.figure(figsize=(12, 10))
numerical_features = df_daily.select_dtypes(include=[np.number]).columns
correlation = df_daily[numerical_features].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, vmin=-1, vmax=1)
plt.title("Daily Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("results/forecast1day/correlation_heatmap.png")
plt.close()

# Plot original daily time series of Total Consumption
plt.figure(figsize=(15, 6))
plt.plot(df_daily.index, df_daily['Total_consumption'], color='blue', alpha=0.6)
plt.title('Daily Power Consumption Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Power Consumption (kWh)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/forecast1day/time_series_plot.png")
plt.close()

print("\nBasic Statistics (Total Consumption):")
print(df_daily['Total_consumption'].describe())

# ---------------------------
# Prepare Data for Modeling (Daily)
# ---------------------------
# Use the log-transformed consumption as the target
target = "log_consumption"
y = df_daily[target]
exog_features = [
    "day_of_week_sin", "day_of_week_cos", "is_weekend", "is_festive",
    "is_summer", "is_winter", "consumption_lag_1d", "consumption_lag_7d",
    "consumption_lag_30d", "consumption_lag_365d", "consumption_lag_14d", "consumption_lag_21d",
    "rolling_avg_3d", "rolling_std_3d"
]
exog = df_daily[exog_features]

# Split data into training and testing sets (80% train, 20% test)
split_index = int(0.80 * len(df_daily))
y_train = y.iloc[:split_index]
exog_train = exog.iloc[:split_index]

print(f"Training data from {df_daily.index[0].date()} to {df_daily.index[split_index-1].date()}")

# Normalize the exogenous features using training data
scaler = StandardScaler()
exog_train_scaled = pd.DataFrame(
    scaler.fit_transform(exog_train),
    columns=exog_train.columns,
    index=exog_train.index
)

# ---------------------------
# Find Optimal ARIMA Parameters (Daily)
# ---------------------------
print("Finding optimal ARIMA parameters with limited search space...")
try:
    model_auto = auto_arima(
        y_train,
        exogenous=exog_train_scaled,
        seasonal=True,
        m=7,  # Weekly seasonality for daily data
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        max_order=3,
        max_p=2,
        max_q=2,
        max_d=1,
        max_P=1,
        max_Q=1,
        max_D=1,
        start_p=1,
        start_q=1,
        start_P=1,
        start_Q=1,
        information_criterion='aic',
        maxiter=50,
        method='lbfgs',
        n_jobs=1
    )
    
    order = model_auto.order
    seasonal_order = model_auto.seasonal_order
    
    print("Optimal ARIMA order:", order)
    print("Optimal Seasonal order:", seasonal_order)
    
except Exception as e:
    print(f"Auto ARIMA failed with error: {e}")
    print("Using predefined ARIMA parameters instead...")
    order = (1, 1, 1)
    seasonal_order = (1, 0, 0, 7)

# ---------------------------
# Fit SARIMAX Model with a Constant Trend (Daily)
# ---------------------------
print("\nFitting SARIMAX model...")
try:
    model = SARIMAX(
        y_train,
        exog=exog_train_scaled,
        order=order,
        seasonal_order=seasonal_order,
        trend='c',  # Include a constant trend
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False, maxiter=200, method='lbfgs')
    
    print("\nModel Summary:")
    print(model_fit.summary().tables[0].as_text())
    print(model_fit.summary().tables[1].as_text())
    
except MemoryError:
    print("Memory error during SARIMAX fitting. Exiting.")
    exit()

# ---------------------------
# Forecast 1 Day Into the Future
# ---------------------------
# Determine the next day's date based on the last date in the dataset
last_date = df_daily.index[-1]
next_date = last_date + pd.Timedelta(days=1)

# Compute exogenous features for the next day
next_day_of_week = next_date.dayofweek
next_is_weekend = 1 if next_day_of_week >= 5 else 0
next_is_festive = 1 if next_date in be_holidays else 0
next_is_summer = 1 if next_date.month in [6, 7, 8] else 0
next_is_winter = 1 if next_date.month in [12, 1, 2] else 0
next_day_of_week_sin = np.sin(2 * np.pi * next_day_of_week / 7)
next_day_of_week_cos = np.cos(2 * np.pi * next_day_of_week / 7)

# For lagged features, use the latest available values from the historical data
lag_1d = df_daily['Total_consumption'].iloc[-1]
lag_7d = df_daily['Total_consumption'].iloc[-7]
lag_30d = df_daily['Total_consumption'].iloc[-30]
lag_365d = df_daily['Total_consumption'].iloc[-365] if len(df_daily) >= 365 else np.nan
lag_14d = df_daily['Total_consumption'].iloc[-14]
lag_21d = df_daily['Total_consumption'].iloc[-21]

# Rolling features based on the last 3 days
rolling_avg_3d = df_daily['Total_consumption'].iloc[-3:].mean()
rolling_std_3d = df_daily['Total_consumption'].iloc[-3:].std()

# Create a DataFrame for the next day's exogenous features
next_exog = pd.DataFrame({
    "day_of_week_sin": [next_day_of_week_sin],
    "day_of_week_cos": [next_day_of_week_cos],
    "is_weekend": [next_is_weekend],
    "is_festive": [next_is_festive],
    "is_summer": [next_is_summer],
    "is_winter": [next_is_winter],
    "consumption_lag_1d": [lag_1d],
    "consumption_lag_7d": [lag_7d],
    "consumption_lag_30d": [lag_30d],
    "consumption_lag_365d": [lag_365d],
    "consumption_lag_14d": [lag_14d],
    "consumption_lag_21d": [lag_21d],
    "rolling_avg_3d": [rolling_avg_3d],
    "rolling_std_3d": [rolling_std_3d]
}, index=[next_date])

# Scale the new exogenous features using the same scaler fitted on the training data
next_exog_scaled = pd.DataFrame(
    scaler.transform(next_exog),
    columns=next_exog.columns,
    index=next_exog.index
)

# Forecast 1 day ahead (on the log-transformed scale)
forecast_obj = model_fit.get_forecast(steps=1, exog=next_exog_scaled)
y_pred_log_next = forecast_obj.predicted_mean.iloc[0]

# Invert the log transformation to get the forecast in the original scale
y_pred_next = np.exp(y_pred_log_next) - shift_val

print(f"\nForecast for {next_date.date()}: {y_pred_next:.2f} kWh")

# ---------------------------
# Save Forecast Result
# ---------------------------
forecast_next_df = pd.DataFrame({
    "Forecasted_Consumption_kWh": [y_pred_next]
}, index=[next_date])
forecast_next_df.to_csv("results/forecast1day/forecast_next_day.csv")
print("Forecast for the next day has been saved to 'results/forecast1day/forecast_next_day.csv'.")
