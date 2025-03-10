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
os.makedirs("results/daily3", exist_ok=True)

# ---------------------------
# Load Daily Data
# ---------------------------
# The CSV is expected to have columns: Day, Chargers, Chargers achteraan, Grid Organi lbc, Solar
df = pd.read_csv('api_data/aggregated_daily_measurements.csv')

# Set 'Day' as datetime index
df.set_index("Day", inplace=True)
df.index = pd.to_datetime(df.index)

# Use only the "Grid Organi lbc" column for total consumption
df['Total_consumption'] = df['Grid Organi lbc']

# Drop unused columns
df = df.drop(['Chargers', 'Chargers achteraan', 'Solar', 'Grid Organi lbc'], axis=1)

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
plt.savefig("results/daily3/correlation_heatmap.png")
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
plt.savefig("results/daily3/time_series_plot.png")
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

# Split data into training and testing sets (80% train)
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
# Recursive 7-Day Forecast into the Future
# ---------------------------
# For recursive forecasting, we update lag and rolling features using our forecasts.
forecast_horizon = 7
forecasted_values = []  # to store forecasted consumption (original scale)
forecast_dates = []    # to store forecast dates

# We begin with the last available historical consumption series.
# We will use its 'Total_consumption' column (not the log-transformed) for lags and rolling stats.
current_history = df_daily['Total_consumption'].copy()
last_date = df_daily.index[-1]

for i in range(1, forecast_horizon + 1):
    forecast_date = last_date + pd.Timedelta(days=i)
    forecast_dates.append(forecast_date)
    
    # --- Date-based exogenous features ---
    dow = forecast_date.dayofweek
    day_of_week_sin = np.sin(2 * np.pi * dow / 7)
    day_of_week_cos = np.cos(2 * np.pi * dow / 7)
    is_weekend = 1 if dow >= 5 else 0
    is_festive = 1 if forecast_date in be_holidays else 0
    is_summer = 1 if forecast_date.month in [6, 7, 8] else 0
    is_winter = 1 if forecast_date.month in [12, 1, 2] else 0

    # --- Lag-based features ---
    # consumption_lag_1d: last available consumption (from history, which may include forecasts)
    lag_1d = current_history.iloc[-1]
    
    # For the other lags, if the forecast date minus the lag offset exists in the historical index, use that;
    # otherwise, use the forecasted value from our recursive history.
    def get_lag(lag_days):
        ref_date = forecast_date - pd.Timedelta(days=lag_days)
        if ref_date in current_history.index:
            return current_history.loc[ref_date]
        else:
            # This branch should not be hit for a 7-day horizon if the dataset is sufficiently long.
            return current_history.iloc[0]
    
    lag_7 = get_lag(7)
    lag_30 = get_lag(30)
    lag_365 = get_lag(365)
    lag_14 = get_lag(14)
    lag_21 = get_lag(21)
    
    # --- Rolling features ---
    # For rolling_avg_3d and rolling_std_3d, use the last 3 available consumption values.
    rolling_window = current_history.iloc[-3:]
    rolling_avg_3d = rolling_window.mean()
    rolling_std_3d = rolling_window.std()
    
    # Build exogenous features DataFrame for the forecast date
    exog_next = pd.DataFrame({
        "day_of_week_sin": [day_of_week_sin],
        "day_of_week_cos": [day_of_week_cos],
        "is_weekend": [is_weekend],
        "is_festive": [is_festive],
        "is_summer": [is_summer],
        "is_winter": [is_winter],
        "consumption_lag_1d": [lag_1d],
        "consumption_lag_7d": [lag_7],
        "consumption_lag_30d": [lag_30],
        "consumption_lag_365d": [lag_365],
        "consumption_lag_14d": [lag_14],
        "consumption_lag_21d": [lag_21],
        "rolling_avg_3d": [rolling_avg_3d],
        "rolling_std_3d": [rolling_std_3d]
    }, index=[forecast_date])
    
    # Scale the new exogenous features using the same scaler
    exog_next_scaled = pd.DataFrame(
        scaler.transform(exog_next),
        columns=exog_next.columns,
        index=exog_next.index
    )
    
    # Forecast one step ahead using the model
    # (The forecast is on the log scale.)
    y_pred_log_next = model_fit.forecast(steps=1, exog=exog_next_scaled).iloc[0]
    # Invert the log transformation
    y_pred_next = np.exp(y_pred_log_next) - shift_val
    
    # Save the forecasted value and update the current history
    forecasted_values.append(y_pred_next)
    # Append the new forecasted consumption (on the original scale) to current_history.
    current_history = current_history.append(pd.Series(y_pred_next, index=[forecast_date]))

# Create a DataFrame of the recursive forecast results
forecast_recursive_df = pd.DataFrame({
    "Forecasted_Consumption_kWh": forecasted_values
}, index=forecast_dates)

print("\n7-Day Recursive Forecast (Original Scale):")
print(forecast_recursive_df)

# Save forecast result
forecast_recursive_df.to_csv("results/daily3/recursive_forecast_next_7_days.csv")
print("7-day recursive forecast has been saved to 'results/daily3/recursive_forecast_next_7_days.csv'.")
