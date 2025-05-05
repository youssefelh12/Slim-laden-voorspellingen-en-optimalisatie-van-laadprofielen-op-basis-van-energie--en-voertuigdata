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
os.makedirs("results/daily2", exist_ok=True)

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

# Drop rows with NaN values resulting from lag features
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
plt.savefig("results/daily2/correlation_heatmap.png")
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
plt.savefig("results/daily2/time_series_plot.png")
plt.close()

print("\nBasic Statistics (Total Consumption):")
print(df_daily['Total_consumption'].describe())

# ---------------------------
# Prepare Data for Modeling (Daily)
# ---------------------------
# Use the log-transformed consumption as the target
target = "log_consumption"
y_orig = df_daily["Total_consumption"]

# Define exogenous features suitable for daily forecasting
exog_features = [
    "day_of_week_sin", "day_of_week_cos", "is_weekend", "is_festive",
    "is_summer", "is_winter", "consumption_lag_1d", "consumption_lag_7d",
    "consumption_lag_30d", "consumption_lag_365d"
]

y = df_daily[target]
exog = df_daily[exog_features]

# Split data into training and testing sets (80% train, 20% test)
split_index = int(0.80 * len(df_daily))
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
exog_train, exog_test = exog.iloc[:split_index], exog.iloc[split_index:]
y_orig_train, y_orig_test = y_orig.iloc[:split_index], y_orig.iloc[split_index:]

print(f"Training data from {df_daily.index[0].date()} to {df_daily.index[split_index-1].date()}")
print(f"Testing data from {df_daily.index[split_index].date()} to {df_daily.index[-1].date()}")

# Normalize the exogenous features using training data
scaler = StandardScaler()
exog_train_scaled = pd.DataFrame(
    scaler.fit_transform(exog_train),
    columns=exog_train.columns,
    index=exog_train.index
)
exog_test_scaled = pd.DataFrame(
    scaler.transform(exog_test),
    columns=exog_test.columns,
    index=exog_test.index
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
    
    # Forecast on the log-transformed scale
    forecast_obj = model_fit.get_forecast(steps=len(y_test), exog=exog_test_scaled)
    y_pred_log = forecast_obj.predicted_mean
    y_pred_log.index = y_test.index

except MemoryError:
    print("Memory error during SARIMAX fitting. Using a simplified approach...")
    window_size = 2000
    y_pred_log = []
    for i in range(0, len(y_test), window_size):
        end_idx = min(i + window_size, len(y_test))
        subset_train = y_train.iloc[-10000:] if len(y_train) > 10000 else y_train
        subset_exog_train = exog_train_scaled.iloc[-10000:] if len(exog_train_scaled) > 10000 else exog_train_scaled
        
        subset_model = SARIMAX(
            subset_train,
            exog=subset_exog_train,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 0, 7),
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        subset_fit = subset_model.fit(disp=False, maxiter=100, method='lbfgs')
        subset_pred = subset_fit.get_forecast(
            steps=end_idx - i,
            exog=exog_test_scaled.iloc[i:end_idx]
        )
        y_pred_log.extend(subset_pred.predicted_mean.tolist())
    
    y_pred_log = pd.Series(y_pred_log, index=y_test.index)

# ---------------------------
# Invert the Log Transformation (Daily)
# ---------------------------
y_pred = np.exp(y_pred_log) - shift_val

# ---------------------------
# Compute Performance Metrics on Original Scale
# ---------------------------
def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-10
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)

mae = mean_absolute_error(y_orig_test, y_pred)
mse = mean_squared_error(y_orig_test, y_pred)
rmse = np.sqrt(mse)
mape = calculate_mape(y_orig_test.values, y_pred.values)
smape = calculate_smape(y_orig_test.values, y_pred.values)
r2 = r2_score(y_orig_test, y_pred)

print("\nModel Performance Metrics (Original Scale):")
print(f"MAE:   {mae:.4f}")
print(f"MSE:   {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAPE:  {mape:.4f}%")
print(f"sMAPE: {smape:.4f}%")
print(f"R²:    {r2:.4f}")

# ---------------------------
# Create Forecast DataFrame and Save Predictions (Daily)
# ---------------------------
forecast_df = pd.DataFrame({
    "Real_Consumption_kWh": y_orig_test,
    "Predicted_Consumption_kWh": y_pred
}, index=y_orig_test.index)

forecast_df["Difference"] = forecast_df["Predicted_Consumption_kWh"] - forecast_df["Real_Consumption_kWh"]
forecast_df["Absolute_Error"] = abs(forecast_df["Difference"])
forecast_df["Percent_Error"] = (forecast_df["Absolute_Error"] / (forecast_df["Real_Consumption_kWh"].abs() + 1e-10)) * 100
forecast_df = forecast_df.round(4)
forecast_df.to_csv("results/daily2/predicted_values_kwh.csv")
print("\nPredicted values have been saved to 'results/daily2/predicted_values_kwh.csv'")

# ---------------------------
# Feature Importance Analysis (Daily)
# ---------------------------
feature_importance = pd.DataFrame({
    'Feature': exog_features,
    'Correlation': [abs(df_daily[feature].corr(df_daily[target])) for feature in exog_features]
})
feature_importance = feature_importance.sort_values('Correlation', ascending=False)
print("\nFeature Importance by Correlation with Target (Log Scale):")
print(feature_importance.head(10))
feature_importance.to_csv("results/daily2/feature_importance.csv", index=False)

# ---------------------------
# Visualization: Forecast Comparison (Daily)
# ---------------------------
plt.figure(figsize=(15, 8))
plt.plot(forecast_df.index, forecast_df["Real_Consumption_kWh"], label="Actual", color="blue", alpha=0.6, linewidth=1)
plt.plot(forecast_df.index, forecast_df["Predicted_Consumption_kWh"], label="Predicted", color="red", alpha=0.6, linewidth=1)
plt.title("Actual vs Predicted Daily Power Consumption", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Consumption (kWh)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/daily2/forecast_comparison.png")
plt.close()

# Plot the residuals
plt.figure(figsize=(15, 6))
plt.plot(forecast_df.index, forecast_df["Difference"], color='green', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title("Daily Prediction Residuals (Predicted - Actual)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Residual", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/daily2/residuals.png")
plt.close()

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(forecast_df["Difference"], bins=30, alpha=0.7, color='skyblue')
plt.title("Distribution of Daily Prediction Errors", fontsize=14)
plt.xlabel("Prediction Error", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/daily2/error_distribution.png")
plt.close()

# Remove index name to avoid ambiguity when grouping
forecast_df.index.name = None

# Add additional column for error analysis: day_of_week
forecast_df['Day_of_Week'] = forecast_df.index.dayofweek

# Error by day of week
plt.figure(figsize=(12, 6))
day_error = forecast_df.groupby('Day_of_Week')['Absolute_Error'].mean()
sns.barplot(x=day_error.index, y=day_error.values)
plt.title('Average Prediction Error by Day of Week', fontsize=14)
plt.xlabel('Day of Week (0=Monday, 6=Sunday)', fontsize=12)
plt.ylabel('Mean Absolute Error', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/daily2/error_by_day.png")
plt.close()

print("\nAll daily visualizations have been saved in the 'results/daily2' directory.")
print("Daily forecasting analysis complete.")

# ---------------------------
# Boxplot: Comparison of Real vs Forecasted Values
# ---------------------------
# Melt the DataFrame to convert it into a long format suitable for a boxplot.
melted_df = forecast_df[['Real_Consumption_kWh', 'Predicted_Consumption_kWh']].reset_index().melt(
    id_vars='index', 
    value_vars=['Real_Consumption_kWh', 'Predicted_Consumption_kWh'],
    var_name='Type',
    value_name='Consumption'
)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='Consumption', data=melted_df)
plt.title("Comparison of Real vs Forecasted Daily Consumption")
plt.xlabel("Consumption Type")
plt.ylabel("Daily Consumption (kWh)")
plt.tight_layout()
plt.show()
