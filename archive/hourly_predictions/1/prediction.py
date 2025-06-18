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
os.makedirs("results", exist_ok=True)

# Load data
df = pd.read_csv('api_data/aggregated_hourly_measurements.csv')

# Ensure the index is datetime and set it
df.set_index("Hour", inplace=True)
df.index = pd.to_datetime(df.index)

# Map target consumption column and drop unused columns
df['Total_consumption'] = df['Grid Organi lbc']
df = df.drop(['Chargers achteraan', 'Solar', 'Chargers', 'Grid Organi lbc'], axis=1)

# Print basic info about the dataset
print("Dataset Information:")
print(f"Time range: {df.index.min()} to {df.index.max()}")
print(f"Total observations: {len(df)}")
print(f"Missing values: {df['Total_consumption'].isna().sum()}")

# Feature Engineering
be_holidays = holidays.BE()  # Belgian holidays

# Basic time features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# Categorical features
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['is_festive'] = df.index.to_series().apply(lambda x: 1 if x in be_holidays else 0)
df['working_hour'] = df['hour'].apply(lambda x: 1 if 8 <= x <= 18 else 0)

# Seasonal features
df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)
df['is_winter'] = df.index.month.isin([12, 1, 2]).astype(int)

# Create peak hour features
df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)

# Create cyclical features for hour and day of week
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Lagged features (with care to avoid data leakage)
df['consumption_lag_24h'] = df['Total_consumption'].shift(24)   # 24-hour lag
df['consumption_lag_168h'] = df['Total_consumption'].shift(168)  # 7-day lag

# Drop rows with NaN from lag features
df.dropna(inplace=True)

# Create correlation heatmap with correct boolean mask
plt.figure(figsize=(16, 12))
numerical_features = df.select_dtypes(include=[np.number]).columns
correlation = df[numerical_features].corr()

# Create a boolean mask for the upper triangle
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png")
plt.close()

# Plot original time series
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Total_consumption'], color='blue', alpha=0.6)
plt.title('Power Consumption Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Power Consumption (kWh)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/time_series_plot.png")
plt.close()

# Basic statistics
print("\nBasic Statistics:")
print(df['Total_consumption'].describe())

# Define target and features
target = "Total_consumption"
exog_features = [
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "is_weekend", "is_festive", "working_hour", 
    "is_summer", "is_winter", "is_morning_peak", "is_evening_peak",
    "consumption_lag_24h", "consumption_lag_168h"
]

# Prepare data
y = df[target]
exog = df[exog_features]

# Split data into train and test sets - using temporal split to avoid data leakage
split_index = int(0.80 * len(df))
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
exog_train, exog_test = exog.iloc[:split_index], exog.iloc[split_index:]

print(f"Training data from {df.index[0].date()} to {df.index[split_index-1].date()}")
print(f"Testing data from {df.index[split_index].date()} to {df.index[-1].date()}")

# Normalize features to improve model performance (fit scaler only on training data)
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

# Find optimal ARIMA parameters with limited search space
print("Finding optimal ARIMA parameters with limited search space...")
try:
    model_auto = auto_arima(
        y_train,
        exogenous=exog_train_scaled,
        seasonal=True,
        m=24,  # Daily seasonality for hourly data
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
    seasonal_order = (1, 0, 0, 24)

# Fit SARIMAX model with determined or default parameters
print("\nFitting SARIMAX model...")
try:
    model = SARIMAX(
        y_train,
        exog=exog_train_scaled,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend='c'
    )
    model_fit = model.fit(disp=False, maxiter=200, method='lbfgs')
    
    print("\nModel Summary:")
    print(model_fit.summary().tables[0].as_text())
    print(model_fit.summary().tables[1].as_text())
    
    # Make predictions
    forecast_obj = model_fit.get_forecast(steps=len(y_test), exog=exog_test_scaled)
    y_pred = forecast_obj.predicted_mean

    # Align predictions with y_test index to avoid NaNs during merging
    y_pred.index = y_test.index

except MemoryError:
    print("Memory error during SARIMAX fitting. Using a simplified approach...")
    window_size = 2000
    y_pred = []
    for i in range(0, len(y_test), window_size):
        end_idx = min(i + window_size, len(y_test))
        subset_train = y_train.iloc[-10000:] if len(y_train) > 10000 else y_train
        subset_exog_train = exog_train_scaled.iloc[-10000:] if len(exog_train_scaled) > 10000 else exog_train_scaled
        
        subset_model = SARIMAX(
            subset_train,
            exog=subset_exog_train,
            order=(1, 1, 1),  # Simplified model
            seasonal_order=(1, 0, 0, 24),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        subset_fit = subset_model.fit(disp=False, maxiter=100, method='lbfgs')
        subset_pred = subset_fit.get_forecast(
            steps=end_idx - i,
            exog=exog_test_scaled.iloc[i:end_idx]
        )
        y_pred.extend(subset_pred.predicted_mean.tolist())
    
    y_pred = pd.Series(y_pred, index=y_test.index)

# Calculate performance metrics
def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-10
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = calculate_mape(y_test.values, y_pred.values)
smape = calculate_smape(y_test.values, y_pred.values)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"MAE:   {mae:.4f}")
print(f"MSE:   {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAPE:  {mape:.4f}%")
print(f"sMAPE: {smape:.4f}%")
print(f"R²:    {r2:.4f}")

# Ensure predictions have the correct datetime index
if not isinstance(y_pred, pd.Series):
    y_pred = pd.Series(y_pred, index=y_test.index)

# Create the forecast DataFrame
forecast_df = pd.DataFrame({
    "Real_Consumption_kWh": y_test,
    "Predicted_Consumption_kWh": y_pred
}, index=y_test.index)

forecast_df["Difference"] = forecast_df["Predicted_Consumption_kWh"] - forecast_df["Real_Consumption_kWh"]
forecast_df["Absolute_Error"] = abs(forecast_df["Difference"])
forecast_df["Percent_Error"] = (forecast_df["Absolute_Error"] / (forecast_df["Real_Consumption_kWh"].abs() + 1e-10)) * 100
forecast_df = forecast_df.round(4)
forecast_df.to_csv("results/predicted_values_kwh.csv")
print("\nPredicted values have been saved to 'results/predicted_values_kwh.csv'")

# Feature importance analysis based on correlation with target
feature_importance = pd.DataFrame({
    'Feature': exog_features,
    'Correlation': [abs(df[feature].corr(df[target])) for feature in exog_features]
})
feature_importance = feature_importance.sort_values('Correlation', ascending=False)
print("\nFeature Importance by Correlation with Target:")
print(feature_importance.head(10))
feature_importance.to_csv("results/feature_importance.csv", index=False)

# Create heatmap of feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Correlation', y='Feature', data=feature_importance)
plt.title('Features by Correlation with Target', fontsize=14)
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()

# Visualize the predictions: Actual vs Predicted
plt.figure(figsize=(15, 8))
plt.plot(forecast_df.index, forecast_df["Real_Consumption_kWh"], label="Actual", color="blue", alpha=0.6, linewidth=1)
plt.plot(forecast_df.index, forecast_df["Predicted_Consumption_kWh"], label="Predicted", color="red", alpha=0.6, linewidth=1)
plt.title("Actual vs Predicted Power Consumption", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Consumption (kWh)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/forecast_comparison.png")
plt.close()

# Plot the residuals
plt.figure(figsize=(15, 6))
plt.plot(forecast_df.index, forecast_df["Difference"], color='green', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title("Prediction Residuals (Predicted - Actual)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Residual", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/residuals.png")
plt.close()

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(forecast_df["Difference"], bins=30, alpha=0.7, color='skyblue')
plt.title("Distribution of Prediction Errors", fontsize=14)
plt.xlabel("Prediction Error", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/error_distribution.png")
plt.close()

# Remove index name to avoid ambiguity when grouping by 'Hour'
forecast_df.index.name = None

# Create a daily or weekly analysis of errors
forecast_df['Hour'] = forecast_df.index.hour
forecast_df['Day_of_Week'] = forecast_df.index.dayofweek

# Error by hour
plt.figure(figsize=(12, 6))
hour_error = forecast_df.groupby('Hour')['Absolute_Error'].mean()
sns.barplot(x=hour_error.index, y=hour_error.values)
plt.title('Average Prediction Error by Hour of Day', fontsize=14)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Mean Absolute Error', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/error_by_hour.png")
plt.close()

# Error by day of week
plt.figure(figsize=(12, 6))
day_error = forecast_df.groupby('Day_of_Week')['Absolute_Error'].mean()
sns.barplot(x=day_error.index, y=day_error.values)
plt.title('Average Prediction Error by Day of Week', fontsize=14)
plt.xlabel('Day of Week (0=Monday, 6=Sunday)', fontsize=12)
plt.ylabel('Mean Absolute Error', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/error_by_day.png")
plt.close()

print("\nAll visualizations have been saved in the 'results' directory.")
print("Analysis complete.")
