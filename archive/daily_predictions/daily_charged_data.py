import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import calendar
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# ---------------------------
# Directory & Data Loading Functions
# ---------------------------
def create_results_directory(path):
    os.makedirs(path, exist_ok=True)

def load_consumption_data(file_path):
    df = pd.read_csv(file_path)
    df.set_index("Day", inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def load_temperature_data(file_path, index_dates):
    temp_df = pd.read_csv(file_path)
    temp_df['date'] = pd.to_datetime(temp_df['date'], utc=True)
    temp_df.set_index('date', inplace=True)
    # Convert to Europe/Brussels timezone then to a timezone-naive index
    temp_df.index = temp_df.index.tz_convert('Europe/Brussels').tz_localize(None)
    # Reindex to match the consumption dates and fill missing values
    temp_df = temp_df.reindex(index_dates)
    temp_df['temperature_2m_mean'].ffill(inplace=True)
    temp_df['temperature_2m_mean'].bfill(inplace=True)
    return temp_df

def merge_temperature_data(consumption_df, temp_df):
    df = consumption_df.merge(
        temp_df[['temperature_2m_mean']],
        left_index=True,
        right_index=True,
        how='left'
    )
    return df

# ---------------------------
# Feature Engineering Functions
# ---------------------------
def add_cumulative_ev_phev_feature(df):
    from datetime import datetime
    import pandas as pd

    # Dictionary of (date: cumulative count) for EV + PHEV vehicles
    cumulative_data = {
        datetime(2022, 1, 4): 2,
        datetime(2022, 3, 31): 3,
        datetime(2023, 2, 14): 4,
        datetime(2023, 2, 15): 5,
        datetime(2023, 2, 28): 6,
        datetime(2023, 6, 13): 7,
        datetime(2023, 6, 23): 8,
        datetime(2023, 7, 6): 9,
        datetime(2023, 9, 15): 10,
        datetime(2023, 9, 26): 13,
        datetime(2023, 9, 27): 14,
        datetime(2023, 10, 11): 15,
        datetime(2023, 10, 20): 16,
        datetime(2023, 11, 7): 18,
        datetime(2024, 1, 2): 20,
        datetime(2024, 3, 1): 21,
        datetime(2024, 3, 19): 22,
        datetime(2024, 3, 28): 23,
        datetime(2024, 5, 7): 24,
        datetime(2024, 5, 13): 26,
        datetime(2024, 5, 14): 27,
        datetime(2024, 5, 16): 29,
        datetime(2024, 5, 23): 30,
        datetime(2024, 5, 28): 31,
        datetime(2024, 6, 20): 33,
        datetime(2024, 6, 25): 34,
        datetime(2024, 9, 5): 36,
        datetime(2024, 9, 12): 39,
        datetime(2024, 9, 27): 40,
        datetime(2024, 10, 15): 41,
        datetime(2024, 10, 29): 43,
        datetime(2024, 11, 5): 44,
        datetime(2024, 11, 26): 45,
        datetime(2025, 1, 9): 46,
        datetime(2025, 1, 23): 47,
        datetime(2025, 1, 28): 48,
        datetime(2025, 2, 4): 49,
    }

    # Convert to Series and forward-fill over all dates in df
    ev_series = pd.Series(cumulative_data)
    ev_series = ev_series.reindex(df.index.union(ev_series.index)).sort_index().ffill().fillna(0)

    # Assign back to DataFrame
    df["cumulative_ev_phev_count"] = ev_series.reindex(df.index).astype(int)

def add_terugkomdag_feature(df):
    # List of 'terugkomdagen' dates
    terugkomdagen = [
        datetime(2023, 9, 13), datetime(2023, 10, 26), datetime(2023, 11, 14), datetime(2023, 12, 20),
        datetime(2024, 1, 12), datetime(2024, 2, 7), datetime(2024, 3, 14), datetime(2024, 4, 16),
        datetime(2024, 5, 13), datetime(2024, 6, 7), datetime(2024, 3, 16), datetime(2024, 10, 22),
        datetime(2024, 11, 28), datetime(2024, 12, 18), datetime(2025, 1, 10), datetime(2025, 2, 13),
        datetime(2025, 3, 18), datetime(2025, 4, 22), datetime(2025, 5, 12), datetime(2025, 6, 6)
    ]
    df['is_terugkomdag'] = df.index.to_series().dt.date.isin([d.date() for d in terugkomdagen]).astype(int)



def add_temperature_features(df):
    df['temp_mean_lag_1d'] = df['temperature_2m_mean'].shift(1)
    df['temp_mean_rolling_3d'] = df['temperature_2m_mean'].rolling(window=3).mean()

def add_time_features(df, be_holidays):
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_festive'] = df.index.to_series().apply(lambda x: 1 if x in be_holidays else 0)
    df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)
    df['is_winter'] = df.index.month.isin([12, 1, 2]).astype(int)
    
    # Existing cyclical encoding for day-of-week
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Extended cyclical encoding for month and day-of-year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

def add_lag_features(df, be_holidays):
    # Basic lag features
    df['consumption_lag_1d'] = df['Total_consumption'].shift(1)
    df['consumption_lag_7d'] = df['Total_consumption'].shift(7)
    df['consumption_lag_30d'] = df['Total_consumption'].shift(30)
    df['consumption_lag_365d'] = df['Total_consumption'].shift(365)
    # Additional lags and rolling statistics
    df['consumption_lag_14d'] = df['Total_consumption'].shift(14)
    df['consumption_lag_21d'] = df['Total_consumption'].shift(21)
    df['rolling_avg_3d'] = df['Total_consumption'].rolling(window=3).mean()
    df['rolling_std_3d'] = df['Total_consumption'].rolling(window=3).std()
    
    # Holiday-related features: flagging day after a holiday and bridge days
    def is_day_after_holiday(date):
        prev_day = date - pd.Timedelta(days=1)
        return int(prev_day in be_holidays)
    df['is_day_after_holiday'] = df.index.to_series().apply(is_day_after_holiday)
    
    def is_bridge_day(date):
        if date.weekday() == 0 and (date - pd.Timedelta(days=1)) in be_holidays:
            return 1
        elif date.weekday() == 4 and (date + pd.Timedelta(days=1)) in be_holidays:
            return 1
        return 0
    df['is_bridge_day'] = df.index.to_series().apply(is_bridge_day)

def add_holiday_proximity_features(df, be_holidays):
    # Create a sorted list of holiday dates (converted to pd.Timestamp)
    holiday_dates = sorted([pd.Timestamp(date) for date in be_holidays])
    
    def compute_proximity(date):
        prev_diff, next_diff = np.nan, np.nan
        for h in holiday_dates:
            if h <= date:
                prev_diff = (date - h).days
            elif h > date:
                next_diff = (h - date).days
                break
        return prev_diff, next_diff

    df['days_since_holiday'], df['days_until_holiday'] = zip(*df.index.to_series().apply(compute_proximity))

def add_holiday_type_feature(df, be_holidays):
    # Define a mapping of holiday names to their strength weights
    holiday_strength = {
        "Nieuwjaar": 1.0,
        "Paasmaandag": 1.5,
        "Dag van de Arbeid": 1.2,
        "O.L.H. Hemelvaart": 1.3,
        "Pinkstermaandag": 1.2,
        "Nationale feestdag": 1.5,
        "Wapenstilstand": 1.0,
        "Kerstmis": 2.0,
    }
    def get_holiday_strength(date):
        if date in be_holidays:
            holiday_name = be_holidays.get(date)
            return holiday_strength.get(holiday_name, 1.0)  # default to 1.0 if not found
        else:
            return 0.0
    df['holiday_strength'] = df.index.to_series().apply(get_holiday_strength)

# ---------------------------
# Rolling Forecast Cross-Validation
# ---------------------------
def rolling_forecast_cv(y, exog, initial_train_size, forecast_horizon, shift_val):
    """
    Performs time series cross-validation using an expanding window.
    For each fold, the model is trained on data up to time t and a forecast is made for the next 'forecast_horizon' step(s).
    """
    predictions = []
    actuals = []
    for i in range(initial_train_size, len(y) - forecast_horizon + 1):
        train_end = i
        test_end = i + forecast_horizon
        
        # Define training and testing sets for this fold
        y_train = y.iloc[:train_end]
        y_test = y.iloc[train_end:test_end]
        exog_train = exog.iloc[:train_end]
        exog_test = exog.iloc[train_end:test_end]
        
        # Scale exogenous features for this fold
        scaler = StandardScaler()
        exog_train_scaled = pd.DataFrame(scaler.fit_transform(exog_train), 
                                           columns=exog_train.columns, index=exog_train.index)
        exog_test_scaled = pd.DataFrame(scaler.transform(exog_test), 
                                        columns=exog_test.columns, index=exog_test.index)
        
        # Fit the SARIMAX model and forecast for the current fold
        y_pred_log = fit_sarimax_model_basic(y_train, exog_train_scaled, y_test, exog_test_scaled)
        y_pred = invert_log_transformation(y_pred_log, shift_val)
        predictions.append(y_pred)
        
        # Convert the actual log values back to original scale
        actual = np.exp(y_test) - shift_val
        actuals.append(actual)
    
    # Aggregate the forecasts and actuals from each fold
    pred_series = pd.concat(predictions)
    actual_series = pd.concat(actuals)
    return actual_series, pred_series

# ---------------------------
# Visualization Functions
# ---------------------------
def plot_correlation_heatmap(df, output_path):
    plt.figure(figsize=(12, 10))
    numerical_features = df.select_dtypes(include=[np.number]).columns
    correlation = df[numerical_features].corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, vmin=-1, vmax=1)
    plt.title("Daily Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_time_series(df, column, output_path):
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[column], color='blue', alpha=0.6)
    plt.title('Daily Power Consumption Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Power Consumption (kWh)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_forecast_comparison(forecast_df, output_path):
    plt.figure(figsize=(15, 8))
    plt.plot(forecast_df.index, forecast_df["Real_Consumption_kWh"], label="Actual", linewidth=1, color="blue", alpha=0.6)
    plt.plot(forecast_df.index, forecast_df["Predicted_Consumption_kWh"], label="Predicted", linewidth=1, color="red", alpha=0.6)
    plt.title("Actual vs Predicted Daily Power Consumption", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Consumption (kWh)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_residuals(forecast_df, output_path):
    plt.figure(figsize=(15, 6))
    plt.plot(forecast_df.index, forecast_df["Difference"], color='green', alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title("Daily Prediction Residuals (Predicted - Actual)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Residual", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_error_distribution(forecast_df, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(forecast_df["Difference"], bins=30, alpha=0.7, color='skyblue')
    plt.title("Distribution of Daily Prediction Errors", fontsize=14)
    plt.xlabel("Prediction Error", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_error_by_day(forecast_df, output_path):
    forecast_df.index.name = None
    forecast_df['Day_of_Week'] = forecast_df.index.dayofweek
    plt.figure(figsize=(12, 6))
    day_error = forecast_df.groupby('Day_of_Week')['Absolute_Error'].mean()
    sns.barplot(x=day_error.index, y=day_error.values)
    plt.title('Average Prediction Error by Day of Week', fontsize=14)
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_boxplot_comparison(forecast_df):
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

# ---------------------------
# Modeling & Evaluation Functions
# ---------------------------
def invert_log_transformation(y_pred_log, shift_val):
    return np.exp(y_pred_log) - shift_val

def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-10
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)

def compute_performance_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = calculate_mape(y_true.values, y_pred.values)
    smape = calculate_smape(y_true.values, y_pred.values)
    r2 = r2_score(y_true, y_pred)
    
    print("\nModel Performance Metrics (Original Scale):")
    print(f"MAE:   {mae:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAPE:  {mape:.4f}%")
    print(f"sMAPE: {smape:.4f}%")
    print(f"R²:    {r2:.4f}")
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "R2": r2}

def create_forecast_df(y_orig_test, y_pred):
    forecast_df = pd.DataFrame({
        "Real_Consumption_kWh": y_orig_test,
        "Predicted_Consumption_kWh": y_pred
    }, index=y_orig_test.index)
    forecast_df["Difference"] = forecast_df["Predicted_Consumption_kWh"] - forecast_df["Real_Consumption_kWh"]
    forecast_df["Absolute_Error"] = abs(forecast_df["Difference"])
    forecast_df["Percent_Error"] = (forecast_df["Absolute_Error"] / (forecast_df["Real_Consumption_kWh"].abs() + 1e-10)) * 100
    forecast_df = forecast_df.round(4)
    return forecast_df

# Basic SARIMAX model using fixed parameters
def fit_sarimax_model_basic(y_train, exog_train_scaled, y_test, exog_test_scaled):
    # Define fixed SARIMAX parameters
    order = (1, 1, 1)
    seasonal_order = (1, 0, 0, 7)
    
    # Create and fit the SARIMAX model with basic parameters
    model = SARIMAX(y_train, exog=exog_train_scaled, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)  # disp=False to suppress output
    
    # Forecast on the test data
    forecast_obj = model_fit.get_forecast(steps=len(y_test), exog=exog_test_scaled)
    y_pred_log = forecast_obj.predicted_mean
    y_pred_log.index = y_test.index
    return y_pred_log

# ---------------------------
# Main Routine
# ---------------------------
def main():
    # Define directories and file paths
    results_dir = "results/daily_chargers_sarimax"
    create_results_directory(results_dir)
    
    consumption_file = 'api_data/daily_cleaned_chargers.csv'
    temp_file = 'api_data/daily_temperature_data.csv'
    
    # Load consumption data and display basic information
    df = load_consumption_data(consumption_file)
    print("Dataset Information (Daily Data):")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    print(f"Total observations: {len(df)}")
    print(f"Missing values in Total_consumption: {df['Total_consumption'].isna().sum()}")
    
    # Transform target: log consumption (ensure positive values by shifting)
    shift_val = abs(df["Total_consumption"].min()) + 1  
    df["log_consumption"] = np.log(df["Total_consumption"] + shift_val)
    
    # Load, align, and merge temperature data
    temp_df = load_temperature_data(temp_file, df.index)
    df = merge_temperature_data(df, temp_df)
    print(f"\nFinal time range after merging temperature data: {df.index.min()} to {df.index.max()}")
    print(f"Total observations: {len(df)}")
    
    # Add temperature features
    add_temperature_features(df)
    
    add_terugkomdag_feature(df)
    add_cumulative_ev_phev_feature(df)

    # Create Belgian holiday calendar and add time, lag, and holiday features
    be_holidays = holidays.BE()
    add_time_features(df, be_holidays)
    add_lag_features(df, be_holidays)
    add_holiday_proximity_features(df, be_holidays)
    add_holiday_type_feature(df, be_holidays)
    
    # Drop rows with NaN values due to lags/rolling calculations
    df.dropna(inplace=True)
    
    # Visualization: Correlation heatmap and time series plot
    plot_correlation_heatmap(df, os.path.join(results_dir, "correlation_heatmap.png"))
    plot_time_series(df, "Total_consumption", os.path.join(results_dir, "time_series_plot.png"))
    
    print("\nBasic Statistics (Total Consumption):")
    print(df['Total_consumption'].describe())

    df_filtered = df.loc['2025-01-01 12:00:00':'2025-01-03 12:00:00']
    print(df_filtered)

    # Prepare data for modeling
    target = "log_consumption"
    exog_features = [
        "dayofmonth", "weekofyear", "day_of_week_sin", "day_of_week_cos", 
        "is_weekend", "is_festive", "is_summer", "is_winter", 
        "temperature_2m_mean", "temp_mean_lag_1d", "temp_mean_rolling_3d",
        "consumption_lag_1d", "consumption_lag_7d", "consumption_lag_30d", 
        "rolling_avg_3d", "rolling_std_3d",
        "days_since_holiday", "days_until_holiday", "holiday_strength",
        "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos"
        ,'is_terugkomdag','cumulative_ev_phev_count'
    ]
    
    y = df[target]
    exog = df[exog_features]
    
    # ---------------------------
    # Time Series Cross-Validation (Rolling Forecast)
    # ---------------------------
    # Set the forecast horizon to 1 day for a rolling one-step-ahead forecast.
    forecast_horizon = 1
    # Use the first 80% of data as the initial training window.
    initial_train_size = int(0.80 * len(df))
    
    actual_series, pred_series = rolling_forecast_cv(y, exog, initial_train_size, forecast_horizon, shift_val)
    
    # Compute performance metrics on the aggregated forecasts (original scale)
    metrics = compute_performance_metrics(actual_series, pred_series)
    
    # Create forecast DataFrame and save predictions
    forecast_df = create_forecast_df(actual_series, pred_series)
    forecast_csv_path = os.path.join(results_dir, "predicted_values_kwh_cv.csv")
    forecast_df.to_csv(forecast_csv_path)
    print(f"\nCross-validated predicted values have been saved to '{forecast_csv_path}'")
    
    # Additional visualizations using cross-validated forecasts
    plot_forecast_comparison(forecast_df, os.path.join(results_dir, "forecast_comparison_cv.png"))
    plot_residuals(forecast_df, os.path.join(results_dir, "residuals_cv.png"))
    plot_error_distribution(forecast_df, os.path.join(results_dir, "error_distribution_cv.png"))
    plot_error_by_day(forecast_df, os.path.join(results_dir, "error_by_day_cv.png"))
    
    print("\nAll daily visualizations (cross-validation) have been saved in the 'results/daily_chargers_sarimax' directory.")
    print("Daily forecasting cross-validation analysis complete.")
    
    # Boxplot: Comparison of Real vs Forecasted Values
    plot_boxplot_comparison(forecast_df)

if __name__ == '__main__':
    main()
