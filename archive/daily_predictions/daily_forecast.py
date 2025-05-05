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
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

def add_lag_features(df, be_holidays):
    # Basic lag features (removed 365-day lag because the dataset is too short)
    df['consumption_lag_1d'] = df['Total_consumption'].shift(1)
    df['consumption_lag_7d'] = df['Total_consumption'].shift(7)
    df['consumption_lag_30d'] = df['Total_consumption'].shift(30)
    # Additional lags and rolling statistics
    df['consumption_lag_14d'] = df['Total_consumption'].shift(14)
    df['consumption_lag_21d'] = df['Total_consumption'].shift(21)
    df['rolling_avg_3d'] = df['Total_consumption'].rolling(window=3).mean()
    df['rolling_std_3d'] = df['Total_consumption'].rolling(window=3).std()
    
    # Holiday-related features
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

# ---------------------------
# Modeling & Evaluation Functions
# ---------------------------
def invert_log_transformation(y_pred_log, shift_val):
    return np.exp(y_pred_log) - shift_val

# ---------------------------
# Main Routine for Production Forecasting
# ---------------------------
def main():
    # Define directories and file paths
    results_dir = "results/production_forecast"
    create_results_directory(results_dir)
    
    consumption_file = 'api_data/daily_building_consumption_0624.csv'
    temp_file = 'api_data/daily_temperature_data.csv'
    
    # Load consumption data and merge temperature data
    df = load_consumption_data(consumption_file)
    print("Dataset Information (Daily Data):")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    print(f"Total observations: {len(df)}")
    
    # Transform target: log consumption (ensure positive values by shifting)
    shift_val = abs(df["Total_consumption"].min()) + 1  
    df["log_consumption"] = np.log(df["Total_consumption"] + shift_val)
    
    # Load temperature data and merge
    temp_df = load_temperature_data(temp_file, df.index)
    df = merge_temperature_data(df, temp_df)
    print(f"\nFinal time range after merging temperature data: {df.index.min()} to {df.index.max()}")
    
    # Add feature engineering
    be_holidays = holidays.BE()
    add_temperature_features(df)
    add_time_features(df, be_holidays)
    add_lag_features(df, be_holidays)
    
    # Drop any rows with NaN (from lag/rolling calculations)
    df.dropna(inplace=True)
    
    # Define exogenous features (you can adjust this list based on prior feature selection)
    exog_features = [
        "dayofmonth", "weekofyear", "day_of_week_sin", "day_of_week_cos", 
        "is_weekend", "is_festive", "is_summer", "is_winter", 
        "temperature_2m_mean", "temp_mean_lag_1d", "temp_mean_rolling_3d",
        "consumption_lag_1d", "consumption_lag_7d", 
        "rolling_avg_3d", "rolling_std_3d"
    ]
    
    # Prepare modeling variables using 100% of the data
    target = "log_consumption"
    y_full = df[target]
    exog_full = df[exog_features]
    
    # Normalize exogenous features based on historical data
    scaler = StandardScaler()
    exog_full_scaled = pd.DataFrame(
        scaler.fit_transform(exog_full),
        columns=exog_full.columns,
        index=exog_full.index
    )
    
    # Fit the SARIMAX model using all historical data
    order = (1, 1, 1)
    seasonal_order = (1, 0, 0, 7)
    model = SARIMAX(y_full, exog=exog_full_scaled, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    
    # ---- Forecast 1 Day Ahead ----
    # Define forecast day as one day after the last available date
    forecast_day = df.index.max() + pd.Timedelta(days=1)
    print(f"\nForecasting for: {forecast_day.date()}")
    
    # Build exogenous features for the forecast day
    exog_forecast = {}
    exog_forecast["dayofmonth"] = forecast_day.day
    exog_forecast["weekofyear"] = int(forecast_day.isocalendar().week)
    exog_forecast["day_of_week_sin"] = np.sin(2 * np.pi * forecast_day.weekday() / 7)
    exog_forecast["day_of_week_cos"] = np.cos(2 * np.pi * forecast_day.weekday() / 7)
    exog_forecast["is_weekend"] = 1 if forecast_day.weekday() >= 5 else 0
    exog_forecast["is_festive"] = 1 if forecast_day in be_holidays else 0
    exog_forecast["is_summer"] = 1 if forecast_day.month in [6,7,8] else 0
    exog_forecast["is_winter"] = 1 if forecast_day.month in [12,1,2] else 0
    # For temperature, assume the forecast temperature equals the last observed value
    exog_forecast["temperature_2m_mean"] = df["temperature_2m_mean"].iloc[-1]
    # For temp_mean_lag_1d, use the last available temperature
    exog_forecast["temp_mean_lag_1d"] = df["temperature_2m_mean"].iloc[-1]
    # For temp_mean_rolling_3d, compute the mean of the last three days' temperature
    exog_forecast["temp_mean_rolling_3d"] = df["temperature_2m_mean"].iloc[-3:].mean()
    # For consumption lags, use historical data:
    exog_forecast["consumption_lag_1d"] = df["Total_consumption"].iloc[-1]
    # For consumption_lag_7d, try to get the consumption from 7 days ago
    lag7_date = forecast_day - pd.Timedelta(days=7)
    if lag7_date in df.index:
        exog_forecast["consumption_lag_7d"] = df.loc[lag7_date, "Total_consumption"]
    else:
        exog_forecast["consumption_lag_7d"] = df["Total_consumption"].iloc[-1]
    # For rolling average and std over the last 3 days
    exog_forecast["rolling_avg_3d"] = df["Total_consumption"].iloc[-3:].mean()
    exog_forecast["rolling_std_3d"] = df["Total_consumption"].iloc[-3:].std()
    
    # Create a DataFrame for the forecast exogenous features and scale them
    exog_forecast_df = pd.DataFrame(exog_forecast, index=[forecast_day])
    exog_forecast_scaled = pd.DataFrame(
        scaler.transform(exog_forecast_df),
        columns=exog_forecast_df.columns,
        index=exog_forecast_df.index
    )
    
    # Forecast one day ahead using the fitted model
    forecast_obj = model_fit.get_forecast(steps=1, exog=exog_forecast_scaled)
    y_pred_log = forecast_obj.predicted_mean
    y_pred = invert_log_transformation(y_pred_log, shift_val)
    
    print(f"\nForecasted consumption for {forecast_day.date()}: {y_pred.iloc[0]:.2f} kWh")
    
    # Optionally, you can save the forecast to a CSV file
    forecast_df = pd.DataFrame({
        "Forecast_Day": [forecast_day.date()],
        "Predicted_Consumption_kWh": [y_pred.iloc[0]]
    })
    forecast_csv_path = os.path.join(results_dir, "one_day_forecast.csv")
    forecast_df.to_csv(forecast_csv_path, index=False)
    print(f"Forecast saved to: {forecast_csv_path}")

if __name__ == '__main__':
    main()
