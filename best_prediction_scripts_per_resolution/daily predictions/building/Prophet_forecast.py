import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import holidays

# ---------------------------
# Directory & Data Loading Functions
# ---------------------------
def create_results_directory(path):
    os.makedirs(path, exist_ok=True)


def load_consumption_data(file_path):
    df = pd.read_csv(file_path)
    df['Day'] = pd.to_datetime(df['Day'])
    df.set_index('Day', inplace=True)
    return df


def load_temperature_data(file_path, index_dates):
    temp_df = pd.read_csv(file_path)
    temp_df['date'] = pd.to_datetime(temp_df['date'], utc=True)
    temp_df.set_index('date', inplace=True)
    temp_df.index = temp_df.index.tz_convert('Europe/Brussels').tz_localize(None)
    temp_df = temp_df.reindex(index_dates)
    temp_df['temperature_2m_mean'].ffill(inplace=True)
    temp_df['temperature_2m_mean'].bfill(inplace=True)
    return temp_df


def merge_temperature_data(consumption_df, temp_df):
    return consumption_df.merge(
        temp_df[['temperature_2m_mean']],
        left_index=True,
        right_index=True,
        how='left'
    )

# ---------------------------
# Feature Engineering
# ---------------------------
def add_time_and_holiday_features(df):
    be_holidays = holidays.BE()
    df['is_weekend'] = df.index.weekday >= 5
    df['is_festive'] = df.index.to_series().isin(be_holidays)
    df['is_summer'] = df.index.month.isin([6,7,8])
    df['is_winter'] = df.index.month.isin([12,1,2])

# ---------------------------
# Metrics Functions
# ---------------------------
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
    mae_pct = (mae / (np.mean(y_true) + 1e-10)) * 100

    print("\nModel Performance Metrics (Original Scale):")
    print(f"MAE:       {mae:.4f}")
    print(f"MAE %:     {mae_pct:.4f}%")
    print(f"MSE:       {mse:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAPE:      {mape:.4f}%")
    print(f"sMAPE:     {smape:.4f}%")
    print(f"R²:        {r2:.4f}")

    return {
        'MAE': mae,
        'MAE_percent': mae_pct,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAPE': smape,
        'R2': r2
    }

# ---------------------------
# Main Routine
# ---------------------------
def main():
    results_dir = 'results/prophet'
    create_results_directory(results_dir)

    # Load data
    cons_file = 'api_data/daily_cleaned3.csv'
    temp_file = 'api_data/daily_temperature_data.csv'
    df = load_consumption_data(cons_file)

    # Log transform target
    shift_val = abs(df['Total_consumption'].min()) + 1
    df['log_consumption'] = np.log(df['Total_consumption'] + shift_val)

    # Merge temperature and features
    temp_df = load_temperature_data(temp_file, df.index)
    df = merge_temperature_data(df, temp_df)
    add_time_and_holiday_features(df)
    df.dropna(inplace=True)

    # Prepare Prophet dataframe
    prophet_df = df.reset_index()[['Day', 'log_consumption', 'temperature_2m_mean',
                                    'is_weekend', 'is_festive', 'is_summer', 'is_winter']]
    prophet_df.rename(columns={'Day': 'ds', 'log_consumption': 'y'}, inplace=True)

    # Split train/test
    split = int(0.8 * len(prophet_df))
    train_df = prophet_df.iloc[:split]
    test_df = prophet_df.iloc[split:]

    # Initialize and add regressors
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    for reg in ['temperature_2m_mean', 'is_weekend', 'is_festive', 'is_summer', 'is_winter']:
        model.add_regressor(reg)

    # Fit model
    model.fit(train_df)

    # Forecast
    future = model.make_future_dataframe(periods=len(test_df), freq='D')
    future = future.merge(
        prophet_df[['ds', 'temperature_2m_mean', 'is_weekend', 'is_festive', 'is_summer', 'is_winter']],
        on='ds', how='left'
    )
    forecast = model.predict(future)
    forecast_test = forecast.set_index('ds').loc[test_df['ds']]

    # Invert log transform
    y_pred = np.exp(forecast_test['yhat']) - shift_val
    y_true = np.exp(test_df['y']) - shift_val

    # Compute and print metrics
    metrics = compute_performance_metrics(y_true, y_pred)

    # Save results
    result_df = pd.DataFrame({'Real': y_true, 'Predicted': y_pred}, index=test_df['ds'])
    result_df.to_csv(os.path.join(results_dir, 'prophet_forecast.csv'))

    # Plot only forecast period comparison
    plt.figure(figsize=(12,6))
    # Plot actual values for forecast period only
    plt.plot(result_df.index, df.loc[result_df.index, 'Total_consumption'], label='Actual')
    # Plot predictions
    plt.plot(result_df.index, result_df['Predicted'], label='Predicted')
    plt.legend()
    plt.title('Prophet Forecast (Test Period Only)')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (kWh)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'prophet_forecast_plot.png'))
    plt.close()

    print('\nProphet modeling complete.')

if __name__ == '__main__':
    main()
