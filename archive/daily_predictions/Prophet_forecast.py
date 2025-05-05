import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import holidays

# ---------------------------
# Configuration
# ---------------------------
CONSUMPTION_FILE = 'api_data/daily_cleaned3.csv'
TARGET_COL = 'Total_consumption'
LOOKBACK_DAYS = 365      # how many days of history to use
FORECAST_DAYS = 3      # how many days to forecast
SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL = (1, 0, 0, 7)

# ---------------------------
# Data Loading
# ---------------------------
def load_daily_data(path):
    df = pd.read_csv(path, parse_dates=['Day'])
    df.set_index('Day', inplace=True)
    return df[[TARGET_COL]].asfreq('D')

# ---------------------------
# Prophet Forecast
# ---------------------------
def prophet_forecast(df, lookback, horizon):
    df_recent = df.iloc[-(lookback + horizon):].reset_index().rename(columns={'Day':'ds', TARGET_COL:'y'})
    model = Prophet(daily_seasonality=True,
                    weekly_seasonality=True,
                    changepoint_prior_scale=0.05)
    model.fit(df_recent[['ds','y']])
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return forecast[['ds','yhat']].set_index('ds').iloc[-horizon:]

# ---------------------------
# SARIMAX Forecast
# ---------------------------
def sarimax_forecast(df, lookback, horizon):
    # restrict to lookback window
    df_train = df.iloc[-(lookback + horizon):-horizon]
    df_test = df.iloc[-horizon:]

    # exogenous features
    be_holidays = holidays.BE()
    def make_exog(series):
        ex = pd.DataFrame(index=series.index)
        ex['day_of_week'] = series.index.dayofweek
        ex['is_weekend'] = (ex['day_of_week'] >= 5).astype(int)
        ex['is_holiday'] = series.index.to_series().isin(be_holidays).astype(int)
        return ex

    exog_train = make_exog(df_train[TARGET_COL])
    exog_test = make_exog(df_test[TARGET_COL])

    # fit
    model = SARIMAX(df_train[TARGET_COL], exog=exog_train,
                    order=SARIMAX_ORDER, seasonal_order=SARIMAX_SEASONAL)
    fit = model.fit(disp=False)
    pred_obj = fit.get_forecast(steps=horizon, exog=exog_test)
    yhat = pred_obj.predicted_mean
    return yhat, df_test[TARGET_COL]

# ---------------------------
# Main
# ---------------------------
def main():
    df = load_daily_data(CONSUMPTION_FILE)

    # Prophet
    prop_pred = prophet_forecast(df, LOOKBACK_DAYS, FORECAST_DAYS)

    # SARIMAX
    sarimax_pred, actual = sarimax_forecast(df, LOOKBACK_DAYS, FORECAST_DAYS)

    # Metrics comparison
    mae = mean_absolute_error(actual, sarimax_pred)
    rmse = np.sqrt(mean_squared_error(actual, sarimax_pred))
    r2 = r2_score(actual, sarimax_pred)
    print(f"SARIMAX → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # Plotting
    plt.figure(figsize=(12,5))
    # history last lookback window
    history = df.iloc[-7:-FORECAST_DAYS]
    plt.plot(history.index, history[TARGET_COL], label=f'History ({LOOKBACK_DAYS}d)')
    # actual test
    plt.plot(actual.index, actual.values, 'o-', label='Actual')
    # sarimax forecast
    plt.plot(sarimax_pred.index, sarimax_pred.values, 'x--', label='SARIMAX Forecast')

    plt.title(f'SARIMAX {FORECAST_DAYS}-Day Forecast vs Actual', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Daily Consumption (kWh)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
