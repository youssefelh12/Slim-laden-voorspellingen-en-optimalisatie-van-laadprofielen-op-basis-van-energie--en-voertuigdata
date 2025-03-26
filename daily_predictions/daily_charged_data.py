import os
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import calendar
from sklearn.preprocessing import RobustScaler
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.special import inv_boxcox
from scipy.stats import boxcox

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create results directory if it doesn't exist
RESULTS_DIR = "results/daily_cleaned_charger"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    # Load consumption data
    df = pd.read_csv('api_data/daily_cleaned_chargers.csv')
    df.set_index("Day", inplace=True)
    df.index = pd.to_datetime(df.index)
    df_daily = df.copy()
    logging.info(f"Consumption data from {df_daily.index.min()} to {df_daily.index.max()}, observations: {len(df_daily)}")
    
    # Load temperature data
    temp_df = pd.read_csv('api_data/daily_temperature_data.csv')
    temp_df['date'] = pd.to_datetime(temp_df['date'], utc=True)
    temp_df.set_index('date', inplace=True)
    temp_df.index = temp_df.index.tz_convert('Europe/Brussels').tz_localize(None)
    
    # Align temperature data to consumption dates
    full_dates = df_daily.index
    temp_df = temp_df.reindex(full_dates)
    temp_df['temperature_2m_mean'].ffill(inplace=True)
    temp_df['temperature_2m_mean'].bfill(inplace=True)
    
    # Merge temperature data
    df_daily = df_daily.merge(
        temp_df[['temperature_2m_mean']],
        left_index=True,
        right_index=True,
        how='left'
    )
    logging.info(f"Final data from {df_daily.index.min()} to {df_daily.index.max()}, observations: {len(df_daily)}")
    return df_daily

def feature_engineering(df_daily):

    # Zorg ervoor dat de consumptiewaarden strikt positief zijn
    shift_val = abs(df_daily["Total_consumption"].min()) + 1
    df_daily["Total_consumption_shifted"] = df_daily["Total_consumption"] + shift_val

    # Box-Cox transformatie voor een betere normalisatie
    consumption_bc, lam = boxcox(df_daily["Total_consumption_shifted"])
    df_daily["consumption_boxcox"] = consumption_bc
    logging.info(f"Box-Cox lambda: {lam}")

    # Temperatuurfeatures: lags en rollend gemiddelde
    df_daily['temp_mean_lag_1d'] = df_daily['temperature_2m_mean'].shift(1)
    df_daily['temp_mean_rolling_3d'] = df_daily['temperature_2m_mean'].rolling(window=3).mean()

    # Tijdgebaseerde features
    df_daily['year'] = df_daily.index.year
    df_daily['month'] = df_daily.index.month
    df_daily['dayofyear'] = df_daily.index.dayofyear
    df_daily['dayofmonth'] = df_daily.index.day
    df_daily['weekofyear'] = df_daily.index.isocalendar().week.astype(int)
    df_daily['day_of_week'] = df_daily.index.dayofweek

    # Categorieën: weekend flag
    df_daily['is_weekend'] = df_daily['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Holiday features met Belgische feestdagen
    be_holidays = holidays.BE()
    df_daily['is_festive'] = df_daily.index.to_series().apply(lambda x: 1 if x.date() in be_holidays else 0)
    
    # Holiday proximity features: dagen tot de volgende feestdag en dagen sinds de laatste feestdag
    df_daily['days_to_next_holiday'] = df_daily.index.to_series().apply(
        lambda x: min([(h - x.date()).days for h in sorted(be_holidays) if h >= x.date()] or [0])
    )
    df_daily['days_since_last_holiday'] = df_daily.index.to_series().apply(
        lambda x: min([(x.date() - h).days for h in sorted(be_holidays) if h <= x.date()] or [0])
    )

    # Seizoensgebonden flags
    df_daily['is_summer'] = df_daily.index.month.isin([6, 7, 8]).astype(int)
    df_daily['is_winter'] = df_daily.index.month.isin([12, 1, 2]).astype(int)

    # Cyclische features voor dag van de week en maand
    df_daily['day_of_week_sin'] = np.sin(2 * np.pi * df_daily['day_of_week'] / 7)
    df_daily['day_of_week_cos'] = np.cos(2 * np.pi * df_daily['day_of_week'] / 7)
    df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['month'] / 12)
    df_daily['month_cos'] = np.cos(2 * np.pi * df_daily['month'] / 12)

    # Lag features voor de consumptie
    for lag in [1, 7, 14, 21, 30, 365]:
        df_daily[f'consumption_lag_{lag}d'] = df_daily["Total_consumption"].shift(lag)

    # Rollende statistieken
    df_daily['rolling_avg_3d'] = df_daily["Total_consumption"].rolling(window=3).mean()
    df_daily['rolling_std_3d'] = df_daily["Total_consumption"].rolling(window=3).std()
    df_daily['rolling_avg_7d'] = df_daily["Total_consumption"].rolling(window=7).mean()

    # Verwijder rijen met NaN waarden door de lag/rolling berekeningen
    df_daily.dropna(inplace=True)

    # Retourneer de bewerkte dataframe samen met de shift-waarde en Box-Cox lambda (nodig voor inverse transformatie)
    return df_daily, shift_val, lam


def scale_exogenous(exog_train, exog_test):
    scaler = RobustScaler()
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
    # Save the scaler for future use
    with open(os.path.join(RESULTS_DIR, 'robust_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    return exog_train_scaled, exog_test_scaled

def train_sarimax(y_train, exog_train_scaled):
    logging.info("Finding optimal ARIMA parameters with expanded search space...")
    try:
        model_auto = auto_arima(
            y_train,
            exogenous=exog_train_scaled,
            seasonal=True,
            m=7,  # Weekly seasonality
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            max_order=5,  # Expanded order search
            max_p=3,
            max_q=3,
            max_d=2,
            max_P=2,
            max_Q=2,
            max_D=1,
            start_p=1,
            start_q=1,
            start_P=1,
            start_Q=1,
            information_criterion='aic',
            maxiter=100,
            method='lbfgs',
            n_jobs=1
        )
        order = model_auto.order
        seasonal_order = model_auto.seasonal_order
        logging.info(f"Optimal ARIMA order: {order}")
        logging.info(f"Optimal Seasonal order: {seasonal_order}")
    except Exception as e:
        logging.error(f"Auto ARIMA failed: {e}. Using default parameters.")
        order = (1, 1, 1)
        seasonal_order = (1, 0, 0, 7)
    
    logging.info("Fitting SARIMAX model...")
    model = SARIMAX(
        y_train,
        exog=exog_train_scaled,
        order=order,
        seasonal_order=seasonal_order,
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False, maxiter=200, method='lbfgs')
    
    # Save the fitted model
    with open(os.path.join(RESULTS_DIR, 'sarimax_model.pkl'), 'wb') as f:
        pickle.dump(model_fit, f)
    logging.info("Model fitting complete.")
    return model_fit

def forecast_and_evaluate(model_fit, y_test, exog_test_scaled, shift_val, lam):
    logging.info("Generating forecasts...")
    forecast_obj = model_fit.get_forecast(steps=len(y_test), exog=exog_test_scaled)
    y_pred_transformed = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()  # Confidence intervals for uncertainty
    
    # Inverse the Box-Cox transformation and subtract shift
    y_pred = inv_boxcox(y_pred_transformed, lam) - shift_val
    
    # Invert the target transformation for y_test evaluation
    y_test_inverted = inv_boxcox(y_test, lam) - shift_val
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test_inverted, y_pred)
    mse = mean_squared_error(y_test_inverted, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inverted, y_pred)
    
    def calculate_mape(y_true, y_pred):
        epsilon = 1e-10
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    def calculate_smape(y_true, y_pred):
        denominator = np.abs(y_true) + np.abs(y_pred) + 1e-10
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)
    
    mape = calculate_mape(y_test_inverted, y_pred)
    smape = calculate_smape(y_test_inverted, y_pred)
    
    logging.info("Performance Metrics:")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAPE: {mape:.4f}%")
    logging.info(f"sMAPE: {smape:.4f}%")
    logging.info(f"R²: {r2:.4f}")
    
    # Build a DataFrame for forecast results
    forecast_df = pd.DataFrame({
        "Real_Consumption_kWh": y_test_inverted,
        "Predicted_Consumption_kWh": y_pred
    }, index=y_test.index)
    forecast_df["Difference"] = forecast_df["Predicted_Consumption_kWh"] - forecast_df["Real_Consumption_kWh"]
    forecast_df["Absolute_Error"] = forecast_df["Difference"].abs()
    forecast_df["Percent_Error"] = (forecast_df["Absolute_Error"] / (forecast_df["Real_Consumption_kWh"].abs() + 1e-10)) * 100
    forecast_df = forecast_df.round(4)
    forecast_df.to_csv(os.path.join(RESULTS_DIR, "predicted_values_kwh.csv"))
    logging.info("Forecast results saved.")
    return forecast_df, conf_int

def plot_diagnostics(forecast_df, model_fit):
    # Forecast comparison plot
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
    plt.savefig(os.path.join(RESULTS_DIR, "forecast_comparison.png"))
    plt.close()
    
    # Residual plot
    residuals = model_fit.resid
    plt.figure(figsize=(15,6))
    plt.plot(residuals)
    plt.title("Model Residuals")
    plt.xlabel("Time")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_residuals.png"))
    plt.close()
    
    # ACF and PACF plots for residuals
    plt.figure(figsize=(12,6))
    plot_acf(residuals, lags=40)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residuals_acf.png"))
    plt.close()
    
    plt.figure(figsize=(12,6))
    plot_pacf(residuals, lags=40)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residuals_pacf.png"))
    plt.close()
    
    # Error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(forecast_df["Difference"], bins=30, kde=True)
    plt.title("Distribution of Daily Prediction Errors", fontsize=14)
    plt.xlabel("Prediction Error", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"))
    plt.close()
    
    # Error by day of week
    forecast_df['Day_of_Week'] = forecast_df.index.dayofweek
    plt.figure(figsize=(12,6))
    day_error = forecast_df.groupby('Day_of_Week')['Absolute_Error'].mean()
    sns.barplot(x=day_error.index, y=day_error.values)
    plt.title('Average Prediction Error by Day of Week (0=Mon, 6=Sun)', fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_by_day.png"))
    plt.close()

def main():
    # Load and preprocess data
    df_daily = load_data()
    df_daily, shift_val, lam = feature_engineering(df_daily)
    
    # Define target and exogenous features (using the Box-Cox transformed consumption)
    target = "consumption_boxcox"
    y = df_daily[target]
    exog_features = [
        "dayofmonth", "weekofyear", "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos", "is_weekend", "is_festive",
        "is_summer", "is_winter", "days_to_next_holiday", "days_since_last_holiday",
        "temperature_2m_mean", "temp_mean_lag_1d", "temp_mean_rolling_3d",
        "consumption_lag_1d", "consumption_lag_7d", "consumption_lag_14d", 
        "consumption_lag_21d", "consumption_lag_30d", "consumption_lag_365d",
        "rolling_avg_3d", "rolling_std_3d", "rolling_avg_7d"
    ]
    exog = df_daily[exog_features]
    
    # Split data into training and testing sets (80/20 split)
    split_index = int(0.80 * len(df_daily))
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    exog_train, exog_test = exog.iloc[:split_index], exog.iloc[split_index:]
    
    # Scale exogenous features using RobustScaler
    exog_train_scaled, exog_test_scaled = scale_exogenous(exog_train, exog_test)
    
    # Train SARIMAX model with optimal parameters from auto_arima
    model_fit = train_sarimax(y_train, exog_train_scaled)
    
    # Generate forecasts and evaluate performance
    forecast_df, conf_int = forecast_and_evaluate(model_fit, y_test, exog_test_scaled, shift_val, lam)
    
    # Generate diagnostic plots
    plot_diagnostics(forecast_df, model_fit)
    
    # Feature importance (by correlation with target)
    feature_importance = pd.DataFrame({
        'Feature': exog_features,
        'Correlation': [abs(df_daily[feature].corr(df_daily[target])) for feature in exog_features]
    }).sort_values('Correlation', ascending=False)
    logging.info("Top 10 Feature Importance (by correlation):")
    logging.info(feature_importance.head(10))
    feature_importance.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
    
    # Additional: Boxplot comparing actual vs predicted consumption
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
    plt.savefig(os.path.join(RESULTS_DIR, "boxplot_forecast.png"))
    plt.close()
    
    logging.info("Daily forecasting analysis complete.")

if __name__ == "__main__":
    main()
