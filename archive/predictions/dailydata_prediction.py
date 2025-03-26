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

# ---------------------------
# Utility Functions
# ---------------------------
def create_results_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def load_and_preprocess_data(file_path):
    """
    Load the CSV and preprocess the data.
    Expected CSV columns: Day, Chargers, Chargers achteraan, Grid Organi lbc, Solar
    """
    df = pd.read_csv(file_path)
    df.set_index("Day", inplace=True)
    df.index = pd.to_datetime(df.index)
    # Use only the "Grid Organi lbc" column for total consumption
    df['Total_consumption'] = df['Grid Organi lbc']
    # Drop unused columns
    df = df.drop(['Chargers', 'Chargers achteraan', 'Solar', 'Grid Organi lbc'], axis=1)
    return df.copy()

def print_dataset_info(df):
    print("Dataset Information (Daily Data):")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    print(f"Total observations: {len(df)}")
    print(f"Missing values: {df['Total_consumption'].isna().sum()}")

def transform_target(df):
    """
    Compute log consumption ensuring a positive shift.
    Returns the shift value and modified DataFrame.
    """
    shift_val = abs(df["Total_consumption"].min()) + 1  
    df["log_consumption"] = np.log(df["Total_consumption"] + shift_val)
    return shift_val, df

def feature_engineering(df):
    """
    Add time, seasonal, lag, rolling features, and a rolling kurtosis feature.
    """
    be_holidays = holidays.BE()  # Belgian holidays
    
    # Basic time features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Categorical features
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_festive'] = df.index.to_series().apply(lambda x: 1 if x in be_holidays else 0)
    
    # Seasonal features
    df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)
    df['is_winter'] = df.index.month.isin([12, 1, 2]).astype(int)
    
    # Cyclical features for day of week
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Lagged features (avoid data leakage)
    df['consumption_lag_1d'] = df['Total_consumption'].shift(1)   # 1-day lag
    df['consumption_lag_7d'] = df['Total_consumption'].shift(7)   # 7-day lag
    
    # Additional lag features for monthly and yearly
    df['consumption_lag_30d'] = df['Total_consumption'].shift(30)   # Approx. monthly lag
    df['consumption_lag_365d'] = df['Total_consumption'].shift(365)   # Yearly lag
    
    # Advanced Lag Features
    df['consumption_lag_14d'] = df['Total_consumption'].shift(14)   # 14-day lag
    df['consumption_lag_21d'] = df['Total_consumption'].shift(21)   # 21-day lag
    df['rolling_avg_3d'] = df['Total_consumption'].rolling(window=3).mean()  # 3-day moving average
    df['rolling_std_3d'] = df['Total_consumption'].rolling(window=3).std()   # 3-day rolling standard deviation

    
    # Drop rows with NaN values resulting from lag and rolling calculations
    df.dropna(inplace=True)
    return df

def plot_correlation_heatmap(df, save_path):
    plt.figure(figsize=(12, 10))
    numerical_features = df.select_dtypes(include=[np.number]).columns
    correlation = df[numerical_features].corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, vmin=-1, vmax=1)
    plt.title("Daily Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_time_series(df, save_path):
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Total_consumption'], color='blue', alpha=0.6)
    plt.title('Daily Power Consumption Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Power Consumption (kWh)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_basic_statistics(df):
    print("\nBasic Statistics (Total Consumption):")
    print(df['Total_consumption'].describe())

def prepare_model_data(df, target, exog_features, train_ratio=0.8):
    """
    Prepare target and exogenous features for modeling, then split into training and test sets.
    """
    y = df[target]
    exog = df[exog_features]
    y_orig = df["Total_consumption"]
    split_index = int(train_ratio * len(df))
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    exog_train, exog_test = exog.iloc[:split_index], exog.iloc[split_index:]
    y_orig_train, y_orig_test = y_orig.iloc[:split_index], y_orig.iloc[split_index:]
    print(f"Training data from {df.index[0].date()} to {df.index[split_index-1].date()}")
    print(f"Testing data from {df.index[split_index].date()} to {df.index[-1].date()}")
    return y, y_train, y_test, exog, exog_train, exog_test, y_orig_train, y_orig_test

def scale_features(exog_train, exog_test):
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
    return exog_train_scaled, exog_test_scaled, scaler

def find_optimal_arima(y_train, exog_train_scaled):
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
    return order, seasonal_order

def fit_sarimax_model(y_train, exog_train_scaled, order, seasonal_order, y_test, exog_test_scaled):
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
        forecast_obj = model_fit.get_forecast(steps=len(y_test), exog=exog_test_scaled)
        y_pred_log = forecast_obj.predicted_mean
        y_pred_log.index = y_test.index
    except MemoryError:
        print("Memory error during SARIMAX fitting. Using a simplified approach...")
        y_pred_log = []
        window_size = 2000
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
            model_fit = subset_fit
        y_pred_log = pd.Series(y_pred_log, index=y_test.index)
    return y_pred_log, model_fit

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
    return mae, mse, rmse, mape, smape, r2

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

def save_forecast_df(forecast_df, filepath):
    forecast_df.to_csv(filepath)
    print(f"\nPredicted values have been saved to '{filepath}'")

def feature_importance_analysis(df, exog_features, target, filepath):
    feature_importance = pd.DataFrame({
        'Feature': exog_features,
        'Correlation': [abs(df[feature].corr(df[target])) for feature in exog_features]
    })
    feature_importance = feature_importance.sort_values('Correlation', ascending=False)
    print("\nFeature Importance by Correlation with Target (Log Scale):")
    print(feature_importance.head(10))
    feature_importance.to_csv(filepath, index=False)

def visualize_forecast_comparison(forecast_df, results_dir):
    # Forecast Comparison Plot
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
    plt.savefig(os.path.join(results_dir, "forecast_comparison.png"))
    plt.close()
    
    # Residuals Plot
    plt.figure(figsize=(15, 6))
    plt.plot(forecast_df.index, forecast_df["Difference"], color='green', alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title("Daily Prediction Residuals (Predicted - Actual)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Residual", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "residuals.png"))
    plt.close()
    
    # Error Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(forecast_df["Difference"], bins=30, alpha=0.7, color='skyblue')
    plt.title("Distribution of Daily Prediction Errors", fontsize=14)
    plt.xlabel("Prediction Error", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "error_distribution.png"))
    plt.close()
    
    # Error by Day of Week Plot
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
    plt.savefig(os.path.join(results_dir, "error_by_day.png"))
    plt.close()
    
    print("\nAll daily visualizations have been saved in the '{}' directory.".format(results_dir))

def visualize_boxplot_comparison(forecast_df):
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

def forecast_one_day(model_fit, df, scaler, exog_features, shift_val):
    """
    Forecast one day into the future.
    
    This function builds the exogenous features for the next day based on the last
    available date in df, scales them, and then obtains a one-step forecast using the
    provided SARIMAX model.
    """
    next_date = df.index.max() + pd.Timedelta(days=1)
    be_holidays = holidays.BE()
    day_of_week = next_date.dayofweek
    month = next_date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_festive = 1 if next_date in be_holidays else 0
    is_summer = 1 if month in [6, 7, 8] else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Helper function to get lag values
    def get_lag_value(df, next_date, days):
        target_date = next_date - pd.Timedelta(days=days)
        if target_date in df.index:
            return df.loc[target_date, 'Total_consumption']
        else:
            return df['Total_consumption'].iloc[-1]
    
    last_consumption = df['Total_consumption'].iloc[-1]
    
    # Build the exogenous feature dictionary
    exog_dict = {
        "day_of_week_sin": day_of_week_sin,
        "day_of_week_cos": day_of_week_cos,
        "is_weekend": is_weekend,
        "is_festive": is_festive,
        "is_summer": is_summer,
        "is_winter": is_winter,
        "consumption_lag_1d": last_consumption,
        "consumption_lag_7d": get_lag_value(df, next_date, 7),
        "consumption_lag_30d": get_lag_value(df, next_date, 30),
        "consumption_lag_365d": get_lag_value(df, next_date, 365),
        "consumption_lag_14d": get_lag_value(df, next_date, 14),
        "consumption_lag_21d": get_lag_value(df, next_date, 21),
        "rolling_avg_3d": df['Total_consumption'].iloc[-3:].mean(),
        "rolling_std_3d": df['Total_consumption'].iloc[-3:].std()
    }
    
    # Create a DataFrame and ensure the column order matches the training exogenous features
    new_exog = pd.DataFrame(exog_dict, index=[next_date])
    new_exog = new_exog[exog_features]
    
    # Scale the new exogenous features using the previously fitted scaler
    new_exog_scaled = pd.DataFrame(scaler.transform(new_exog),
                                   columns=new_exog.columns,
                                   index=new_exog.index)
    # Forecast one step ahead on the log-transformed scale
    forecast_obj = model_fit.get_forecast(steps=1, exog=new_exog_scaled)
    y_pred_log = forecast_obj.predicted_mean.iloc[0]
    # Invert the log transformation to return forecast on the original scale
    y_pred = np.exp(y_pred_log) - shift_val
    return next_date, y_pred

# ---------------------------
# Main Function
# ---------------------------
def main():
    results_dir = "results/daily3"
    create_results_dir(results_dir)
    
    # Load and preprocess data
    data_path = 'api_data/aggregated_daily_measurements.csv'
    df = load_and_preprocess_data(data_path)
    print_dataset_info(df)
    
    # Transform target
    shift_val, df = transform_target(df)
    
    # Feature engineering (with added rolling kurtosis)
    df = feature_engineering(df)
    
    # Visualizations: Correlation Heatmap & Time Series Plot
    plot_correlation_heatmap(df, os.path.join(results_dir, "correlation_heatmap.png"))
    plot_time_series(df, os.path.join(results_dir, "time_series_plot.png"))
    print_basic_statistics(df)
    
    # Prepare data for modeling
    target = "log_consumption"
    exog_features = [
        "day_of_week_sin", "day_of_week_cos", "is_weekend", "is_festive",
        "is_summer", "is_winter", "consumption_lag_1d", "consumption_lag_7d",
        "consumption_lag_30d", "consumption_lag_365d", "consumption_lag_14d", "consumption_lag_21d",
        "rolling_avg_3d", "rolling_std_3d"  
    ]
    y, y_train, y_test, exog, exog_train, exog_test, y_orig_train, y_orig_test = prepare_model_data(df, target, exog_features)
    
    # Scale exogenous features using training data
    exog_train_scaled, exog_test_scaled, scaler = scale_features(exog_train, exog_test)
    
    # Find optimal ARIMA parameters
    order, seasonal_order = find_optimal_arima(y_train, exog_train_scaled)
    
    # Fit SARIMAX model and forecast (on log-transformed scale)
    y_pred_log, model_fit = fit_sarimax_model(y_train, exog_train_scaled, order, seasonal_order, y_test, exog_test_scaled)
    
    # Invert the log transformation to get predictions on the original scale
    y_pred = invert_log_transformation(y_pred_log, shift_val)
    
    # Compute performance metrics
    compute_performance_metrics(y_orig_test, y_pred)
    
    # Create forecast DataFrame and save predictions
    forecast_df = create_forecast_df(y_orig_test, y_pred)
    save_forecast_df(forecast_df, os.path.join(results_dir, "predicted_values_kwh.csv"))
    
    # Feature importance analysis
    feature_importance_analysis(df, exog_features, target, os.path.join(results_dir, "feature_importance.csv"))
    
    # Visualization: Forecast comparison and error analysis plots
    visualize_forecast_comparison(forecast_df, results_dir)
    
    # Boxplot: Compare Real vs Forecasted Values
    visualize_boxplot_comparison(forecast_df)
    
    # Forecast one day into the future
    forecast_date, next_day_pred = forecast_one_day(model_fit, df, scaler, exog_features, shift_val)
    print(f"\nForecast for {forecast_date.date()}: {next_day_pred:.4f} kWh")
    
    print("Daily forecasting analysis complete.")

if __name__ == "__main__":
    main()
