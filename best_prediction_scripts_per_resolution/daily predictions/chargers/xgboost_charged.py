import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import seaborn as sns

# ──────────────────────────── config ──────────────────────────────
CHARGING_FILE    = './api_data/daily_cleaned_chargers.csv'
TEMPERATURE_FILE = './api_data/daily_temperature_data.csv'
CALENDAR_FILE    = './layout1_full_calendar_2023-2025.csv'
TOTAL_HEADCOUNT  = 105

# ---------------------------
# Directory & Data Loading Functions
# ---------------------------
def create_results_directory(path):
    os.makedirs(path, exist_ok=True)


def load_consumption_data(file_path):
    df = pd.read_csv(file_path)
    df.set_index('Day', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def load_temperature_data(file_path, index_dates):
    temp_df = pd.read_csv(file_path)
    temp_df['date'] = pd.to_datetime(temp_df['date'], utc=True)
    temp_df.set_index('date', inplace=True)
    temp_df.index = temp_df.index.tz_convert('Europe/Brussels').tz_localize(None)
    temp_df = temp_df.reindex(index_dates)
    temp_df['temperature_2m_mean'] = temp_df['temperature_2m_mean'].ffill().bfill()
    return temp_df


def merge_temperature_data(consumption_df, temp_df):
    return consumption_df.merge(
        temp_df[['temperature_2m_mean']],
        left_index=True,
        right_index=True,
        how='left'
    )

# ---------------------------
# Workforce Calendar & Feature
# ---------------------------
def load_workforce_calendar(file_path):
    cal = pd.read_csv(file_path, parse_dates=['Datum'])
    cal = cal.rename(columns={
        'Totaal_Vakantiedagen': 'vacation_cnt',
        'Totaal_Thuiswerkdagen': 'homework_cnt'
    })
    cal['cal_date'] = cal['Datum'].dt.date
    return cal.set_index('cal_date')[['vacation_cnt','homework_cnt']]


def add_work_at_office(df, cal_df, total_headcount):
    df = df.copy()
    df['cal_date'] = df.index.date
    df = df.merge(cal_df, left_on='cal_date', right_index=True, how='left')
    df[['vacation_cnt','homework_cnt']] = df[['vacation_cnt','homework_cnt']].fillna(0).astype(int)
    is_workday = (df.index.dayofweek < 5) & (df['is_festive'] == 0)
    df['work_at_office'] = np.where(
        is_workday,
        (total_headcount - df['vacation_cnt'] - df['homework_cnt']).clip(lower=0),
        0
    )
    return df.drop(columns='cal_date')

# ---------------------------
# Feature Engineering
# ---------------------------
def add_cumulative_ev_phev_feature(df):
    cumulative_data = {
        datetime(2022,1,4):2, datetime(2022,3,31):3, datetime(2023,2,14):4,
        datetime(2023,2,15):5, datetime(2023,2,28):6, datetime(2023,6,13):7,
        datetime(2023,6,23):8, datetime(2023,7,6):9, datetime(2023,9,15):10,
        datetime(2023,9,26):13, datetime(2023,9,27):14, datetime(2023,10,11):15,
        datetime(2023,10,20):16, datetime(2023,11,7):18, datetime(2024,1,2):20,
        datetime(2024,3,1):21, datetime(2024,3,19):22, datetime(2024,3,28):23,
        datetime(2024,5,7):24, datetime(2024,5,13):26, datetime(2024,5,14):27,
        datetime(2024,5,16):29, datetime(2024,5,23):30, datetime(2024,5,28):31,
        datetime(2024,6,20):33, datetime(2024,6,25):34, datetime(2024,9,5):36,
        datetime(2024,9,12):39, datetime(2024,9,27):40, datetime(2024,10,15):41,
        datetime(2024,10,29):43, datetime(2024,11,5):44, datetime(2024,11,26):45,
        datetime(2025,1,9):46, datetime(2025,1,23):47, datetime(2025,1,28):48,
        datetime(2025,2,4):49,
    }
    ev_series = pd.Series(cumulative_data)
    ev_series = ev_series.reindex(df.index.union(ev_series.index)).sort_index().ffill().fillna(0)
    df['cumulative_ev_phev_count'] = ev_series.reindex(df.index).astype(int)


def add_terugkomdag_feature(df):
    terugkomdagen = [
        datetime(2023,9,13), datetime(2023,10,26), datetime(2023,11,14), datetime(2023,12,20),
        datetime(2024,1,12), datetime(2024,2,7), datetime(2024,3,14), datetime(2024,4,16),
        datetime(2024,5,13), datetime(2024,6,7), datetime(2024,10,22), datetime(2024,11,28),
        datetime(2024,12,18), datetime(2025,1,10), datetime(2025,2,13), datetime(2025,3,18),
        datetime(2025,4,22), datetime(2025,5,12), datetime(2025,6,6)
    ]
    df['is_terugkomdag'] = df.index.to_series().dt.date.isin([d.date() for d in terugkomdagen]).astype(int)


def add_temperature_features(df):
    df['temp_mean_lag_1d'] = df['temperature_2m_mean'].shift(1)
    df['temp_mean_rolling_3d'] = df['temperature_2m_mean'].rolling(window=3).mean()


def add_time_features(df, be_holidays):
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_festive'] = df.index.to_series().apply(lambda x: int(x in be_holidays))
    df['is_summer'] = df.index.month.isin([6,7,8]).astype(int)
    df['is_winter'] = df.index.month.isin([12,1,2]).astype(int)
    df['day_of_week_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
    df['month_sin'] = np.sin(2*np.pi*df.index.month/12)
    df['month_cos'] = np.cos(2*np.pi*df.index.month/12)
    df['dayofyear_sin'] = np.sin(2*np.pi*df.index.dayofyear/365)
    df['dayofyear_cos'] = np.cos(2*np.pi*df.index.dayofyear/365)


def add_lag_features(df, be_holidays):
    df['rolling_avg_3d'] = df['Total_consumption'].rolling(window=3).mean()
    df['rolling_std_3d'] = df['Total_consumption'].rolling(window=3).std()
    df['is_day_after_holiday'] = df.index.to_series().apply(lambda x: int((x - pd.Timedelta(days=1)) in be_holidays))
    df['is_bridge_day'] = df.index.to_series().apply(
        lambda x: int((x.weekday()==0 and (x - pd.Timedelta(days=1)) in be_holidays) or
                      (x.weekday()==4 and (x + pd.Timedelta(days=1)) in be_holidays))
    )


def add_holiday_proximity_features(df, be_holidays):
    hd = sorted(pd.to_datetime(list(be_holidays.keys())))
    def prox(d):
        prev = max([(d - h).days for h in hd if h <= d], default=np.nan)
        nxt = min([(h - d).days for h in hd if h > d], default=np.nan)
        return prev, nxt
    df['days_since_holiday'], df['days_until_holiday'] = zip(*df.index.to_series().apply(prox))


def add_holiday_type_feature(df, be_holidays):
    strength = {
        "Nieuwjaar":1.0,"Paasmaandag":1.5,"Dag van de Arbeid":1.2,
        "O.L.H. Hemelvaart":1.3,"Pinkstermaandag":1.2,"Nationale feestdag":1.5,
        "Wapenstilstand":1.0,"Kerstmis":2.0
    }
    df['holiday_strength'] = df.index.to_series().apply(lambda d: strength.get(be_holidays.get(d), 0.0) if be_holidays.get(d) else 0.0)

# ---------------------------
# Rolling Forecast CV with XGBoost
# ---------------------------
def rolling_forecast_cv_xgb(df, features, initial_train_size, forecast_horizon=1):
    preds, acts = [], []
    for i in range(initial_train_size, len(df) - forecast_horizon + 1):
        train = df.iloc[:i].copy()
        test  = df.iloc[i:i+forecast_horizon].copy()

        avg_wd = train['Total_consumption'].groupby(train.index.dayofweek).mean()
        train['avg_consumption_wd'] = train.index.to_series().dt.dayofweek.map(avg_wd)
        test['avg_consumption_wd']  = test.index.to_series().dt.dayofweek.map(avg_wd)

        X_train = train[features + ['avg_consumption_wd']]
        y_train = train['Total_consumption']
        X_test  = test[features + ['avg_consumption_wd']]
        y_test  = test['Total_consumption']

        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        preds.extend(y_pred)
        acts.extend(y_test.values)

    idx = df.index[initial_train_size:len(df) - forecast_horizon + 1]
    return pd.Series(acts, index=idx), pd.Series(preds, index=idx)

# ---------------------------
# Metric Calculation
# ---------------------------
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100


def calculate_smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

# ---------------------------
# Extra plots : EV fleet size & office occupancy
# ---------------------------
def plot_ev_and_occupancy(df, out_dir):
    """
    Saves two time–series charts:
    1. cumulative EV / PHEV count          →  <out_dir>/evcount_over_time.png
    2. head-count physically in the office →  <out_dir>/occupancy_over_time.png
    """
    # EV / PHEV count ----------------------------------------------------------
    plt.figure(figsize=(13,6))
    plt.plot(df.index, df['cumulative_ev_phev_count'],
             linewidth=2, label='Fleet size (EV + PHEV)')
    plt.title('Cumulative EV / PHEV count over time')
    plt.xlabel('Date')
    plt.ylabel('Number of vehicles')
    plt.grid(alpha=.3);  plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'evcount_over_time.png'))
    plt.close()

    # Office occupancy ---------------------------------------------------------
    plt.figure(figsize=(13,6))
    plt.plot(df.index, df['work_at_office'],
             linewidth=2, label='Employees at the office',)
    plt.title('Office occupancy (head-count) over time')
    plt.xlabel('Date')
    plt.ylabel('People on-site')
    plt.grid(alpha=.3);  plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'occupancy_over_time.png'))
    plt.close()


def compute_performance_metrics(y_true, y_pred):
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    mape  = calculate_mape(y_true.values, y_pred.values)
    smape = calculate_smape(y_true.values, y_pred.values)
    r2    = r2_score(y_true, y_pred)
    print("\nModel Performance Metrics:")
    print(f"MAE:   {mae:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAPE:  {mape:.4f}%")
    print(f"sMAPE: {smape:.4f}%")
    print(f"R\u00b2:    {r2:.4f}")
    return {"MAE":mae, "MSE":mse, "RMSE":rmse, "MAPE":mape, "sMAPE":smape, "R2":r2}

# ---------------------------
# Plotting Functions
# ---------------------------
def plot_forecast_comparison(df, path):
    plt.figure(figsize=(15,8))
    plt.plot(df.index, df['Real_Consumption_kWh'], label='Actual', alpha=0.6)
    plt.plot(df.index, df['Predicted_Consumption_kWh'], label='Predicted', alpha=0.6)
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(path); plt.close()



def plot_residuals(df, path):
    plt.figure(figsize=(15,6))
    plt.plot(df.index, df['Difference'], alpha=0.6)
    plt.axhline(0, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_error_distribution(df, path):
    plt.figure(figsize=(10,6))
    plt.hist(df['Difference'],bins=30,alpha=0.7)
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_error_by_day(df, path):
    d = df.copy(); d['dow'] = d.index.dayofweek; me = d.groupby('dow')['Absolute_Error'].mean()
    plt.figure(figsize=(10,6)); plt.bar(me.index, me.values); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_boxplot_comparison(df, path):
    melted = df[['Real_Consumption_kWh','Predicted_Consumption_kWh']].melt(var_name='Type', value_name='Consumption')
    plt.figure(figsize=(8,6)); sns.boxplot(x='Type',y='Consumption',data=melted)
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_feature_importance(importances, feature_names, path):
    plt.figure(figsize=(10,8))
    nz = importances>0
    names = [n for n, nzv in zip(feature_names, nz) if nzv]
    imps = importances[nz]
    plt.barh(names, imps)
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout(); plt.savefig(path); plt.close()

# ---------------------------
# Main Routine
# ---------------------------
def main():
    results_dir = 'results/daily_chargers_xgb'
    create_results_directory(results_dir)  

    df = load_consumption_data(CHARGING_FILE)
    temp_df = load_temperature_data(TEMPERATURE_FILE, df.index)
    df = merge_temperature_data(df, temp_df)

    be_holidays = holidays.BE()
    add_temperature_features(df)
    add_terugkomdag_feature(df)
    add_cumulative_ev_phev_feature(df)
    

    add_time_features(df, be_holidays)
    add_lag_features(df, be_holidays)
    add_holiday_proximity_features(df, be_holidays)
    add_holiday_type_feature(df, be_holidays)

    cal_df = load_workforce_calendar(CALENDAR_FILE)
    df = add_work_at_office(df, cal_df, TOTAL_HEADCOUNT)
    df.dropna(inplace=True)

    # keep everything *from* 1 Jan 2023 onward
    df_after_jan_2023 = df.loc[df.index >= '2023-01-01']

    # (A) — quickest: call the plotting helper with the filtered DataFrame
    plot_ev_and_occupancy(df_after_jan_2023, results_dir)
    features = [
        'dayofmonth','weekofyear','day_of_week_sin','day_of_week_cos',
        'is_weekend','is_festive','is_summer','is_winter',
        'temperature_2m_mean','temp_mean_lag_1d','temp_mean_rolling_3d',
        'rolling_avg_3d','rolling_std_3d','days_since_holiday','days_until_holiday',
        'holiday_strength','month_sin','month_cos','dayofyear_sin','dayofyear_cos',
        'is_terugkomdag','cumulative_ev_phev_count','work_at_office'
    ]

    initial_train_size = int(0.8 * len(df))
    actuals, predictions = rolling_forecast_cv_xgb(df, features, initial_train_size)
    compute_performance_metrics(actuals, predictions)

    # train final model for importances
    final_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
    final_model.fit(df[features], df['Total_consumption'])
    importances = final_model.feature_importances_

    used_features = [f for f, imp in zip(features , importances) if imp > 0]
    print("Features used by XGBoost (non-zero importance):\n", used_features)
    plot_feature_importance(importances, features + ['avg_consumption_wd'], os.path.join(results_dir, 'feature_importance.png'))

    # rerun CV with selected features
    actuals2, predictions2 = rolling_forecast_cv_xgb(df, used_features, initial_train_size)
    compute_performance_metrics(actuals2, predictions2)

    forecast_df = pd.DataFrame({
        'Real_Consumption_kWh': actuals2,
        'Predicted_Consumption_kWh': predictions2
    }, index=actuals2.index)
    forecast_df['Difference'] = forecast_df['Predicted_Consumption_kWh'] - forecast_df['Real_Consumption_kWh']
    forecast_df['Absolute_Error'] = forecast_df['Difference'].abs()
    forecast_df.to_csv(os.path.join(results_dir, 'predicted_values_kwh_cv_xgb.csv'))

    plot_forecast_comparison(forecast_df, os.path.join(results_dir, 'forecast_comparison.png'))
    plot_residuals(forecast_df, os.path.join(results_dir, 'residuals.png'))
    plot_error_distribution(forecast_df, os.path.join(results_dir, 'error_distribution.png'))
    plot_error_by_day(forecast_df, os.path.join(results_dir, 'error_by_day.png'))
    plot_boxplot_comparison(forecast_df, os.path.join(results_dir, 'boxplot_comparison.png'))

if __name__ == '__main__':
    main()
