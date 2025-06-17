import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
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
    temp_df.index = temp_df.index.tz_convert('Europe/Brussels').tz_localize(None)
    temp_df = temp_df.reindex(index_dates)
    temp_df['temperature_2m_mean'] = temp_df['temperature_2m_mean'].ffill()
    temp_df['temperature_2m_mean'] = temp_df['temperature_2m_mean'].bfill()
    return temp_df


def merge_temperature_data(consumption_df, temp_df):
    return consumption_df.merge(
        temp_df[['temperature_2m_mean']],
        left_index=True,
        right_index=True,
        how='left'
    )

# ---------------------------
# Feature Engineering Functions
# ---------------------------
def add_cumulative_ev_phev_feature(df):
    cumulative_data = {datetime(2022,1,4):2, datetime(2022,3,31):3, datetime(2023,2,14):4,
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
                       datetime(2025,2,4):49}
    s = pd.Series(cumulative_data)
    s = s.reindex(df.index.union(s.index)).sort_index().ffill().fillna(0)
    df['cumulative_ev_phev_count'] = s.reindex(df.index).astype(int)


def add_terugkomdag_feature(df):
    terug = [datetime(2023,9,13), datetime(2023,10,26), datetime(2023,11,14), datetime(2023,12,20),
             datetime(2024,1,12), datetime(2024,2,7), datetime(2024,3,14), datetime(2024,4,16),
             datetime(2024,5,13), datetime(2024,6,7), datetime(2024,3,16), datetime(2024,10,22),
             datetime(2024,11,28), datetime(2024,12,18), datetime(2025,1,10), datetime(2025,2,13),
             datetime(2025,3,18), datetime(2025,4,22), datetime(2025,5,12), datetime(2025,6,6)]
    df['is_terugkomdag'] = df.index.to_series().dt.date.isin([d.date() for d in terug]).astype(int)


def add_temperature_features(df):
    df['temp_mean_lag_1d'] = df['temperature_2m_mean'].shift(1)
    df['temp_mean_rolling_3d'] = df['temperature_2m_mean'].rolling(3).mean()


def add_time_features(df, hol):
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week']>=5).astype(int)
    df['is_festive'] = df.index.to_series().apply(lambda x: int(x in hol))
    df['is_summer'] = df.index.month.isin([6,7,8]).astype(int)
    df['is_winter'] = df.index.month.isin([12,1,2]).astype(int)
    df['day_of_week_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
    df['month_sin'] = np.sin(2*np.pi*df.index.month/12)
    df['month_cos'] = np.cos(2*np.pi*df.index.month/12)
    df['dayofyear_sin'] = np.sin(2*np.pi*df.index.dayofyear/365)
    df['dayofyear_cos'] = np.cos(2*np.pi*df.index.dayofyear/365)


def add_lag_features(df, hol):
    df['consumption_lag_1d'] = df['Total_consumption'].shift(1)
    df['consumption_lag_7d'] = df['Total_consumption'].shift(7)
    df['rolling_avg_3d'] = df['Total_consumption'].rolling(3).mean()
    df['is_day_after_holiday'] = df.index.to_series().apply(lambda x: int((x-pd.Timedelta(1, 'd')) in hol))


def add_holiday_proximity(df, hol):
    hd = sorted(pd.to_datetime(list(hol.keys())))
    def prox(d):
        prev = max([(d-h).days for h in hd if h<=d], default=np.nan)
        nxt = min([(h-d).days for h in hd if h>d], default=np.nan)
        return prev, nxt
    df['days_since_holiday'], df['days_until_holiday'] = zip(*df.index.to_series().apply(prox))


def add_holiday_strength(df, hol):
    strength = {"Nieuwjaar":1.0, "Paasmaandag":1.5, "Dag van de Arbeid":1.2,
                "O.L.H. Hemelvaart":1.3, "Pinkstermaandag":1.2, "Nationale feestdag":1.5,
                "Wapenstilstand":1.0, "Kerstmis":2.0}
    df['holiday_strength'] = df.index.to_series().apply(lambda d: strength.get(hol.get(d),0.0))

# ---------------------------
# Rolling Forecast CV with SARIMAX
# ---------------------------
def rolling_forecast_cv_sarimax(y, exog, initial_train, horizon=1):
    preds, acts = [], []
    for i in range(initial_train, len(y)-horizon+1):
        y_train = y.iloc[:i]
        y_test  = y.iloc[i:i+horizon]
        ex_train = exog.iloc[:i]
        ex_test  = exog.iloc[i:i+horizon]

        scaler = StandardScaler()
        ex_train_s = pd.DataFrame(scaler.fit_transform(ex_train), index=ex_train.index, columns=ex_train.columns)
        ex_test_s  = pd.DataFrame(scaler.transform(ex_test), index=ex_test.index,  columns=ex_test.columns)

        model = SARIMAX(y_train, exog=ex_train_s, order=(1,1,1), seasonal_order=(1,0,0,7))
        res = model.fit(disp=False)
        f = res.get_forecast(steps=horizon, exog=ex_test_s)
        y_pred = f.predicted_mean
        preds.extend(y_pred.values)
        acts.extend(y_test.values)

    idx = y.index[initial_train:len(y)-horizon+1]
    return pd.Series(acts, index=idx), pd.Series(preds, index=idx)

# ---------------------------
# Metrics
# ---------------------------
def calculate_mape(y_true, y_pred): return np.mean(np.abs((y_true-y_pred)/(y_true+1e-10))) *100

def calculate_smape(y_true, y_pred): return 100*np.mean(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+1e-10))

# ---------------------------
# Plotting
# ---------------------------
# [reuse seaborn plots as before]

# ---------------------------
# Main
# ---------------------------
def main():
    res_dir = 'results/daily_chargers_sarimax'
    create_results_directory(res_dir)
    df = load_consumption_data('api_data/daily_cleaned_chargers.csv')
    temp = load_temperature_data('api_data/daily_temperature_data.csv', df.index)
    df = merge_temperature_data(df, temp)

    hol = holidays.BE()
    add_temperature_features(df)
    add_terugkomdag_feature(df)
    add_cumulative_ev_phev_feature(df)
    add_time_features(df, hol)
    add_lag_features(df, hol)
    add_holiday_proximity(df, hol)
    add_holiday_strength(df, hol)
    df.dropna(inplace=True)

    exogs = ['temperature_2m_mean','temp_mean_lag_1d','temp_mean_rolling_3d',
             'consumption_lag_1d','consumption_lag_7d','rolling_avg_3d',
             'dayofmonth','weekofyear','day_of_week','is_weekend',
             'is_festive','is_summer','is_winter','month_sin','month_cos',
             'dayofyear_sin','dayofyear_cos','days_since_holiday',
             'days_until_holiday','holiday_strength','is_terugkomdag',
             'cumulative_ev_phev_count']

    y = df['Total_consumption']
    X = df[exogs]

    init = int(0.8*len(df))
    acts, preds = rolling_forecast_cv_sarimax(y, X, init)

    mae = mean_absolute_error(acts, preds)
    mse = mean_squared_error(acts, preds)
    rmse = np.sqrt(mse)
    mape = calculate_mape(acts.values, preds.values)
    smape = calculate_smape(acts.values, preds.values)
    r2 = r2_score(acts, preds)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, sMAPE: {smape:.2f}%, R2: {r2:.4f}")

    df_fc = pd.DataFrame({'Real_Consumption_kWh':acts, 'Predicted_Consumption_kWh':preds}, index=acts.index)
    df_fc['Difference'] = df_fc['Predicted_Consumption_kWh'] - df_fc['Real_Consumption_kWh']
    df_fc['Absolute_Error'] = df_fc['Difference'].abs()
    df_fc.to_csv(os.path.join(res_dir,'predicted_values_kwh_cv_sarimax.csv'))

    # plotting omitted for brevity

if __name__=='__main__':
    main()
