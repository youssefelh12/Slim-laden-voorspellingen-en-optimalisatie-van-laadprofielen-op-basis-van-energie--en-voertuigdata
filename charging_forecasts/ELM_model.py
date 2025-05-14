import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------------
# 1.  LOAD YOUR CLEANED DATA  ------------------------------------------------------
csv_path = "./Charging_data_cleaned.csv"   # <- already uploaded
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.rename(columns={'Date': 'Timestamp', 'Chargers': 'kWh'}, inplace=True)

# ---------------------------------------------------------------------------------
# 2.  AUXILIARY CALENDARS  ---------------------------------------------------------
belgian_holidays = {
    # 2024 official + Easter Monday & Whit Monday
    date(2024,1,1), date(2024,4,1), date(2024,5,1), date(2024,5,9),
    date(2024,5,20), date(2024,7,21), date(2024,8,15), date(2024,11,1),
    date(2024,11,11), date(2024,12,25),
    # 2025
    date(2025,1,1), date(2025,4,21), date(2025,5,1), date(2025,5,29),
    date(2025,6,9), date(2025,7,21), date(2025,8,15), date(2025,11,1),
    date(2025,11,11), date(2025,12,25)
}

back_to_work_days = {
    date(2023,9,13), date(2023,10,26), date(2023,11,14), date(2023,12,20),
    date(2024,1,12), date(2024,2,7),  date(2024,3,14),  date(2024,4,16),
    date(2024,5,13), date(2024,6,7),  date(2024,10,22), date(2024,11,28),
    date(2024,12,18), date(2025,1,10), date(2025,2,13), date(2025,3,18),
    date(2025,4,22), date(2025,5,12), date(2025,6,6)
}

fleet_timeline = {
    date(2024,6,20):35, date(2024,6,25):36, date(2024,9,5):38,  date(2024,9,12):41,
    date(2024,9,27):42, date(2024,10,15):43, date(2024,10,29):45, date(2024,11,5):46,
    date(2024,11,26):47, date(2025,1,9):48, date(2025,1,23):49, date(2025,1,28):50,
    date(2025,2,4):51
}

def fleet_size(d):
    past = [k for k in fleet_timeline if k <= d]
    return fleet_timeline[max(past)] if past else 35

def is_bridge(d):
    # Friday after holiday or Monday before holiday
    if d.weekday()==4 and (d - timedelta(days=1)) in belgian_holidays:
        return True
    if d.weekday()==0 and (d + timedelta(days=1)) in belgian_holidays:
        return True
    return False

# ---------------------------------------------------------------------------------
# 3.  FEATURE ENGINEERING  ---------------------------------------------------------
df['date'] = df['Timestamp'].dt.date
df['hour'] = df['Timestamp'].dt.hour
df['weekday'] = df['Timestamp'].dt.weekday
df['is_weekend']   = (df['weekday']>=5).astype(int)
df['is_holiday']   = df['date'].isin(belgian_holidays).astype(int)
df['is_backwork']  = df['date'].isin(back_to_work_days).astype(int)
df['is_bridge']    = df['date'].apply(is_bridge).astype(int)
df['fleet_size']   = df['date'].apply(fleet_size)

# cyclic hour features
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

# 24-hour lag kWh
df['lag_24h'] = df['kWh'].shift(24)
df['lag_24h'].fillna(method='bfill', inplace=True)

feature_cols = ['hour_sin','hour_cos','weekday','is_weekend','is_holiday',
                'is_bridge','is_backwork','fleet_size','lag_24h']

# ---------------------------------------------------------------------------------
# 4.  TRAIN / TEST SPLIT (last full day as test)  ---------------------------------
last_ts = df['Timestamp'].iloc[-1]
candidate = (last_ts - timedelta(days=1)).date()

for i in range(10):
    target_day = candidate - timedelta(days=i)
    if (df['date'] == target_day).sum() == 24:
        test_day = target_day
        break

train_df = df[df['date'] < test_day]
test_df  = df[df['date'] == test_day]

X_train = train_df[feature_cols].values
y_train = train_df['kWh'].values
X_test  = test_df[feature_cols].values
y_test  = test_df['kWh'].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ---------------------------------------------------------------------------------
# 5.  SIMPLE EXTREME LEARNING MACHINE (direct)  ----------------------------------
rng = np.random.default_rng(42)
n_hidden = 120
W_hidden = rng.normal(size=(X_train_s.shape[1], n_hidden))
b_hidden = rng.normal(size=(n_hidden,))
H_train = np.tanh(X_train_s @ W_hidden + b_hidden)
H_test  = np.tanh(X_test_s  @ W_hidden + b_hidden)

# ridge-regularised pseudo-inverse
reg = 1e-3
W_out = np.linalg.solve(H_train.T @ H_train + reg*np.eye(n_hidden), H_train.T @ y_train)
y_pred = H_test @ W_out

# ---------------------------------------------------------------------------------
# 6.  METRICS  --------------------------------------------------------------------
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100

metrics_df = pd.DataFrame({
    'Metric':['MAE (kWh)','RMSE (kWh)','R2','WAPE (%)'],
    'Value':[round(mae,2), round(rmse,2), round(r2,3), round(wape,2)]
})

print(metrics_df)


# ---------------------------------------------------------------------------------
# 7.  ACTUAL VS PREDICTED TABLE  ---------------------------------------------------
comp_df = test_df[['Timestamp','kWh']].copy().rename(columns={'kWh':'Actual_kWh'})
comp_df['Forecast_kWh'] = y_pred


# ---------------------------------------------------------------------------------
# 8.  PLOT  -----------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(comp_df['Timestamp'], comp_df['Actual_kWh'], label='Actual')
plt.plot(comp_df['Timestamp'], comp_df['Forecast_kWh'], label='Forecast')
plt.title(f'Charging Load – Actual vs Forecast ({test_day})')
plt.xlabel('Time')
plt.ylabel('kWh')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
