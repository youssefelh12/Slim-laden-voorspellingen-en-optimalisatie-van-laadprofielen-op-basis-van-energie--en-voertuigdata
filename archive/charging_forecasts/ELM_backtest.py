import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ---------------------------------------------------------
CSV_PATH = "./Charging_data_cleaned.csv"
N_HIDDEN = 500
RIDGE_REG = 1e-3
N_BACKTEST_DAYS = 30
# ---------------------------------------------------------

# 1. Load & preprocess (same as before, condensed)
df = pd.read_csv(CSV_PATH, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
df.rename(columns={'Date':'Timestamp','Chargers':'kWh'}, inplace=True)

belgian_holidays = {
    date(2024,1,1), date(2024,4,1), date(2024,5,1), date(2024,5,9), date(2024,5,20),
    date(2024,7,21), date(2024,8,15), date(2024,11,1), date(2024,11,11), date(2024,12,25),
    date(2025,1,1), date(2025,4,21), date(2025,5,1), date(2025,5,29), date(2025,6,9),
    date(2025,7,21), date(2025,8,15), date(2025,11,1), date(2025,11,11), date(2025,12,25)
}

back_to_work_days = {
    date(2023,9,13), date(2023,10,26), date(2023,11,14), date(2023,12,20),
    date(2024,1,12), date(2024,2,7), date(2024,3,14), date(2024,4,16),
    date(2024,5,13), date(2024,6,7), date(2024,10,22), date(2024,11,28),
    date(2024,12,18), date(2025,1,10), date(2025,2,13), date(2025,3,18),
    date(2025,4,22), date(2025,5,12), date(2025,6,6)
}

fleet_timeline = {
    date(2024,6,20):35, date(2024,6,25):36, date(2024,9,5):38, date(2024,9,12):41,
    date(2024,9,27):42, date(2024,10,15):43, date(2024,10,29):45, date(2024,11,5):46,
    date(2024,11,26):47, date(2025,1,9):48, date(2025,1,23):49, date(2025,1,28):50,
    date(2025,2,4):51
}
def fleet_size(d):
    older = [k for k in fleet_timeline if k<=d]
    return fleet_timeline[max(older)] if older else 35

def is_bridge(d):
    return (d.weekday()==4 and (d-timedelta(days=1)) in belgian_holidays) or \
           (d.weekday()==0 and (d+timedelta(days=1)) in belgian_holidays)

df['date']=df['Timestamp'].dt.date
df['hour']=df['Timestamp'].dt.hour
df['weekday']=df['Timestamp'].dt.weekday
df['is_weekend']=(df['weekday']>=5).astype(int)
df['is_holiday']=df['date'].isin(belgian_holidays).astype(int)
df['is_backwork']=df['date'].isin(back_to_work_days).astype(int)
df['is_bridge']=df['date'].apply(is_bridge).astype(int)
df['fleet_size']=df['date'].apply(fleet_size)
df['hour_sin']=np.sin(2*np.pi*df['hour']/24)
df['hour_cos']=np.cos(2*np.pi*df['hour']/24)
df['lag_24h']=df['kWh'].shift(24).bfill()

feat_cols=['hour_sin','hour_cos','weekday','is_weekend','is_holiday',
           'is_bridge','is_backwork','fleet_size','lag_24h']

# 2. Determine back-test days
last_ts=df['Timestamp'].iloc[-1]
days=[]
cur=(last_ts - timedelta(days=1)).date()
while len(days)<N_BACKTEST_DAYS:
    if (df['date']==cur).sum()==24:
        days.append(cur)
    cur-=timedelta(days=1)
days=sorted(days)

results=[]
pred_frames=[]

for idx, test_day in enumerate(days):
    train_df=df[df['date']<test_day]
    test_df=df[df['date']==test_day]
    X_train=train_df[feat_cols].values
    y_train=train_df['kWh'].values
    X_test=test_df[feat_cols].values
    y_test=test_df['kWh'].values
    
    scaler=StandardScaler()
    X_train_s=scaler.fit_transform(X_train)
    X_test_s=scaler.transform(X_test)
    
    rng=np.random.default_rng(idx)
    W=rng.normal(size=(X_train_s.shape[1], N_HIDDEN))
    b=rng.normal(size=(N_HIDDEN,))
    H_train=np.tanh(X_train_s@W + b)
    H_test=np.tanh(X_test_s @W + b)
    W_out=np.linalg.solve(H_train.T@H_train + RIDGE_REG*np.eye(N_HIDDEN), H_train.T@y_train)
    y_pred=H_test@W_out
    
    mae=mean_absolute_error(y_test,y_pred)
    rmse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    wape=np.sum(np.abs(y_test-y_pred))/np.sum(np.abs(y_test))*100
    
    results.append({'Date':test_day,'MAE':mae,'RMSE':rmse,'R2':r2,'WAPE':wape})
    
    temp_df=test_df[['Timestamp','kWh']].copy()
    temp_df.rename(columns={'kWh':'Actual_kWh'}, inplace=True)
    temp_df['Forecast_kWh']=y_pred
    pred_frames.append(temp_df)

results_df=pd.DataFrame(results)
summary_df=pd.DataFrame({
    'Metric':['MAE','RMSE','R2','WAPE'],
    'Mean':[results_df['MAE'].mean(),
            results_df['RMSE'].mean(),
            results_df['R2'].mean(),
            results_df['WAPE'].mean()]
})


# 3. Plot: Actual vs Forecast entire back-test window
full_df=pd.concat(pred_frames).sort_values('Timestamp').reset_index(drop=True)

plt.figure(figsize=(12,4))
plt.plot(full_df['Timestamp'], full_df['Actual_kWh'], label='Actual')
plt.plot(full_df['Timestamp'], full_df['Forecast_kWh'], label='Forecast')
plt.title('Actual vs Forecast – last 30 days (24-h ahead rolling)')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Plot: R2 by day of week
weekday_map={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
results_df['weekday']=pd.to_datetime(results_df['Date']).dt.weekday
r2_by_day=results_df.groupby('weekday')['R2'].mean().reindex(range(7))
labels=[weekday_map[i] for i in r2_by_day.index]

# ... same code up to r2_by_day calculation ...
r2_by_day = results_df.groupby('weekday')['R2'].mean().reindex(range(7))

# --- clip negative values to zero for plotting only ---
r2_plot = r2_by_day.clip(lower=0)

labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
plt.figure(figsize=(6,4))
plt.bar(labels, r2_plot.values, color='steelblue')
plt.title('Average R² by weekday (clipped at 0)')
plt.ylabel('R² ( 0 – 1 )')
plt.ylim(0, 1)          # keep y-axis 0‒1
plt.tight_layout()
plt.show()
