"""
30-day walk-forward back-test with a stacked forecast:

Base learners
-------------
1) SARIMAX      (statsmodels)
2) CatBoost     (catboost)
3) Prophet      (prophet 1.x  –  pip install prophet)
4) XGBoost      (xgboost)

Meta learner
------------
Extreme Learning Machine (single hidden layer, tanh, ridge-solved output)

Outputs
-------
• CSV-like tables in the notebook (ace_tools.display_dataframe_to_user)
• Two plots: full 30-day Actual-vs-Forecast + R² by weekday
"""

# ------------------------------------------------------------------
# 0.  Imports & CONSTANTS
# ------------------------------------------------------------------
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from catboost import CatBoostRegressor, Pool
from prophet import Prophet
from xgboost import XGBRegressor


CSV_PATH      = "./Charging_data_cleaned.csv"   # ← update if needed
BACKTEST_DAYS = 5
META_HIDDEN   = 40
RIDGE_LAMBDA  = 1e-3

# ------------------------------------------------------------------
# 1.  Load & basic cleanup
# ------------------------------------------------------------------
df = (pd.read_csv(CSV_PATH, parse_dates=['Date'])
        .sort_values('Date')
        .reset_index(drop=True)
        .rename(columns={'Date':'Timestamp', 'Chargers':'kWh'}))

# ------------------------------------------------------------------
# 2.  Calendar + fleet helpers
# ------------------------------------------------------------------
belgian_holidays = {                           # add more if needed
    date(2024,1,1), date(2024,4,1), date(2024,5,1), date(2024,5,9),
    date(2024,5,20), date(2024,7,21), date(2024,8,15), date(2024,11,1),
    date(2024,11,11), date(2024,12,25),
    date(2025,1,1), date(2025,4,21), date(2025,5,1), date(2025,5,29),
    date(2025,6,9), date(2025,7,21), date(2025,8,15), date(2025,11,1),
    date(2025,11,11), date(2025,12,25)
}

back_to_work_days = {
    date(2023,9,13), date(2023,10,26), date(2023,11,14), date(2023,12,20),
    date(2024,1,12), date(2024,2,7), date(2024,3,14), date(2024,4,16),
    date(2024,5,13), date(2024,6,7), date(2024,10,22), date(2024,11,28),
    date(2024,12,18), date(2025,1,10), date(2025,2,13), date(2025,3,18),
    date(2025,4,22), date(2025,5,12), date(2025,6,6)
}

fleet_timeline = {           # stepwise growth of company EV fleet
    date(2024,6,20):35, date(2024,6,25):36, date(2024,9,5):38,
    date(2024,9,12):41, date(2024,9,27):42, date(2024,10,15):43,
    date(2024,10,29):45, date(2024,11,5):46, date(2024,11,26):47,
    date(2025,1,9):48,  date(2025,1,23):49, date(2025,1,28):50,
    date(2025,2,4):51
}
def fleet_size(d):
    pts = [k for k in fleet_timeline if k <= d]
    return fleet_timeline[max(pts)] if pts else 35

def is_bridge(d):
    return (d.weekday()==4 and (d-timedelta(days=1)) in belgian_holidays) or \
           (d.weekday()==0 and (d+timedelta(days=1)) in belgian_holidays)

# ------------------------------------------------------------------
# 3.  Feature engineering
# ------------------------------------------------------------------
df['date']       = df['Timestamp'].dt.date
df['hour']       = df['Timestamp'].dt.hour
df['weekday']    = df['Timestamp'].dt.weekday
df['is_weekend'] = (df['weekday']>=5).astype(int)
df['is_holiday'] = df['date'].isin(belgian_holidays).astype(int)
df['is_backwk']  = df['date'].isin(back_to_work_days).astype(int)
df['is_bridge']  = df['date'].apply(is_bridge).astype(int)
df['fleet_size'] = df['date'].apply(fleet_size)
df['hour_sin']   = np.sin(2*np.pi*df['hour']/24)
df['hour_cos']   = np.cos(2*np.pi*df['hour']/24)
df['lag_24h']    = df['kWh'].shift(24).bfill()

FEATS = ['hour_sin','hour_cos','weekday','is_weekend','is_holiday',
         'is_bridge','is_backwk','fleet_size','lag_24h']

# ------------------------------------------------------------------
# 4.  Choose last N complete days for rolling back-test
# ------------------------------------------------------------------
last_ts = df['Timestamp'].iloc[-1]
days    = []
d       = (last_ts - timedelta(days=1)).date()
while len(days) < BACKTEST_DAYS:
    if (df['date']==d).sum() == 24:
        days.append(d)
    d -= timedelta(days=1)
days = sorted(days)

# Storage for results & predictions
metrics, pred_frames = [], []

# ------------------------------------------------------------------
# 5.  Walk-forward loop
# ------------------------------------------------------------------
for fold, test_day in enumerate(days):
    tr = df[df['date'] < test_day]
    te = df[df['date'] == test_day]
    Xtr, ytr = tr[FEATS].values, tr['kWh'].values
    Xte, yte = te[FEATS].values, te['kWh'].values
    
    # scale numeric inputs once per fold
    scaler  = StandardScaler()
    Xtr_s   = scaler.fit_transform(Xtr)
    Xte_s   = scaler.transform(Xte)
    
    # ───── Base learner #1 – SARIMAX ──────────────────────────────
    sar_model = SARIMAX(ytr, exog=Xtr_s,
                        order=(1,0,1), seasonal_order=(0,1,1,24),
                        enforce_stationarity=False, enforce_invertibility=False)
    sar_fit   = sar_model.fit(disp=False)
    sar_pred  = sar_fit.predict(start=len(ytr), end=len(ytr)+23, exog=Xte_s)
    sar_train = sar_fit.fittedvalues[-len(ytr):]          # in-sample
    
    # ───── Base learner #2 – CatBoost ─────────────────────────────
    cat = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05,
                            loss_function='RMSE', verbose=False,
                            random_seed=fold)
    cat.fit(Xtr_s, ytr)
    cat_pred  = cat.predict(Xte_s)
    cat_train = cat.predict(Xtr_s)
    
    # ───── Base learner #3 – Prophet ──────────────────────────────
    ph_tr = pd.DataFrame({
        'ds'      : tr['Timestamp'],
        'y'       : ytr,
        'hour'    : tr['hour'],
        'weekday' : tr['weekday'],
        'holiday' : tr['is_holiday']
    })
    ph_te = pd.DataFrame({
        'ds'      : te['Timestamp'],
        'hour'    : te['hour'],
        'weekday' : te['weekday'],
        'holiday' : te['is_holiday']
    })
    prophet = Prophet(weekly_seasonality=False, daily_seasonality=False,
                      yearly_seasonality=False)
    for extra in ['hour','weekday','holiday']:
        prophet.add_regressor(extra)
    prophet.fit(ph_tr)
    ph_pred  = prophet.predict(ph_te)['yhat'].values
    ph_train = prophet.predict(ph_tr)['yhat'].values
    
    # ───── Base learner #4 – XGBoost ──────────────────────────────
    xgb = XGBRegressor(n_estimators=400, learning_rate=0.05,
                       max_depth=6, subsample=0.8, colsample_bytree=0.8,
                       objective='reg:squarederror', random_state=fold)
    xgb.fit(Xtr_s, ytr)
    xgb_pred  = xgb.predict(Xte_s)
    xgb_train = xgb.predict(Xtr_s)
    
    # Stack matrices (same order)
    train_stack = np.column_stack([sar_train, cat_train, ph_train, xgb_train])
    test_stack  = np.column_stack([sar_pred, cat_pred, ph_pred, xgb_pred])
    
    # ───── Meta Extreme Learning Machine ─────────────────────────
    rng  = np.random.default_rng(fold+123)
    W    = rng.normal(size=(train_stack.shape[1], META_HIDDEN))
    b    = rng.normal(size=(META_HIDDEN,))
    Htr  = np.tanh(train_stack @ W + b)
    Hte  = np.tanh(test_stack  @ W + b)
    Wout = np.linalg.solve(Htr.T @ Htr + RIDGE_LAMBDA*np.eye(META_HIDDEN),
                           Htr.T @ ytr)
    yhat = Hte @ Wout
    
    # ───── Store metrics & predictions ───────────────────────────
    mae  = mean_absolute_error(yte, yhat)
    rmse = mean_squared_error(yte, yhat)
    r2   = r2_score(yte, yhat)
    wape = np.sum(np.abs(yte - yhat)) / np.sum(np.abs(yte)) * 100
    metrics.append({'Date':test_day,'MAE':mae,'RMSE':rmse,'R2':r2,'WAPE':wape})
    
    tmp = te[['Timestamp','kWh']].copy()
    tmp.rename(columns={'kWh':'Actual_kWh'}, inplace=True)
    tmp['Forecast_kWh'] = yhat
    pred_frames.append(tmp)

# ------------------------------------------------------------------
# 6.  Display tables & plots
# ------------------------------------------------------------------
metrics_df = pd.DataFrame(metrics)
mean_df = pd.DataFrame({
    'Metric':['MAE','RMSE','R2','WAPE'],
    'Mean':[metrics_df['MAE'].mean(),
            metrics_df['RMSE'].mean(),
            metrics_df['R2'].mean(),
            metrics_df['WAPE'].mean()]
})



# Full 30-day series
full = pd.concat(pred_frames).sort_values('Timestamp').reset_index(drop=True)
plt.figure(figsize=(13,4))
plt.plot(full['Timestamp'], full['Actual_KWh' if 'Actual_KWh' in full else 'Actual_kWh'],
         label='Actual', lw=1)
plt.plot(full['Timestamp'], full['Forecast_kWh'], label='Stack forecast', lw=1)
plt.title('Actual vs Forecast – stacked model (last 30 days)')
plt.xlabel('Date'); plt.ylabel('kWh')
plt.xticks(rotation=45); plt.legend(); plt.tight_layout(); plt.show()

# R² by weekday (clipped at 0)
week_map = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
metrics_df['weekday'] = pd.to_datetime(metrics_df['Date']).dt.weekday
r2_by_wd = metrics_df.groupby('weekday')['R2'].mean().clip(lower=0).reindex(range(7))
plt.figure(figsize=(6,4))
plt.bar(week_map, r2_by_wd.values, color='teal')
plt.title('Average R² by weekday – stacked model')
plt.ylabel('R² (clipped)')
plt.ylim(0,1); plt.tight_layout(); plt.show()
