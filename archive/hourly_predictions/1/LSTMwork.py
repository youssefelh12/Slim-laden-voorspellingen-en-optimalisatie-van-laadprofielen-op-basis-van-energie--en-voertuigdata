import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import holidays  # pip install holidays
from joblib import dump

# ----------------------------------------------------------
# 0) Configuration
# ----------------------------------------------------------
RAW_PATH     = pathlib.Path("api_data/hourly_building_consumption.csv")
MODEL_DIR    = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True)
FEATURE_COLS = [
    "Total_consumption", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "lag1", "lag10", "lag50",
    "roll1", "roll10", "roll50", "time_delta",
    "is_month_start", "is_month_end", "is_holiday"
]
TARGET_COL   = "Total_consumption"

# ----------------------------------------------------------
# 1) Helper functions
# ----------------------------------------------------------
def add_features(df):
    df = df.copy()
    df['hour']         = df.index.hour
    df['day_of_week']  = df.index.dayofweek

    # cyclic encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # lags & rolling stats (causal)
    df['lag1']  = df[TARGET_COL].shift(1)
    df['lag10'] = df[TARGET_COL].shift(10)
    df['lag50'] = df[TARGET_COL].shift(50)
    df['roll1']  = df[TARGET_COL].shift(1).rolling(1,  min_periods=1).mean()
    df['roll10'] = df[TARGET_COL].shift(1).rolling(10, min_periods=1).mean()
    df['roll50'] = df[TARGET_COL].shift(1).rolling(50, min_periods=1).mean()

    # time delta (hours)
    df['time_delta'] = df.index.to_series().diff().dt.total_seconds().div(3600)

    # calendar flags
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end']   = df.index.is_month_end.astype(int)

    # holiday flag (Swiss)
    swiss_hols = holidays.Switzerland()
    df['is_holiday'] = df.index.normalize().isin(swiss_hols).astype(int)

    return df.dropna()





def make_sequences(df, features, target, seq_len):
    values  = df[features].values
    targets = df[target].values
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(values[i-seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

# ----------------------------------------------------------
# 2) Load & split
# ----------------------------------------------------------
df = pd.read_csv(RAW_PATH, parse_dates=['Hour'], index_col='Hour').sort_index()
n_total = len(df)
n_train = int(n_total * 0.7)
n_val   = int(n_total * 0.1)

raw_train = df.iloc[:n_train]
raw_val   = df.iloc[n_train : n_train + n_val]
raw_test  = df.iloc[n_train + n_val :]  # ← corrected

for name, split in [('train', raw_train), ('val', raw_val), ('test', raw_test)]:
    print(f"{name} size before filtering = {len(split)}")


# ----------------------------------------------------------
# 3) Filter workhours & add features
# ----------------------------------------------------------
train_wh = keep_workhours(raw_train)
val_wh   = keep_workhours(raw_val)
test_wh  = keep_workhours(raw_test)

train_df = add_features(train_wh)
val_df   = add_features(val_wh)
test_df  = add_features(test_wh)

# ----------------------------------------------------------
# 4) Scaling
# ----------------------------------------------------------
# after keep_workhours & add_features
assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0, \
       "One of the splits is empty—check your indexing/filtering!"

scaler = MinMaxScaler()
scale_cols = ['Total_consumption', 'lag1', 'lag10', 'lag50', 'roll1', 'roll10', 'roll50']
train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
val_df[scale_cols]   = scaler.transform(val_df[scale_cols])
test_df[scale_cols]  = scaler.transform(test_df[scale_cols])
# save scaler
dump(scaler, MODEL_DIR / 'scaler_workhours.joblib')

# ----------------------------------------------------------
# 5) Hyperparameter tuning: sequence length
# ----------------------------------------------------------
from tensorflow.keras import backend as K

seq_lengths = [10, 20, 50, 100]
results = []

for seq in seq_lengths:
    print(f"\n=== Tuning: SEQUENCE_LEN = {seq} ===")
    # clear session to free memory
    K.clear_session()

    # prepare sequences
    X_train, y_train = make_sequences(train_df, FEATURE_COLS, TARGET_COL, seq)
    X_val,   y_val   = make_sequences(val_df,   FEATURE_COLS, TARGET_COL, seq)

    # build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq, len(FEATURE_COLS))),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])

    # train
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )

    # evaluate on validation
    y_pred_val = model.predict(X_val).flatten()
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val  = r2_score(y_val, y_pred_val)
    print(f"Validation MAE: {mae_val:.4f}, R²: {r2_val:.4f}")

    results.append({'seq_len': seq, 'val_mae': mae_val, 'val_r2': r2_val})

# summarize
df_results = pd.DataFrame(results).set_index('seq_len')
print("\nTuning Results:\n", df_results)

# ----------------------------------------------------------
# 6) (Optional) Choose best seq and retrain on train+val, evaluate on test
# ----------------------------------------------------------
best_seq = df_results['val_mae'].idxmin()
print(f"\nBest sequence length: {best_seq}")

# retrain on combined train+val
combined_df = pd.concat([train_df, val_df])
X_comb, y_comb = make_sequences(combined_df, FEATURE_COLS, TARGET_COL, best_seq)
X_test, y_test = make_sequences(test_df, FEATURE_COLS, TARGET_COL, best_seq)

K.clear_session()
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(best_seq, len(FEATURE_COLS))),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])
model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])

mc = ModelCheckpoint(
    filepath=MODEL_DIR / 'best_workhours_model.keras',
    monitor='val_loss', save_best_only=True
)
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    X_comb, y_comb,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[es, mc],
    verbose=1
)

# final evaluation
y_pred_test = model.predict(X_test).flatten()
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test  = r2_score(y_test, y_pred_test)
print(f"Test MAE: {mae_test:.4f}, Test R²: {r2_test:.4f}")

# inverse scale & plot
y_pad = np.zeros((len(y_test), len(scale_cols)-1))
y_true_inv = scaler.inverse_transform(np.hstack([y_test.reshape(-1,1), y_pad]))[:,0]
y_pred_inv = scaler.inverse_transform(np.hstack([y_pred_test.reshape(-1,1), y_pad]))[:,0]
print(f"Test MAE (kWh): {np.mean(np.abs(y_true_inv - y_pred_inv)):.2f}")

plt.figure(figsize=(12,4))
plt.plot(y_true_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Actual vs Predicted (kWh)')
plt.legend()
plt.tight_layout()
plt.show()

# save final model and scaler
model.save(MODEL_DIR / 'final_workhours_model.keras')
print("Model and scaler saved.")
