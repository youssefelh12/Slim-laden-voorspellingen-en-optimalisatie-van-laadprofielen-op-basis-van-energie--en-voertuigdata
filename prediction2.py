import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import holidays
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# A. Load Energy Data and Merge Weather Data
# -----------------------------
df = pd.read_csv("Data/15min2024_consumption.csv", parse_dates=["Date"])
df = df.sort_values("Date")

# Convert charger columns to float.
charger_cols = ["Chargers (L1) [kW]", "Chargers (L2) [kW]", "Chargers (L3) [kW]"]
for col in charger_cols:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

# Compute total charger consumption.
df["total_chargers"] = df["Chargers (L1) [kW]"] + df["Chargers (L2) [kW]"] + df["Chargers (L3) [kW]"]

# Set Date as index.
df.set_index("Date", inplace=True)

# Load weather data (pre-fetched using Open-Meteo API)
weather_df = pd.read_csv("weather_data.csv", parse_dates=["date"])
# Rename the date column to match the energy data
weather_df = weather_df.rename(columns={"date": "Date"})
weather_df.set_index("Date", inplace=True)

# Merge weather data with energy data - now with matching index names
df = df.join(weather_df, how='left')
df[['temperature_2m', 'relative_humidity_2m']] = df[['temperature_2m', 'relative_humidity_2m']].ffill()

# -----------------------------
# B. Feature Engineering (No Grid-Based Features)
# -----------------------------
be_holidays = holidays.BE()

# Time-based cyclical features
df['hour'] = df.index.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week'] = df.index.dayofweek
df['week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Work schedule & holiday effects
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['is_festive'] = df.index.to_series().apply(lambda x: 1 if x in be_holidays else 0)
df['in_work_hours'] = df['hour'].apply(lambda h: 1 if 8 <= h < 18 else 0)

# Lag & rolling window features
df['lag_1'] = df['total_chargers'].shift(1)
df['rolling_mean_1h'] = df['total_chargers'].rolling(window=4).mean()
df['rolling_4h_mean'] = df['total_chargers'].rolling(window=16).mean()
df['rolling_1d_mean'] = df['total_chargers'].rolling(window=96).mean()
df['rolling_std_4h'] = df['total_chargers'].rolling(window=16).std()

# Weather-based features
df['temp_variance'] = df['temperature_2m'].rolling(window=24).std()

# Drop rows with NaN values (caused by rolling calculations)
df.dropna(inplace=True)

# -----------------------------
# C. Correlation Heatmap (Check Feature Relationships)
# -----------------------------
selected_features = ['total_chargers', 'hour_sin', 'hour_cos', 'week_sin', 'week_cos',
                     'is_weekend', 'is_festive', 'in_work_hours', 'lag_1', 'rolling_mean_1h',
                     'rolling_4h_mean', 'rolling_1d_mean', 'rolling_std_4h', 'temperature_2m', 'temp_variance']

plt.figure(figsize=(12,10))
corr = df[selected_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: Safe Features for Training")
plt.show()

# -----------------------------
# D. Scaling and Preparing Data for LSTM
# -----------------------------
scaler_target = MinMaxScaler()
df['total_chargers'] = scaler_target.fit_transform(df[['total_chargers']])

scaler_exog = MinMaxScaler()
df[selected_features[1:]] = scaler_exog.fit_transform(df[selected_features[1:]])

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length, 0])
    return np.array(X_seq), np.array(y_seq)

seq_length = 96  # 1 day (96 time steps at 15-min intervals)
X_seq, y_seq = create_sequences(df[selected_features].values, seq_length)

# -----------------------------
# E. Building and Training LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, len(selected_features))))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_seq, y_seq, epochs=20, batch_size=64)

# -----------------------------
# F. Forecasting the Next 3 Days
# -----------------------------
num_predictions = 288  # 3 days * 24 hours * 4 intervals/hour
predictions_scaled = []
current_seq = X_seq[-1].reshape(1, seq_length, len(selected_features))

for _ in range(num_predictions):
    pred = model.predict(current_seq)
    predictions_scaled.append(pred[0, 0])
    new_row = np.hstack([[pred[0, 0]], df[selected_features[1:]].values[-1]])
    new_row = new_row.reshape(1, 1, len(selected_features))
    current_seq = np.concatenate((current_seq[:, 1:, :], new_row), axis=1)

# Inverse transform predictions
predictions_inv = scaler_target.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

# Generate future timestamps
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=15), periods=num_predictions, freq='15T')

# Save forecast to CSV
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Total_Charger_Consumption": predictions_inv})
forecast_df.to_csv("forecast_next_3days_safe.csv", index=False)

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(df.index, scaler_target.inverse_transform(df[['total_chargers']]), label="Historical Total Chargers")
plt.plot(future_dates, predictions_inv, label="Predicted Total Chargers", color="red")
plt.xlabel("Date")
plt.ylabel("Total Charger Consumption (kW)")
plt.title("Safe Feature-Based Forecast for Next 3 Days")
plt.legend()
plt.xticks(rotation=45)
plt.show()