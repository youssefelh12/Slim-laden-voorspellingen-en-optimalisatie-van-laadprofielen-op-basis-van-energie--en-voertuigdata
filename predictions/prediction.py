import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import holidays
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# 1. Data Loading and Preprocessing
# -----------------------------
# Replace with your dataset path.
df = pd.read_csv("Data/15min2024_consumption.csv", parse_dates=["Date"])   
df = df.sort_values("Date")

# Convert grid consumption columns (which use comma as decimal separator) to float.
grid_cols = ["Grid Organi lbc (L1) [kW]", "Grid Organi lbc (L2) [kW]", "Grid Organi lbc (L3) [kW]"]
for col in grid_cols:
    df[col] = df[col].str.replace(",", ".").astype(float)

# Compute total grid consumption by summing the three grid features.
df["total_consumption"] = (df["Grid Organi lbc (L1) [kW]"] +
                           df["Grid Organi lbc (L2) [kW]"] +
                           df["Grid Organi lbc (L3) [kW]"])

# Set the Date column as index.
df.set_index("Date", inplace=True)

# -----------------------------
# 2. Feature Engineering
# -----------------------------
# Create additional time-based features:
#   - Hour and minute of the day.
#   - Day of week (0=Monday ... 6=Sunday).
#   - is_weekend: 1 if Saturday or Sunday, else 0.
#   - is_festive: 1 if the day is a Belgian holiday.
be_holidays = holidays.BE()  # Belgian holidays

df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['is_festive'] = df.index.to_series().apply(lambda x: 1 if x in be_holidays else 0)

# -----------------------------
# 3. Scaling and Creating Combined Features
# -----------------------------
# For our multi-feature model, we will use:
#   - The target: total_consumption.
#   - Exogenous features: hour, minute, day_of_week, is_weekend, is_festive.
# We use separate scalers for the consumption (target) and for the exogenous features.

# Prepare target (consumption) as a 2D array.
consumption = df["total_consumption"].values.reshape(-1, 1)
scaler_consumption = MinMaxScaler()
consumption_scaled = scaler_consumption.fit_transform(consumption)

# Prepare exogenous features.
exog_features = df[['hour', 'minute', 'day_of_week', 'is_weekend', 'is_festive']]
scaler_exog = MinMaxScaler()
exog_scaled = scaler_exog.fit_transform(exog_features)

# Combine the scaled consumption and exogenous features.
# Each row will have 6 features: [consumption, hour, minute, day_of_week, is_weekend, is_festive]
X_all = np.hstack([consumption_scaled, exog_scaled])  # shape: (n_samples, 6)

# -----------------------------
# 4. Preparing Sequences for LSTM
# -----------------------------
def create_sequences(data, target, seq_length):
    """
    data: combined feature array (n_samples, num_features)
    target: target consumption column (n_samples, 1)
    """
    X_seq = []
    y_seq = []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(target[i+seq_length])  # target is consumption only
    return np.array(X_seq), np.array(y_seq)

seq_length = 96  # one day of past data at 15-min intervals
X_seq, y_seq = create_sequences(X_all, consumption_scaled, seq_length)
print("X_seq shape:", X_seq.shape)  # (num_samples, 96, 6)
print("y_seq shape:", y_seq.shape)  # (num_samples, 1)

# -----------------------------
# 5. Building and Training the LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, X_all.shape[1])))
model.add(Dense(1))  # Predict scaled consumption
model.compile(optimizer='adam', loss='mse')

# Train the model. Adjust epochs and batch_size as needed.
model.fit(X_seq, y_seq, epochs=20, batch_size=64)

# -----------------------------
# 6. Forecasting the Next 3 Days with Exogenous Features
# -----------------------------
num_predictions = 288  # 3 days * 24 hours * 4 intervals per hour
predictions_scaled = []  # to store predicted consumption (scaled)

# The current sequence: use the last available sequence from training.
current_seq = X_all[-seq_length:]  # shape: (seq_length, 6)
current_seq = current_seq.reshape(1, seq_length, X_all.shape[1])

# Generate future timestamps (15-min intervals)
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=15), periods=num_predictions, freq='15T')

# Pre-compute the exogenous features for each future timestamp.
future_exog = []
for dt in future_dates:
    hour = dt.hour
    minute = dt.minute
    day_of_week = dt.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    is_festive = 1 if dt in be_holidays else 0
    future_exog.append([hour, minute, day_of_week, is_weekend, is_festive])
future_exog = np.array(future_exog)  # shape: (num_predictions, 5)
# Scale future exogenous features using the fitted scaler.
future_exog_scaled = scaler_exog.transform(future_exog)

# Forecast iteratively.
forecast_features = []  # to store full multi-feature rows for forecasting (for debugging if needed)
for i in range(num_predictions):
    # Predict the next consumption value (scaled)
    pred = model.predict(current_seq)
    pred_val = pred[0, 0]  # scaled predicted consumption
    predictions_scaled.append(pred_val)
    
    # Get the exogenous features for the current forecast step (already scaled)
    exog_scaled_future = future_exog_scaled[i]  # shape: (5,)
    
    # Create a new row vector: [predicted_consumption, exog features]
    new_row = np.hstack([[pred_val], exog_scaled_future])  # shape: (6,)
    forecast_features.append(new_row)
    
    # Append new_row to the current sequence and drop the first time step.
    new_row = new_row.reshape(1, 1, X_all.shape[1])  # shape: (1,1,6)
    current_seq = np.concatenate((current_seq[:, 1:, :], new_row), axis=1)

# Inverse-transform predicted consumption values.
predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
predictions_inv = scaler_consumption.inverse_transform(predictions_scaled).flatten()

# -----------------------------
# 7. Plotting the Forecast
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["total_consumption"], label="Historical Total Consumption")
plt.plot(future_dates, predictions_inv, label="Predicted Total Consumption", color="red")
plt.xlabel("Date")
plt.ylabel("Total Consumption (kW)")
plt.title("Forecast for Next 3 Days with Exogenous Features")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Saving the Forecast to CSV
# -----------------------------
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Total_Consumption": predictions_inv
})
forecast_df.to_csv("forecast_next_3days.csv", index=False)
print(forecast_df.head())
