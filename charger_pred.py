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

# Convert grid consumption columns to float (if needed)
grid_cols = ["Grid Organi lbc (L1) [kW]", "Grid Organi lbc (L2) [kW]", "Grid Organi lbc (L3) [kW]"]
for col in grid_cols:
    df[col] = df[col].str.replace(",", ".").astype(float)

# Convert charger columns to float.
charger_cols = ["Chargers (L1) [kW]", "Chargers (L2) [kW]", "Chargers (L3) [kW]"]
for col in charger_cols:
    # Ensure values are strings before replacing comma and converting to float.
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

# Compute total charger consumption by summing the three charger columns.
df["total_chargers"] = df["Chargers (L1) [kW]"] + df["Chargers (L2) [kW]"] + df["Chargers (L3) [kW]"]

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
# We now want to forecast total charger consumption.
# Prepare the target (total_chargers) as a 2D array.
chargers = df["total_chargers"].values.reshape(-1, 1)
scaler_chargers = MinMaxScaler()
chargers_scaled = scaler_chargers.fit_transform(chargers)

# Prepare exogenous features: hour, minute, day_of_week, is_weekend, is_festive.
exog_features = df[['hour', 'minute', 'day_of_week', 'is_weekend', 'is_festive']]
scaler_exog = MinMaxScaler()
exog_scaled = scaler_exog.fit_transform(exog_features)

# Combine the scaled charger target and exogenous features.
# Each time step now has 6 features: [total_chargers, hour, minute, day_of_week, is_weekend, is_festive]
X_all = np.hstack([chargers_scaled, exog_scaled])  # shape: (n_samples, 6)

# -----------------------------
# 4. Preparing Sequences for LSTM
# -----------------------------
def create_sequences(data, target, seq_length):
    """
    data: combined feature array (n_samples, num_features)
    target: target column (n_samples, 1) for consumption (or chargers)
    """
    X_seq = []
    y_seq = []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(target[i+seq_length])  # target is the charger consumption
    return np.array(X_seq), np.array(y_seq)

seq_length = 96  # one day of past data at 15-min intervals
X_seq, y_seq = create_sequences(X_all, chargers_scaled, seq_length)
print("X_seq shape:", X_seq.shape)  # (num_samples, 96, 6)
print("y_seq shape:", y_seq.shape)  # (num_samples, 1)

# -----------------------------
# 5. Building and Training the LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, X_all.shape[1])))
model.add(Dense(1))  # Predict scaled total charger consumption
model.compile(optimizer='adam', loss='mse')

# Train the model. Adjust epochs and batch_size as needed.
model.fit(X_seq, y_seq, epochs=20, batch_size=64)

# -----------------------------
# 6. Forecasting the Next 3 Days with Exogenous Features
# -----------------------------
num_predictions = 288  # 3 days * 24 hours * 4 intervals per hour
predictions_scaled = []  # to store predicted (scaled) charger consumption

# Use the last available sequence from the training data.
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
for i in range(num_predictions):
    # Predict the next value (scaled charger consumption)
    pred = model.predict(current_seq)
    pred_val = pred[0, 0]  # scaled prediction
    predictions_scaled.append(pred_val)
    
    # Retrieve the exogenous features for the current future time step (already scaled)
    exog_scaled_future = future_exog_scaled[i]  # shape: (5,)
    
    # Form a new row with predicted charger and exogenous features: shape (6,)
    new_row = np.hstack([[pred_val], exog_scaled_future])
    
    # Append new_row to the sequence (maintaining 3 dimensions)
    new_row = new_row.reshape(1, 1, X_all.shape[1])
    current_seq = np.concatenate((current_seq[:, 1:, :], new_row), axis=1)

# Inverse-transform the predictions to the original scale.
predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
predictions_inv = scaler_chargers.inverse_transform(predictions_scaled).flatten()

# -----------------------------
# 7. Save the predictions to a CSV file
# -----------------------------
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Total_Chargers': predictions_inv
})

# Save to CSV
predictions_df.to_csv('predictions_chargers.csv', index=False)

print("Predictions saved to predictions_chargers.csv")



