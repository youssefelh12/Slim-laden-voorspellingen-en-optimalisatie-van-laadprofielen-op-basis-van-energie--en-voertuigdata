import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import holidays
import calendar

# -------------------------------
# 1. Data Loading and Cleaning
# -------------------------------

# Load the monthly data.
df = pd.read_csv("api_data/monthly_api_data.csv")

# Convert the Month column to datetime (format: YYYY-MM) and sort the data.
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df.sort_values("Month", inplace=True)

# Filter to use only rows with Measurement "Grid Organi lbc" and Phase "PHASEA"
df = df[(df["Measurement"] == "Grid Organi lbc") & (df["Phase"] == "PHASEA")]

# Drop columns that are no longer needed.
df.drop(columns=["Measurement", "Phase"], inplace=True)

# -------------------------------
# 2. Feature Engineering (Monthly Level)
# -------------------------------

# Using Belgium holidays.
be_holidays = holidays.BE()

def count_holidays_and_weekends(ts):
    """
    Given a Timestamp representing a month, count the number of holidays and weekend days in that month.
    """
    year = ts.year
    month = ts.month
    _, last_day = calendar.monthrange(year, month)
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = pd.Timestamp(year=year, month=month, day=last_day)
    days = pd.date_range(start_date, end_date, freq="D")
    
    holiday_count = sum(1 for day in days if day in be_holidays)
    weekend_count = sum(1 for day in days if day.weekday() >= 5)  # Saturday=5, Sunday=6
    
    return pd.Series({"holiday_count": holiday_count, "weekend_count": weekend_count})

# Apply the function to create new features based on the Month.
holiday_weekend_features = df["Month"].apply(count_holidays_and_weekends)
df = pd.concat([df, holiday_weekend_features], axis=1)

# (Optional) Add additional time features such as month number and year.
df["month_number"] = df["Month"].dt.month
df["year"] = df["Month"].dt.year

# -------------------------------
# 3. Setting the Index
# -------------------------------

# Set the Month column as the index. The data is already monthly aggregated.
df.set_index("Month", inplace=True)

# -------------------------------
# 4. Creating Lag Features
# -------------------------------

# To capture temporal dependencies, create lag features using the previous 4 months’ consumption.
for lag in range(1, 5):
    df[f'lag_{lag}'] = df["Consumption"].shift(lag)

# Drop the initial rows that have NaN values due to lag creation.
df.dropna(inplace=True)

# -------------------------------
# 5. Define Features and Target
# -------------------------------

# Our target is the monthly Consumption.
target = "Consumption"

# Define the feature columns (all columns except the target).
feature_cols = [col for col in df.columns if col != target]

X = df[feature_cols]
y = df[target]

# -------------------------------
# 6. Train-Test Split (80/20 Split)
# -------------------------------

# Calculate the index for an 80/20 split.
split_index = int(0.8 * len(df))
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print(f"Training data from {X_train.index.min().date()} to {X_train.index.max().date()}")
print(f"Testing data from {X_test.index.min().date()} to {X_test.index.max().date()}")

# -------------------------------
# 7. Data Normalization
# -------------------------------

# Initialize scalers for features and target.
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit the scalers on the training data and transform.
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Note: y is a 1D array; we reshape it for scaling.
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# -------------------------------
# 8. Model Training
# -------------------------------

# Initialize and train a RandomForestRegressor on the scaled data.
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train_scaled)

# -------------------------------
# 9. Model Evaluation
# -------------------------------

# Make predictions on the scaled test set.
y_pred_scaled = model.predict(X_test_scaled)

# Inverse transform predictions to original scale.
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Calculate evaluation metrics in the original scale.
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Plot actual vs. predicted consumption.
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual Consumption", marker="o")
plt.plot(y_test.index, y_pred, label="Predicted Consumption", marker="x")
plt.xlabel("Month")
plt.ylabel("Monthly Energy Consumption")
plt.title("Monthly Energy Consumption Prediction for Grid Organi lbc (PHASEA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
