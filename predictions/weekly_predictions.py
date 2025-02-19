import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays

# -------------------------------
# 1. Data Loading and Cleaning
# -------------------------------

# Assuming your CSV file is named "weekly_consumption.csv" in the "Data" folder.
# Numbers use spaces for thousands and a comma as the decimal separator.
df = pd.read_csv("Data/weekly_consumption.csv", thousands=" ", decimal=",")

# Convert the Date column to datetime and sort by date
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

# -------------------------------
# 2. Feature Engineering (Daily Level)
# -------------------------------

# Add a holiday flag.
# Here we assume the data is from Belgium. Adjust as needed.
be_holidays = holidays.BE()
df["is_holiday"] = df["Date"].apply(lambda d: 1 if d in be_holidays else 0)

# Add a weekend flag (Saturday=5, Sunday=6)
df["is_weekend"] = df["Date"].dt.weekday.isin([5, 6]).astype(int)

# -------------------------------
# 3. Aggregate to Weekly Data
# -------------------------------

# Set Date as the index for resampling.
df.set_index("Date", inplace=True)

# Define how to aggregate each column.
# For energy readings, use sum (to get total energy per week).
# For the holiday/weekend flags, summing counts the number of days with the flag.
agg_dict = {
    "Consumption [kWh]": "sum",
    "Grid Organi lbc (L1) [kWh]": "sum",
    "Grid Organi lbc (L2) [kWh]": "sum",
    "Grid Organi lbc (L3) [kWh]": "sum",
    "Chargers (L1) [kWh]": "sum",
    "Chargers (L2) [kWh]": "sum",
    "Chargers (L3) [kWh]": "sum",
    "Solar (L1) [kWh]": "sum",
    "Solar (L2) [kWh]": "sum",
    "Solar (L3) [kWh]": "sum",
    "Chargers achteraan (L1) [kWh]": "sum",
    "Chargers achteraan (L2) [kWh]": "sum",
    "Chargers achteraan (L3) [kWh]": "sum",
    "is_holiday": "sum",
    "is_weekend": "sum",
}

# Resample the data by week (default: weeks ending on Sunday)
df_weekly = df.resample("W").agg(agg_dict)

# Add some time-related features from the weekly index
df_weekly["weekofyear"] = df_weekly.index.isocalendar().week.astype(int)
df_weekly["year"] = df_weekly.index.year

# -------------------------------
# 4. Creating Lag Features
# -------------------------------

# To help the model capture temporal dependencies, create lag features for the target.
# Here we add the past 4 weeks' consumption as features.
for lag in range(1, 5):
    df_weekly[f'lag_{lag}'] = df_weekly["Consumption [kWh]"].shift(lag)

# Drop the first few rows with NaN values (because of lag creation)
df_weekly.dropna(inplace=True)

# -------------------------------
# 5. Define Features and Target
# -------------------------------

# Our target is the weekly total consumption.
target = "Consumption [kWh]"

# Define a list of features for the model.
feature_cols = [col for col in df_weekly.columns if col != target]

X = df_weekly[feature_cols]
y = df_weekly[target]

# -------------------------------
# 6. Train-Test Split using Last Month as Test Set
# -------------------------------

# Instead of a fixed 70/30 split, reserve the last month of data for testing.
cutoff_date = df_weekly.index.max() - pd.DateOffset(months=1)

X_train = X[X.index < cutoff_date]
y_train = y[y.index < cutoff_date]
X_test = X[X.index >= cutoff_date]
y_test = y[y.index >= cutoff_date]

print(f"Training data from {X_train.index.min().date()} to {X_train.index.max().date()}")
print(f"Testing data from {X_test.index.min().date()} to {X_test.index.max().date()}")

# -------------------------------
# 7. Model Training
# -------------------------------

# Initialize and train a RandomForestRegressor
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 8. Model Evaluation
# -------------------------------

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Plot actual vs. predicted consumption
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual Consumption", marker="o")
plt.plot(y_test.index, y_pred, label="Predicted Consumption", marker="x")
plt.xlabel("Date")
plt.ylabel("Weekly Energy Consumption [kWh]")
plt.title("Weekly Energy Consumption Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
