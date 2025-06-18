import os
import pandas as pd
import numpy as np
import holidays
import calendar
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load, Sum & Compute Building Consumption
# -------------------------------
df = pd.read_csv("api_data/monthly_api_data.csv")

# Parse Month & sort
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df.sort_values("Month", inplace=True)

# Sommeer per Month en Measurement
df_grouped = (
    df
    .groupby(["Month", "Measurement"])["Consumption"]
    .sum()
    .unstack(fill_value=0)
)

# Zorg dat alle kolommen bestaan
for col in ["Grid Organi lbc", "Chargers", "Chargers achteraan", "Solar"]:
    if col not in df_grouped.columns:
        df_grouped[col] = 0

# Bereken gebouwverbruik = grid - chargers - chargers achteraan + solar
df_grouped["Building_Consumption"] = (
    df_grouped["Grid Organi lbc"]
    - df_grouped["Chargers"]
    - df_grouped["Chargers achteraan"]
    + df_grouped["Solar"]
)

# Reset index en selecteer alleen Month & Building_Consumption
df = (
    df_grouped
    .reset_index()[["Month", "Building_Consumption"]]
    .rename(columns={"Building_Consumption": "Total_charger_consumption"})
)

# Zet om van kWh naar MWh
df["Total_charger_consumption"] /= 1000

# -------------------------------
# 2. Safe Feature Engineering
# -------------------------------
be_holidays = holidays.BE()

def count_holidays_and_weekends(ts):
    year, month = ts.year, ts.month
    _, last_day = calendar.monthrange(year, month)
    days = pd.date_range(ts.replace(day=1), ts.replace(day=last_day), freq="D")
    return pd.Series({
        "holiday_count": sum(d in be_holidays for d in days),
        "weekend_count": sum(d.weekday() >= 5 for d in days)
    })

df[["holiday_count", "weekend_count"]] = df["Month"].apply(count_holidays_and_weekends)
df["month_number"] = df["Month"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month_number"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_number"] / 12)
df["trend_index"] = np.arange(len(df))
df["lag_1"]  = df["Total_charger_consumption"].shift(1)
df["lag_12"] = df["Total_charger_consumption"].shift(12)
df.dropna(subset=["lag_1", "lag_12"], inplace=True)

# -------------------------------
# 3. Define target & features
# -------------------------------
target = "Total_charger_consumption"
features = [
    "holiday_count",
    "weekend_count",
    "month_sin",
    "month_cos",
    "trend_index",
    "lag_1",
    "lag_12",
]
X = df[features]
y = df[target]

# -------------------------------
# 4. Train/Test Split (80/20)
# -------------------------------
split = int(0.8 * len(df))
X_train, X_test   = X.iloc[:split], X.iloc[split:]
y_train, y_test   = y.iloc[:split], y.iloc[split:]
months_test       = df["Month"].iloc[split:]

# -------------------------------
# 5. Train Random Forest
# -------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# -------------------------------
# 6. Forecast & Evaluate
# -------------------------------
y_pred = rf.predict(X_test)

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

def calculate_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-10
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)

def calculate_mase(y_true, y_pred, y_train):
    d = np.abs(np.diff(y_train)).mean() if len(y_train) > 1 else 1e-10
    return np.mean(np.abs(y_true - y_pred)) / d

mae   = mean_absolute_error(y_test, y_pred)
mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)
nrmse = rmse / (y_test.max() - y_test.min())
mape_ = calculate_mape(y_test.values, y_pred)
smape_= calculate_smape(y_test.values, y_pred)
mase_ = calculate_mase(y_test.values, y_pred, y_train.values)
r2    = r2_score(y_test, y_pred)
n, p  = len(y_test), X_test.shape[1]
r2_adj= 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\n=== Random Forest Forecast Metrics (MWh) ===")
print(f"MAE:          {mae:.2f}")
print(f"MSE:          {mse:.2f}")
print(f"RMSE:         {rmse:.2f}")
print(f"NRMSE:        {nrmse:.4f}")
print(f"MAPE:         {mape_:.2f}%")
print(f"sMAPE:        {smape_:.2f}%")
print(f"MASE:         {mase_:.2f}")
print(f"R²:           {r2:.4f}")
print(f"Adjusted R²:  {r2_adj:.4f}")

# -------------------------------
# 7. Bar Plot Actual vs Forecast & Save
# -------------------------------
os.makedirs("results", exist_ok=True)
x     = np.arange(len(months_test))
width = 0.35

plt.figure(figsize=(12, 5))
plt.bar(x - width/2, y_test,  width, label="Actual")
plt.bar(x + width/2, y_pred,  width, label="Forecast")

plt.xlabel("Month")
plt.ylabel("Building Consumption (MWh)")
plt.title("Random Forest: Verbruik Gebouw (Bar Plot)")
plt.xticks(x, [m.strftime("%Y-%m") for m in months_test], rotation=45)
plt.legend()
plt.grid(axis='y')
plt.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig("results/random_forest_building_consumption_MWh_bar.png")
plt.show()
