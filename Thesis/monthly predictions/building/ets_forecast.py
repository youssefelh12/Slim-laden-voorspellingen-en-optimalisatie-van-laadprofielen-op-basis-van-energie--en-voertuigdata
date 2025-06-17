# ===============================================================
# Monthly electricity – SES & Theta benchmark on 12 observations
# ===============================================================
import os
import pandas as pd
import numpy as np
import holidays, calendar
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load, Sum & Compute Building Consumption
# -------------------------------
df = pd.read_csv("api_data/monthly_api_data.csv")

# Parse Month & sort
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df.sort_values("Month", inplace=True)

# Sum per Month & Measurement
df_grouped = (
    df
    .groupby(["Month", "Measurement"])["Consumption"]
    .sum()
    .unstack(fill_value=0)
)

# Ensure all needed cols exist
# Ensure the four measurement columns exist
for col in ["Grid Organi lbc", "Chargers", "Chargers achteraan", "Solar"]:
    if col not in df_grouped.columns:      # <- use .columns, not .setdefault
        df_grouped[col] = 0


# Building consumption = grid − chargers − chargers achteraan + solar
df_grouped["Building_Consumption"] = (
    df_grouped["Grid Organi lbc"]
    - df_grouped["Chargers"]
    - df_grouped["Chargers achteraan"]
    + df_grouped["Solar"]
)

# Keep Month & total, convert kWh ➔ MWh
df = (df_grouped.reset_index()[["Month", "Building_Consumption"]]
      .rename(columns={"Building_Consumption": "Total_building_consumption"}))
df["Total_building_consumption"] /= 1000

# -------------------------------
# 2. *Optional* Feature Engineering (not used by the models)
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

# -------------------------------
# 3. Train/Test Split (80/20)
# -------------------------------
target = "Total_building_consumption"
y = df[target]

split = max(1, int(0.8 * len(df)))        # at least one point in the test set
y_train, y_test = y.iloc[:split], y.iloc[split:]
months_test     = df["Month"].iloc[split:]

print(f"Training: {df['Month'].iloc[0].date()} → {df['Month'].iloc[split-1].date()}")
print(f"Testing:  {months_test.iloc[0].date()} → {months_test.iloc[-1].date()}")

# -------------------------------
# 4. Fit Simple Exponential Smoothing
# -------------------------------
ses_model     = SimpleExpSmoothing(y_train, initialization_method="estimated").fit()
ses_forecast  = ses_model.forecast(len(y_test))

# -------------------------------
# 5. Fit Theta Method (level + drift)
# -------------------------------
theta_model   = ThetaModel(y_train, deseasonalize=False).fit()
theta_forecast = theta_model.forecast(len(y_test))

# -------------------------------
# 6. Evaluation helpers
# -------------------------------
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-10
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)

def mase(y_true, y_pred, y_train):
    d = np.abs(np.diff(y_train)).mean() if len(y_train) > 1 else 1e-10
    return np.mean(np.abs(y_true - y_pred)) / d

def metrics(y_true, y_pred, y_train):
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    nrmse = rmse / (y_true.max() - y_true.min())
    return {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "NRMSE": nrmse,
        "MAPE": mape(y_true, y_pred), "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train),
        "R²": r2_score(y_true, y_pred)
    }

ses_metrics   = metrics(y_test, ses_forecast, y_train)
theta_metrics = metrics(y_test, theta_forecast, y_train)

print("\n=== Forecast metrics (MWh) ===")
for model_name, m in {"SES": ses_metrics, "Theta": theta_metrics}.items():
    print(f"\n{model_name}")
    for k, v in m.items():
        print(f"{k:8}: {v:.4f}")

# -------------------------------
# 7. Bar Plot Actual vs Forecasts
# -------------------------------
os.makedirs("results/monthly_building", exist_ok=True)
x     = np.arange(len(months_test))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, y_test,         width, label="Actual")
plt.bar(x,          ses_forecast,  width, label="SES forecast")
plt.bar(x + width,  theta_forecast,width, label="Theta forecast")

plt.xlabel("Month")
plt.ylabel("Building Consumption (MWh)")
plt.title("SES & Theta Forecast vs Actual (Bar Plot)")
plt.xticks(x, [m.strftime("%Y-%m") for m in months_test], rotation=45)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()

out_path = "results/monthly_building/ses_theta_building_consumption_MWh_bar.png"
plt.savefig(out_path)
plt.show()
print(f"\nBar chart saved to {out_path}")
