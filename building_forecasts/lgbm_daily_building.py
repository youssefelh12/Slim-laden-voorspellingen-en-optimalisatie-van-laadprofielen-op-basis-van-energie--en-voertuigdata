# ───────────────────────── imports ─────────────────────────
import warnings, holidays, numpy as np, pandas as pd, lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────── config ───────────────────────────
CONSUMP_FILE    = "./api_data/daily_building_consumption_0624.csv"
TEMP_FILE       = "./api_data/daily_temperature_data2.csv"        # optional
CALENDAR_FILE   = "./layout1_full_calendar_2023-2025.csv"
TOTAL_HEADCOUNT = 105
TRIM_START      = pd.Timestamp("2024-10-01")   # ← keep data from 1 Oct 2024
RESULTS_DIR     = "results/daily_lgbm_backtest"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# ───────────── 1. load & trim consumption ─────────────────
df = pd.read_csv(CONSUMP_FILE, parse_dates=["Day"]).set_index("Day")
df.rename(columns={"Total_consumption": "y"}, inplace=True)
df = df[df.index >= TRIM_START].sort_index()   # ← *** trim here ***

# ───────────── 2. merge temperature (tz-safe) ────────────
if Path(TEMP_FILE).exists():
    tdf = (pd.read_csv(TEMP_FILE, parse_dates=["date"])
             .rename(columns={"date": "Day"})
             .set_index("Day"))
    if tdf.index.tz is not None:
        tdf.index = tdf.index.tz_convert("Europe/Brussels").tz_localize(None)
    df = df.join(tdf["temperature_2m_mean"], how="left")

# ───────────── 3. calendar counts & engineered flags ─────
cal = (pd.read_csv(CALENDAR_FILE, parse_dates=["Datum"])
         .rename(columns={
             "Datum": "Day",
             "Totaal_Vakantiedagen":  "vacation_cnt",
             "Totaal_Thuiswerkdagen": "homework_cnt"})
         .set_index("Day"))
df = df.join(cal, how="left").fillna({"vacation_cnt": 0, "homework_cnt": 0})

be_holidays        = holidays.BE()
df["weekday"]      = df.index.dayofweek
df["is_weekend"]   = (df["weekday"] >= 5).astype(int)
df["is_holiday"]   = df.index.to_series().isin(be_holidays).astype(int)

tkd_dates = pd.to_datetime([
    "2023-09-13","2023-10-26","2023-11-14","2023-12-20",
    "2024-01-12","2024-02-07","2024-03-14","2024-04-16",
    "2024-05-13","2024-06-07","2024-10-22","2024-11-28",
    "2024-12-18","2025-01-10","2025-02-13","2025-03-18",
    "2025-04-22","2025-05-12","2025-06-06"
]).normalize()
df["is_terugkomdag"] = df.index.normalize().isin(tkd_dates).astype(int)

mask_workday = (df["weekday"] < 5) & (df["is_holiday"] == 0)
df["work_at_office"] = np.where(
    mask_workday,
    (TOTAL_HEADCOUNT - df["vacation_cnt"] - df["homework_cnt"]).clip(lower=0),
    0
)

# ───────────── 4. lag / rolling stats ─────────────────────
df["lag_1"]       = df["y"].shift(1)
df["lag_7"]       = df["y"].shift(7)
df["roll_mean_7"] = df["y"].rolling(7).mean()
df["roll_std_7"]  = df["y"].rolling(7).std()
df.dropna(inplace=True)          # drop rows lost to lags

# ───────────── 5. 30-day rolling back-test ───────────────
cutoff   = df.index.max() - pd.Timedelta(days=30)
days_iter = pd.date_range(cutoff, df.index.max(), freq="D")

base_cols = [
    "is_weekend","is_holiday","is_terugkomdag","weekday",
    "lag_1","lag_7","roll_mean_7","roll_std_7",
    "work_at_office","avg_cons_weekday"
]
if "temperature_2m_mean" in df.columns:
    base_cols.append("temperature_2m_mean")

preds, reals, wds = [], [], []

for day in days_iter:
    train = df[df.index < day].copy()
    test  = df.loc[[day]].copy()

    avg_wd = (train.groupby("weekday")["y"]
                    .mean().rename("avg_cons_weekday")
                    .reset_index())
    train = train.merge(avg_wd, on="weekday", how="left")
    test  =  test.merge(avg_wd, on="weekday", how="left")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=600, learning_rate=0.05,
        random_state=42
    ).fit(train[base_cols], train["y"])

    preds.append(model.predict(test[base_cols])[0])
    reals.append(test["y"].iloc[0])
    wds.append(test["weekday"].iloc[0])

# ───────────── 6. metrics & CSV ───────────────────────────
mae  = mean_absolute_error(reals, preds)
rmse = np.sqrt(mean_squared_error(reals, preds))
r2   = r2_score(reals, preds)

print("\n--- 30-Day Daily Back-test (data ≥ 2024-10-01) ---")
print(f"MAE : {mae:.2f} kWh")
print(f"RMSE: {rmse:.2f} kWh")
print(f"R²  : {r2:.2f}")

out = pd.DataFrame({"y": reals, "yhat": preds, "weekday": wds}, index=days_iter)
out.to_csv(Path(RESULTS_DIR) / "30day_backtest.csv")

# ───────────── 7. plots ──────────────────────────────────
labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
r2_by_wd = [
    r2_score(out.loc[out.weekday == i, "y"],
             out.loc[out.weekday == i, "yhat"]) if (out.weekday == i).any() else np.nan
    for i in range(7)
]

plt.figure(figsize=(9,4))
bars = plt.bar(labels + ["All"], r2_by_wd + [r2], color="steelblue")
plt.axhline(0, color="k", lw=0.7)
plt.ylabel("R²")
plt.title("R² by Weekday (Last 30 Days, data ≥ Oct 2024)")
for bar, v in zip(bars, r2_by_wd + [r2]):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
             ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(Path(RESULTS_DIR) / "r2_by_weekday.png", dpi=120)
plt.show()

plt.figure(figsize=(12,5))
plt.plot(out.index, out["y"],    label="Actual",    alpha=0.7)
plt.plot(out.index, out["yhat"], label="Predicted", alpha=0.8)
plt.title("Actual vs Predicted — Last 30 Days (data ≥ Oct 2024)")
plt.xlabel("Date"); plt.ylabel("Daily kWh")
plt.legend(); plt.tight_layout()
plt.savefig(Path(RESULTS_DIR) / "actual_vs_predicted.png", dpi=120)
plt.show()

print(f"\nOutputs saved in {RESULTS_DIR}")
