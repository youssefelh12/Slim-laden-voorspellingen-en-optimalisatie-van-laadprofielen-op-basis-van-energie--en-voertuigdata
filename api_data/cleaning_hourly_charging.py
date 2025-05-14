import pandas as pd

# -------------------- I/O paths --------------------
input_file  = "./api_data/hourly_year_13-05.csv"
output_file = "Charging_data_13-05.csv"

# -------------------- Load & tidy ------------------
df = pd.read_csv(input_file, sep=",", decimal=",", quotechar='"')
df["Date"] = pd.to_datetime(df["Date"])

# -------------------- Helper lists -----------------
charger_cols = [c for c in df.columns if c.lower().startswith("chargers")]
charger_achteraan_cols = [c for c in charger_cols if "achteraan" in c.lower()]
charger_other_cols     = [c for c in charger_cols if c not in charger_achteraan_cols]

grid_cols  = ["Grid Organi lbc (L1) [kWh]",
              "Grid Organi lbc (L2) [kWh]",
              "Grid Organi lbc (L3) [kWh]"]
solar_cols = ["Solar (L1) [kWh]",
              "Solar (L2) [kWh]",
              "Solar (L3) [kWh]"]

# -------------------- Column totals ----------------
df["Grid"]  = df[grid_cols].sum(axis=1)
df["Solar"] = df[solar_cols].sum(axis=1)

# sum of other chargers and achteraan chargers
other_sum    = df[charger_other_cols].sum(axis=1)
achteraan_sum = df[charger_achteraan_cols].sum(axis=1)

# cutoff date: October 6, 2024
cutoff = pd.Timestamp("2024-10-04")

# only double achteraan before cutoff
extra_achteraan = achteraan_sum.where(df["Date"] < cutoff, 0)

df["Chargers"] = other_sum + achteraan_sum +  4 * extra_achteraan

# -------------------- Result -----------------------
df.sort_values("Date", inplace=True)
df_result = df[["Date", "Chargers"]]
df_result.to_csv(output_file, index=False, sep=",", decimal=".")
