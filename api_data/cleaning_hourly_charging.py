import pandas as pd

# -------------------- I/O paths --------------------
input_file  = "./api_data/elek_hourly.csv"
output_file = "Charging_data_new.csv"

# -------------------- Load & tidy ------------------
df = pd.read_csv(input_file, sep=",", decimal=",", quotechar='"')
df["Date"] = pd.to_datetime(df["Date"])

# -------------------- Helper lists -----------------
# Anything beginning with “Chargers”, e.g.
#   “Chargers (L1) [kWh]”, “Chargers achteraan (L2) [kWh]”, etc.
charger_cols = [c for c in df.columns if c.lower().startswith("chargers")]

# If you still want Solar/Grid, leave these as they were
grid_cols   = ["Grid Organi lbc (L1) [kWh]",
               "Grid Organi lbc (L2) [kWh]",
               "Grid Organi lbc (L3) [kWh]"]

solar_cols  = ["Solar (L1) [kWh]",
               "Solar (L2) [kWh]",
               "Solar (L3) [kWh]"]

# -------------------- Column totals ----------------
df["Grid"]     = df[grid_cols].sum(axis=1)
df["Solar"]    = df[solar_cols].sum(axis=1)
df["Chargers"] = df[charger_cols].sum(axis=1)

# -------------------- Result -----------------------
df.sort_values("Date", inplace=True)
df_result = df[["Date", "Chargers"]]

df_result.to_csv(output_file, index=False, sep=",", decimal=".")
