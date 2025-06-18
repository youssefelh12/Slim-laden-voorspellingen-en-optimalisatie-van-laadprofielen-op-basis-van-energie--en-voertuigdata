import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- I/O paths --------------------
input_file  = "./api_data/hourly_year_13-05.csv"
output_file = "./building_forecasts/building_data.csv"

# -------------------- Load & initial filter --------------------
df = pd.read_csv(input_file, sep=",", decimal=",", quotechar='"')
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date and remove data before June 27, 2024
df.sort_values("Date", inplace=True)
df = df[df["Date"] >= pd.Timestamp("2024-10-10")]

# -------------------- Column groups -------------------
# Grid and Solar columns (explicit)
grid_cols = [
    "Grid Organi lbc (L1) [kWh]",
    "Grid Organi lbc (L2) [kWh]",
    "Grid Organi lbc (L3) [kWh]"
]
solar_cols = [
    "Solar (L1) [kWh]",
    "Solar (L2) [kWh]",
    "Solar (L3) [kWh]"
]
# Charger columns (explicit)
charger_other_cols = [
    "Chargers (L1) [kWh]",
    "Chargers (L2) [kWh]",
    "Chargers (L3) [kWh]"
]
charger_achteraan_cols = [
    "Chargers achteraan (L1) [kWh]",
    "Chargers achteraan (L2) [kWh]",
    "Chargers achteraan (L3) [kWh]"
]

# -------------------- Column totals ------------------
# Sum grid and solar
df["Grid"]  = df[grid_cols].sum(axis=1)
df["Solar"] = df[solar_cols].sum(axis=1)
df["Chargers_vooraan"] = df[charger_other_cols].sum(axis=1)
df["Chargers_achteraan"] = df[charger_achteraan_cols].sum(axis=1)



# -------------------- Sum all chargers ------------------
df["Chargers"] = df["Chargers_vooraan"] + df["Chargers_achteraan"]

# -------------------- Compute consumption ------------------
df['Consumption'] = df['Grid'] - df['Chargers'] + df['Solar']

# -------------------- Save results -------------------
df.sort_values("Date", inplace=True)
df_result = df[["Date", "Consumption"]]
df_result.to_csv(output_file, index=False, sep=",", decimal=".")

# -------------------- Plot --------------------------
# Use Date as index to plot time series
df.set_index("Date", inplace=True)
plt.figure(figsize=(14, 4))
plt.plot(df["Consumption"], label="Consumption")
plt.title("Hourly building consumption", fontsize=16)
plt.xlabel("Time", fontsize=16)
plt.ylabel("kWh", fontsize=16)
plt.tight_layout()
plt.legend()
plt.show()
