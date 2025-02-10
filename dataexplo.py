import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read data
csv_path = "Data/week_consumption.csv"
data = pd.read_csv(csv_path, parse_dates=["Date"], thousands=',')
data = data.sort_values(by="Date")

# Summing L1, L2, L3 values for chargers
chargers_total = data[["Chargers (L1) [kW]", "Chargers (L2) [kW]", "Chargers (L3) [kW]"]].sum(axis=1)
chargers_achteraan_total = data[["Chargers achteraan (L1) [kW]", "Chargers achteraan (L2) [kW]", "Chargers achteraan (L3) [kW]"]].sum(axis=1)

# Combine both charger totals
data["Total Chargers Combined"] = chargers_total + chargers_achteraan_total

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data["Date"], data["Total Chargers Combined"], color='green', label="Total Chargers Combined [kW]")
ax.set_title("Total Chargers Combined [kW]")
ax.set_xlabel("Date")
ax.set_ylabel("Power (kW)")
ax.grid(True)
ax.legend()

# Improve date formatting on the x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
