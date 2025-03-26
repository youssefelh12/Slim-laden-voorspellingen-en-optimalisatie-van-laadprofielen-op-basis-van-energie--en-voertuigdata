import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read data
csv_path = "Data/week_consumption.csv"
data = pd.read_csv(csv_path, parse_dates=["Date"], thousands=',')
data = data.sort_values(by="Date")

# Summing L1, L2, L3 values for each category
data["Grid Organi Total"] = data[["Grid Organi lbc (L1) [kW]", "Grid Organi lbc (L2) [kW]", "Grid Organi lbc (L3) [kW]"]].sum(axis=1)
data["Chargers Total"] = data[["Chargers (L1) [kW]", "Chargers (L2) [kW]", "Chargers (L3) [kW]"]].sum(axis=1)
data["Solar Total"] = data[["Solar (L1) [kW]", "Solar (L2) [kW]", "Solar (L3) [kW]"]].sum(axis=1)
data["Chargers achteraan Total"] = data[["Chargers achteraan (L1) [kW]", "Chargers achteraan (L2) [kW]", "Chargers achteraan (L3) [kW]"]].sum(axis=1)

# Create subplots (2 rows x 2 columns)
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Plot each variable on its own subplot
axs[0, 0].plot(data["Date"], data["Grid Organi Total"], color='blue')
axs[0, 0].set_title("Grid Organi Total [kW]")
axs[0, 0].grid(True)

axs[0, 1].plot(data["Date"], data["Chargers Total"], color='green')
axs[0, 1].set_title("Chargers Total [kW]")
axs[0, 1].grid(True)

axs[1, 0].plot(data["Date"], data["Solar Total"], color='red')
axs[1, 0].set_title("Solar Total [kW]")
axs[1, 0].grid(True)

axs[1, 1].plot(data["Date"], data["Chargers achteraan Total"], color='purple')
axs[1, 1].set_title("Chargers achteraan Total [kW]")
axs[1, 1].grid(True)

# Improve date formatting on the x-axis
for ax in axs.flat:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
