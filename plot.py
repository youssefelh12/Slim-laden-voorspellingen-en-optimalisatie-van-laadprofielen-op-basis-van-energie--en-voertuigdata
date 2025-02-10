import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read data
csv_path = "Data/daily_consumption.csv"
data = pd.read_csv(csv_path, parse_dates=["Date"], thousands=',')
data = data.sort_values(by="Date")

# Create subplots (2 rows x 2 columns)
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Plot each variable on its own subplot
axs[0, 0].plot(data["Date"], data["Consumption [kWh]"], color='blue')
axs[0, 0].set_title("Consumption [kW]")
axs[0, 0].grid(True)

axs[0, 1].plot(data["Date"], data["Production [kWh]"], color='green')
axs[0, 1].set_title("Production [kW]")
axs[0, 1].grid(True)

axs[1, 0].plot(data["Date"], data["Import [kWh]"], color='red')
axs[1, 0].set_title("Import [kW]")
axs[1, 0].grid(True)

axs[1, 1].plot(data["Date"], data["Export [kWh]"], color='purple')
axs[1, 1].set_title("Export [kW]")
axs[1, 1].grid(True)

# Improve date formatting on the x-axis
for ax in axs.flat:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
