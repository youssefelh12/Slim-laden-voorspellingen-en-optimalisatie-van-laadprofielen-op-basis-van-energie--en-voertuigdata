import pandas as pd
import matplotlib.pyplot as plt

# File path
csv_path = "Data/Electricity consumption - 2023-01-01 - 2025-02-04 - 5 minutes (1).csv"

# Read data into a DataFrame
data = pd.read_csv(csv_path, parse_dates=["Date"], thousands=',')

# Sort by date
data = data.sort_values(by="Date")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(data["Date"], data["Consumption [kW]"], label="Consumption", marker='o')
plt.plot(data["Date"], data["Production [kW]"], label="Production", marker='s')
plt.plot(data["Date"], data["Import [kW]"], label="Import", marker='^')
plt.plot(data["Date"], data["Export [kW]"], label="Export", marker='v')

# Formatting
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("Energy Consumption and Production Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

# Show plot
plt.show()
