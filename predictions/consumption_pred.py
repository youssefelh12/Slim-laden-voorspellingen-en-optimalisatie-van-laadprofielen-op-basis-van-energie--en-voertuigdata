import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read the CSV file; ensure the Date column is parsed as datetime
df = pd.read_csv('predictions_chargers.csv', parse_dates=['Date'])

# Optionally, sort by Date if not already sorted
df.sort_values('Date', inplace=True)

# Create the plot
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Predicted_Total_Chargers'], marker='o', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Predicted Total Chargers (kW)')
plt.title('Forecast for Next 3 Days (15-Minute Intervals)')

# Set the x-axis major locator to show a tick every 15 minutes
locator = mdates.MinuteLocator(interval=15)
formatter = mdates.DateFormatter('%H:%M')
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Rotate the tick labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
