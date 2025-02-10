import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read your data with proper datetime parsing
df = pd.read_csv('./Data/week_consumption.csv', decimal=',', parse_dates=['Date'], index_col='Date')

# List of columns to plot (example for grid, chargers, and solar on phase L1)
columns_to_plot = [
    "Grid Organi lbc (L1) [kW]",
    "Chargers (L1) [kW]",
    "Solar (L1) [kW]"
]

plt.figure(figsize=(14, 8))
for col in columns_to_plot:
    plt.plot(df.index, pd.to_numeric(df[col], errors='coerce'), label=col)

plt.xlabel('Date')
plt.ylabel('Power [kW]')
plt.title('Time Series Plot for Grid, Chargers, and Solar (L1)')
plt.legend()
plt.show()


"""
# Convert a sample column to numeric
df['Chargers (L1) [kW]'] = pd.to_numeric(df["Chargers (L1) [kW]"], errors='coerce')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Chargers (L1) [kW]'], kde=True)
plt.title('Histogram and Density of Chargers (L1)')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Chargers (L1) [kW]'])
plt.title('Box Plot of Chargers (L1)')
plt.show()

"""
