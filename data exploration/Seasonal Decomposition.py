import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Read your data with proper datetime parsing
df = pd.read_csv('./Data/15min2024_consumption.csv', decimal=',', parse_dates=['Date'], index_col='Date')
data = df.sort_values(by="Date")

# Choose one series, e.g., "Chargers (L1) [kW]"
series = pd.to_numeric(df["Chargers (L1) [kW]"], errors='coerce')

# Decompose the series (assuming an additive model and known seasonal period)
# For 15-minute data, if you expect a daily seasonality: period = 24*4 = 96
decomposition = seasonal_decompose(series.dropna(), model='additive', period=96)
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.show()
