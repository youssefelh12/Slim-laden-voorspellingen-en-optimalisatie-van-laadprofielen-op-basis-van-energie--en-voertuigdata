import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Read your data, parsing the 'Hour' column as datetime and using it as the index
df = pd.read_csv('./api_data/hourly_api_data.csv', parse_dates=['Hour'], index_col='Hour')

# Sort the DataFrame by the datetime index
df = df.sort_index()

# Optionally, filter for a specific measurement.
# Here we select the first unique measurement found in the 'Measurement' column.
selected_measurement = df['Measurement'].unique()[0]
print(f"Selected Measurement: {selected_measurement}")

# Filter the data for the selected measurement and select the 'Consumption' column
series_data = df[df['Measurement'] == selected_measurement]['Consumption']

# Ensure the data is numeric
series = pd.to_numeric(series_data, errors='coerce')

# Decompose the series. For hourly data with daily seasonality, set period = 24.
decomposition = seasonal_decompose(series.dropna(), model='additive', period=24)

# Plot the decomposition
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.suptitle(f'Seasonal Decomposition of Consumption for {selected_measurement}', fontsize=16)
plt.show()
