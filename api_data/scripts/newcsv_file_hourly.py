import pandas as pd

# Read the CSV file and parse the 'Hour' column as datetime
df = pd.read_csv('./api_data/hourly_api_data.csv', parse_dates=['Hour'])

# Group by 'Hour' and 'Measurement', summing the 'Consumption' values
grouped = df.groupby(['Hour', 'Measurement'])['Consumption'].sum()

# Pivot the table so that each measurement type becomes a column
pivot_df = grouped.unstack('Measurement')

# Fill NaN values with 0 (if desired)
pivot_df = pivot_df.fillna(0)

# Divide all values by 1000 and round to 4 decimal places
pivot_df = (pivot_df / 1000).round(4)

# Save the pivoted DataFrame to a new CSV file
pivot_df.to_csv('./api_data/aggregated_hourly_measurements.csv')

print("New CSV file 'aggregated_measurements.csv' has been created.")

