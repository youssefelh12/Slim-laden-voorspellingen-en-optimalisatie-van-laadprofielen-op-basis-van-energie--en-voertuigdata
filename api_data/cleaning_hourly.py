import pandas as pd

# Replace with the path to your actual CSV file
input_file = './api_data/aggregated_hourly_measurements.csv'
output_file = 'hourly_charging_data.csv'

# Read the CSV; adjust 'sep' and 'decimal' if necessary.
df = pd.read_csv(input_file, sep=',', decimal='.', quotechar='"')

# Convert Date column to datetime for accurate sorting.
df['Hour'] = pd.to_datetime(df['Hour'])

# Sum the relevant columns to create Grid, Solar, and Chargers
df['Grid'] = (
    df["Grid Organi lbc"] 
  + df["Grid Organi lbc"] 
  + df["Grid Organi lbc"]
)

df['Solar'] = (
    df["Solar"] 
  + df["Solar"] 
  + df["Solar"]
)

df['Chargers'] = (
    df["Chargers"]
  + df["Chargers"]
  + df["Chargers"]
  + df["Chargers achteraan"]
  + df["Chargers achteraan"]
  + df["Chargers achteraan"]
)

# Create the new column total_consumption by:
# Grid minus Chargers plus Solar
df['total_consumption'] = df['Grid'] - df['Chargers'] + df['Solar']

df['Chargers'] = (
    df["Chargers"]
  + df["Chargers achteraan"]

)

# Drop the original Consumption [kWh] column
#df.drop(columns=["Consumption [kWh]"], inplace=True)

# Sort by Date in chronological order
df.sort_values(by='Hour', inplace=True)

# Build the final result DataFrame
df_result = df[['Hour', 'Chargers']]

# Write out to a new CSV. Adjust separator and decimal as needed.
df_result.to_csv(output_file, index=False, sep=',', decimal='.')
