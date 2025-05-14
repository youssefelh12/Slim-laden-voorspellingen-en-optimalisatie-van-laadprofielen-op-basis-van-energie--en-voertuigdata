import pandas as pd
import numpy as np
from datetime import timedelta


# Load the uploaded file
path = "./Charging_data_cleaned.csv"
df = pd.read_csv(path, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
df.rename(columns={'Date':'Timestamp', 'Chargers':'kWh'}, inplace=True)

# -------------------------------------------------
# 1) Check frequency & missing timestamps (hourly expected)
# -------------------------------------------------
full_idx = pd.date_range(start=df['Timestamp'].min(),
                         end=df['Timestamp'].max(),
                         freq='H')
missing_ts = full_idx.difference(df['Timestamp'])

# -------------------------------------------------
# 2) Duplicate timestamps?
# -------------------------------------------------
dupes = df[df.duplicated('Timestamp', keep=False)]

# -------------------------------------------------
# 3) Missing target values
# -------------------------------------------------
missing_vals = df['kWh'].isna().sum()

# -------------------------------------------------
# 4) Basic outlier scan (IQR)
# -------------------------------------------------
q1, q3 = df['kWh'].quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 3*iqr
lower_fence = q1 - 3*iqr
outliers = df[(df['kWh'] > upper_fence) | (df['kWh'] < lower_fence)]

# Summaries for display
summary = pd.DataFrame({
    'Check': [
        'Start date', 'End date', 'Expected rows (hourly)', 'Actual rows',
        'Missing timestamps', 'Duplicate timestamps', 'NaN target values',
        'Upper outlier fence (kWh)', 'Lower outlier fence (kWh)',
        'Outlier rows detected'
    ],
    'Value': [
        df['Timestamp'].min(), df['Timestamp'].max(), len(full_idx), len(df),
        len(missing_ts), len(dupes), missing_vals, round(upper_fence,2), round(lower_fence,2),
        len(outliers)
    ]
})



print(summary)