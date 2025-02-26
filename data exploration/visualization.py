import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read the aggregated CSV file with 'Hour' as a datetime index
df = pd.read_csv('./api_data/aggregated_measurements.csv', parse_dates=['Hour'], index_col='Hour')

# Loop through each column and plot its time series data
for col in df.columns:
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[col], label=col, color='blue')
    plt.title(f'Time Series Plot for {col}', fontsize=16)
    plt.xlabel('Hour', fontsize=14)
    plt.ylabel(f'{col} Consumption (in thousands)', fontsize=14)
    
    # Format the x-axis for date display
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
