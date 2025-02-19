import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
data = pd.read_csv("monthly_api_data.csv")  # Replace with your CSV file path if needed

# Get the unique measurements to create separate plots
measurements = data['Measurement'].unique()

# Create a figure with subplots (one per measurement)
fig, axes = plt.subplots(nrows=len(measurements), ncols=1, figsize=(10, 5 * len(measurements)), sharex=True)

# If there is only one measurement, ensure axes is iterable
if len(measurements) == 1:
    axes = [axes]

# Plot each measurement separately
for ax, measurement in zip(axes, measurements):
    subset = data[data["Measurement"] == measurement]
    sns.barplot(data=subset, x="Month", y="Consumption", hue="Phase", ax=ax)
    ax.set_title(f"Consumption for {measurement}")
    ax.set_ylabel("Consumption")
    ax.set_xlabel("Month")
    ax.legend(title="Phase")

plt.tight_layout()
plt.show()
