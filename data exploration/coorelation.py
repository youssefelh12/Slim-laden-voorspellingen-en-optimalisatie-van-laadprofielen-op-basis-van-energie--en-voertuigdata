import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read your data with proper datetime parsing
df = pd.read_csv('./Data/week_consumption.csv', decimal=',', parse_dates=['Date'], index_col='Date')

# First, convert all necessary columns to numeric
cols = df.columns
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
