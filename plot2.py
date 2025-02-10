import pandas as pd
import plotly.express as px

# Read data
csv_path = "Data/Electricity consumption - 2023-01-01 - 2025-02-04 - 5 minutes (1).csv"
data = pd.read_csv(csv_path, parse_dates=["Date"], thousands=',')
data = data.sort_values(by="Date")

# Create an interactive line plot for all variables
fig = px.line(
    data, 
    x="Date", 
    y=["Consumption [kW]", "Production [kW]", "Import [kW]", "Export [kW]"],
    labels={'value': 'Power (kW)', 'variable': 'Measurement'},
    title="Energy Consumption and Production Over Time"
)

# Enable range slider and selectors
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

fig.show()
