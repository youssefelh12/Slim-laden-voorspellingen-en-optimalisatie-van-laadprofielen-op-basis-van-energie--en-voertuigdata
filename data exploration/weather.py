import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# API request parameters (note the hourly parameter is now a comma-separated string)
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 51.2205,
    "longitude": 4.4003,
    "hourly": "temperature_2m,relative_humidity_2m",
    "start_date": "2024-08-11",
    "end_date": "2024-12-31"
}

responses = openmeteo.weather_api(url, params=params)

# Process the first response (assuming the API returns a list)
response = responses[0]

# Print basic weather information
print(f"Coordinates: {response.Latitude()}°N, {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()} ({response.TimezoneAbbreviation()})")
print(f"UTC Offset: {response.UtcOffsetSeconds()} seconds")

# Process hourly data
hourly = response.Hourly()

# Retrieve the time arrays (assumed to be list-like)
time_array = hourly.Time()       # Expected to be a list or array of start times
time_end_array = hourly.TimeEnd()  # Expected to be a list or array of end times

# Create start and end timestamps for the date range using the first and last elements
start_timestamp = pd.to_datetime(time_array[0], unit="s", utc=True)
end_timestamp = pd.to_datetime(time_end_array[-1], unit="s", utc=True)

# Get hourly variable data as numpy arrays
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

# Calculate the interval (in seconds) for the hourly data
interval_seconds = hourly.Interval()

# Create a date range using the interval.
# Note: If you're on an older pandas version, replace "inclusive='left'" with "closed='left'"
date_range = pd.date_range(
    start=start_timestamp,
    end=end_timestamp,
    freq=pd.Timedelta(seconds=interval_seconds),
    inclusive="left"
)

# Build a DataFrame from the hourly data
hourly_data = {
    "date": date_range,
    "temperature_2m": hourly_temperature_2m,
    "relative_humidity_2m": hourly_relative_humidity_2m
}

hourly_dataframe = pd.DataFrame(data=hourly_data)
print(hourly_dataframe)

# Save the DataFrame to a CSV file
csv_filename = "weather_data.csv"
hourly_dataframe.to_csv(csv_filename, index=False)
print(f"Weather data saved to {csv_filename}")
