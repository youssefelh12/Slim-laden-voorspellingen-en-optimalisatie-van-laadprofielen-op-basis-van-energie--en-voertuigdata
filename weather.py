import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# API request parameters
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 51.2205,
    "longitude": 4.4003,
    "hourly": ["temperature_2m", "relative_humidity_2m"],
    "start_date": "2024-08-11",
    "end_date": "2024-12-31"
}
responses = openmeteo.weather_api(url, params=params)

# Process response
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "temperature_2m": hourly_temperature_2m,
    "relative_humidity_2m": hourly_relative_humidity_2m
}

hourly_dataframe = pd.DataFrame(data=hourly_data)
print(hourly_dataframe)

# Save to CSV
csv_filename = "weather_data.csv"
hourly_dataframe.to_csv(csv_filename, index=False)
print(f"Weather data saved to {csv_filename}")
