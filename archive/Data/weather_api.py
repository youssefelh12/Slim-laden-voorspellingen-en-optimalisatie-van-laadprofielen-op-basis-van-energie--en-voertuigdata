import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 51.15579,
    "longitude": 4.377204,
    "start_date": "2022-08-11",
    "end_date": "2025-05-22",
    "daily": "temperature_2m_max,temperature_2m_min",
    "temperature_unit": "celsius",
    "timezone": "auto"  # Required for daily data
}

responses = openmeteo.weather_api(url, params=params)

# Process the first location
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process daily data
daily = response.Daily()
daily_temperature_max = daily.Variables(0).ValuesAsNumpy()
daily_temperature_min = daily.Variables(1).ValuesAsNumpy()

# Create date range for daily data
date_range = pd.date_range(
    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=daily.Interval()),
    inclusive="left"
)

# Calculate mean temperature
daily_temperature_mean = (daily_temperature_max + daily_temperature_min) / 2

# Create DataFrame
daily_data = {
    "date": date_range,
    "temperature_2m_max": daily_temperature_max,
    "temperature_2m_min": daily_temperature_min,
    "temperature_2m_mean": daily_temperature_mean
}

daily_dataframe = pd.DataFrame(data=daily_data)

# Save to CSV
daily_dataframe.to_csv('./api_data/daily_temperature_data2.csv', index=False)

print("\nDaily Temperature Data:")
print(daily_dataframe.head())