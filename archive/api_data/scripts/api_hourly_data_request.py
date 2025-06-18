#!/usr/bin/env python3
import csv
import json
import logging
import os
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta  # pip install python-dateutil
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION SECTION ---
# Replace these with your actual credentials and service location identifier.
CLIENT_ID = config.CLIENT_ID
CLIENT_SECRET = config.CLIENT_SECRET
USERNAME = config.USERNAME
PASSWORD = config.PASSWORD
SMAPPEE_LOCATION_ID = config.SMAPPEE_LOCATION_ID

# Base URLs for Smappee API endpoints
TOKEN_URL = "https://app1pub.smappee.net/dev/v3/oauth2/token"
BASE_URL = "https://app1pub.smappee.net/dev/v3/servicelocation"

# CSV output filename for hourly data
CSV_FILENAME = "api_data/hourly_api_data.csv"

def get_access_token():
    """
    Retrieve an access token from Smappee using user credentials.
    """
    payload = {
        "grant_type": "password",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "username": USERNAME,
        "password": PASSWORD
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}
    try:
        response = requests.post(TOKEN_URL, data=payload, headers=headers)
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            logger.error("No access token in response: %s", token_data)
            return None
        logger.info("Successfully retrieved access token.")
        return access_token
    except requests.RequestException as e:
        logger.error("Error fetching access token: %s", e)
        return None

def get_metering_configuration(smappee_location_id, token):
    """
    Retrieve the metering configuration for the given location.
    """
    url = f"{BASE_URL}/{smappee_location_id}/meteringconfiguration"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        logger.info("Metering configuration fetched successfully.")
        return response.json()
    else:
        logger.error("Failed to fetch metering configuration: %s - %s", response.status_code, response.text)
        return None

def get_electricity_consumption(smappee_location_id, start_date, end_date, token, aggregation=2):
    """
    Retrieve electricity consumption data between start_date and end_date.
    
    Parameters:
      - aggregation: 2 for hourly values.
      - 'from' and 'to' are UTC timestamps in milliseconds.
    """
    from_ts = int(start_date.timestamp() * 1000)
    to_ts = int(end_date.timestamp() * 1000)
    params = {
        "aggregation": aggregation,
        "from": from_ts,
        "to": to_ts,
    }
    url = f"{BASE_URL}/{smappee_location_id}/consumption"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        logger.info("Consumption data fetched for period %s to %s.", start_date, end_date)
        return response.json()
    else:
        logger.error("Failed to fetch consumption data: %s - %s", response.status_code, response.text)
        return None

def build_channel_lookup(metering_config):
    """
    Build a lookup dictionary from the metering configuration.
    
    Maps consumptionIndex (integer) -> (measurement name, phase).
    """
    lookup = {}
    if "measurements" in metering_config:
        for measurement in metering_config["measurements"]:
            meas_name = measurement.get("name")
            for channel in measurement.get("channels", []):
                consumption_index = channel.get("consumptionIndex")
                phase = channel.get("phase")
                if consumption_index is not None:
                    lookup[consumption_index] = (meas_name, phase)
    else:
        logger.error("Metering configuration does not contain 'measurements'.")
    return lookup

def extract_consumption_per_phase(consumption_json, channel_lookup):
    """
    Extract and aggregate consumption data per (measurement, phase) pair.
    
    Groups data by hour using the format "YYYY-MM-DD HH:00:00".
    """
    hourly_consumption = {}

    if not consumption_json or "consumptions" not in consumption_json:
        logger.error("Consumption JSON missing 'consumptions' key.")
        return hourly_consumption

    for record in consumption_json["consumptions"]:
        ts = record.get("timestamp")
        if not ts:
            continue
        record_date = datetime.fromtimestamp(ts / 1000)
        hour_str = record_date.strftime("%Y-%m-%d %H:00:00")
        if hour_str not in hourly_consumption:
            hourly_consumption[hour_str] = {}
        active_values = record.get("active", [])
        for consumption_index, (meas_name, phase) in channel_lookup.items():
            # If the consumption_index is 0, assume it is already 0-based.
            idx = consumption_index if consumption_index == 0 else consumption_index - 1
            if idx < len(active_values):
                value = active_values[idx]
                if value is not None:
                    key = (meas_name, phase)
                    hourly_consumption[hour_str][key] = hourly_consumption[hour_str].get(key, 0) + value

    return hourly_consumption

def main():
    # Retrieve access token
    token = get_access_token()
    if not token:
        logger.error("Access token retrieval failed. Exiting.")
        return

    # Fetch metering configuration
    metering_config = get_metering_configuration(SMAPPEE_LOCATION_ID, token)
    if metering_config is None:
        logger.error("Metering configuration not available. Exiting.")
        return

    # Build lookup for channels using the metering configuration.
    channel_lookup = build_channel_lookup(metering_config)
    if not channel_lookup:
        logger.error("Channel lookup could not be built. Exiting.")
        return
    logger.info("Channel lookup: %s", channel_lookup)

    # Define the period for which to extract data.
    # Example: starting from January 1, 2020 until now.
    period_start = datetime(2022, 9, 1)
    now = datetime.now()

    aggregated_data = {}

    current_start = period_start
    while current_start < now:
        current_end = current_start + relativedelta(hours=1)
        if current_end > now:
            current_end = now

        # Fetch consumption data with aggregation=2 (hourly values)
        consumption_json = get_electricity_consumption(SMAPPEE_LOCATION_ID, current_start, current_end, token, aggregation=2)
        if consumption_json is None:
            logger.error("Skipping period %s to %s due to API error.", current_start, current_end)
            current_start += relativedelta(hours=1)
            continue

        hourly_data = extract_consumption_per_phase(consumption_json, channel_lookup)
        for hour, data in hourly_data.items():
            if hour not in aggregated_data:
                aggregated_data[hour] = {}
            for key, value in data.items():
                aggregated_data[hour][key] = aggregated_data[hour].get(key, 0) + value

        logger.info("Processed data for hour starting %s.", current_start.strftime("%Y-%m-%d %H:00:00"))
        current_start += relativedelta(hours=1)

    # Write the aggregated hourly data to CSV.
    with open(CSV_FILENAME, "w", newline="") as csvfile:
        fieldnames = ["Hour", "Measurement", "Phase", "Consumption"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for hour, channels in aggregated_data.items():
            for (meas_name, phase), consumption in channels.items():
                writer.writerow({
                    "Hour": hour,
                    "Measurement": meas_name,
                    "Phase": phase,
                    "Consumption": consumption
                })
    logger.info("CSV file generated: %s", CSV_FILENAME)

if __name__ == "__main__":
    main()
