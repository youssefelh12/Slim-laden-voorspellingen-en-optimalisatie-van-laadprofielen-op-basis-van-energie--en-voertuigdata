import random
import json
import logging
import os
import threading
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from paho.mqtt import client as mqtt_client
from main.models import AuthTokens, Location, Measurement
from datetime import datetime, timedelta
from django.conf import settings
import requests

from main.scripts.Smappee_oauth2 import get_access_token

# Connection settings for broker
broker = '5124c399ff494ec682e231c08f65c060.s2.eu.hivemq.cloud'
port = 8883  # Use the TLS port
topic = "servicelocation/826fafdc-3d0b-4657-8f8b-5a0cb4ea66ab/realtime"
client_id = f'subscribe-{random.randint(0, 1000)}'

# Set up logging
filename = os.path.splitext(os.path.basename(__file__))[0]
logger = logging.getLogger(filename)
logger.setLevel(logging.DEBUG)

# Create a file handler
log_file = f"{filename}.log"
file_handler = logging.FileHandler(log_file)

# Create a formatter and set it to the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


class Command(BaseCommand):
    help = 'Run the MQTT listener command'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_saved_time = None
        self.measurement_data_buffer = []
        self.batch_mode = 'real-time'  # Configurable mode: 'real-time', 'batch', 'average'
        self.buffer_lock = threading.Lock()
        self.buffer_ttl = 300  # 5 minutes in seconds
        self.batch_timer = None

        # Dynamically gather servicelocations and corresponding Location UUIDs
        self.servicelocations = {
            location.smappee_id: location.smappee_location_uuid
            for location in
            Location.objects.filter(smappee_id__isnull=False, MQTT_is_active=True, smappee_location_uuid__isnull=False)
        }
        logger.info(f"Loaded servicelocations: {self.servicelocations}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT Broker!")
            # Subscribe to all topics using smappee uuid
            for smappee_location_uuid in self.servicelocations.values():
                topic = f"servicelocation/{smappee_location_uuid}/realtime"
                client.subscribe(topic)
                logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, message):
        logger.info(f"Message received on topic: {message.topic}")  # Add this log
        payload = message.payload.decode()
        logger.debug("Received payload: %s", payload)

        try:
            smappee_measurements = json.loads(payload)
            logger.info(f"Parsed JSON payload successfully.")

            # Extract the servicelocation ID from the topic
            topic_parts = message.topic.split('/')
            servicelocation_id = topic_parts[1]
            logger.debug(f"Extracted servicelocation ID: {servicelocation_id}")

            location = Location.objects.get(smappee_location_uuid=servicelocation_id)
            timestamp = make_aware(datetime.now())
            logger.info(f"Processing data for location: {location.name} at {timestamp}")

            # Collect data for each channel
            combined_data = self.process_smappee_data(smappee_measurements, location, timestamp)
            logger.info(f"Collected combined data: {combined_data}")

            # In on_message method
            for data_point in combined_data:
                data_point['location_id'] = location.id

            # Mode Handling: Real-Time, Batch, or Average
            if self.batch_mode == 'real-time':
                logger.info(f"Saving data in real-time mode for location: {location.id}")
                self.save_measurements(combined_data, timestamp, location)
            else:
                logger.info(f"Buffering data for batch or average mode for location: {location.id}")
                self.buffer_measurements(combined_data, timestamp)

        except json.JSONDecodeError:
            logger.error("Failed to parse JSON payload: %s", payload)
        except Exception as e:
            logger.error(f"An error occurred while processing the message: {e}")

    def process_smappee_data(self, smappee_measurements, location, timestamp):
        combined_data = []
        ev_total_power = 0
        solar_total_power = 0
        grid_total_power = 0

        api_headers = {'Authorization': f'Bearer {AuthTokens.objects.get(name="smappee").access_token}'}
        api_config_url = f'https://app1pub.smappee.net/dev/v3/servicelocation/{location.smappee_id}/meteringconfiguration'
        api_config_response = requests.get(api_config_url, headers=api_headers)

        if api_config_response.status_code == 200:
            api_config_response_json = api_config_response.json()
            logger.info(f"Successfully fetched smappee metering configuration for location {location.id}")
            logger.debug(f"API Config Response: {api_config_response_json}")
            channel_powers = smappee_measurements['channelPowers']
            for channel in channel_powers:
                publish_index = channel.get('publishIndex')
                if publish_index is not None:
                    power_value = channel.get('power')
                    phase_id = channel.get('phaseId')
                    current = channel.get('current')
                    for input_channel in api_config_response_json['channelsConfiguration']['inputChannels']:
                        if input_channel['ctInput'] == publish_index:
                            input_channel_name = input_channel['name']
                            input_channel_type = input_channel['inputChannelType']
                            combined_data.append({
                                'name': input_channel_name,
                                'input_channel_type': input_channel_type,
                                'power_value': power_value,
                                'phase_id': phase_id,
                                'current': current
                            })
                            break
        else:
            logger.error(f"Failed to get smappee metering configuration: {api_config_response.status_code}")

        return combined_data

    def buffer_measurements(self, data, timestamp):
        # Add data to buffer
        with self.buffer_lock:
            self.measurement_data_buffer.append((data, timestamp))

        # Start the 5-minute timer if not already started
        if not self.batch_timer:
            self.batch_timer = threading.Timer(self.buffer_ttl, self.process_buffer)
            self.batch_timer.start()

    def process_buffer(self):
        # Process buffered data based on the mode: Batch or Average
        with self.buffer_lock:
            # Group the buffered data by location
            location_buffers = {}
            for data, timestamp in self.measurement_data_buffer:
                for data_point in data:
                    location_id = data_point['location_id']  # Add location_id to each data point in `on_message`
                    if location_id not in location_buffers:
                        location_buffers[location_id] = []
                    location_buffers[location_id].append((data, timestamp))

            # Process each location's buffer
            for location_id, buffer_data in location_buffers.items():
                location = Location.objects.get(id=location_id)
                if self.batch_mode == 'batch':
                    # Batch mode: Save all buffered data
                    for data, timestamp in buffer_data:
                        self.save_measurements(data, timestamp, location)
                elif self.batch_mode == 'average':
                    # Average mode: Calculate and save the averages
                    self.save_average_measurements(location)

            # Clear the buffer
            self.measurement_data_buffer.clear()

        # Reset the timer
        self.batch_timer = None

    def save_measurements(self, combined_data, timestamp, location):
        ev_total_power = 0
        solar_total_power = 0
        grid_total_power = 0
        has_production_data = False  # Flag to check for 'PRODUCTION' data points

        # Save each individual measurement
        for data_point in combined_data:
            if any(keyword in data_point['name'].lower() for keyword in
                   ['laadpalen', 'charger', 'chargers', 'laadpaal']):
                ev_total_power += data_point['power_value']

            if data_point['input_channel_type'] == 'PRODUCTION':
                solar_total_power += data_point['power_value']
                has_production_data = True  # Flag to check for 'PRODUCTION' data points

            if 'grid' in data_point['name'].lower() or 'net' in data_point['name'].lower():
                grid_total_power += data_point['power_value']

            Measurement.objects.create(
                timestamp=timestamp,
                location=location,
                name=data_point['name'],
                power_value=data_point['power_value'],
                phase_id=data_point['phase_id'],
                current=data_point['current'],
                type=data_point['input_channel_type']
            )

        # Save total measurements

        if has_production_data:
            Measurement.objects.create(timestamp=timestamp, location=location, name='Solar total',
                                       power_value=solar_total_power, type='PRODUCTION')
        else:
            solar_total_power = 0

        Measurement.objects.create(timestamp=timestamp, location=location, name='Building exclusive power',
                                   power_value=(grid_total_power + solar_total_power) - ev_total_power,
                                   type='CONSUMPTION')

        Measurement.objects.create(timestamp=timestamp, location=location, name='EV total',
                                   power_value=ev_total_power, type='CONSUMPTION')

        Measurement.objects.create(timestamp=timestamp, location=location, name='Grid total',
                                   power_value=grid_total_power, type='GRID')

    def save_average_measurements(self, location):
        # Calculate average values for the last 5 minutes
        timestamp = make_aware(datetime.now())

        # Aggregate and calculate the averages
        aggregated_data = {}
        count = 0

        # Collect only data points relevant to the current location
        for data, _timestamp in self.measurement_data_buffer:
            for data_point in data:
                key = (data_point['name'], location.id)  # Use (name, location.id) as unique key
                if key not in aggregated_data:
                    aggregated_data[key] = {
                        'total_power': 0,
                        'phase_id': data_point['phase_id'],
                        'current': 0
                    }
                aggregated_data[key]['total_power'] += data_point['power_value']
                aggregated_data[key]['current'] += data_point['current']
            count += 1

        # Calculate the averages and store them
        for (name, _), values in aggregated_data.items():
            Measurement.objects.create(
                timestamp=timestamp,
                location=location,
                name=name,
                power_value=values['total_power'] / count,
                phase_id=values['phase_id'],
                current=values['current'] / count,
                type='AVERAGE'
            )
            logger.info(f"Stored average power for {name} in location {location.id}: {values['total_power'] / count}")

    def handle(self, *args, **options):
        get_access_token()
        client = mqtt_client.Client(client_id)

        # Set SSL/TLS settings
        client.tls_set()  # Use default CA certificates from the system

        password = settings.HIVEMQ_PASSWORD
        username = settings.HIVEMQ_USERNAME
        client.username_pw_set(username=username, password=password)

        # Set the connection callbacks
        client.on_connect = self.on_connect
        client.on_message = self.on_message

        # Connect to the HiveMQ Cloud broker
        client.connect(broker, port)
        client.loop_forever()
