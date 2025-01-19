# Used to collect collect meteorlogical data from Envirnment Canada's Weather API + Power Demand data from IESO

### Imports ###
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import asyncio  # Import asyncio library for async operations
import aiohttp # Import aiohttp library for making HTTP requests
import nest_asyncio # Allows for asyncio to be nested

### Functions ###

# get_weather_data - For a given day and FSA (through the lat and long), get the weather data for that day
async def get_weather_data(session: aiohttp.ClientSession, current_date: datetime, next_date: datetime, lat: float, lon: float):

    # Set Up
    weather_data = []   # List to store weather data 
    bbox_limit = 0.1   # Bounding box limit for data collection

    while True: # Loop to get data until enough data is collected (need at least 24 data points for day)
        bbox = f'{lon-bbox_limit},{lat-bbox_limit},{lon+bbox_limit},{lat+bbox_limit}' # Bounding box for data collection
        url = f'https://api.weather.gc.ca/collections/climate-hourly/items?bbox={bbox}&datetime={current_date.isoformat()}/{next_date.isoformat()}' # URL to get data from API

        # Make request to API
        response = await session.get(url) # Make function wait for response
        response_data = await response.json() # Get data from response

        # If enough data was found, extract data and stron in weather_data list
        if 'features' in response_data and len(response_data['features']) > 23:
            for data_point in response_data['features']:
                weather_data.append({
                    #'Station Name': data_point['properties'].get('STATION_NAME'),
                    'Date/Time (LST)': data_point['properties'].get('LOCAL_DATE'),
                    'Year': data_point['properties'].get('LOCAL_YEAR'),
                    'Month': data_point['properties'].get('LOCAL_MONTH'),
                    'Day': data_point['properties'].get('LOCAL_DAY'),
                    'Hour': data_point['properties'].get('LOCAL_HOUR'),
                    'Temperature': data_point['properties'].get('TEMP'),
                    'Dew Point Temperature': data_point['properties'].get('DEW_POINT_TEMP'),
                    'Wind_Speed': data_point['properties'].get('WIND_SPEED'),
                    'Station Pressure': data_point['properties'].get('STATION_PRESSURE'),
                    'Humidity': data_point['properties'].get('RELATIVE_HUMIDITY'),
                    'Windchill': data_point['properties'].get('WINDCHILL'),
                })
            
            break # Break loop if enough data is found

        else: # If not enough data is found, increase the bounding box to get more data
            bbox_limit += 0.05
            await asyncio.sleep(0.5) # Add a slight delay to avoid overloading the server

    # Turn list of dictionaries into a pandas dataframe
    weather_data_df = pd.DataFrame(weather_data)

    # Average out data points for the same hour - Usefull for when multiple data points are collected for the same hour from different stations
    weather_data_df = weather_data_df.groupby(['Date/Time (LST)', 'Year', 'Month', 'Day', 'Hour']).mean(numeric_only=True).reset_index()

    return weather_data_df  # Return weather data for the day

# get_power_data - Get power data for a given date range and FSA by picking up data from saved CSVs
def get_power_data(data_path, start_date: datetime, end_date: datetime, fsa):
    
    # Set up
    target_dir = data_path + "/Hourly_Demand_Data"
    filtered_monthly_power_data = [pd.DataFrame()]
    power_data = pd.DataFrame()
    monthly_power_data = []

    # Get data from IESO
    for year in range(start_date.year, end_date.year + 1):
        for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']: # Loop through all months
            file_path = f'{target_dir}/PUB_HourlyConsumptionByFSA_{year}{month}_v1.csv'
            if os.path.exists(file_path): # Check if file exists

                # Use try and catch if problems reading input data
                try: 
                    power_data = pd.read_csv(file_path, skiprows=3, header = 0, usecols= ['FSA', 'DATE', 'HOUR', 'CUSTOMER_TYPE', 'TOTAL_CONSUMPTION'])
                except FileNotFoundError: # not all months had a file (for example, 2024 only has up to may)
                    continue
                except ValueError: # skiprows=x does not match the "normal sequence" of 3. For example, 2023 08 data had a different skip_row value
                    power_data = pd.read_csv(file_path, skiprows=7, header = 0, usecols= ['FSA', 'DATE', 'HOUR', 'CUSTOMER_TYPE', 'TOTAL_CONSUMPTION'])

                # Filter power data for residential power and for terget FSA
                monthly_power_data.append(power_data[(power_data['FSA'] == fsa) & (power_data['CUSTOMER_TYPE'] == 'Residential')])
                
    # Concatenate all monthly power data into one dataframe
    monthly_power_data_df = pd.concat(monthly_power_data)
    filtered_power_data_df = monthly_power_data_df.groupby(["FSA", "CUSTOMER_TYPE", "HOUR", "DATE"]).sum().reset_index()

    # Modify Calander Columns
    filtered_power_data_df['DATE'] = pd.to_datetime(filtered_power_data_df['DATE'])
    filtered_power_data_df['YEAR'] = filtered_power_data_df['DATE'].dt.year
    filtered_power_data_df['MONTH'] = filtered_power_data_df['DATE'].dt.month
    filtered_power_data_df['DAY'] = filtered_power_data_df['DATE'].dt.day
    
    return filtered_power_data_df

# get_data_for_time_range - get data for a given time range for a given latitude and longitude
async def get_data_for_time_range(data_path, start_date: datetime, end_date: datetime, fsa, lat: float, lon: float):

    # Weather Data Collection - Done asynchronously wtih Weather API #
    async with aiohttp.ClientSession() as session: # Using session to make requests
        
        # List to store all the jobs to be run asynchronously
        weather_jobs = []
        
        # Get current date
        current_date = start_date

        # Collect weather one day at a time asychronously for all days in range
        while current_date <= end_date:
            # Get datatime object for end of current day
            next_date = current_date + timedelta(days=1) - timedelta(hours=1) 

            # Get weather data for current day
            weather_jobs.append(get_weather_data(session, current_date, next_date, lat, lon))

            # Move to next day
            current_date = current_date + timedelta(days=1)  

        # Run all jobs asynchronously   
        retuned_data = await asyncio.gather(*weather_jobs)

        # Turn list of weather dataframes for each day into one dataframe
        weather_data_list = [data for data in retuned_data if not data.empty] # Get all data that is not empty
        weather_data = pd.concat(weather_data_list) if weather_data_list else pd.DataFrame() # Concatenate all dataframes into one dataframe

    # Collect Power Data - Pick up data from saved CSV from IESO
    power_data = get_power_data(data_path, start_date, end_date, fsa)

    return weather_data, power_data



### Set Up ###
# Student Directory
hanad_run = ["c:/Users/hanad/capstone_github/power-forecasting-capstone/data", 1]
clover_run = ["./data", 2]
joseph_laptop_run = ["C:\\Users\\sposa\\Documents\\GitHub\\power-forecasting-capstone\\data", 3]
joseph_pc_run = ["D:\\Users\\Joseph\\Documents\\GitHub\\power-forecasting-capstone\\data", 3]
janna_run = ["./data", 4]

# Paste student name_run for whoever is running the code
run_student = hanad_run
if (run_student[1] == joseph_laptop_run[1]):
    print("JOSEPH IS RUNNING!")
elif (run_student[1] == hanad_run[1]):
    print("HANAD IS RUNNING!")
elif (run_student[1] == janna_run[1]):
    print("JANNA IS RUNNING!")
elif (run_student[1] == clover_run[1]):
    print("CLOVER IS RUNNING!")
else:
    print("ERROR!! NO ELIGIBLE STUDENT!")
    sys.exit() # Kill script

# Set Up data direcotry path for data collection
data_path = run_student[0]
target_dir = data_path + "/raw_data"

if not os.path.exists(target_dir): # Create directory if it does not exist
    os.makedirs(target_dir)

# Set up FSA Dictionary that will map FSA names to a latitude and longitude
fsa_map = {
    "L9G": {"lat": 43.27, "lon": -79.95}, # Target FSA for Tests
    "M9M": {"lat": 43.74, "lon": -79.54}, # Jane and Finch
    "L9H": {"lat": 43.32, "lon": -79.98}, # Dundas
}


# Link - f'https://api.weather.gc.ca/collections/climate-hourly/items?bbox={bbox}&datetime={current_date.isoformat()}/{next_date.isoformat()}'
# bbox = boubding box around a given latitude and longitude, gets all data within bbox given
# datetime = start_date.isoformat() / end_date.isoformat() - gets all data within the given time frame

### Calling Data ###
# Choose FSA for data collection + Get latitude and longitude of chosen fsa
fsa = "L9G"
lat = fsa_map[fsa]["lat"]
lon = fsa_map[fsa]["lon"]

# Choose date range for data collection
start_year = 2018
start_month = 1
start_day = 1
start_hour = 0

end_year = 2018
end_month = 12
end_day = 31
end_hour = 23

# Making datetime objects for start and end dates
start_date = datetime(start_year, start_month, start_day, start_hour,0,0)
end_date = datetime(end_year, end_month, end_day, end_hour,0,0)

# Collect data - Using asynchronous functions
weather_data, power_data = asyncio.run(get_data_for_time_range(data_path, start_date, end_date, fsa, lat, lon))

# Save data to CSV
weather_data.to_csv(f'{target_dir}/weather_data_{fsa}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_V4.csv', index=False)
power_data.to_csv(f'{target_dir}/power_data_{fsa}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_V5.csv', index=False)
