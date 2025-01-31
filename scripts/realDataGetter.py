# Used to collect collect meteorlogical data from Envirnment Canada's Weather API + Power Demand data from IESO

### Imports ###
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import asyncio  # Import asyncio library for async operations
import aiohttp # Import aiohttp library for making HTTP requests
import nest_asyncio # Allows for asyncio to be nested
import canada_holiday # Import library to get Canadian Holidays

### Functions ###

# calculate_windchill - Calculate windchill given temperature and wind speed
def calculate_windchill(weather_data: pd.DataFrame):

    # Calculate Windchill + Retunr Dataframe
    weather_data['Windchill'] = 13.12 + 0.6215 * weather_data['Temperature'] - 11.37 * np.power(weather_data['Wind_Speed'], 0.16) + 0.3965 * weather_data['Temperature'] * np.power(weather_data['Wind_Speed'], 0.16)    

    return weather_data

# fill_missing_data - Fill missing data in given dataframe  
def fill_missing_data(data: pd.DataFrame):

    # Fill missing data with forward fill and back fill
    data = data.ffill()
    data = data.bfill()

    return data

# add_calendar_columns - Add Weekends, Holidays, and Seasons calendar columns to dataframe
def add_calendar_columns(data: pd.DataFrame):

    # Add temporary DATE coliumn to dataframe to perform operations
    data['DATE'] = pd.to_datetime(data['Date/Time (LST)'])

    # Add Weekend Column to dataframe (0 = Weekday, 1 = Weekend) - // = Floor Division (returns integer and drops remainder)
    data['Weekend'] = data['DATE'].dt.dayofweek // 5  # Wekdays = 0 - 4, Weekends = 5 - 6

    # Add Season Column to dataframe (1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall) - NO MAKE 0 OR 1 
    data['Season'] = (data['Month'] % 12 + 3) // 3 

    # # Add Holiday Column
    # data['Holiday'] = 0 # Initialize Holiday Column
    # temp_value = 0 # Temporary value to store holiday value for the day
    # for index, row in  data.iterrows(): # Loop through all rows in dataframe
    #     date_temp = date(row['Year'], row['Month'], row['Day'])
    #     if (row["Hour"] == 0):
    #         if canada_holiday.is_holiday(date_temp, "Ontario"): # Check if date is a holiday
    #             temp_value = 1
    #             data.loc[index, 'Holiday'] = temp_value
    #             print("################### HOLIDAY ###################")
    #             print(" -> " + str(date_temp) + ", " + str(row['Hour']) + ", " + str(temp_value))
    #         else:
    #             temp_value = 0
    #             print(" -> " + str(date_temp) + ", " + str(row['Hour']) + ", " + str(temp_value))
    #     else:
    #         data.loc[index, 'Holiday'] = temp_value # Fill in holdiay column for rest of day
    #         print(" -> " + str(date_temp)  + ", " + str(row['Hour']) + ", " + str(temp_value))

    # Drop temporary DATE column   
    data = data.drop(columns=['DATE'])

    return data

# add_lags_to_weather_data - Add lagged columns to weather   
def add_lags_to_weather_data(data: pd.DataFrame, lag: int):
    
    # Add lag columns to weather dataframe
    for i in range(1, lag + 1):
        data[f'Hour_{i}'] = data['Hour'].shift(i)
        data[f'Temperature_Lag_{i}'] = data['Temperature'].shift(i)
        data[f'Dew Point Temperature_Lag_{i}'] = data['Dew Point Temperature'].shift(i)
        data[f'Wind_Speed_Lag_{i}'] = data['Wind_Speed'].shift(i)
        data[f'Station Pressure_Lag_{i}'] = data['Station Pressure'].shift(i)
        data[f'Humidity_Lag_{i}'] = data['Humidity'].shift(i)
        data[f'Windchill_Lag_{i}'] = data['Windchill'].shift(i)

    # Using bfill to fill in missing data in lag columns - QUICK AND DIRTY
    data = data.bfill()

    return data

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
def get_power_data(data_path, start_date: datetime, end_date: datetime, fsa: str):
    
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

                # # Filter power data for residential power and for terget FSA
                monthly_power_data.append(power_data)
                
    # Concatenate all monthly power data into one dataframe
    monthly_power_data_df = pd.concat(monthly_power_data)

    # Modify Calander Columns - Add Year, Month, Day to Power Dataframe
    monthly_power_data_df['DATE'] = pd.to_datetime(monthly_power_data_df['DATE'])
    monthly_power_data_df['YEAR'] = monthly_power_data_df['DATE'].dt.year
    monthly_power_data_df['MONTH'] = monthly_power_data_df['DATE'].dt.month
    monthly_power_data_df['DAY'] = monthly_power_data_df['DATE'].dt.day

    # Filter power data for residential power and for target FSA
    monthly_power_data_df = monthly_power_data_df.loc[monthly_power_data_df['CUSTOMER_TYPE'] == "Residential"] # Filter for Residential Data
    monthly_power_data_df = monthly_power_data_df.loc[monthly_power_data_df['FSA'] == fsa] # Filter for FSA Data
    filtered_power_data_df = monthly_power_data_df.groupby(["FSA", "CUSTOMER_TYPE", "DATE","YEAR", "MONTH", "DAY", "HOUR"]).TOTAL_CONSUMPTION.sum().reset_index()
    
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

    # Fill missing data in weather data + Calculate Windchill + Add Calendar Columns (Weekend, Season, Holiday)
    weather_data = fill_missing_data(weather_data)
    weather_data = calculate_windchill(weather_data)
    weather_data = add_calendar_columns(weather_data)
    weather_data = add_lags_to_weather_data(weather_data, 1) # Add lagged columns to weather data - TESTING 

    # Collect Power Data - Pick up data from saved CSV from IESO
    power_data = get_power_data(data_path, start_date, end_date, fsa)

    # Fill missing data in power data
    power_data = fill_missing_data(power_data)

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
    "L7G": {"lat": 43.27, "lon": -79.95}, # Ancaster
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

end_year = 2023
end_month = 12
end_day = 31
end_hour = 23

# Making datetime objects for start and end dates
start_date = datetime(start_year, start_month, start_day, start_hour,0,0)
end_date = datetime(end_year, end_month, end_day, end_hour,0,0)

# Collect data - Using asynchronous functions
weather_data, power_data = asyncio.run(get_data_for_time_range(data_path, start_date, end_date, fsa, lat, lon))

# Save data to CSV
weather_data.to_csv(f'{target_dir}/weather_data_{fsa}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_lagAnal_lag1.csv', index=False)
power_data.to_csv(f'{target_dir}/power_data_{fsa}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_lagAnal_lag1.csv', index=False)
