# Used to collect collect meteorlogical data from Envirnment Canada's Weather API + Power Demand data from IESO
# Libraries
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import asyncio  # Import asyncio library for async operations
import aiohttp # Import aiohttp library for making HTTP requests
import nest_asyncio # Allows for asyncio to be nested
nest_asyncio.apply() # Apply nest_asyncio to allow for nested asyncio operations
from sklearn import preprocessing # Import preprocessing library for data normalization
import joblib # Import joblib to save and load models
import canada_holiday

### Functions ###

# calculate_windchill - Calculate windchill given temperature and wind speed
def calculate_windchill(weather_data: pd.DataFrame):

    # Calculate Windchill + Return Dataframe
    weather_data['Windchill'] = 13.12 + 0.6215 * weather_data['Temperature'] - 11.37 * np.power(weather_data['Wind Speed'], 0.16) + 0.3965 * weather_data['Temperature'] * np.power(weather_data['Wind Speed'], 0.16)    

    return weather_data

# fill_missing_data - Fill missing data in given dataframe  
def fill_missing_data(data: pd.DataFrame):

    # Fill missing data with forward fill and back fill
    data = data.ffill()
    data = data.bfill()

    return data

# add_calendar_columns - Add Weekends, Holidays, and Seasons calendar columns to dataframe
def add_calendar_columns(data: pd.DataFrame):
    
    # Add temporary DATE column to dataframe to perform operations
    try:
        data['DATE'] = pd.to_datetime(data['Date/Time (LST)'])
    except:
        data['DATE'] = pd.to_datetime(data[["Year", "Month", "Day","Hour"]])

    # Add Weekend Column to dataframe (0 = Weekday, 1 = Weekend) - // = Floor Division (returns integer and drops remainder)
    data['Weekend'] = data['DATE'].dt.dayofweek // 5  # Wekdays = 0 - 4, Weekends = 5 - 6

    # Add Season Column to dataframe
    # Winter(1) = November (Including) to April (Including)
    # Summer(0) = May (Including) to October (Including)
    data['Season'] = ((data['Month'] <5) | (data['Month']>10))
    data['Season'] =  data['Season'].astype(int)

    # Add Holiday Column
    data['Holiday'] = 0 # Initialize Holiday Column
    holiday_array = []
    temp_value = 0 # Temporary value to store holiday value for the day
    for index, row in  data.iterrows(): # Loop through all rows in dataframe
        date_temp = date(row['Year'], row['Month'], row['Day'])
        if (row["Hour"] == 0):
            if canada_holiday.is_holiday(date_temp, "Ontario"): # Check if date is a holiday
                temp_value = 1
                holiday_array.append(temp_value)
                #data.loc[index, 'Holiday'] = temp_value
            else:
                temp_value = 0
                holiday_array.append(temp_value)
                
        else:
            holiday_array.append(temp_value)
            #data.loc[index, 'Holiday'] = temp_value # Fill in holdiay column for rest of day
    data['Holiday'] = holiday_array
    
    # Convert year, month, day, hour to boolean values
    # Day range 1 to 31 (subtract last day for 0 condition)
    days = [*range(1, 31)]
    # Hour range 0 to 23 (subtract last hour for 0 condition)
    hours = [*range(0, 23)]
    # Hour range 0 to 23 (subtract last month for 0 condition)
    months = [*range(1, 12)]
    
    # Year range from 2018 to 2050 of training dataset
    years = [*range(2018, 2051)]
    
    for year in years:
        data["Year_" + str(year)] = data["Year"]==int(year)
        data["Year_" + str(year)] = data["Year_" + str(year)].astype(int)

    for month in months:
        data["Month_" + str(month)] = data["Month"]==int(month)
        data["Month_" + str(month)] = data["Month_" + str(month)].astype(int)

    for day in days:
        data["Day_"+str(day)] = data["Day"]==int(day)
        data["Day_"+str(day)] = data["Day_"+str(day)].astype(int)

    for hour in hours:
        data["Hour_"+str(hour)] = data["Hour"]==int(hour)
        data["Hour_"+str(hour)] = data["Hour_"+str(hour)].astype(int)
    
    

    # Drop temporary DATE column   
    data = data.drop(columns=['DATE'])
    
    # Drop Date/Time (LST) column
    try:
        data = data.drop(columns=['Date/Time (LST)'])
        return data
    except:
        return data

# add_lags_to_weather_data - Add lagged columns to weather   
def add_lags_to_weather_data(data: pd.DataFrame, lag: int):
    
    # Add lag columns to weather dataframe
    for i in range(1, lag + 1):
        for hour in range(0, 23):
            data[f'Hour_{hour}_Lag_{i}'] = data[f'Hour_{hour}'].shift(i)
            
        data[f'Temperature_Lag_{i}'] = data['Temperature'].shift(i)
        data[f'Dew Point Temperature_Lag_{i}'] = data['Dew Point Temperature'].shift(i)
        data[f'Wind Speed_Lag_{i}'] = data['Wind Speed'].shift(i)
        data[f'Station Pressure_Lag_{i}'] = data['Station Pressure'].shift(i)
        data[f'Humidity_Lag_{i}'] = data['Humidity'].shift(i)
        data[f'Windchill_Lag_{i}'] = data['Windchill'].shift(i)

    # Using bfill to fill in missing data in lag columns - QUICK AND DIRTY
    data = data.bfill()

    return data



# find_fsa_index - Find index of FSA in fsa_map dictiionary
def find_fsa_index(fsa: str, fsa_map: dict):
    index = 0
    for key in fsa_map:
        if key == fsa:
            return index
        index += 1
    return 272 # Return 272 if FSA not found (default case and good location since it is near good weather stations)

# find_fsa_lat_lon - Find lat and lon of FSA given the index in the fsa_map dictionary
def find_fsa_lat_lon(index: int, fsa_map : dict):
    for i, key in enumerate(fsa_map):
        if i == index:
            return fsa_map[key]["lat"], fsa_map[key]["lon"], key
    return 43.27, -79.95, "M2M" # Return default lat and lon if index not found (default case and good location since it is near good weather stations)

# get_weather_data - For a given day and FSA (through the lat and long), get the weather data for that day
async def get_weather_data(session: aiohttp.ClientSession, current_date: datetime, next_date: datetime, lat: float, lon: float, fsa: str, fsa_map: dict):

    # Set Up
    weather_data = []   # List to store weather data 
    bbox_limit = 0.2   # Bounding box limit for data collection
    current_lat = lat # Current latitude
    current_lon = lon # Current longitude
    current_fsa = fsa
    hard_reset = 20 # If the code can't get data after 20 tries, it will hard reset the lat and lon to a neighboring FSA
    reset_counter = 0 # Counter to keep track of how many times the code has tried to get data

    while True: # Loop to get data until enough data is collected (need at least 24 data points for day)
        reset_counter += 1 # Increment counter
        
        # If counter is greater than hard reset, reset lat and lon to a neighboring FSA
        if reset_counter > hard_reset:
            reset_counter = 0 # Reset counter
            print("FAILED TO FIND WEATHER DATA FOR - " + current_fsa + " - " + str(current_date) + " - " + str(next_date) + " - WILL GET FROM NEIGHBORING FSA") # alert user of hard reset
            
            # Find index of current FSA in fsa_map dictionary
            fsa_index = find_fsa_index(current_fsa, fsa_map)

            # Get the lat long of a neighboring FSA  index
            if fsa_index > 230: # If index number is greater than 230, get the lat long of the previous index
                current_lat, current_lon, current_fsa = find_fsa_lat_lon(fsa_index-1, fsa_map)
            else: # If index number is less than 230, get the lat long of the previous index
                current_lat, current_lon, current_fsa = find_fsa_lat_lon(fsa_index+1, fsa_map)
            bbox_limit = 0.2 # Reset bounding box limit

        bbox = f'{current_lon-bbox_limit},{current_lat-bbox_limit},{current_lon+bbox_limit},{current_lat+bbox_limit}' # Bounding box for data collection
        url = f'https://api.weather.gc.ca/collections/climate-hourly/items?bbox={bbox}&datetime={current_date.isoformat()}/{next_date.isoformat()}' # URL to get data from API

        # Make request to API
        response = await session.get(url) # Make function wait for response
        
        # Check if response data is in JSON format
        try:
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
                        'Wind Speed': data_point['properties'].get('WIND_SPEED'),
                        'Station Pressure': data_point['properties'].get('STATION_PRESSURE'),
                        'Humidity': data_point['properties'].get('RELATIVE_HUMIDITY'),
                        'Windchill': data_point['properties'].get('WINDCHILL'),
                    })
                
                weather_data_df_temp = pd.DataFrame(weather_data)
                hour_unique_count = weather_data_df_temp['Hour'].nunique()
                if(hour_unique_count>23):
                    break # Break loop if enough data is found

            else: # If not enough data is found, increase the bounding box to get more data
                bbox_limit += 0.2
                await asyncio.sleep(0.25) # Add a slight delay to avoid overloading the server

        except: # If response data is not in JSON format
            
            reset_counter = 0 # Reset counter

            print("EDGE CASE FOUND - " + current_fsa + " - " + str(current_date) + " - " + str(next_date)) # print edge case found

            # Find index of current FSA in fsa_map dictionary
            fsa_index = find_fsa_index(current_fsa, fsa_map)

            # Get the lat long of a neighboring FSA  index
            if fsa_index > 230: # If index number is greater than 230, get the lat long of the previous index
                current_lat, current_lon, current_fsa = find_fsa_lat_lon(fsa_index-1, fsa_map)
            else: # If index number is less than 230, get the lat long of the previous index
                current_lat, current_lon, current_fsa = find_fsa_lat_lon(fsa_index+1, fsa_map)
            bbox_limit = 0.2 # Reset bounding box limit
            await asyncio.sleep(0.25) # Add a slight delay to avoid overloading the server


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
async def get_data_for_time_range(data_path, start_date: datetime, end_date: datetime, fsa, lat: float, lon: float, fsa_map: dict):
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
            weather_jobs.append(get_weather_data(session, current_date, next_date, lat, lon, fsa, fsa_map))

            # Move to next day
            current_date = current_date + timedelta(days=1)  

        # Run all jobs asynchronously   
        retuned_data = await asyncio.gather(*weather_jobs)

        # Turn list of weather dataframes for each day into one dataframe
        weather_data_list = [data for data in retuned_data if not data.empty] # Get all data that is not empty
        weather_data = pd.concat(weather_data_list) if weather_data_list else pd.DataFrame() # Concatenate all dataframes into one dataframe

    # Fill missing data in weather data + Calculate Windchill + Add Calendar Columns (Weekend, Season, Holiday) + Add 24 Lags
    weather_data = fill_missing_data(weather_data)
    weather_data = calculate_windchill(weather_data)
    weather_data = add_calendar_columns(weather_data)

    weather_data = add_lags_to_weather_data(weather_data, 23)

    # Collect Power Data - Pick up data from saved CSV from IESO
    power_data = get_power_data(data_path, start_date, end_date, fsa)

    # Fill missing data in power data
    power_data = fill_missing_data(power_data)

    return weather_data, power_data

# normalize_data - Normalize data using MinMaxScaler
def normalize_data(weather_data: pd.DataFrame, power_data: pd.DataFrame, scaler=None):
    
    weather_data = weather_data.drop(columns = ["Year", "Month", "Day","Hour"])
    
    # Initialize Scaler if not provided
    if not scaler:
        weather_scaler = preprocessing.MinMaxScaler()
        power_scaler = preprocessing.MinMaxScaler()
        
        #scaler = preprocessing.StandardScaler()

    # Normalize Weather Data
    weather_data_normalized = weather_data.copy()
    
    # Do not normalize the categorical variables
    features = weather_data_normalized.columns
    # normalized_features = []
    # for feature in features:
    #     if (feature == "Weekend" or feature == "Season" or ("Year" in feature) or ("Month" in feature) or ("Day" in feature) or ("Hour" in feature)):
    #         continue
    #     else:
    #         normalized_features.append(feature)
        
    weather_data_normalized[features] = weather_scaler.fit_transform(weather_data_normalized[features])

    # Normalize Power Data - Only TOTAL_CONSUMPTION column
    power_data_normalized = power_data.copy()
    power_data_normalized['TOTAL_CONSUMPTION'] = power_scaler.fit_transform(power_data_normalized['TOTAL_CONSUMPTION'].values.reshape(-1, 1))

    return weather_data_normalized, power_data_normalized, weather_scaler, power_scaler

def setup_fsa_map(fsa_map_path: str):
    #Set up FSA Dictionary that will map FSA names to a latitude and longitude
    #Set up file path to FSA Map
    
    #Open file and read in data into a dataframe
    fsa_map_df = pd.read_csv(fsa_map_path)
    
    #Create dictionary
    fsa_map = {}
    for index, row in fsa_map_df.iterrows():
        fsa_map[row["FSA"]] = {"lat": row["LATITUDE"], "lon": row["LONGITUDE"]}
    return(fsa_map)