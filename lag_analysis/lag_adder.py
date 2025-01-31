# Script to read in an X frame, add lags to it and save file

# Imports
import sys
import os
import pandas as pd
import numpy as np

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

# Set Up
data_path = "c:/Users/hanad/capstone_github/power-forecasting-capstone/data"
target_dir = data_path + "/raw_data"
file_name = "weather_data_L9G_20180101_20231231_lagAnal_lag0.csv"
target_path = os.path.join(target_dir, file_name)

# Read in file into data frame
data = pd.read_csv(target_path)

# Add lags to DF and save as new file in target_path
for i in range (1, 25):
    lagged_data = add_lags_to_weather_data(data, i)
    lagged_data.to_csv(f'{target_dir}/weather_data_L9G_20180101_20231231_lagAnal_lag{i}.csv', index=False)
