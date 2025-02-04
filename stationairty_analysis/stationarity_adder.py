# Add stationarity to weather data frame

# Imports
import sys
import os
import pandas as pd
import numpy as np

# add_stationarity - Add stationarity to the weather data frame + Return newly sized weather df
def add_stationarity(weather_data: pd.DataFrame, power_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    
    # Add stationarity to the weather data frame
    weather_data['stationary_temp'] = weather_data['Temperature'].diff(window_size)
    weather_data['stationary_dewpoint'] = weather_data['Dew Point Temperature'].diff(window_size)
    weather_data['stationary_humid'] = weather_data['Humidity'].diff(window_size)
    weather_data['stationary_windspeed'] = weather_data['Wind_Speed'].diff(window_size)
    weather_data['stationary_pressure'] = weather_data['Station Pressure'].diff(window_size)
    weather_data['stationary_windchill'] = weather_data['Windchill'].diff(window_size)
    

    # Use bfill to fill in the NaN values
    weather_data = weather_data.bfill()
    
    # Return the modified weather_data
    return weather_data, power_data

# Set Up
data_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/data/"
anal_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/stationairty_analysis"
df_directory = os.path.join(data_dir, "raw_data")
fsa = "L9G"

# Open target weather and power data frames
weather_data = pd.read_csv(os.path.join(df_directory, f'weather_data_{fsa}_20180101_20231231_Anal.csv'))
power_data = pd.read_csv(os.path.join(df_directory, f'power_data_{fsa}_20180101_20231231_Anal.csv'))
weather_data.to_csv(f'{anal_dir}/weather_{fsa}_{0}.csv', index=False)
power_data.to_csv(f'{anal_dir}/power_{fsa}_{0}.csv', index=False)

for window_size in range(1, 25):
    mod_weather_data, mod_power_data = add_stationarity(weather_data, power_data, window_size)
    mod_weather_data.to_csv(f'{anal_dir}/weather_{fsa}_{window_size}.csv', index=False)
    mod_power_data.to_csv(f'{anal_dir}/power_{fsa}_{window_size}.csv', index=False)
