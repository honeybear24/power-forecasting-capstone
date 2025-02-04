# Goal: Add rolling statistics to the weather data frame

# Imports
import sys
import os
import pandas as pd
import numpy as np

# add_rolling_stats - Add rolling statistics to the weather data frame + Return newly sized weather df
def add_rolling_stats(weather_data: pd.DataFrame, power_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    
    # Add rolling statistics (mean) to the weather data frame
    weather_data['rolling_mean_temp'] = weather_data['Temperature'].rolling(window_size).mean()
    weather_data['rolling_mean_dewpoint'] = weather_data['Dew Point Temperature'].rolling(window_size).mean()
    weather_data['rolling_mean_humid'] = weather_data['Humidity'].rolling(window_size).mean()
    weather_data['rolling_mean_windspeed'] = weather_data['Wind_Speed'].rolling(window_size).mean()
    weather_data['rolling_mean_pressure'] = weather_data['Station Pressure'].rolling(window_size).mean()
    weather_data['rolling_mean_windchill'] = weather_data['Windchill'].rolling(window_size).mean()

    # Add rolling statistics (std) to the weather data frame
    weather_data['rolling_std_temp'] = weather_data['Temperature'].rolling(window_size).std()
    weather_data['rolling_std_dewpoint'] = weather_data['Dew Point Temperature'].rolling(window_size).std()
    weather_data['rolling_std_humid'] = weather_data['Humidity'].rolling(window_size).std()
    weather_data['rolling_std_windspeed'] = weather_data['Wind_Speed'].rolling(window_size).std()
    weather_data['rolling_std_pressure'] = weather_data['Station Pressure'].rolling(window_size).std()
    weather_data['rolling_std_windchill'] = weather_data['Windchill'].rolling(window_size).std()

    # Add rolling statistics (var) to the weather data frame
    weather_data['rolling_var_temp'] = weather_data['Temperature'].rolling(window_size).var()
    weather_data['rolling_var_dewpoint'] = weather_data['Dew Point Temperature'].rolling(window_size).var()
    weather_data['rolling_var_humid'] = weather_data['Humidity'].rolling(window_size).var()
    weather_data['rolling_var_windspeed'] = weather_data['Wind_Speed'].rolling(window_size).var()
    weather_data['rolling_var_pressure'] = weather_data['Station Pressure'].rolling(window_size).var()
    weather_data['rolling_var_windchill'] = weather_data['Windchill'].rolling(window_size).var()

    # Use bfill to fill in the NaN values
    weather_data = weather_data.bfill()
    
    # Return the modified weather_data
    return weather_data, power_data

# Set Up
data_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/data/"
anal_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/rolling_stats_analysis"
df_directory = os.path.join(data_dir, "raw_data")
fsa = "L9G"

# Open target weather and power data frames
weather_data = pd.read_csv(os.path.join(df_directory, f'weather_data_{fsa}_20180101_20231231_Anal.csv'))
power_data = pd.read_csv(os.path.join(df_directory, f'power_data_{fsa}_20180101_20231231_Anal.csv'))
weather_data.to_csv(f'{anal_dir}/weather_{fsa}_{0}.csv', index=False)
power_data.to_csv(f'{anal_dir}/power_{fsa}_{0}.csv', index=False)

for window_size in range(1, 25):
    mod_weather_data, mod_power_data = add_rolling_stats(weather_data, power_data, window_size)
    mod_weather_data.to_csv(f'{anal_dir}/weather_{fsa}_{window_size}.csv', index=False)
    mod_power_data.to_csv(f'{anal_dir}/power_{fsa}_{window_size}.csv', index=False)