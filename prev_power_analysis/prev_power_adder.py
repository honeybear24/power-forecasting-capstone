# Add power from the previous year to the X dataframe

# Imports
import sys
import os
import pandas as pd
import numpy as np

# add_lagged_power - Add power consumption from the previous year (the previous 8760 data points) to weather data frame + Return newly sized weather and power df
def add_lagged_power(weather_data: pd.DataFrame, power_data: pd.DataFrame):

    # Remove the first 8760 rows from weather_data and power_data
    weather_data = weather_data.iloc[8760:]
    power_data_modified = power_data.iloc[8760:]

    # Reset the index of weather_data
    weather_data = weather_data.reset_index(drop=True)

    # Remove last 8760 rows from power_data and store in new dataframe
    power_data_lagged = power_data.iloc[:-8760]

    # Reset the index of power_data_lagged
    power_data_lagged = power_data_lagged.reset_index(drop=True)

    # Add power_data_lagged to weather_data
    weather_data['TOTAL_CONSUMPTION'] = power_data_lagged['TOTAL_CONSUMPTION']
    
    # Return the modified weather_data and power_data
    return weather_data, power_data_modified


# Set Up
data_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/data/"
anal_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/prev_power_analysis"
df_directory = os.path.join(data_dir, "raw_data")

# Open target weather and power data frames
weather_data = pd.read_csv(os.path.join(df_directory, "weather_data_K0K_20180101_20231231_prevPowerAnal.csv"))
power_data = pd.read_csv(os.path.join(df_directory, "power_data_K0K_20180101_20231231_prevPowerAnal.csv"))

weather_data, power_data = add_lagged_power(weather_data, power_data)
weather_data.to_csv(f'{anal_dir}/weather_M2M.csv', index=False)
power_data.to_csv(f'{anal_dir}/power_M2M.csv', index=False)



