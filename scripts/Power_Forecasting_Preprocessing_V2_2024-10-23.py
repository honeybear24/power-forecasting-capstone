# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Libraries
import pandas as pd
import datetime
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import os

#%% Student directory
joseph_run = "C:\\Users\\sposa\\OneDrive - Toronto Metropolitan University (TMU)\\"
hanad_run = "C:\\Users\\hanad\\OneDrive - Toronto Metropolitan University (TMU)\\"
# janna_run
# clover_run

###############################################################################
############### MAKE SURE TO CHANGE BEFORE RUNNING CODE #######################
###############################################################################
# Paste student name_run for whoever is running the code
run_student = joseph_run
if (run_student == joseph_run):
    print("JOSEPH IS RUNNING!")
elif (run_student == hanad_run):
    print("HANAD IS RUNNING!")
# elif (run_student == janna_run):
#     print("JANNA IS RUNNING!")
# elif (run_student == clover_run):
#     print("CLOVER IS RUNNING!")
else:
    print("ERROR!! NO ELIGIBLE STUDENT!")
            
#%% User input

# FSA - Forward Section Area (first 3 characters of Postal Code)
#       L8K = Neighborhood in Hamilton (Link to FSA Map: https://www.prospectsinfluential.com/wp-content/uploads/2017/09/Interactive-Canadian-FSA-Map.pdf)
fsa_list = ['L9G']

# GUI INPUT
fsa_chosen = "L9G"

#years = ['2018', '2019', '2020', '2021', '2022', '2023']
years = ['2018']

# Jan - 01
# Feb - 02
# Mar - 03
# Apr - 04
# May - 05
# Jun - 06
# Jul - 07
# Aug - 08
# Sep - 09
# Oct - 10
# Nov - 11
# Dec - 12
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

#%% Input files (PRE-PROCESSING)
# dirs_inputs = run_student + "ELE-70A (Capstone)\\Inputs\\"
dirs_inputs = "data"

dirs_hourly_consumption_demand = os.path.join(dirs_inputs, "Hourly_Demand_Data")

# Dictionary for reading in hourly consumption by FSA
# FSA -> Year -> Month -> Value
hourly_consumption_data_dic_by_month = {}

for fsa in fsa_list:
    hourly_consumption_data_dic_by_month[fsa] = {} # Initialize FSA dictionary
    for year in years:
        hourly_consumption_data_dic_by_month[fsa][year] = {} # Initialize yearly dictionary
        
        for month in months:
            hourly_consumption_data_dic_by_month[fsa][year][month] = {} # Initialize monthly dictionary
            
            # Initialize dataframes to be used
            hourly_data_date = pd.DataFrame()
            hourly_data_res = pd.DataFrame()
            hourly_data_res_fsa = pd.DataFrame()
            hourly_data_hour_sum = pd.DataFrame()
            
            hourly_data_string = "PUB_HourlyConsumptionByFSA_"+year+month+"_v1.csv"
            
            # Use try and catch if problems reading input data
            try:
                # Not cooked yet, we are going to let it COOK below
                file_path = os.path.join(dirs_hourly_consumption_demand, hourly_data_string)
                hourly_data_raw = pd.read_csv(file_path, skiprows=3, header = 0, usecols= ['FSA', 'DATE', 'HOUR', 'CUSTOMER_TYPE', 'TOTAL_CONSUMPTION'])
            except FileNotFoundError: # not all months had a file (for example, 2024 only has up to may)
                continue
            except ValueError: # skiprows=x does not match the "normal sequence" of 3. For example, 2023 08 data had a different skip_row value
                hourly_data_raw = pd.read_csv(dirs_hourly_consumption_demand+hourly_data_string, skiprows=7, header = 0, usecols= ['FSA', 'DATE', 'HOUR', 'CUSTOMER_TYPE', 'TOTAL_CONSUMPTION'])
       
            # Convert Date into year, month, day
            hourly_data_fix_date = hourly_data_raw
            hourly_data_fix_date['DATE'] = pd.to_datetime(hourly_data_raw['DATE'])
            hourly_data_fix_date['YEAR'] = hourly_data_fix_date['DATE'].dt.year
            hourly_data_fix_date['MONTH'] = hourly_data_fix_date['DATE'].dt.month
            hourly_data_fix_date['DAY'] = hourly_data_fix_date['DATE'].dt.day
            
            # Filter out only residential data
            hourly_data_res = hourly_data_fix_date.loc[hourly_data_fix_date['CUSTOMER_TYPE'] == "Residential"].reset_index(drop=True)
            
            # Then filter out by the fsa
            hourly_data_res_fsa = hourly_data_res.loc[hourly_data_res['FSA'] == fsa].reset_index(drop=True)
            
            # Take the sum if fsa has more than 1 date (this is because there are different pay codes in residential loads)
            hourly_data_hour_sum = hourly_data_res_fsa.groupby(["FSA", "CUSTOMER_TYPE", "YEAR", "MONTH", "DAY", "HOUR"]).TOTAL_CONSUMPTION.sum().reset_index()
            
            
            hourly_consumption_data_dic_by_month[fsa][year][month] = hourly_data_hour_sum
            
            
            print(hourly_data_string)


# Conversion between FSA and latitude longitude - NOT IMPORTANT NOW
dirs_hourly_weather = os.path.join(dirs_inputs, "Weather_Data\\Hamilton_Weather\\")

# Dictionary for reading in weather data
# FSA -> Year -> Month -> Value

hourly_weather_data_dic_by_month = {}

for fsa in fsa_list:
    hourly_weather_data_dic_by_month[fsa] = {} # Initialize FSA dictionary
    for year in years:
        hourly_weather_data_dic_by_month[fsa][year] = {} # Initialize yearly dictionary
        
        for month in months:
            hourly_weather_data_dic_by_month[fsa][year][month] = {} # Initialize monthly dictionary
            
            # # Initialize dataframes to be used
            # hourly_data_date = pd.DataFrame()
            # hourly_data_res = pd.DataFrame()
            # hourly_data_res_fsa = pd.DataFrame()
            # hourly_data_hour_sum = pd.DataFrame()
            climate_id = "6153193"
            weather_data_string = "en_climate_hourly_ON_" + climate_id + "_" + month + "-" + year + "_P1H.csv"
            
            # Use try and catch if problems reading input data
            try:
                # Not cooked yet, we are going to let it COOK below
                ## TO DO: Calculate Hmdx using existing data (Link: https://weather.mcmaster.ca/parameter_calculation)
                file_path = os.path.join(dirs_hourly_weather, weather_data_string)
                hourly_data_raw = pd.read_csv(file_path, skiprows=0, header = 0, usecols= ['Climate ID', 'Date/Time (LST)', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Dir (10s deg)', 'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)', 'Wind Chill', 'Weather'])
            except FileNotFoundError: # not all months had a file (for example, 2024 only has up to may)
                continue
            # except ValueError: # skiprows=x does not match the "normal sequence" of 3. For example, 2023 08 data had a different skip_row value
            #     hourly_data_raw = pd.read_csv(dirs_hourly_weather+weather_data_string, skiprows=7, header = 0, usecols= ['FSA', 'DATE', 'HOUR', 'CUSTOMER_TYPE', 'TOTAL_CONSUMPTION'])
       
            # Convert Date into year, month, day
            hourly_data_fix_date = hourly_data_raw
            hourly_data_fix_date['DATE'] = pd.to_datetime(hourly_data_raw['Date/Time (LST)'])
            hourly_data_fix_date['YEAR'] = hourly_data_fix_date['DATE'].dt.year
            hourly_data_fix_date['MONTH'] = hourly_data_fix_date['DATE'].dt.month
            hourly_data_fix_date['DAY'] = hourly_data_fix_date['DATE'].dt.day
            hourly_data_fix_date['HOUR'] = hourly_data_fix_date['DATE'].dt.hour
            hourly_data_fix_date['HOUR'] = hourly_data_fix_date['HOUR']+1   
            
            # Manual Calculation for WindChill
            hourly_data_fix_date['WIND CHILL CALCULATION'] = 13.12 + hourly_data_fix_date['Temp (°C)']*0.6125 - 11.37 * hourly_data_fix_date['Wind Spd (km/h)']**0.16 + 0.3965 * hourly_data_fix_date['Temp (°C)'] * hourly_data_fix_date['Wind Spd (km/h)']**0.16
            
            hourly_weather_data_dic_by_month[fsa][year][month] = hourly_data_fix_date
            
            print(weather_data_string)


# Calendar Variables

# X Variables
X_df = pd.DataFrame()
Y_df = pd.DataFrame()

# Combine hourly data by month into the X_df
for year in years:        
    for month in months:
        # Extract the monthly hourly data for x variables
        hourly_data_consumption_by_month_X = hourly_consumption_data_dic_by_month[fsa_chosen][year][month].drop(["FSA", "CUSTOMER_TYPE", "TOTAL_CONSUMPTION"], axis=1)
        hourly_data_weather_by_month_X = hourly_weather_data_dic_by_month[fsa_chosen][year][month].drop(['Climate ID', 'Date/Time (LST)', 'Wind Dir (10s deg)', 'Visibility (km)', 'Stn Press (kPa)', 'Weather', 'DATE', 'YEAR', 'MONTH', 'DAY', 'HOUR'], axis=1)
        
        
        # Combine all hourly data for x variables
        hourly_data_by_month_X = pd.concat([hourly_data_consumption_by_month_X, hourly_data_weather_by_month_X], axis = 1)
        
        # Extract the monthly hourly data for y variables
        hourly_data_by_month_Y = hourly_consumption_data_dic_by_month[fsa_chosen][year][month].drop(["FSA", "CUSTOMER_TYPE", "YEAR", "MONTH", "DAY", "HOUR"], axis=1)
        
        
        
        X_df = pd.concat([X_df, hourly_data_by_month_X], ignore_index=True)
        Y_df = pd.concat([Y_df, hourly_data_by_month_Y], ignore_index=True)



#%% Plot Input Data
year_plot = "2018"
month_plot =  "02"
day_plot = "03"

# First day
hourly_data_month_day = hourly_consumption_data_dic_by_month[fsa][year_plot][month_plot]
hourly_data_month_day = hourly_data_month_day[hourly_data_month_day['DAY'] == int(day_plot)]

plot = plt.subplot(1, 3, 1)
plot = plt.plot(hourly_data_month_day["HOUR"], hourly_data_month_day["TOTAL_CONSUMPTION"], 'o-')

plt.title("HOURLY THREE DAY CONSUMPTION STARTING " + year_plot + "/" + month_plot + "/" + day_plot)
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")


# Second day
hourly_data_month_day = hourly_consumption_data_dic_by_month[fsa][year_plot][month_plot]
hourly_data_month_day = hourly_data_month_day[hourly_data_month_day['DAY'] == int(day_plot)+1]

plot = plt.subplot(1, 3, 2)
plot = plt.plot(hourly_data_month_day["HOUR"], hourly_data_month_day["TOTAL_CONSUMPTION"], 'o-')

# Third day
hourly_data_month_day = hourly_consumption_data_dic_by_month[fsa][year_plot][month_plot]
hourly_data_month_day = hourly_data_month_day[hourly_data_month_day['DAY'] == int(day_plot)+2]

plot = plt.subplot(1, 3, 3)
plot = plt.plot(hourly_data_month_day["HOUR"], hourly_data_month_day["TOTAL_CONSUMPTION"], 'o-')


plt.show()

# TRY TO PLOT WEATHER WITH X = HOUR, Y = TEMP
# Plot weather data with X = HOUR, Y = TEMP

# First day
hourly_weather_month_day = hourly_weather_data_dic_by_month[fsa][year_plot][month_plot]
hourly_weather_month_day = hourly_weather_month_day[hourly_weather_month_day['DAY'] == int(day_plot)]

plot = plt.subplot(1, 3, 1)
plot = plt.plot(hourly_weather_month_day["HOUR"], hourly_weather_month_day[hourly_weather_month_day.columns[2]], 'o-')

plt.title("HOURLY THREE DAY TEMPERATURE STARTING " + year_plot + "/" + month_plot + "/" + day_plot)
plt.xlabel("HOUR")
plt.ylabel("TEMPERATURE in °C")

# Second day
hourly_weather_month_day = hourly_weather_data_dic_by_month[fsa][year_plot][month_plot]
hourly_weather_month_day = hourly_weather_month_day[hourly_weather_month_day['DAY'] == int(day_plot)+1]

plot = plt.subplot(1, 3, 2)
plot = plt.plot(hourly_weather_month_day["HOUR"], hourly_weather_month_day[hourly_weather_month_day.columns[2]], 'o-')

# Third day
hourly_weather_month_day = hourly_weather_data_dic_by_month[fsa][year_plot][month_plot]
hourly_weather_month_day = hourly_weather_month_day[hourly_weather_month_day['DAY'] == int(day_plot)+2]

plot = plt.subplot(1, 3, 3)
plot = plt.plot(hourly_weather_month_day["HOUR"], hourly_weather_month_day[hourly_weather_month_day.columns[2]], 'o-')

plt.show()













