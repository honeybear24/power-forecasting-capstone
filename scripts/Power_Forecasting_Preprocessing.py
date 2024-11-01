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
import math
import numpy as np

#%% Student directory
hanad_run = ["./data", 1]
clover_run = ["./data", 2]
joseph_laptop_run = ["C:\\Users\\sposa\\Documents\\GitHub\\power-forecasting-capstone\\data", 3]
joseph_pc_run = ["D:\\Users\\Joseph\\Documents\\GitHub\\power-forecasting-capstone\\data", 3]
janna_run = ["./data", 4]

###############################################################################
############### MAKE SURE TO CHANGE BEFORE RUNNING CODE #######################
###############################################################################
# Paste student name_run for whoever is running the code
run_student = joseph_pc_run
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
            
#%% User input

# FSA - Forward Section Area (first 3 characters of Postal Code)
#       L8K = Neighborhood in Hamilton (Link to FSA Map: https://www.prospectsinfluential.com/wp-content/uploads/2017/09/Interactive-Canadian-FSA-Map.pdf)
fsa_list = ['L9G']

# GUI INPUT
fsa_chosen = "L9G"

years = ['2018', '2019', '2020', '2021', '2022', '2023']
#years = ['2018']

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
dirs_inputs = run_student[0]

dirs_hourly_consumption_demand = os.path.join(dirs_inputs, "Hourly_Demand_Data")

###############################################################################
# Dictionary for reading in hourly consumption by FSA
###############################################################################
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
                hourly_data_raw = pd.read_csv(file_path, skiprows=7, header = 0, usecols= ['FSA', 'DATE', 'HOUR', 'CUSTOMER_TYPE', 'TOTAL_CONSUMPTION'])
       
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

###############################################################################
# Conversion between FSA and latitude longitude - NOT IMPORTANT NOW
###############################################################################
dirs_hourly_weather = os.path.join(dirs_inputs, "Weather_Data\\Hamilton_Weather\\")

###############################################################################
# Dictionary for reading in weather data
###############################################################################
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


###############################################################################
# Calendar Variables
###############################################################################


###############################################################################
# X and Y Dataframes
###############################################################################
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
        hourly_data_by_month_Y = hourly_consumption_data_dic_by_month[fsa_chosen][year][month].drop(["FSA", "CUSTOMER_TYPE"], axis=1)
        
        
        
        X_df = pd.concat([X_df, hourly_data_by_month_X], ignore_index=True)
        Y_df = pd.concat([Y_df, hourly_data_by_month_Y], ignore_index=True)

###############################################################################
# Cleaning up X_df dataframe
###############################################################################

# Exclude "Wind Chill" column
X_without_bad_windchill = X_df.drop(columns="Wind Chill")

## Smooth out missing data
#    METHOD FOR MISSING WEATHER DATA: If value is blank, set black as average of pervious and next values
# Get columns in dataframe and cycle through them in a for loop
X_columns = X_without_bad_windchill.columns
for column in X_columns:
    # Skip over date/time columns
    if column == "YEAR" or column == "MONTH" or column == "DAY" or column == "HOUR":
        continue

    #print("Current Column: " + column) # Checking what columns make it here
    #print("\n")
    counter_adjacent_nan = 0
    # For current column, iterate through all rough
    for index, row in X_without_bad_windchill.iterrows():
        # Check if there are more than 2 nan values
        if (counter_adjacent_nan>1):
            counter_adjacent_nan -= 1
            continue
        # Check if value for current index and column is missing
        #if row[column] != row[column]:
        if math.isnan(row[column]):
            #print("MISSING DATA FOUND AT Index: " , str(index) , " | Column: " + column + " | Value: " + str(row[column]))
            
            # If found, take need to take linear interpolation between last actual value and next actual value (need to interpolate any consectutively missing data points too!)
            # Finding last actual data point
            last_data_point = X_without_bad_windchill.loc[index-1, column]
            #print("Last viable data point: " + str(last_data_point)  + "\n")
            
            # Find next available data point
            counter = 0
            not_found = True
            while not_found == True:
                counter += 1

                try: # Try to get value of next data point
                    next_data_point = X_without_bad_windchill.loc[index+counter, column]
                except KeyError: # If there are NaN values at the end of the data frame, set the value of the next data point to be eqaul last data point
                    next_data_point = last_data_point

                # Check if variable is a real value (NaN values wil ALWAYS fail this condition)
                if next_data_point == next_data_point:
                    # If found, set not_found to False and leave loop
                    not_found = False
            
            # Get linear interpolation of data
            interpolated_data = np.linspace(start=last_data_point, stop=next_data_point, num=counter+1, endpoint = False)[1:]
            

            # Give interpolated values to missing data points
            counter = 0
            counter_adjacent_nan=0
            for new_value in interpolated_data:
                X_without_bad_windchill.loc[index+counter, column] = new_value
                #print(" NEW INTERPOLATED VALUE = " + str(X_without_bad_windchill.loc[index+counter, column]))
                counter += 1
                counter_adjacent_nan += 1
X_df_cleaned = X_without_bad_windchill

#%% Regression Model
# HANAD FILLS IN CODE HERE ON A NEW BRANCH




#%% SVR Model
# CLOVER FILLS IN CODE HERE ON A NEW BRANCH



#%% KNN Model
# JOSEPH FILLS IN CODE HERE ON A NEW BRANCH
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

X_train, X_test, Y_train, Y_test = train_test_split(X_df_cleaned, Y_df['TOTAL_CONSUMPTION'], test_size=0.20, shuffle = False)

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, Y_train)

Y_pred = knn_model.predict(X_test)

# # TRY GRID SEARCH
# pipeline_knn = Pipeline([("model", KNeighborsRegressor(n_neighbors=1))])

# knn_model = GridSearchCV(estimator = pipeline_knn, param_grid = {'model__n_neighbors': range(1,50)}, cv=3)

# knn_model.fit(X_train, Y_train)
# output = pd.DataFrame(knn_model.cv_results_)

# Y_pred = knn_model.predict(X_test)

mape = mean_absolute_percentage_error(Y_test, Y_pred)
print("MAPE = " + str(round(mape*100, 3)) + "%.")
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)


## PLOTTING
year_plot = 2023
month_plot = 1
day_plot = 1



Y_pred_df = pd.DataFrame(Y_pred, columns=['TOTAL_CONSUMPTION'], index = Y_test.index)
Y_test_df = pd.DataFrame(Y_test, columns=['TOTAL_CONSUMPTION'])

Y_pred_df['YEAR'] = X_test['YEAR']
Y_pred_df['MONTH'] = X_test['MONTH']
Y_pred_df['DAY'] = X_test['DAY']
Y_pred_df['HOUR'] = X_test['HOUR']

Y_test_df['YEAR'] = X_test['YEAR']
Y_test_df['MONTH'] = X_test['MONTH']
Y_test_df['DAY'] = X_test['DAY']
Y_test_df['HOUR'] = X_test['HOUR']


X_test_year = X_test[X_test['YEAR'] == year_plot]
X_test_year_month = X_test_year[X_test_year['MONTH'] == month_plot]
X_test_year_month_day = X_test_year_month[X_test_year_month['DAY'] == day_plot]

Y_test_year = Y_test_df[Y_test_df['YEAR'] == year_plot]
Y_test_year_month = Y_test_year[Y_test_year['MONTH'] == month_plot]
Y_test_year_month_day = Y_test_year_month[Y_test_year_month['DAY'] == day_plot]

Y_pred_year = Y_pred_df[Y_pred_df['YEAR'] == year_plot]
Y_pred_year_month = Y_pred_year[Y_pred_year['MONTH'] == month_plot]
Y_pred_year_month_day = Y_pred_year_month[Y_pred_year_month['DAY'] == day_plot]

# Yearly Plot
plt.title(str(year_plot) + " Prediction VS Actual of KNN Model")

plt.plot(X_test_year['HOUR'].index, Y_pred_year['TOTAL_CONSUMPTION'], color="blue", linewidth=3)
plt.plot(X_test_year['HOUR'].index, Y_test_year['TOTAL_CONSUMPTION'], color="black")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()

plt.title(str(year_plot) + " Prediction VS Actual of KNN Model")

plt.plot(X_test_year['HOUR'].index, Y_test_year['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.plot(X_test_year['HOUR'].index, Y_pred_year['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.legend(loc="upper left")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()

# Monthly Plot
plt.title(str(year_plot) + ", Month = " + str(month_plot) + " Prediction VS Actual of KNN Model")

plt.plot(Y_test_year_month['HOUR'].index, Y_pred_year_month['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.plot(Y_test_year_month['HOUR'].index, Y_test_year_month['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.legend(loc="upper left")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()

# Daily Plot
plt.title(str(year_plot) + "/" + str(month_plot) + "/" + str(day_plot) + " Prediction VS Actual of KNN Model")

plt.plot(X_test_year_month_day['HOUR'].index, Y_pred_year_month_day['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.plot(X_test_year_month_day['HOUR'].index, Y_test_year_month_day['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.legend(loc="upper left")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()

#%% Neural Network Model
# JANNA FILLS IN CODE HERE ON A NEW BRANCH











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













