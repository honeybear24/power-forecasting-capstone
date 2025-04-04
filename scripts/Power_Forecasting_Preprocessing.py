#%% Libraries
import pandas as pd
import datetime
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import canada_holiday


from pyhelpers.store import save_fig, save_svg_as_emf
import subprocess
import  aspose.cells 
from aspose.cells import Workbook

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

months_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

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
            hourly_data_hour_sum = hourly_data_res_fsa.groupby(["FSA", "CUSTOMER_TYPE", "YEAR", "MONTH", "DAY", "HOUR", "DATE"]).TOTAL_CONSUMPTION.sum().reset_index()
            
            
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
            
            # Change Temperature column names so it does not have symbols anymore
            hourly_data_fix_date.rename(columns = {'Temp (°C)':'Temp (C)', 'Dew Point Temp (°C)':'Dew Point Temp (C)'}, inplace = True)
            
            hourly_weather_data_dic_by_month[fsa][year][month] = hourly_data_fix_date
            
            print(weather_data_string)

###############################################################################
# Calendar Variables
###############################################################################

# Add weekdays to hourly consumption dataframe
# Monday = 0 
# Tuesday = 1
# Wednesday = 2
# Thursday = 3
# Friday = 4 
# Saturday = 5
# Sunday = 6

# Add season to hourly consumption dataframe
# As per OEB:
# Winter(1) = November (Including) to April (Including)
# Summer(0) = May (Including) to October (Including)
for year in years:        
    for month in months:
        try:
            # Get day of week and check if it is a weekend or weekday
            hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["DAY_OF_WEEK"] = hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["DATE"].dt.weekday
            hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["WEEKEND"] = hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["DAY_OF_WEEK"]>4
            hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["WEEKDAY"] = hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["DAY_OF_WEEK"]<5
            
            # Convert boolean of weekend or weekday to integer numbers (1-True, 0-False)
            hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["WEEKEND"] = hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["WEEKEND"].astype(int)
            hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["WEEKDAY"] = hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["WEEKDAY"].astype(int)
            
            # Get Season and check if it is winter or summer
            hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["SEASON"] = (hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["MONTH"]<5) | (hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["MONTH"]>10)
            
            # Convert boolean of weekend or weekday to integer numbers (1-Winter, 0-Summer)
            hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["SEASON"] = hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["SEASON"].astype(int)
            
            
            
            
            # Pad the holiday column with zeros to initialize it
            hourly_consumption_data_dic_by_month[fsa_chosen][year][month]["HOLIDAY"] = 0
            
            for index, row in  hourly_consumption_data_dic_by_month[fsa_chosen][year][month].iterrows():
                date_temp = date(row["YEAR"], row["MONTH"], row["DAY"])
                if (row["HOUR"] == 1):
                    if canada_holiday.is_holiday(date_temp, "Ontario"):
                        hourly_consumption_data_dic_by_month[fsa_chosen][year][month].loc[index, "HOLIDAY"] = 1
                        temp_value = 1
                    else:
                        temp_value = 0
                else:
                    hourly_consumption_data_dic_by_month[fsa_chosen][year][month].loc[index, "HOLIDAY"] = temp_value
        except KeyError:
            continue

###############################################################################
# X and Y Dataframes
###############################################################################
X_df = pd.DataFrame()
Y_df = pd.DataFrame()

# Combine hourly data by month into the X_df
for year in years:        
    for month in months:
        try:
            # Extract the monthly hourly data for x variables
            hourly_data_consumption_by_month_X = hourly_consumption_data_dic_by_month[fsa_chosen][year][month].drop(["FSA", "CUSTOMER_TYPE", "TOTAL_CONSUMPTION", "DATE"], axis=1)
            hourly_data_weather_by_month_X = hourly_weather_data_dic_by_month[fsa_chosen][year][month].drop(['Climate ID', 'Date/Time (LST)', 'Wind Dir (10s deg)', 'Visibility (km)', 'Stn Press (kPa)', 'Weather', 'DATE', 'YEAR', 'MONTH', 'DAY', 'HOUR'], axis=1)
            
            
            # Combine all hourly data for x variables
            hourly_data_by_month_X = pd.concat([hourly_data_consumption_by_month_X, hourly_data_weather_by_month_X], axis = 1)
            
            # Extract the monthly hourly data for y variables
            hourly_data_by_month_Y = hourly_consumption_data_dic_by_month[fsa_chosen][year][month].drop(["FSA", "CUSTOMER_TYPE", "DATE", "DAY_OF_WEEK", "WEEKEND", "WEEKDAY", "SEASON", "HOLIDAY"], axis=1)
            
            
            X_df = pd.concat([X_df, hourly_data_by_month_X], ignore_index=True)
            Y_df = pd.concat([Y_df, hourly_data_by_month_Y], ignore_index=True)
        except KeyError:
            continue

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

# Add Date back into X_df_cleaned variable
X_df_cleaned["DATE"] = 0
for index, row in  X_df_cleaned.iterrows():
    X_df_cleaned.loc[index, "DATE"] = date(int(row["YEAR"]),int(row["MONTH"]),int(row["DAY"]))

# Convert year, month, day, hour to boolean values

# Day range 1 to 31
days = [*range(1, 32)]
# Hour range 1 to 24
hours = [*range(1, 25)]

for year in years:
    X_df_cleaned[year] = X_df_cleaned["YEAR"]==int(year)
    X_df_cleaned[year] = X_df_cleaned[year].astype(int)

for month in months_name:
    X_df_cleaned[month] = X_df_cleaned["MONTH"]==int(months_name.index(month)+1)
    X_df_cleaned[month] = X_df_cleaned[month].astype(int)

for day in days:
    X_df_cleaned["DAY_"+str(day)] = X_df_cleaned["DAY"]==int(day)
    X_df_cleaned["DAY_"+str(day)] = X_df_cleaned["DAY_"+str(day)].astype(int)

for hour in hours:
    X_df_cleaned["HOUR_"+str(hour)] = X_df_cleaned["HOUR"]==int(hour)
    X_df_cleaned["HOUR_"+str(hour)] = X_df_cleaned["HOUR_"+str(hour)].astype(int)

X_df_cleaned_output = X_df_cleaned.drop(['YEAR', 'MONTH', 'DAY', 'HOUR', 'DATE', 'WEEKDAY', 'DAY_OF_WEEK'], axis=1)

###############################################################################
# Export X and Y Dataframes to CSV
###############################################################################
dirs_dataframes = os.path.join(dirs_inputs, "X_Y_Inputs")
X_df_output_string =  os.path.join(dirs_dataframes, "X_df_"+fsa_chosen+".csv")
X_df_cleaned_output.to_csv(X_df_output_string, index=False)
Y_df_output_string =  os.path.join(dirs_dataframes, "Y_df_"+fsa_chosen+".csv")
Y_df["TOTAL_CONSUMPTION"].to_csv(Y_df_output_string, index=False)

#%% Regression Model
# HANAD FILLS IN CODE HERE ON A NEW BRANCH



#%% SVR Model
# CLOVER FILLS IN CODE HERE ON A NEW BRANCH


#%% KNN Model
# JOSEPH FILLS IN CODE HERE ON A NEW BRANCH
# ADDING STUFF




#%% Neural Network Model
# JANNA FILLS IN CODE HERE ON A NEW BRANCH

#%% Plot All Input Variables Over Power Consumption
 

dirs_plots = os.path.join(dirs_inputs, "Input_Plots")
save_plots = True


for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)] 
    plt.scatter(X_year["YEAR"], Y_year["TOTAL_CONSUMPTION"], label = year, rasterized=True)
    plt.title("Year Versus Consumption.")
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Year")
    plt.legend() 
    plot_svg =  os.path.join(dirs_plots, "Year_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]   
    plt.scatter(X_year["MONTH"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Month Versus Consumption.")
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Month")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "Month_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]   
    plt.scatter(X_year["DAY_OF_WEEK"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Day of Week Versus Consumption.")
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Day of Week")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "DOW_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]   
    plt.scatter(X_year["SEASON"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Season Versus Consumption.")
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Season (1-Winter, 0-Summer)")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "Season_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]   
    plt.scatter(X_year["HOUR"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Hour Versus Consumption.")
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Hour")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "Hour_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]   
    plt.scatter(X_year["WEEKEND"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Weekend Versus Consumption.")
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Weekend (Boolean)")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "Weekend_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]     
    plt.scatter(X_year["HOLIDAY"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Holiday Versus Consumption.")
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Holiday (Boolean)")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "Holiday_VS_Consumption.png") 
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]     
    plt.scatter(X_year["Temp (C)"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Temperature Versus Consumption.")
    plt.legend()
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Temperature (°C)")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "Temp_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]     
    plt.scatter(X_year["Dew Point Temp (C)"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Dew Point Temperature Versus Consumption.")
    plt.legend()
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Dew Point Temperature (°C)")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "Dew_Point_Temp_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]     
    plt.scatter(X_year["Rel Hum (%)"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Relative Humidity Versus Consumption.")
    plt.legend()
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Relative Humidity (%)")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "Relative_Humidity_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]     
    plt.scatter(X_year["Wind Spd (km/h)"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Wind Speed Versus Consumption.")
    plt.legend()
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Wind Speed (km/h)")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "Wind_Speed_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

for year in years:
    X_year = X_df_cleaned.loc[X_df_cleaned['YEAR'] == int(year)]
    Y_year = Y_df.loc[Y_df['YEAR'] == int(year)]   
    plt.scatter(X_year["WIND CHILL CALCULATION"], Y_year["TOTAL_CONSUMPTION"], label = year)
    plt.title("Wind Chill Versus Consumption.")
    plt.legend()
    plt.ylabel("Consumption (KW)")
    plt.xlabel("Wind Speed (°C)")
    plt.legend()
    plot_svg =  os.path.join(dirs_plots, "Wind_Chill_VS_Consumption.png")
if save_plots:
    plt.savefig(plot_svg)
plt.show()

#%% MAY DELETE LATER Plot Power Consumption and Temperature over time period
# year_plot = "2018"
# month_plot =  "02"
# day_plot = "03"

# # First day
# hourly_data_month_day = hourly_consumption_data_dic_by_month[fsa][year_plot][month_plot]
# hourly_data_month_day = hourly_data_month_day[hourly_data_month_day['DAY'] == int(day_plot)]

# plot = plt.subplot(1, 3, 1)
# plot = plt.plot(hourly_data_month_day["HOUR"], hourly_data_month_day["TOTAL_CONSUMPTION"], 'o-')

# plt.title("HOURLY THREE DAY CONSUMPTION STARTING " + year_plot + "/" + month_plot + "/" + day_plot)
# plt.xlabel("HOUR")
# plt.ylabel("CONSUMPTION in KW")


# # Second day
# hourly_data_month_day = hourly_consumption_data_dic_by_month[fsa][year_plot][month_plot]
# hourly_data_month_day = hourly_data_month_day[hourly_data_month_day['DAY'] == int(day_plot)+1]

# plot = plt.subplot(1, 3, 2)
# plot = plt.plot(hourly_data_month_day["HOUR"], hourly_data_month_day["TOTAL_CONSUMPTION"], 'o-')

# # Third day
# hourly_data_month_day = hourly_consumption_data_dic_by_month[fsa][year_plot][month_plot]
# hourly_data_month_day = hourly_data_month_day[hourly_data_month_day['DAY'] == int(day_plot)+2]

# plot = plt.subplot(1, 3, 3)
# plot = plt.plot(hourly_data_month_day["HOUR"], hourly_data_month_day["TOTAL_CONSUMPTION"], 'o-')


# plt.show()

# # TRY TO PLOT WEATHER WITH X = HOUR, Y = TEMP
# # Plot weather data with X = HOUR, Y = TEMP

# # First day
# hourly_weather_month_day = hourly_weather_data_dic_by_month[fsa][year_plot][month_plot]
# hourly_weather_month_day = hourly_weather_month_day[hourly_weather_month_day['DAY'] == int(day_plot)]

# plot = plt.subplot(1, 3, 1)
# plot = plt.plot(hourly_weather_month_day["HOUR"], hourly_weather_month_day[hourly_weather_month_day.columns[2]], 'o-')

# plt.title("HOURLY THREE DAY TEMPERATURE STARTING " + year_plot + "/" + month_plot + "/" + day_plot)
# plt.xlabel("HOUR")
# plt.ylabel("TEMPERATURE in °C")

# # Second day
# hourly_weather_month_day = hourly_weather_data_dic_by_month[fsa][year_plot][month_plot]
# hourly_weather_month_day = hourly_weather_month_day[hourly_weather_month_day['DAY'] == int(day_plot)+1]

# plot = plt.subplot(1, 3, 2)
# plot = plt.plot(hourly_weather_month_day["HOUR"], hourly_weather_month_day[hourly_weather_month_day.columns[2]], 'o-')

# # Third day
# hourly_weather_month_day = hourly_weather_data_dic_by_month[fsa][year_plot][month_plot]
# hourly_weather_month_day = hourly_weather_month_day[hourly_weather_month_day['DAY'] == int(day_plot)+2]

# plot = plt.subplot(1, 3, 3)
# plot = plt.plot(hourly_weather_month_day["HOUR"], hourly_weather_month_day[hourly_weather_month_day.columns[2]], 'o-')
# plt.show()






