### Test script that can be used to fill in missing data points in X dataframe ###
## Method: When a missing value has been found, take the linear interpolation between the last real data point and next real data point and assign values to the inbetween missing variables ##
## NOTE: If there is one value missing between two data points, this method effectively assigns the last realy data point to the missing value
## NOTE: This method can also handle missing data points at the end of the file with a brute force approach of assigning all end of file missing variable to be the same as the last real data point ##

import os
import pandas as pd
import numpy as np
import math
## Import saved CSV into script as dataframes
data_dir = "D:\\Users\\Joseph\\Documents\\GitHub\\power-forecasting-capstone\\data"
df_directory = os.path.join(data_dir, "Data_Frames")

X = pd.read_csv(os.path.join(df_directory, "X_2018.csv"))
Y = pd.read_csv(os.path.join(df_directory, "Y_2018.csv"))


## Cleaning up X dataframe
# Exclude "Wind Chill" column
X_without_bad_windchill = X.drop(columns="Wind Chill")
print(X_without_bad_windchill.to_string(max_rows=5))

## Smooth out missing data
#    METHOD FOR MISSING WEATHER DATA: If value is blank, set black as average of pervious and next values
# Get columns in dataframe and cycle through them in a for loop
X_columns = X_without_bad_windchill.columns
for column in X_columns:
    # Skip over date/time columns
    if column == "YEAR" or column == "MONTH" or column == "DAY" or column == "HOUR":
        continue

    print("Current Column: " + column) # Checking what columns make it here
    print("\n")
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
            print("MISSING DATA FOUND AT Index: " , str(index) , " | Column: " + column + " | Value: " + str(row[column]))
            
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
                print(" NEW INTERPOLATED VALUE = " + str(X_without_bad_windchill.loc[index+counter, column]))
                counter += 1
                counter_adjacent_nan += 1
            
# Save modified data frame to csv for validation
df_dir_path = os.path.join(".\data\Data_Frames\\")
if not os.path.exists(df_dir_path): # If Data_Frames directory does NOT exists, create it
    os.makedirs(df_dir_path)
X_without_bad_windchill.to_csv(df_dir_path + "X_2018_interpolated.csv", index=False)
            
            

