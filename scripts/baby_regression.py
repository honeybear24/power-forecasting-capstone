### This script contains Hanad's first attempt at making a regression model based off of 2018 Data ###

import os
import pandas as pd
import matplotlib as plt
#from sklearn import 

## Import saved CSV into script as dataframes
data_dir = "./data"
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

    # For current column, iterate through all rough
    for index, row in X_without_bad_windchill.iterrows():
        # Check if value for current index and column is missing
        if row[column] != row[column]:
            print("MISSING DATA FOUND AT Index: " , str(index) , " | Column: " + column + " | Value: " + str(row[column]))

            # If found, take the average of the previous 
            

            

