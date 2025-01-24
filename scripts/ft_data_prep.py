## Goal: To combine Janna's feature transformation csv with existing data frame containing calander info
# Imports
import os
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import SplineTransformer

## Preprocessing Functions
# fill_missing_data - Fill missing data in given dataframe  
def fill_missing_data(data: pd.DataFrame):

    # Fill missing data with forward fill and back fill
    data = data.ffill()
    data = data.bfill()

    return data



# File Path Set Up
## Import saved CSV into script as dataframes
data_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/data/"
temp_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/temp/"
janna_directory = os.path.join(data_dir, "janna_data")
onehotcal_directory = os.path.join(data_dir, "X_Y_Inputs")

# Getting saved data and storing in dataframe
X_transformed_weather = pd.read_csv(os.path.join(janna_directory, "Transformed_Features_Normalized.csv"))
X_onehotcal = pd.read_csv(os.path.join(onehotcal_directory, "X_df_L9G.csv")).drop(["Temp (C)", "Dew Point Temp (C)", "Rel Hum (%)", "Wind Spd (km/h)", "WIND CHILL CALCULATION"], axis=1)
X_orig = pd.read_csv(os.path.join(onehotcal_directory, "X_df_L9G_orig.csv")).drop(["Temp (C)", "Dew Point Temp (C)", "Rel Hum (%)", "Wind Spd (km/h)", "WIND CHILL CALCULATION", "DATE"], axis=1)

# Concatinating dataframes and saving to CSVs
X_transWithOneHot = X_transformed_weather.join(X_onehotcal)
X_transWithOrig = X_transformed_weather.join(X_orig)

# Do Preprocessing of data
X_transWithOneHot = fill_missing_data(X_transWithOneHot)
X_transWithOrig = fill_missing_data(X_transWithOrig)

# Saving to CSVs in Janna's Directory
X_transWithOneHot.to_csv(f'{janna_directory}/X_transformed_with_oneHotCalVariables.csv', index=False)
X_transWithOrig.to_csv(f'{janna_directory}/X_transformed_with_origCalVariables.csv', index=False)


