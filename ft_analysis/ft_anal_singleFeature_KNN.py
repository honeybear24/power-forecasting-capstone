# Feature Transformation Analysis, Single Features - This script is used to analyze effect individual features have on KNN model forecasting

"""
    Results of Single Feature Analysis: Same impact seen in linear regression model

"""


# Set Up
import os
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
import joblib
from sklearn.neighbors import KNeighborsRegressor


# Import saved CSV into script as dataframes
data_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/data/"
temp_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/temp/"
anal_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/ft_analysis"
df_directory = os.path.join(data_dir, "janna_data")
anal_directory = os.path.join(data_dir, "ft_analysis")
#X = pd.read_csv(os.path.join(df_directory, "X_transformed_with_oneHotCalVariables.csv"))
X = pd.read_csv(os.path.join(df_directory, "X_transformed_with_origCalVariables.csv"))
Y = pd.read_csv(os.path.join(df_directory, "Y_df_normalized.csv"))

# Create pipeline containing ridge regression and spline transformer    
pipe = KNeighborsRegressor(n_neighbors=12, weights = 'distance')


### Evaluating Model with only Base Features ### 
# Extract main features from X dataframe
X_main = X[['Year', 'Month', 'Day', 'Hour', 'WEEKEND', 'WEEKDAY', 'HOLIDAY', 'SEASON', 'Temp (C) - Original', 'Dew Point Temp (C) - Original', 'Rel Hum (%) - Original', 'Wind Spd (km/h) - Original', 'WIND CHILL CALCULATION - Original']]

### Testing to see if the main features are picked up correctly
#print(X_main.head())

# Make train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_main, Y['TOTAL_CONSUMPTION'], test_size=1/(5), shuffle=False)

# Fit model
pipe.fit(X_train, Y_train)

# Predict
Y_pred = pipe.predict(X_test)

# Open min_max_scaler_y.pkl using joblib - Used to denormalize Y_pred and Y_test
scaler_path = os.path.join(df_directory, "min_max_scaler_y.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
scaler = joblib.load(scaler_path)

# Ensure Y_pred and Y_test are reshaped correctly
Y_pred = Y_pred.reshape(-1, 1) # Reshape Y_pred to correct size for denormalization (-1 = infer number of rows, 1 = 1 column)
Y_test = Y_test.values.reshape(-1, 1)

# Denormalize Y_pred and Y_test
Y_pred_denorm = scaler.inverse_transform(Y_pred)
Y_pred_denorm_df = pd.DataFrame(Y_pred_denorm, columns=['TOTAL_CONSUMPTION'])
Y_test_denorm = scaler.inverse_transform(Y_test)
Y_test_denorm_df = pd.DataFrame(Y_test_denorm, columns=['TOTAL_CONSUMPTION'])

# Evaluate Model - Always make sure to denormalize Y_pred and Y_test before evaluating
mape = mean_absolute_percentage_error(Y_test_denorm_df, Y_pred_denorm_df)
mae = mean_absolute_error(Y_test_denorm_df, Y_pred_denorm_df)
r2 = r2_score(Y_test_denorm_df, Y_pred_denorm_df)
mse = mean_squared_error(Y_test_denorm_df, Y_pred_denorm_df)
rmse = root_mean_squared_error(Y_test_denorm_df, Y_pred_denorm_df)

# Print to STDOUT
print("Features, MAPE, MAE, R^2, MSE, RMSE")
print("Base Features, ", mape, ", ", mae, ", ", r2, ", ", mse, ", ", rmse)


### Evaluating Model with 1 Selected Feature Features ###
# Make list of all base features and list of tags
list_of_base_features_without_tags = ['Temp (C)', 'Dew Point Temp (C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'WIND CHILL CALCULATION']
list_of_transformations = ['Square', 'Cube', 'Sine', 'Cosine', 'Tangent', 'Logarithm', 'Square Root', 'Inverse']

# Iterate through each feature and transformation
for base_ft in list_of_base_features_without_tags:
    for tag in list_of_transformations:

        # Create new feature name
        new_feature_name = base_ft + " - " + tag

        # Make copy of X_main
        X_new = X_main.copy() 

        # Extract target feature from X dataframe and concantenate with X_new
        X_target = X[[new_feature_name]]
        X_new = pd.concat([X_new, X_target], axis=1)

        # ### Testing to see if the target features are picked up correctly
        # print(X_new.head(n=2))
        # print("\n\n")

        # Make train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y['TOTAL_CONSUMPTION'], test_size=1/(5), shuffle=False)

        # Fit model
        pipe.fit(X_train, Y_train)

        # Predict
        Y_pred = pipe.predict(X_test)

        # Reshape Y_pred
        Y_pred = Y_pred.reshape(-1, 1)

        # Denormalize Y_pred
        Y_pred_denorm = scaler.inverse_transform(Y_pred)
        Y_pred_denorm_df = pd.DataFrame(Y_pred_denorm, columns=['TOTAL_CONSUMPTION'])
        
        # Evaluate Model - Always make sure to denormalize Y_pred and Y_test before evaluating
        mape = mean_absolute_percentage_error(Y_test_denorm_df, Y_pred_denorm_df)
        mae = mean_absolute_error(Y_test_denorm_df, Y_pred_denorm_df)
        r2 = r2_score(Y_test_denorm_df, Y_pred_denorm_df)
        mse = mean_squared_error(Y_test_denorm_df, Y_pred_denorm_df)
        rmse = root_mean_squared_error(Y_test_denorm_df, Y_pred_denorm_df)

        # Print name of added feature to base X dataframe and evaluation scores to STDOUT
        print(new_feature_name, ", ", mape, ", ", mae, ", ", r2, ", ", mse, ", ", rmse)
