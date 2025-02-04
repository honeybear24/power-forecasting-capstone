# Rolling Stats Analysis - Studying impact of that lags to weather variables have on forecasting accuracy of linear regression model

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

# Set up file paths and import Y data frame
data_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/data/"
anal_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/rolling_stats_analysis"
#X = pd.read_csv(os.path.join(df_directory, "weather_data_L9G_20180101_20231231_lagAnal_lag0.csv"))
#Y = pd.read_csv(os.path.join(df_directory, "power_data_L9G_20180101_20231231_lagAnal_lag0.csv"))

# Create pipeline containing ridge regression and spline transformer    
pipe = make_pipeline(SplineTransformer(n_knots=6, degree=3, knots='quantile'), linear_model.Ridge(alpha=2.7755102040816326))

# Print out header to CSV to STDOUT
print("Window Size, MAPE, MAE, R^2, MSE, RMSE")

# Choose FSA
fsa = "L9G"

# Cycle through the number of lags we want to look at
for i in range(0,25):

    # Skip i = 1 because it is has missing data
    if i == 1:
        continue

    # Import X data frame that has desired number of lags
    X = pd.read_csv(os.path.join(anal_dir, f'{anal_dir}/weather_{fsa}_{i}.csv')).drop(["Date/Time (LST)"],axis=1)
    Y = pd.read_csv(os.path.join(anal_dir, f'{anal_dir}/power_{fsa}_{i}.csv'))

    # Make train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y['TOTAL_CONSUMPTION'], test_size=1/(5), shuffle=False)

    # Fit model
    pipe.fit(X_train, Y_train)

    # Predict
    Y_pred = pipe.predict(X_test)

    # Turn Y_pred and Y_test into dataframes
    Y_pred_df = pd.DataFrame(Y_pred, columns=['TOTAL_CONSUMPTION'])
    Y_test_df = pd.DataFrame(Y_test, columns=['TOTAL_CONSUMPTION'])

    # Evaluate Model - Always make sure to denormalize Y_pred and Y_test before evaluating
    mape = mean_absolute_percentage_error(Y_test_df, Y_pred_df)
    mae = mean_absolute_error(Y_test_df, Y_pred_df)
    r2 = r2_score(Y_test_df, Y_pred_df)
    mse = mean_squared_error(Y_test_df, Y_pred_df)
    rmse = root_mean_squared_error(Y_test_df, Y_pred_df)

    # Print to STDOUT
    print("" , i , ", ", mape, ", ", mae, ", ", r2, ", ", mse, ", ", rmse)