# Feature Transformation Analysis, Top Helpful Features - This script is used to analyze effect of the top supposedly benifical features (one from each base feature, if they help) have on linear regression model forecasting

"""
    Results of Feature Analysis: All "helpful" features did not provide any significant benifit. 
    Features, MAPE, MAE, R^2, MSE, RMSE
    Base + Top "Helpful" Tranformations,  0.06933582666360207 ,  640.7476152766686 ,  0.8923422558201818 ,  744978.2425820982 ,  863.1212212557969
        0.01% decrease in MAPE compared to base features
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
pipe = make_pipeline(SplineTransformer(n_knots=6, degree=3, knots='quantile'), linear_model.Ridge(alpha=2.7755102040816326))

# Extract main + benifical transformed features from X dataframe
# Benificial Featurs : Temp (C) - Square,Temp (C) - Cube,Temp (C) - Sine,Temp (C) - Cosine,Temp (C) - Tangent,Temp (C) - Square Root,Temp (C) - Logarithm,Temp (C) - Inverse,Dew Point Temp (C) - Inverse,Wind Spd (km/h) - Square,Wind Spd (km/h) - Cube,Wind Spd (km/h) - Sine,Wind Spd (km/h) - Cosine,Wind Spd (km/h) - Tangent,Wind Spd (km/h) - Logarithm,Wind Spd (km/h) - Square Root,Wind Spd (km/h) - Inverse,WIND CHILL CALCULATION - Cube,WIND CHILL CALCULATION - Sine,WIND CHILL CALCULATION - Tangent,WIND CHILL CALCULATION - Inverse
X_main = X[['Year', 'Month', 'Day', 'Hour', 'WEEKEND', 'WEEKDAY', 'HOLIDAY', 'SEASON', 'Temp (C) - Original', 'Dew Point Temp (C) - Original', 'Rel Hum (%) - Original', 'Wind Spd (km/h) - Original', 'WIND CHILL CALCULATION - Original', 'Temp (C) - Cosine', 'Dew Point Temp (C) - Inverse', 'Wind Spd (km/h) - Square Root', 'WIND CHILL CALCULATION - Sine' ]]

### Testing to see if the main features are picked up correctly
#print(X_main.head(n = 2))

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
print("Base + Top \"Helpful\" Tranformations, ", mape, ", ", mae, ", ", r2, ", ", mse, ", ", rmse)