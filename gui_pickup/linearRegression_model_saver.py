# Script to read in data from gui_pickup, train linear regression model, test linear regression model (will need to denormalize output),and save model to gui_pickup folder

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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import MinMaxScaler
import pickle
import joblib


# Import saved CSV into script as dataframes
home_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/"
df_directory = os.path.join(home_dir, "gui_pickup")
X = pd.read_csv(os.path.join(df_directory, "X_transformed_with_origCalVariables.csv"))
Y = pd.read_csv(os.path.join(df_directory, "Y_df_normalized.csv"))

# Create pipeline containing ridge regression and spline transformer
pipe = make_pipeline(SplineTransformer(n_knots=6, degree=3, knots='quantile'), linear_model.Ridge(alpha=2.7755102040816326))

# Make train test split for all data in X and Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y['TOTAL_CONSUMPTION'], test_size=1/(5), shuffle=False)

# Train model
pipe.fit(X_train, Y_train)

# Fit model
Y_pred = pipe.predict(X_test)

# Denormalize Y_pred and Y_test with min_max_scaler_y.pkl using joblib
scaler_path = os.path.join(df_directory, "min_max_scaler_y.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
scaler = joblib.load(scaler_path)

# Suggestion
# Ensure Y_pred and Y_test are reshaped correctly
Y_pred = Y_pred.reshape(-1, 1) # Reshape Y_pred to correct size for denormalization (-1 = infer number of rows, 1 = 1 column)
Y_test = Y_test.values.reshape(-1, 1)

Y_pred_denorm = scaler.inverse_transform(Y_pred)
Y_pred_denorm_df = pd.DataFrame(Y_pred_denorm, columns=['TOTAL_CONSUMPTION'])
Y_test_denorm = scaler.inverse_transform(Y_test)
Y_test_denorm_df = pd.DataFrame(Y_test_denorm, columns=['TOTAL_CONSUMPTION'])


# Original
# Y_pred_denorm = scaler.inverse_transform(Y_pred[['TOTAL_CONSUMPTION']])
# Y_pred_denorm_df = pd.DataFrame(Y_pred_denorm, columns=['TOTAL_CONSUMPTION'], index = Y_test.index)
# Y_test_denorm = scaler.inverse_transform(Y_test)
# Y_test_denorm_df = pd.DataFrame(Y_test_denorm, columns=['TOTAL_CONSUMPTION'])

# Evaluate Model    
mape = mean_absolute_percentage_error(Y_test_denorm_df, Y_pred_denorm_df)
mae = mean_absolute_error(Y_test_denorm_df, Y_pred_denorm_df)
r2 = r2_score(Y_test_denorm_df, Y_pred_denorm_df)
mse = mean_squared_error(Y_test_denorm_df, Y_pred_denorm_df)
rmse = root_mean_squared_error(Y_test_denorm_df, Y_pred_denorm_df)

# Print out result to STDOUT - Before Saving Model
print("Before Saving Model")
print(f"MAPE: {mape}, MAE: {mae}, R^2: {r2}, MSE: {mse}, RMSE: {rmse}")

# Save model to gui_pickup folder using joblib
joblib.dump(pipe, os.path.join(df_directory, "ridge_regression_model.pkl"))

# Load model from gui_pickup folder using joblib
pipe_saved = joblib.load(os.path.join(df_directory, "ridge_regression_model.pkl"))

# Predict using loaded model
Y_pred_saved = pipe_saved.predict(X_test)

# Ensure Y_pred and Y_test are reshaped correctly
Y_pred_saved = Y_pred_saved.reshape(-1, 1)
#Y_test = Y_test.values.reshape(-1, 1) already reshaped to correct size for denormalization

# Denormalize Y_pred_saved and Y_test with min_max_scaler.pkl
Y_pred_denorm_saved = scaler.inverse_transform(Y_pred_saved)
Y_pred_denorm_saved_df = pd.DataFrame(Y_pred_denorm_saved, columns=['TOTAL_CONSUMPTION'])
Y_test_denorm_saved = scaler.inverse_transform(Y_test)
Y_test_denorm_saved_df = pd.DataFrame(Y_test_denorm_saved, columns=['TOTAL_CONSUMPTION'])

# Evaluate Model    
mape_saved = mean_absolute_percentage_error(Y_test_denorm_saved_df, Y_pred_denorm_saved_df)
mae_saved = mean_absolute_error(Y_test_denorm_saved_df, Y_pred_denorm_saved_df)
r2_saved = r2_score(Y_test_denorm_saved_df, Y_pred_denorm_saved_df)
mse_saved = mean_squared_error(Y_test_denorm_saved_df, Y_pred_denorm_saved_df)
rmse_saved = root_mean_squared_error(Y_test_denorm_saved_df, Y_pred_denorm_saved_df)

# Print out result to STDOUT - After Saving Model
print("After Saving Model")
print(f"MAPE: {mape_saved}, MAE: {mae_saved}, R^2: {r2_saved}, MSE: {mse_saved}, RMSE: {rmse_saved}")