# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Libraries
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests

from sklearn import preprocessing # Import preprocessing library for data normalization
import joblib # Import joblib to save and load models
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import xgboost as xgb # model 
from sklearn.model_selection import train_test_split

def save_xgb_model(X_df_XGB: pd.DataFrame, Y_df_XGB: pd.DataFrame, power_scaler, fsa, file_path, selected_features):
    
    # Split data into training and testing sets (80-20 split)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_df_XGB, 
        Y_df_XGB["TOTAL_CONSUMPTION"],
        test_size=730/(61320),
        shuffle=False  # Keep time series order
    )

    # Create and train the XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Train XGB model with optimized parameters
    xgb_model.fit(X_train, Y_train)

    # Save model
    file_path_model = os.path.join(file_path, "XGB_" + fsa + "_Model_" + "_".join(selected_features) + ".pkl")
    joblib.dump(xgb_model, file_path_model)
    
    # Make predictions
    Y_pred = xgb_model.predict(X_test)
    
    # Denomalize nomalized power data to check and see if matching with original data
    Y_pred_denormalized = Y_pred.copy()
    Y_pred_denormalized = power_scaler.inverse_transform(Y_pred_denormalized.reshape(-1, 1))
    
    # Denomalize nomalized power data to check and see if matching with original data
    Y_test_denormalized = Y_test.copy()
    Y_test_denormalized = power_scaler.inverse_transform(Y_test_denormalized.values.reshape(-1, 1))
    
    mape = mean_absolute_percentage_error(Y_test_denormalized, Y_pred_denormalized)
    mae = mean_absolute_error(Y_test_denormalized, Y_pred_denormalized)
    mse = mean_squared_error(Y_test_denormalized, Y_pred_denormalized)
    r2 = r2_score(Y_test_denormalized, Y_pred_denormalized)
    rmse = root_mean_squared_error(Y_test_denormalized, Y_pred_denormalized)
    
    # #%% Export metrix evaluation to csv
    metrix_evaluation = pd.DataFrame({
                                "Model": ["XGB"],
                                "MAPE (%)": [mape*100],
                                "MAE (MW)": [mae*0.001],
                                "r2": [r2],
                                "MSE (MW Squared)": [mse*0.001*0.001],
                                "RMSE (MW)" : [rmse*0.001],  
                                    })
    
    file_path_metrics = os.path.join(file_path, "XGB_" + fsa + "_Metrics_" + "_".join(selected_features) + ".csv")   
    metrix_evaluation.to_csv(file_path_metrics, index=False)


