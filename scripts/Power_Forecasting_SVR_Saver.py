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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib

def save_svr_model(X_df_SVR: pd.DataFrame, Y_df_SVR: pd.DataFrame, power_scaler, fsa, file_path):
    
    # Split data into training and testing sets (80-20 split)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_df_SVR, 
        Y_df_SVR["TOTAL_CONSUMPTION"],
        test_size=730/(61320),
        shuffle=False  # Keep time series order
    )

    #Implement grid search     
    param_grid = {
        'kernel': ['rbf'],      # Test different kernels
        'C': [0.1, 1],             # Regularization parameter
        'gamma': [0.001, 0.01, 0.1, 1, 'scale']   # Kernel coefficient for rbf and poly
    }
    
    #set my SVR model 
    svr = SVR()
    
    #get the best parameters 
    
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=10, scoring='neg_mean_absolute_percentage_error')
    grid_search.fit(X_train, Y_train)
    
    best_params = grid_search.best_params_
    best_svr = grid_search.best_estimator_
    
    print("Best Parameters:", best_params)
    
    # Train SVR model with optimized parameters
    best_svr.fit(X_train, Y_train)

    # Save model
    file_path_model = os.path.join(file_path, "SVR_" + fsa + "_Model.pkl")
    joblib.dump(best_svr, file_path_model)
    
    # Make predictions
    Y_pred = best_svr.predict(X_test)
    
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
                                "Model": ["SVR"],
                                "MAPE (%)": [mape*100],
                                "MAE (MW)": [mae*0.001],
                                "r2": [r2],
                                "MSE (MW Squared)": [mse*0.001*0.001],
                                "RMSE (MW)" : [rmse*0.001],  
                                    })
    
    file_path_metrics = os.path.join(file_path, "SVR_" + fsa + "_Metrics.csv")   
    metrix_evaluation.to_csv(file_path_metrics, index=False)


