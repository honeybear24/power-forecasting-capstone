# Ridge Regression model with splines (using optimized values)

# Set Up
import os
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import SplineTransformer
import joblib

def save_lr_model(X: pd.DataFrame, Y: pd.DataFrame, power_scaler, fsa, file_path):
    # Create pipeline containing linear regression model and standard scalar
    pipe = make_pipeline(SplineTransformer(n_knots=6, degree=3, knots='quantile'), linear_model.Ridge(alpha=2.7755102040816326))
    
    # Train model - Extract only January from X dataframe
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y['TOTAL_CONSUMPTION'], test_size=1/(5), shuffle=False)
    pipe.fit(X_train, Y_train)
    
    # Save model
    file_path_model = os.path.join(file_path, "LR_" + fsa + "_Model.pkl")
    joblib.dump(pipe, file_path_model)
    
    # Fit model
    Y_pred = pipe.predict(X_test)
    
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
                                "Model": ["LR"],
                                "MAPE (%)": [mape*100],
                                "MAE (MW)": [mae*0.001],
                                "r2": [r2],
                                "MSE (MW Squared)": [mse*0.001*0.001],
                                "RMSE (MW)" : [rmse*0.001],  
                                    })
    
    file_path_metrics = os.path.join(file_path, "LR_" + fsa + "_Metrics.csv")   
    metrix_evaluation.to_csv(file_path_metrics, index=False)