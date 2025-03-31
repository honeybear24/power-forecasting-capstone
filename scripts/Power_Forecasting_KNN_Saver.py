# Libraries
import pandas as pd
import datetime
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def save_knn_model(X_df_knn: pd.DataFrame, Y_df_knn: pd.DataFrame, power_scaler, fsa, file_path, selected_features):
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_df_knn, Y_df_knn['TOTAL_CONSUMPTION'], test_size=730/(61320), shuffle = False)
    
    # TRY GRID SEARCH
    pipeline_knn = Pipeline([("model", KNeighborsRegressor(n_neighbors=1, weights = 'distance'))])
    
    knn_model = GridSearchCV(estimator = pipeline_knn,
                             scoring = ['neg_mean_absolute_percentage_error'],
                             param_grid = {'model__n_neighbors': range(10,30)},
                             refit = 'neg_mean_absolute_percentage_error', 
                             cv = 10)
    
    knn_model.fit(X_train, Y_train)
    
    # Save model
    file_path_model = os.path.join(file_path, "KNN_" + fsa + "_Model_" + "_".join(selected_features) + ".pkl")
    joblib.dump(knn_model, file_path_model)

    # Get metric evaluation
    Y_pred = knn_model.predict(X_test)
    
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
                                "Model": ["KNN"],
                                "MAPE (%)": [mape*100],
                                "MAE (MW)": [mae*0.001],
                                "r2": [r2],
                                "MSE (MW Squared)": [mse*0.001*0.001],
                                "RMSE (MW)" : [rmse*0.001],  
                                    })
    
    file_path_metrics = os.path.join(file_path, "KNN_" + fsa + "_Metrics_" + "_".join(selected_features) + ".csv")   
    metrix_evaluation.to_csv(file_path_metrics, index=False)












