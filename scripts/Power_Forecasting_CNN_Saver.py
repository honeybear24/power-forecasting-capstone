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

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from keras import models, layers, optimizers, callbacks
import gc

def save_cnn_model(X_df_CNN: pd.DataFrame, Y_df_CNN: pd.DataFrame, power_scaler, fsa, file_path, selected_features, cnn_param_file_path):
    # Read in CNN parameters excel file
    file_path_cnn_params = os.path.join(cnn_param_file_path, "CNN_Hyperparameters.xlsx")   
    cnn_params = pd.read_excel(file_path_cnn_params, sheet_name="User_Input", header = 0, usecols = "A:F")
    
    # Convert all parameters to desired type
    epochs_param = cnn_params["Epochs"].iloc[0]
    epochs_param = epochs_param.astype(int)
    
    iterations_param = cnn_params["Iterations"].iloc[0]
    iterations_param = iterations_param.astype(int)
    
    batch_param = cnn_params["Batch Size"].iloc[0]
    batch_param = batch_param.astype(int)
    
    activation_param = cnn_params["Activation Function"].iloc[0]
    activation_param = str(activation_param)
    
    loss_param = cnn_params["Loss Function"].iloc[0]
    loss_param = str(loss_param)
    
    layer_param = cnn_params["Hidden Layers"].iloc[0]
    layer_param = layer_param.astype(int)
    
    # Choose loss function based on excel input
    if loss_param == "huber":
        loss_object_param = keras.losses.Huber()
    if loss_param == "mean absolute error":
        loss_object_param = keras.losses.MeanAbsoluteError()
    if loss_param == "mean absolute percentage error":
        loss_object_param = keras.losses.MeanAbsolutePercentageError()
    if loss_param == "mean squared error":
        loss_object_param = keras.losses.MeanSquaredError()
    if loss_param == "mean squared logarithmic error":
        loss_object_param = keras.losses.MeanSquaredLogarithmicError()
    
    # Preprocesing data for CNN
    Y_df_CNN = Y_df_CNN['TOTAL_CONSUMPTION']
    
    for feature in X_df_CNN.columns:
      if (feature == "Weekend" or feature == "Season" or feature == "Holiday" or ("Year_" in feature) or ("Month_" in feature) or ("Day_" in feature) or ("Hour_" in feature)):
        X_df_CNN[feature].astype('bool')
      if ("Lag" in feature):
        X_df_CNN = X_df_CNN.drop(columns = [feature])
    
    # Create dataframe without humidity and speed
    data = X_df_CNN
    
    #window_size = 168  # Last 168 hours (one week)
    window_size = 48  # Last 24 hours (one day)
    forecast_horizon = 24  # Next 24 hours
    
    X_data = []
    y_data = []
    
    # Create input-output pairs using a sliding window
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X_data.append(data.iloc[i:i + window_size + forecast_horizon].values)  # Collect 168 hours of feature data
        # Collect the next 24 hours of output data from Y_df
        y_data.append(Y_df_CNN.iloc[i + window_size:i + window_size + forecast_horizon].values.flatten())  # Flatten to ensure it's 1D
    
    # Convert to numpy arrays
    X_data, y_data = np.array(X_data, dtype=np.float64), np.array(y_data, dtype=np.float64)
    X_data = np.expand_dims(X_data, axis=-1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=730/(61320), shuffle=False, random_state=42)
    
    gc.collect()
    mape_list = []
    
    # List for sequential
    sequential_list = []
    sequential_list.append(layers.Input(shape=X_train.shape[-3:]))
    sequential_list.append(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation=activation_param))
    # Keep the width the same size, pool the height
    sequential_list.append(layers.MaxPool2D(pool_size=(1, 2))) 
    sequential_list.append(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activation_param))
    sequential_list.append(layers.MaxPool2D(pool_size=(1, 2)))
    
    for i in range(layer_param-2):
        sequential_list.append(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activation_param))
        if i<4:
            sequential_list.append(layers.MaxPool2D(pool_size=(1, 2)))
    
    # Flatten the layers to feed into dense network
    sequential_list.append(layers.Flatten()) 
    
    # Add dense layers
    sequential_list.append(layers.Dense(128, activation=activation_param))
    sequential_list.append(layers.Dense(24))
    
    for i in range(iterations_param):
        # Print what iteration is being conducted
        print("Iteration " + str(i+1) + "/" + str(iterations_param))
        
        # Define CNN model
        cnn_model = models.Sequential(sequential_list)
        
        # Customizing the Adam optimizer
        optimizer = optimizers.Adam(
            # learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07
        )
        
        # Compile the model
        cnn_model.compile(optimizer=optimizer, loss=loss_object_param, metrics=["mae"])
        
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            patience=8,  # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
        )
        
        # Train the model
        history = cnn_model.fit(
            X_train,
            Y_train,
            epochs = epochs_param,
            batch_size = batch_param,
            validation_split=0.2,
            callbacks=[early_stopping],
        )
        
        # Get metric evaluation
        Y_pred_temp = cnn_model.predict(X_test)
        
        # Denomalize nomalized power data to check and see if matching with original data
        Y_pred_denormalized_temp = pd.DataFrame(Y_pred_temp.copy())
        Y_pred_denormalized_temp = power_scaler.inverse_transform(Y_pred_denormalized_temp.values.reshape(-1, 1))
        
        # Denomalize nomalized power data to check and see if matching with original data
        Y_test_denormalized_temp = pd.DataFrame(Y_test.copy())
        Y_test_denormalized_temp = power_scaler.inverse_transform(Y_test_denormalized_temp.values.reshape(-1, 1))

        mape = mean_absolute_percentage_error(Y_test_denormalized_temp, Y_pred_denormalized_temp)
        mape_list.append(mape*100)
        
        if (i == 0):
            smallest_mape = mape
            
            # Save model
            file_path_model = os.path.join(file_path, "CNN_" + fsa + "_Model_" + "_".join(selected_features) + ".keras")
            cnn_model.save(file_path_model)
            
            Y_test_denormalized = Y_test_denormalized_temp
            Y_pred_denormalized = Y_pred_denormalized_temp
        elif (mape < smallest_mape):
            smallest_mape = mape
            
            # Save model
            file_path_model = os.path.join(file_path, "CNN_" + fsa + "_Model_" + "_".join(selected_features) + ".keras")
            cnn_model.save(file_path_model)
            
            Y_test_denormalized = Y_test_denormalized_temp
            Y_pred_denormalized = Y_pred_denormalized_temp
            
        keras.backend.clear_session()
        print("\n")
    
    mape = smallest_mape
    mape_list_df = pd.DataFrame(mape_list)
    mape_list_df = mape_list_df.rename(columns={0: 'MAPE (%)'})
    
    mae = mean_absolute_error(Y_test_denormalized, Y_pred_denormalized)
    mse = mean_squared_error(Y_test_denormalized, Y_pred_denormalized)
    r2 = r2_score(Y_test_denormalized, Y_pred_denormalized)
    rmse = root_mean_squared_error(Y_test_denormalized, Y_pred_denormalized)
    
    # #%% Export metrix evaluation to csv
    metrix_evaluation = pd.DataFrame({
                                "Model": ["CNN"],
                                "MAPE (%)": [mape*100],
                                "MAE (MW)": [mae*0.001],
                                "r2": [r2],
                                "MSE (MW Squared)": [mse*0.001*0.001],
                                "RMSE (MW)" : [rmse*0.001],  
                                    })
    
    file_path_metrics = os.path.join(file_path, "CNN_" + fsa + "_Metrics_" + "_".join(selected_features) + ".csv")   
    metrix_evaluation.to_csv(file_path_metrics, index=False)
    
    file_path_mape = os.path.join(file_path, "CNN_" + fsa + "_MAPE_Tracker_" + "_".join(selected_features) + ".csv")   
    mape_list_df.to_csv(file_path_mape, index=False)
    
    


