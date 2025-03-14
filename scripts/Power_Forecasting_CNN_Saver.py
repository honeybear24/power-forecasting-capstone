
#%% Libraries
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

def save_cnn_model(X_df_CNN: pd.DataFrame, Y_df_CNN: pd.DataFrame, power_scaler, fsa, file_path):

    Y_df_CNN = Y_df_CNN['TOTAL_CONSUMPTION']
    
    for feature in X_df_CNN.columns:
      if (feature == "Weekend" or feature == "Season" or feature == "Holiday" or ("Year_" in feature) or ("Month_" in feature) or ("Day_" in feature) or ("Hour_" in feature)):
        X_df_CNN[feature].astype('bool')
      if ("Lag" in feature):
        X_df_CNN = X_df_CNN.drop(columns = [feature])
    
    # Create dataframe without humidity and speed
    data = X_df_CNN
    
    window_size = 168  # Last 168 hours (one week)
    forecast_horizon = 24  # Next 24 hours
    
    X_data = []
    y_data = []
    
    # Create input-output pairs using a sliding window
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X_data.append(data.iloc[i:i + window_size + forecast_horizon].values)  # Collect 168 hours of feature data
        # Collect the next 24 hours of output data from Y_df
        y_data.append(Y_df_CNN.iloc[i + window_size:i + window_size + forecast_horizon].values.flatten())  # Flatten to ensure it's 1D
    
    # Convert to numpy arrays
    X_data, y_data = np.array(X_data, dtype=np.float16), np.array(y_data, dtype=np.float16)
    X_data = np.expand_dims(X_data, axis=-1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=730/(61320), shuffle=False, random_state=42)
    
    gc.collect()
  
    # Better model
    cnn_model = models.Sequential(
        [
            layers.Input(shape=X_train.shape[-3:]),
            layers.Conv2D(
                filters=32, kernel_size=(3, 3), padding="same", activation="relu"
            ),
            layers.MaxPool2D(
                pool_size=(1, 2)
            ),  # Keep the width the same size, pool the height
            layers.Conv2D(
                filters=64, kernel_size=(3, 3), padding="same", activation="relu"
            ),
            layers.MaxPool2D(pool_size=(1, 2)),
            layers.Conv2D(
                filters=128, kernel_size=(3, 3), padding="same", activation="relu"
            ),
            layers.MaxPool2D(pool_size=(1, 2)),
            # this is added
            layers.Conv2D(
                filters=128, kernel_size=(3, 3), padding="same", activation="relu"
            ),
            layers.MaxPool2D(pool_size=(1, 2)),
            layers.Conv2D(
                filters=128, kernel_size=(3, 3), padding="same", activation="relu"
            ),
            layers.Conv2D(
                filters=128, kernel_size=(3, 3), padding="same", activation="relu"
            ),
            # Flatten the layers to feed into dense network
            layers.Flatten(),
            layers.BatchNormalization(),
            # Add dense layers
            layers.Dense(128, activation="relu"),
            layers.Dense(
                24
            ),  # This is the output layer, we are predicting 24 hours!!!
        ]
    )
    
    # Customizing the Adam optimizer
    optimizer = optimizers.Adam(
        # learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07
    )
    
    # Compile the model
    cnn_model.compile(optimizer=optimizer, loss=keras.losses.Huber(), metrics=["mae"])
    
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
    )
    
    # Train the model
    history = cnn_model.fit(
        X_train,
        Y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
    )

    
    # Save model
    file_path_model = os.path.join(file_path, "CNN_" + fsa + "_Model.keras")
    cnn_model.save(file_path_model)
    
    # Get metric evaluation
    Y_pred = cnn_model.predict(X_test)

    # Denomalize nomalized power data to check and see if matching with original data
    Y_pred_denormalized = pd.DataFrame(Y_pred.copy())
    Y_pred_denormalized = power_scaler.inverse_transform(Y_pred_denormalized.values.reshape(-1, 1))
    
    # Denomalize nomalized power data to check and see if matching with original data
    Y_test_denormalized = pd.DataFrame(Y_test.copy())
    Y_test_denormalized = power_scaler.inverse_transform(Y_test_denormalized.values.reshape(-1, 1))
    
    mape = mean_absolute_percentage_error(Y_test_denormalized, Y_pred_denormalized)
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
    
    file_path_metrics = os.path.join(file_path, "CNN_" + fsa + "_Metrics.csv")   
    metrix_evaluation.to_csv(file_path_metrics, index=False)


