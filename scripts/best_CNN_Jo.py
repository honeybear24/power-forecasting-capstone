import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

from tensorflow import keras
from keras import models, layers, optimizers, callbacks

import datetime
from datetime import datetime, date, timedelta
import math
import canada_holiday
from pyhelpers.store import save_fig, save_svg_as_emf
import subprocess
import  aspose.cells 
from aspose.cells import Workbook

fsa_chosen = "L9G"
years = ['2018', '2019', '2020', '2021', '2022', '2023']
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

# Setting up paths for data loading
dirs_x_y_input = os.path.join("..", "data", "X_Y_Inputs")
file_path_x = os.path.join(dirs_x_y_input, "X_df_"+fsa_chosen+".csv")
file_path_y = os.path.join(dirs_x_y_input, "Y_df_"+fsa_chosen+".csv")

# Load the data
X_df_cnn = pd.read_csv(file_path_x)
Y_df_cnn = pd.read_csv(file_path_y)

# Drop unrequired columns
drop_list = [
    "Rel Hum (%)",
    "Wind Spd (km/h)",
    #"DATE",
]
feature_columns = X_df_cnn.columns.to_list()
for col in drop_list:
    feature_columns.remove(col)

# Create dataframe without hummus and spread
data = X_df_cnn[feature_columns]

window_size = 168  # Last 168 hours (one week)
forecast_horizon = 24  # Next 24 hours

X_data = []
y_data = []

# Create input-output pairs using a sliding window
for i in range(len(data) - window_size - forecast_horizon + 1):
    X_data.append(data.iloc[i:i + window_size + forecast_horizon].values)  # Collect 168 hours of feature data
    # Collect the next 24 hours of output data from Y_df
    y_data.append(Y_df_cnn.iloc[i + window_size:i + window_size + forecast_horizon].values.flatten())  # Flatten to ensure it's 1D

# Convert to numpy arrays
X_data, y_data = np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)
X_data = np.expand_dims(X_data, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=False, random_state=42)

# Better model
model = models.Sequential(
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
        # Flatten the layers to feed into dense network
        layers.Flatten(),
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
model.compile(optimizer=optimizer, loss=keras.losses.Huber(), metrics=["mae"])

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",  # Metric to monitor
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
)

model_dir = os.path.join("..", "models", "cnn_best_model.keras")
model.save(model_dir)

# Make predictions
model_dir = os.path.join("..", "models", "cnn_best_model.keras")
model = models.load_model(model_dir)

Y_pred = model.predict(X_test)  # Predict on X_test

# Convert predictions to a DataFrame
Y_pred_df = pd.DataFrame(Y_pred)


# Extract loss and MAE values from the history object
loss = history.history['loss']
mae = history.history['mae']
val_loss = history.history['val_loss']  # Validation loss
val_mae = history.history['val_mae']    # Validation MAE

# Plot the training and validation loss
plt.figure(figsize=(12, 6))

# Plot for loss
plt.subplot(1, 2, 1)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot for MAE
plt.subplot(1, 2, 2)
plt.plot(mae, label='Training MAE')
plt.plot(val_mae, label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

#Evaluate the model
print(f'##Model Evaluation##')
# X_test = np.expand_dims(X_test, axis=-1)
test_loss, test_mae = model.evaluate(np.expand_dims(X_test, axis=-1), y_test)  # Use X_test and y_test
print(f'Test Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# Function to calculate Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, Y_pred):
    y_true, Y_pred = np.array(y_true), np.array(Y_pred)
    return np.mean(np.abs((y_true - Y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, Y_pred)
print("MAPE = " + str(round(mape, 3)) + "%.")
