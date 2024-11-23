import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import models, layers, optimizers

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# %%
# Pre-processing data into input for CNN
# Load the CSV file for features with the correct encoding
data = pd.read_csv("X_df.csv", encoding="utf-8")  # Use UTF-8 encoding

# Print the actual column names to diagnose the issue
print("Columns in DataFrame:", data.columns)

# Load the actual output data
Y_df = pd.read_csv("Y_df.csv")  # Replace with the actual filename for your output data

# Assuming your features are in the following columns
feature_columns = [
    "YEAR",
    "MONTH",
    "DAY",
    "HOUR",
    "Temp (Â°C)",
    "Dew Point Temp (Â°C)",
    "Rel Hum (%)",
    "Wind Spd (km/h)",
    "WIND CHILL CALCULATION",
]
data = data[feature_columns]  # Keep only the relevant feature columns

window_size = 168  # Last 168 hours (one week)
forecast_horizon = 24  # Next 24 hours

X_train = []
y_train = []

# Create input-output pairs using a sliding window
for i in range(len(data) - window_size - forecast_horizon + 1):
    X_train.append(
        data.iloc[i : i + window_size].values
    )  # Collect 168 hours of feature data
    # Collect the next 24 hours of output data from Y_df
    y_train.append(
        Y_df.iloc[i + window_size : i + window_size + forecast_horizon].values.flatten()
    )  # Flatten to ensure it's 1D

# Convert to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=False, random_state=42
)  # Split data

print("X_train shape:", X_train.shape)  # Should be (samples, 168, 9)
print("y_train shape:", y_train.shape)  # Should be (samples, 24)

# X_train is already in the required shape for CNN

# Define the number of timesteps
timesteps = 168  # Last 168 hours (one week)

# Define the number of features (columns) in your input data
features = X_train.shape[
    2
]  # This will get the number of features from the shape of X_train

# Add additional dimension to X_train to make it possible to use Conv2D layers
X_train = np.expand_dims(X_train, axis=-1)

# Better model
model = models.Sequential(
    [
        layers.Input(shape=X_train.shape[-3:]),
        # Convolutional layers 1-3
        layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding="same", activation="relu"
        ),
        layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding="same", activation="relu"
        ),
        layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding="same", activation="relu"
        ),
        layers.MaxPool2D(
            pool_size=(1, 2)
        ),  # Keep the width the same size, pool the height
        # Convolutional layers 4-6
        layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu"
        ),
        layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu"
        ),
        layers.Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu"
        ),
        layers.MaxPool2D(pool_size=(1, 2)),
        # Convolutional layers 7-9
        layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        ),
        layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        ),
        layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        ),
        layers.MaxPool2D(pool_size=(1, 2)),
        # Flatten the layers to feed into dense network
        layers.Flatten(),
        # Add dense layers
        layers.Dense(128, activation="relu"),
        layers.Dense(24),  # This is the output layer, we are predicting 24 hours!!!!!
    ]
)


# Define the CNN model
# model = Sequential()
# model.add(
#     Conv1D(
#         filters=64, kernel_size=3, activation="relu", input_shape=(timesteps, features)
#     )
# )
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32, kernel_size=3, activation="relu"))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32, kernel_size=3, activation="relu"))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32, kernel_size=3, activation="relu"))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(50, activation="relu"))
# model.add(Dropout(0.3))
# model.add(Dense(1))

# Customizing the Adam optimizer
optimizer = optimizers.Adam(
    # learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07
)

# Compile the model
model.compile(optimizer=optimizer, loss=keras.losses.Huber(), metrics=["mae"])

# Train the model
history = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)

# %%
# Make predictions
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