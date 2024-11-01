import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

#%%
#Pre-processing data into input for CNN
# Load the CSV file for features with the correct encoding
data = pd.read_csv('X_df.csv', encoding='utf-8')  # Use UTF-8 encoding

# Print the actual column names to diagnose the issue
print("Columns in DataFrame:", data.columns)

# Load the actual output data
Y_df = pd.read_csv('Y_df.csv')  # Replace with the actual filename for your output data

# Assuming your features are in the following columns
feature_columns = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'Temp (Â°C)', 'Dew Point Temp (Â°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'WIND CHILL CALCULATION']
data = data[feature_columns]  # Keep only the relevant feature columns

window_size = 168  # Last 168 hours (one week)
forecast_horizon = 24  # Next 24 hours

X_train = []
y_train = []

# Create input-output pairs using a sliding window
for i in range(len(data) - window_size - forecast_horizon + 1):
    X_train.append(data.iloc[i:i + window_size].values)  # Collect 168 hours of feature data
    # Collect the next 24 hours of output data from Y_df
    y_train.append(Y_df.iloc[i + window_size:i + window_size + forecast_horizon].values.flatten())  # Flatten to ensure it's 1D

# Convert to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  # Split data

print("X_train shape:", X_train.shape)  # Should be (samples, 168, 9)
print("y_train shape:", y_train.shape)  # Should be (samples, 24)

# X_train is already in the required shape for CNN

# Define the number of timesteps
timesteps = 168  # Last 168 hours (one week)

# Define the number of features (columns) in your input data
features = X_train.shape[2]  # This will get the number of features from the shape of X_train

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

# Customizing the Adam optimizer
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# Compile the model
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
validation_data = (X_train, y_train)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

#%%
#Evaluate the model
print(f'##Model Evaluation##')
test_loss, test_mae = model.evaluate(X_test, y_test)  # Use X_test and y_test
print(f'Test Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')

#%%
#Make predictions
predictions = model.predict(X_test)  # Predict on X_test

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions)

# Save to a CSV
predictions_df.to_csv('predictions_CNN.csv', index=False)

# Plot predictions vs actual values
#plt.plot(predictions, label="Predictions")
#plt.plot(y_test, label="Actual Values")  # Use y_test for comparison


# Select the first 24 hours for plotting
num_hours_to_plot = 24  # Number of hours to plot (1 day)
predictions_to_plot = predictions[:num_hours_to_plot]
actual_values_to_plot = y_test[:num_hours_to_plot]

# Plot predictions vs actual values for 1 day
plt.figure(figsize=(12, 6))
plt.plot(predictions_to_plot, label="Predictions", color='blue')
plt.plot(actual_values_to_plot, label="Actual Values", color='orange')
plt.title("Predictions vs Actual Values for 1 Day")
plt.xlabel("Hours")
plt.ylabel("Values")
plt.legend()
plt.show()

# Assuming you want to predict for a specific day, for example, the last day in your dataset
# Extract the last 24 hours of data for prediction
single_day_data = data.iloc[-window_size:]  # Get the last week of data for input
single_day_output = Y_df.iloc[-forecast_horizon:]  # Get the actual output for the next 24 hours

# Prepare the input for the model
X_single_day = single_day_data.values.reshape(1, window_size, -1)  # Reshape for the model input

# Make predictions for that single day
predictions_single_day = model.predict(X_single_day)  # Predict on the reshaped input

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions_single_day)

# Save to a CSV (optional)
predictions_df.to_csv('predictions_single_day.csv', index=False)

# Extract the actual values for the specific day (the next 24 hours)
actual_values_to_plot = single_day_output.values.flatten()  # Flatten to 1D for plotting

# Flatten predictions for plotting
predictions_to_plot = predictions_single_day.flatten()  # Flatten to 1D for plotting

# Plot predictions vs actual values for that single day
plt.figure(figsize=(12, 6))
plt.plot(predictions_to_plot, label="Predictions", color='blue')  # Predictions for the day
plt.plot(actual_values_to_plot, label="Actual Values", color='orange')  # Actual values for the day
plt.title("Predictions vs Actual Values for a Single Day")
plt.xlabel("Hours")
plt.ylabel("Values")
plt.xticks(ticks=np.arange(24), labels=np.arange(1, 25))  # Label x-axis from 1 to 24 hours
plt.legend()
plt.show()