import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scikeras.wrappers import KerasRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)  # Split data

print("X_train shape:", X_train.shape)  # Should be (samples, 168, 9)
print("y_train shape:", y_train.shape)  # Should be (samples, 24)

# X_train is already in the required shape for CNN

# Define the number of timesteps
timesteps = 168  # Last 168 hours (one week)

# Define the number of features (columns) in your input data
features = X_train.shape[2]  # This will get the number of features from the shape of X_train

# Define the model creation function without filters
def create_model(dropout_rate=0.3, dense_units=50):
    model = Sequential()
    model.add(Dense(dense_units, activation='relu', input_shape=(timesteps, features)))  # Adjust input shape as needed
    model.add(Dropout(dropout_rate))
    
    # Add a single Dense layer
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Wrap the model using KerasRegressor
model = KerasRegressor(model=create_model, verbose=0)

# Define the parameter grid without filters
param_grid = {
    'dropout_rate': [0.2, 0.3, 0.5],
    'dense_units': [50, 100],  # Keep dense_units for tuning
    'epochs': [10, 20, 40],
    'batch_size': [16, 32, 64]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, random_state=42)

# Fit the model
random_search_result = random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best Parameters: ", random_search_result.best_params_)
print("Best Score: ", random_search_result.best_score_)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
validation_data = (X_train, y_train)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

#%%
#Make predictions
Y_pred = model.predict(X_test)  # Predict on X_test

# Convert predictions to a DataFrame
Y_pred_df = pd.DataFrame(Y_pred)

# Save to a CSV
Y_pred_df.to_csv('predictions_CNN.csv', index=False)

#%%
#Evaluate the model
print(f'##Model Evaluation##')
test_loss, test_mae = model.evaluate(X_test, y_test)  # Use X_test and y_test
print(f'Test Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# Function to calculate Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, Y_pred):
    y_true, Y_pred = np.array(y_true), np.array(Y_pred)
    return np.mean(np.abs((y_true - Y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, Y_pred)
print("MAPE = " + str(round(mape, 3)) + "%.")

# Plot predictions vs actual values
#plt.plot(predictions, label="Predictions")
#plt.plot(y_test, label="Actual Values")  # Use y_test for comparison
