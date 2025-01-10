import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import models, layers, optimizers 


#%% Student directory
hanad_run = ["./data", 1]
clover_run = ["./data", 2]
joseph_laptop_run = ["C:/Users/sposa/Documents/GitHub/power-forecasting-capstone/data", 3]
joseph_pc_run = ["D:/Users/Joseph/Documents/GitHub/power-forecasting-capstone/data", 3]
janna_run = ["./data", 4]

###############################################################################
############### MAKE SURE TO CHANGE BEFORE RUNNING CODE #######################
###############################################################################
#Paste student name_run for whoever is running the code
run_student = janna_run
if (run_student[1] == joseph_pc_run[1]):
    print("JOSEPH IS RUNNING!")
elif (run_student[1] == hanad_run[1]):
    print("HANAD IS RUNNING!")
elif (run_student[1] == janna_run[1]):
    print("JANNA IS RUNNING!")
elif (run_student[1] == clover_run[1]):
    print("CLOVER IS RUNNING!")
else:
    print("ERROR!! NO ELIGIBLE STUDENT!")

#%% User input

#FSA - Forward Section Area (first 3 characters of Postal Code)
#L8K = Neighborhood in Hamilton (Link to FSA Map: https://www.prospectsinfluential.com/wp-content/uploads/2017/09/Interactive-Canadian-FSA-Map.pdf)
fsa_list = ['L9G']

#GUI INPUT
fsa_chosen = "L9G"

years = ['2018', '2019', '2020', '2021', '2022', '2023']
#years = ['2018']

'''
Jan - 01
Feb - 02
Mar - 03
Apr - 04
May - 05
Jun - 06
Jul - 07
Aug - 08
Sep - 09
Oct - 10
Nov - 11
Dec - 12
'''

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

#Input Files
dirs_inputs = run_student[0]

dirs_x_y_input = os.path.join(dirs_inputs, "X_Y_Inputs")

file_path_x = os.path.join(dirs_x_y_input, "X_df_"+fsa_chosen+".csv")
file_path_y = os.path.join(dirs_x_y_input, "Y_df_"+fsa_chosen+".csv")
X_df_cnn = pd.read_csv(file_path_x)
Y_df_cnn = pd.read_csv(file_path_y)

'''
#%%
#Pre-processing data into input for CNN
# Load the CSV file for features with the correct encoding
data = pd.read_csv('X_df.csv', encoding='utf-8')  # Use UTF-8 encoding

# Print the actual column names to diagnose the issue
print("Columns in DataFrame:", data.columns)

# Load the actual output data
Y_df = pd.read_csv('Y_df.csv')  # Replace with the actual filename for your output data
'''

# Assuming your features are in the following columns
feature_columns = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'Temp (C)', 'Dew Point Temp (C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'WIND CHILL CALCULATION']
data = X_df_cnn[feature_columns]  # Keep only the relevant feature columns

window_size = 168  # Last 168 hours (one week)
forecast_horizon = 24  # Next 24 hours

X_train = []
y_train = []

# Create input-output pairs using a sliding window
for i in range(len(data) - window_size - forecast_horizon + 1):
    X_train.append(data.iloc[i:i + window_size].values)  # Collect 168 hours of feature data
    # Collect the next 24 hours of output data from Y_df
    y_train.append(Y_df_cnn.iloc[i + window_size:i + window_size + forecast_horizon].values.flatten())  # Flatten to ensure it's 1D

# Convert to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=False, random_state=42)  # Split data

print("X_train shape:", X_train.shape)  # Should be (samples, 168, 9)
print("y_train shape:", y_train.shape)  # Should be (samples, 24)

# X_train is already in the required shape for CNN

# Define the number of timesteps
timesteps = 168  # Last 168 hours (one week)

# Define the number of features (columns) in your input data
features = X_train.shape[2]  # This will get the number of features from the shape of X_train

# Define the CNN model
model = models.Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
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
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2)
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

# Define the date for filtering
year_plot = 2023
month_plot = 1
day_plot = 1

X_test_dataframe_plot = pd.DataFrame(X_test[:, 0, 0:4], columns = ['YEAR', 'MONTH', 'DAY', 'HOUR'])


Y_pred_df = pd.DataFrame(Y_pred, columns=['TOTAL_CONSUMPTION'])
Y_test_df = pd.DataFrame(y_test[:,0], columns=['TOTAL_CONSUMPTION'])

Y_pred_df['YEAR'] = X_test_dataframe_plot['YEAR']
Y_pred_df['MONTH'] = X_test_dataframe_plot['MONTH']
Y_pred_df['DAY'] = X_test_dataframe_plot['DAY']
Y_pred_df['HOUR'] = X_test_dataframe_plot['HOUR']

Y_test_df['YEAR'] = X_test_dataframe_plot['YEAR']
Y_test_df['MONTH'] = X_test_dataframe_plot['MONTH']
Y_test_df['DAY'] = X_test_dataframe_plot['DAY']
Y_test_df['HOUR'] = X_test_dataframe_plot['HOUR']


X_test_year = X_test_dataframe_plot[X_test_dataframe_plot['YEAR'] == year_plot]
X_test_year_month = X_test_year[X_test_year['MONTH'] == month_plot]
X_test_year_month_day = X_test_year_month[X_test_year_month['DAY'] == day_plot]

Y_test_year = Y_test_df[Y_test_df['YEAR'] == year_plot]
Y_test_year_month = Y_test_year[Y_test_year['MONTH'] == month_plot]
Y_test_year_month_day = Y_test_year_month[Y_test_year_month['DAY'] == day_plot]

Y_pred_year = Y_pred_df[Y_pred_df['YEAR'] == year_plot]
Y_pred_year_month = Y_pred_year[Y_pred_year['MONTH'] == month_plot]
Y_pred_year_month_day = Y_pred_year_month[Y_pred_year_month['DAY'] == day_plot]

# Yearly Plot
plt.title(str(year_plot) + " Prediction VS Actual of CNN Model")

plt.plot(X_test_year['HOUR'].index, Y_pred_year['TOTAL_CONSUMPTION'], color="blue", linewidth=3)
plt.plot(X_test_year['HOUR'].index, Y_test_year['TOTAL_CONSUMPTION'], color="black")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()

plt.title(str(year_plot) + " Prediction VS Actual of CNN Model")

plt.plot(X_test_year['HOUR'].index, Y_test_year['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.plot(X_test_year['HOUR'].index, Y_pred_year['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.legend(loc="upper left")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()

# Monthly Plot
plt.title(str(year_plot) + ", Month = " + str(month_plot) + " Prediction VS Actual of CNN Model")

plt.plot(Y_test_year_month['HOUR'].index, Y_pred_year_month['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.plot(Y_test_year_month['HOUR'].index, Y_test_year_month['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.legend(loc="upper left")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()

# Daily Plot
plt.title(str(year_plot) + "/" + str(month_plot) + "/" + str(day_plot) + " Prediction VS Actual of KNN Model")

plt.plot(X_test_year_month_day['HOUR'].index, Y_pred_year_month_day['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.plot(X_test_year_month_day['HOUR'].index, Y_test_year_month_day['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.legend(loc="upper left")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()
