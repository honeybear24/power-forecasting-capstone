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

#%% Student directory
hanad_run = ["./data", 1]
clover_run = ["./data", 2]
joseph_laptop_run = ["C:\\Users\\sposa\\Documents\\GitHub\\power-forecasting-capstone\\data", 3]
joseph_pc_run = ["D:\\Users\\Joseph\\Documents\\GitHub\\power-forecasting-capstone\\data", 3]
janna_run = ["./data", 4]

###############################################################################
############### MAKE SURE TO CHANGE BEFORE RUNNING CODE #######################
###############################################################################
# Paste student name_run for whoever is running the code
run_student = clover_run
if (run_student[1] == joseph_laptop_run[1]):
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

# FSA - Forward Section Area (first 3 characters of Postal Code)
#       L8K = Neighborhood in Hamilton (Link to FSA Map: https://www.prospectsinfluential.com/wp-content/uploads/2017/09/Interactive-Canadian-FSA-Map.pdf)
fsa_list = ['L9G']

# GUI INPUT
fsa_chosen = "L9G"

years = ['2018', '2019', '2020', '2021', '2022', '2023']
#years = ['2018']

# Jan - 01
# Feb - 02
# Mar - 03
# Apr - 04
# May - 05
# Jun - 06
# Jul - 07
# Aug - 08
# Sep - 09
# Oct - 10
# Nov - 11
# Dec - 12
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


# SVR Model
#these are the libraries needed for the SVR model
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

#Input files 
dirsinputs = run_student[0]

dirs_x_y_input = os.path.join(dirs_inputs, "X_Y_Inputs")

file_path_x = os.path.join(dirs_x_y_input, "X_df"+fsachosen+".csv")
file_path_y = os.path.join(dirs_x_y_input, "Y_df"+fsa_chosen+".csv")
X_df_SVR = pd.read_csv(file_path_x)
Y_df_SVR = pd.read_csv(file_path_y)




X_df_SVR = X_df_SVR.drop (['DATE','WEEKDAY',], axi = 1)




# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_df_SVR, 
    Y_df_SVR,
    test_size=0.2,
    shuffle=False  # Keep time series order
)

# Scale the features specifically for the SVR model
sc_X = StandardScaler()
sc_y = StandardScaler()

# Scale the training and testing data for the SVR model
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = sc_y.transform(y_test.values.reshape(-1, 1)).ravel()

#Implement grid search 

param_grid = {
    'kernel': ['rbf'],      # Test different kernels
    'C': [0.1, 1, 10, 100, 1000],             # Regularization parameter
    'gamma': [0.001, 0.01, 0.1, 1, 'scale']   # Kernel coefficient for rbf and poly
}

#set my SVR model 
svr = SVR()

#get the best parameters 

grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_percentage_error')
grid_search.fit(X_train_scaled, y_train_scaled)

best_params = grid_search.best_params_
best_svr = grid_search.best_estimator_

print("Best Parameters:", best_params)

# Train SVR model with optimized parameters

best_svr.fit(X_train_scaled, y_train_scaled)
#regressor = SVR(kernel='poly')
#regressor.fit(X_train_scaled, y_train_scaled.ravel())

# Make predictions
y_train_pred = sc_y.inverse_transform(best_svr.predict(X_train_scaled).reshape(-1, 1))
y_test_pred = sc_y.inverse_transform(best_svr.predict(X_test_scaled).reshape(-1, 1))

#y_train_pred = sc_y.inverse_transform(regressor.predict(X_train_scaled).reshape(-1, 1))
#y_test_pred = sc_y.inverse_transform(regressor.predict(X_test_scaled).reshape(-1, 1))

X_test.to_csv('X_df.csv', index=False)
y_test.to_csv('Y_df.csv', index=False) 


# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Training R²: {train_r2:.2f}")
print(f"Test R²: {test_r2:.2f}")

mape = mean_absolute_percentage_error(y_test, y_test_pred)
print("MAPE = " + str(round(mape*100, 3)) + "%.")

## PLOTTING
year_plot = 2023
month_plot = 1
day_plot = 1



Y_pred_df = pd.DataFrame(y_test_pred, columns=['TOTAL_CONSUMPTION'], index = y_test.index)
Y_test_df = pd.DataFrame(y_test, columns=['TOTAL_CONSUMPTION'])

Y_pred_df['YEAR'] = X_test['YEAR']
Y_pred_df['MONTH'] = X_test['MONTH']
Y_pred_df['DAY'] = X_test['DAY']
Y_pred_df['HOUR'] = X_test['HOUR']

Y_test_df['YEAR'] = X_test['YEAR']
Y_test_df['MONTH'] = X_test['MONTH']
Y_test_df['DAY'] = X_test['DAY']
Y_test_df['HOUR'] = X_test['HOUR']


X_test_year = X_test[X_test['YEAR'] == year_plot]
X_test_year_month = X_test_year[X_test_year['MONTH'] == month_plot]
X_test_year_month_day = X_test_year_month[X_test_year_month['DAY'] == day_plot]

Y_test_year = Y_test_df[Y_test_df['YEAR'] == year_plot]
Y_test_year_month = Y_test_year[Y_test_year['MONTH'] == month_plot]
Y_test_year_month_day = Y_test_year_month[Y_test_year_month['DAY'] == day_plot]

Y_pred_year = Y_pred_df[Y_pred_df['YEAR'] == year_plot]
Y_pred_year_month = Y_pred_year[Y_pred_year['MONTH'] == month_plot]
Y_pred_year_month_day = Y_pred_year_month[Y_pred_year_month['DAY'] == day_plot]

# Yearly Plot
plt.title(str(year_plot) + " Prediction VS Actual of SVR Model")

plt.plot(X_test_year['HOUR'].index, Y_pred_year['TOTAL_CONSUMPTION'], color="blue", linewidth=3)
plt.plot(X_test_year['HOUR'].index, Y_test_year['TOTAL_CONSUMPTION'], color="black")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()

plt.title(str(year_plot) + " Prediction VS Actual of SVR Model")

plt.scatter(X_test_year['HOUR'].index, Y_test_year['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.scatter(X_test_year['HOUR'].index, Y_pred_year['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.legend(loc="upper left")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()

# Monthly Plot
plt.title(str(year_plot) + ", Month = " + str(month_plot) + " Prediction VS Actual of SVR Model")

plt.plot(Y_test_year_month['HOUR'].index, Y_pred_year_month['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.plot(Y_test_year_month['HOUR'].index, Y_test_year_month['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.legend(loc="upper left")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()

# Daily Plot
plt.title(str(year_plot) + "/" + str(month_plot) + "/" + str(day_plot) + " Prediction VS Actual of SVR Model")

plt.plot(X_test_year_month_day['HOUR'].index, Y_pred_year_month_day['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.plot(X_test_year_month_day['HOUR'].index, Y_test_year_month_day['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.legend(loc="upper left")
plt.xlabel("HOUR")
plt.ylabel("CONSUMPTION in KW")
plt.xticks(())
plt.yticks(())
plt.show()




#%% KNN Model
# JOSEPH FILLS IN CODE HERE ON A NEW BRANCH
# ADDING STUFF




#%% Neural Network Model
# JANNA FILLS IN CODE HERE ON A NEW BRANCH











