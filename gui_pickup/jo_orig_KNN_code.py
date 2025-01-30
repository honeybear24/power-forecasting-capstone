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
run_student = joseph_pc_run
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

#%% Input files (PRE-PROCESSING)
dirs_inputs = run_student[0]

dirs_x_y_input = os.path.join(dirs_inputs, "X_Y_Inputs")

file_path_x = os.path.join(dirs_x_y_input, "X_df_"+fsa_chosen+".csv")
file_path_y = os.path.join(dirs_x_y_input, "Y_df_"+fsa_chosen+".csv")
X_df_knn = pd.read_csv(file_path_x)
Y_df_knn = pd.read_csv(file_path_y)

#%% Regression Model
# HANAD FILLS IN CODE HERE ON A NEW BRANCH




#%% SVR Model
# CLOVER FILLS IN CODE HERE ON A NEW BRANCH



#%% KNN Model
# JOSEPH FILLS IN CODE HERE ON A NEW BRANCH
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X_df_knn = X_df_knn.drop(["DATE", "WEEKDAY", "Rel Hum (%)", "Wind Spd (km/h)"], axis = 1)


X_train, X_test, Y_train, Y_test = train_test_split(X_df_knn, Y_df_knn, test_size=0.2, shuffle = False)

X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

# knn_model = KNeighborsRegressor(n_neighbors=12, weights = 'distance')
# knn_model = KNeighborsRegressor(n_neighbors=12)
# knn_model.fit(X_train, Y_train)
# Y_pred = knn_model.predict(X_test)
# mape = mean_absolute_percentage_error(Y_test, Y_pred)
# print("MAPE1 = " + str(round(mape*100, 3)) + "%.")
# mae = mean_absolute_error(Y_test, Y_pred)
# print("MAE = " + str(mae) + ".")
# mse = mean_squared_error(Y_test, Y_pred)
# r2 = r2_score(Y_test, Y_pred)


# TRY GRID SEARCH
pipeline_knn = Pipeline([("model", KNeighborsRegressor(n_neighbors=1, weights = 'distance'))])

knn_model = GridSearchCV(estimator = pipeline_knn,
                         scoring = ['neg_mean_absolute_percentage_error'],
                         param_grid = {'model__n_neighbors': range(1,50)},
                         refit = 'neg_mean_absolute_percentage_error')

knn_model.fit(X_train, Y_train)
output = pd.DataFrame(knn_model.cv_results_)

Y_pred = knn_model.predict(X_test)

print(str(knn_model.best_params_))
mape = mean_absolute_percentage_error(Y_test, Y_pred)
print("MAPE = " + str(round(mape*100, 3)) + "%.")
mae = mean_absolute_error(Y_test, Y_pred)
print("MAE = " + str(mae) + ".")
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)


#%% PLOTTING
year_plot = 2023
month_plot = 4
day_plot = 23



Y_pred_df = pd.DataFrame(Y_pred, columns=['TOTAL_CONSUMPTION'], index = Y_test.index)
Y_test_df = pd.DataFrame(Y_test, columns=['TOTAL_CONSUMPTION'])

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

# # Yearly Plot
# plt.title(str(year_plot) + " Prediction VS Actual of KNN Model")

# plt.plot(X_test_year['HOUR'].index, Y_pred_year['TOTAL_CONSUMPTION'], color="blue", linewidth=3)
# plt.plot(X_test_year['HOUR'].index, Y_test_year['TOTAL_CONSUMPTION'], color="black")
# plt.xlabel("HOUR")
# plt.ylabel("CONSUMPTION in KW")
# plt.xticks(())
# plt.yticks(())
# plt.show()

plt.title(str(year_plot) + " Prediction VS Actual of KNN Model")
plt.plot(X_test_year['HOUR'].index, Y_test_year['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.xlabel("INDEX")
plt.ylabel("CONSUMPTION in KW")
plt.xlim(0, 50)
plt.ylim(0, 1)
plt.axes()
plt.plot(X_test_year['HOUR'].index, Y_pred_year['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.plot(X_test_year['HOUR'].index, Y_test_year['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.legend(loc="upper left") 
plt.xticks(())
plt.yticks(())
plt.show()

# Monthly Plot
plt.title(str(year_plot) + ", Month = " + str(month_plot) + " Prediction VS Actual of KNN Model")

plt.plot(Y_test_year_month['HOUR'].index, Y_pred_year_month['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.xlabel("INDEX")
plt.ylabel("CONSUMPTION in KW")
plt.axes()
plt.plot(Y_test_year_month['HOUR'].index, Y_pred_year_month['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label="PREDICTION")
plt.plot(Y_test_year_month['HOUR'].index, Y_test_year_month['TOTAL_CONSUMPTION'], color="black", label="ACTUAL")
plt.legend(loc="upper left")

plt.xticks(())
plt.yticks(())
plt.show()

# Daily Plot

plt.title(str(day_plot) + "/" + str(month_plot) + "/" + str(year_plot) + " Prediction VS Actual of KNN Model")

plt.plot(X_test_year_month_day['HOUR'].index, Y_pred_year_month_day['TOTAL_CONSUMPTION'], 'o-', color="blue", linewidth=3, label="PREDICTION")
plt.plot(X_test_year_month_day['HOUR'].index, Y_test_year_month_day['TOTAL_CONSUMPTION'], 'o-', color="black", label="ACTUAL")
plt.xlabel("Hour")
plt.ylabel("Power Consumption [KW]")
plt.xticks(np.arange(0, 25, 2))
plt.xlim(0, 25)
plt.axes()
plt.plot(X_test_year_month_day['HOUR'].index, Y_pred_year_month_day['TOTAL_CONSUMPTION'], 'o-', color="blue", linewidth=3, label="PREDICTION")
plt.plot(X_test_year_month_day['HOUR'].index, Y_test_year_month_day['TOTAL_CONSUMPTION'], 'o-', color="black", label="ACTUAL")
plt.legend(loc="upper left")
plt.xticks(())
plt.yticks(())
plt.show()









