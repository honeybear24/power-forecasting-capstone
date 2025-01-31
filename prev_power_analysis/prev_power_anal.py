# Previous Power Analysis - Analzye impact that previous power consumption (power consumption from last year - 8760 hours ago) has on forecasting accuracy of linear regression model

# Set Up
import os
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
import joblib

# Set up file paths and import Y data frame
anal_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/prev_power_analysis"
X = pd.read_csv(os.path.join(anal_dir, "weather_L9G.csv")).drop(["Date/Time (LST)"],axis=1)
Y = pd.read_csv(os.path.join(anal_dir, "power_L9G.csv"))

# Create pipeline containing ridge regression and spline transformer    
pipe = make_pipeline(SplineTransformer(n_knots=6, degree=3, knots='quantile'), linear_model.Ridge(alpha=2.7755102040816326))

# Get train/test split of data set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y['TOTAL_CONSUMPTION'], test_size=1/(5), shuffle=False)

# Fit model
pipe.fit(X_train, Y_train)

# Predict
Y_pred = pipe.predict(X_test)

# Modify Y dataframes
Y_pred_df = pd.DataFrame(Y_pred, columns=['TOTAL_CONSUMPTION'])
Y_test_df = pd.DataFrame(Y_test, columns=['TOTAL_CONSUMPTION'])

# Evaluate Model
mape = mean_absolute_percentage_error(Y_test_df, Y_pred_df)
mae = mean_absolute_error(Y_test_df, Y_pred_df)
r2 = r2_score(Y_test_df, Y_pred_df)
mse = mean_squared_error(Y_test_df, Y_pred_df)
rmse = root_mean_squared_error(Y_test_df, Y_pred_df)

# Print to STDOUT
print("Input Data, MAPE, MAE, R^2, MSE, RMSE")
print("With Year Ago Power" , ", ", mape, ", ", mae, ", ", r2, ", ", mse, ", ", rmse)

# # Adding dates to Output Prediction Dataframe
# Y_pred_df['Year'] = X_test['Year']
# Y_pred_df['Month'] = X_test['Month']
# Y_pred_df['Day'] = X_test['Day']
# Y_pred_df['Hour'] = X_test['Hour']

# # Adding dates to Output Test Dataframe
# Y_test_df['Year'] = X_test['Year']
# Y_test_df['Month'] = X_test['Month']
# Y_test_df['Day'] = X_test['Day']
# Y_test_df['Hour'] = X_test['Hour']

# # Plotting
# year_plot = 2023
# month_plot = 4
# day_plot = 19

# Y_pred_df = pd.DataFrame(Y_pred, columns=['TOTAL_CONSUMPTION'], index = Y_test.index)
# Y_test_df = pd.DataFrame(Y_test, columns=['TOTAL_CONSUMPTION'])

# Y_pred_df['Year'] = X_test['Year']
# Y_pred_df['Month'] = X_test['Month']
# Y_pred_df['Day'] = X_test['Day']
# Y_pred_df['Hour'] = X_test['Hour']

# Y_test_df['Year'] = X_test['Year']
# Y_test_df['Month'] = X_test['Month']
# Y_test_df['Day'] = X_test['Day']
# Y_test_df['Hour'] = X_test['Hour']


# X_test_year_month_day = X_test[X_test['Year'] == year_plot]
# X_test_year_month_day = X_test_year_month_day[X_test_year_month_day['Month'] == month_plot]
# X_test_year_month_day = X_test_year_month_day[X_test_year_month_day['Day'] == day_plot]

# Y_test_year_month_day = Y_test_df[Y_test_df['Year'] == year_plot]
# Y_test_year_month_day = Y_test_year_month_day[Y_test_year_month_day['Month'] == month_plot]
# Y_test_year_month_day = Y_test_year_month_day[Y_test_year_month_day['Day'] == day_plot]

# Y_pred_year_month_day = Y_pred_df[Y_pred_df['Year'] == year_plot]
# Y_pred_year_month_day = Y_pred_year_month_day[Y_pred_year_month_day['Month'] == month_plot]
# Y_pred_year_month_day = Y_pred_year_month_day[Y_pred_year_month_day['Day'] == day_plot]

# # Plotting comparison for randomly selected day
# fig1 = plt.figure("Figure 1")
# plt.plot(X_test_year_month_day['Hour'], Y_test_year_month_day['TOTAL_CONSUMPTION'], 'o-', color="black",  label='Actual')
# plt.plot(X_test_year_month_day['Hour'], Y_pred_year_month_day['TOTAL_CONSUMPTION'], 'o-', color="blue", linewidth=3, label='Predicted')
# plt.xlabel("Hour")
# plt.ylabel("Power Consumption [kW]")
# plt.title("Comparison between Actual and Predicted Power Consumption on " + str(day_plot) + "/" + str(month_plot) + "/" + str(year_plot))
# plt.legend()
# plt.axes([0, max(X_test_year_month_day['Hour'])+1, min(min(Y_test_year_month_day['TOTAL_CONSUMPTION']),min(Y_pred_year_month_day['TOTAL_CONSUMPTION'])), max(max(Y_test_year_month_day['TOTAL_CONSUMPTION']),max(Y_pred_year_month_day['TOTAL_CONSUMPTION']))])
# plt.xticks(())
# plt.yticks(())
# plt.show()

# # Ploting comparison for a whole month 
# fig2 = plt.figure("Figure 2")
# plt.plot(X_test.index, Y_test_df['TOTAL_CONSUMPTION'], color="black",  label='Actual')
# plt.plot(X_test.index, Y_pred_df['TOTAL_CONSUMPTION'], color="blue",  label='Prediction')
# plt.title("Comparison between Actual and Predicted Power Consumption for entire Test Set")
# plt.xlabel("Index")
# plt.ylabel("Power Consumption [kW]")
# plt.legend()
# plt.axes([0, 53000, min(min(Y_test_df['TOTAL_CONSUMPTION']),min(Y_pred_df['TOTAL_CONSUMPTION'])), max(max(Y_test_df['TOTAL_CONSUMPTION']),max(Y_pred_df['TOTAL_CONSUMPTION']))])
# plt.xticks(())
# plt.yticks(())
# plt.show()