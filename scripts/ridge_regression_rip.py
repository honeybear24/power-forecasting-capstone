# Ridge Regression model with splines (using optimized values)

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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import SplineTransformer


## Import saved CSV into script as dataframes
data_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/data/"
temp_dir = "c:/Users/hanad/capstone_github/power-forecasting-capstone/temp/"
df_directory = os.path.join(data_dir, "raw_data")
X = pd.read_csv(os.path.join(df_directory, "weather_data_L7G_20180101_20231231_V19s.csv")).drop(["Date/Time (LST)"],axis=1)
Y = pd.read_csv(os.path.join(df_directory, "power_data_L7G_20180101_20231231_V19.csv"))
# df_directory = os.path.join(data_dir, "X_Y_Inputs")
# X = pd.read_csv(os.path.join(df_directory, "X_df_L9G_orig.csv")).drop(["Rel Hum (%)", "Wind Spd (km/h)", "DATE"],axis=1)
# Y = pd.read_csv(os.path.join(df_directory, "Y_df_L9G_orig.csv"))

# Create pipeline containing linear regression model and standard scalar
pipe = make_pipeline(SplineTransformer(n_knots=6, degree=3, knots='quantile'), linear_model.Ridge(alpha=2.7755102040816326))

# Train model - Extract only January from X dataframe
X_train, X_test, Y_train, Y_test = train_test_split(X, Y['TOTAL_CONSUMPTION'], test_size=1/(5), shuffle=False)
X_train.to_csv(os.path.join(temp_dir,'X_train.csv'), index=False) 
Y_train.to_csv(os.path.join(temp_dir,'Y_train.csv'), index=False) 
X_test.to_csv(os.path.join(temp_dir,'X_test.csv'), index=False) 
Y_test.to_csv(os.path.join(temp_dir,'Y_test.csv'), index=False) 
pipe.fit(X_train, Y_train)

# Fit model
Y_pred = pipe.predict(X_test)

# Modify Y dataframes
Y_pred_df = pd.DataFrame(Y_pred, columns=['TOTAL_CONSUMPTION'], index = Y_test.index)
Y_pred_df.to_csv(os.path.join(temp_dir,'Y_pred.csv'), index=False) 
Y_test_df = pd.DataFrame(Y_test, columns=['TOTAL_CONSUMPTION'])

# Adding dates to Output Prediction Dataframe
Y_pred_df['Year'] = X_test['Year']
Y_pred_df['Month'] = X_test['Month']
Y_pred_df['Day'] = X_test['Day']
Y_pred_df['Hour'] = X_test['Hour']

# Adding dates to Output Test Dataframe
Y_test_df['Year'] = X_test['Year']
Y_test_df['Month'] = X_test['Month']
Y_test_df['Day'] = X_test['Day']
Y_test_df['Hour'] = X_test['Hour']


# Evaluate Model
mape = mean_absolute_percentage_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = root_mean_squared_error(Y_test, Y_pred)

# Print out result to STDOUT
print("## Model Evaluation ##")
print("MAPE: ", str(mape))
print("MAE: " , str(mae))
print("R^2 ", str(r2))
print("MSE: ", str(mse))
print("RMSE: " , str(rmse))



# Plotting
year_plot = 2023
month_plot = 4
day_plot = 19

Y_pred_df = pd.DataFrame(Y_pred, columns=['TOTAL_CONSUMPTION'], index = Y_test.index)
Y_test_df = pd.DataFrame(Y_test, columns=['TOTAL_CONSUMPTION'])

Y_pred_df['Year'] = X_test['Year']
Y_pred_df['Month'] = X_test['Month']
Y_pred_df['Day'] = X_test['Day']
Y_pred_df['Hour'] = X_test['Hour']

Y_test_df['Year'] = X_test['Year']
Y_test_df['Month'] = X_test['Month']
Y_test_df['Day'] = X_test['Day']
Y_test_df['Hour'] = X_test['Hour']


X_test_year_month_day = X_test[X_test['Year'] == year_plot]
X_test_year_month_day = X_test_year_month_day[X_test_year_month_day['Month'] == month_plot]
X_test_year_month_day = X_test_year_month_day[X_test_year_month_day['Day'] == day_plot]

Y_test_year_month_day = Y_test_df[Y_test_df['Year'] == year_plot]
Y_test_year_month_day = Y_test_year_month_day[Y_test_year_month_day['Month'] == month_plot]
Y_test_year_month_day = Y_test_year_month_day[Y_test_year_month_day['Day'] == day_plot]

Y_pred_year_month_day = Y_pred_df[Y_pred_df['Year'] == year_plot]
Y_pred_year_month_day = Y_pred_year_month_day[Y_pred_year_month_day['Month'] == month_plot]
Y_pred_year_month_day = Y_pred_year_month_day[Y_pred_year_month_day['Day'] == day_plot]

# Plotting comparison for randomly selected day
fig1 = plt.figure("Figure 1")
plt.plot(X_test_year_month_day['Hour'], Y_test_year_month_day['TOTAL_CONSUMPTION'], 'o-', color="black",  label='Actual')
plt.plot(X_test_year_month_day['Hour'], Y_pred_year_month_day['TOTAL_CONSUMPTION'], 'o-', color="blue", linewidth=3, label='Predicted')
plt.xlabel("Hour")
plt.ylabel("Power Consumption [kW]")
plt.title("Comparison between Actual and Predicted Power Consumption on " + str(day_plot) + "/" + str(month_plot) + "/" + str(year_plot))
plt.legend()
plt.axes([0, max(X_test_year_month_day['Hour'])+1, min(min(Y_test_year_month_day['TOTAL_CONSUMPTION']),min(Y_pred_year_month_day['TOTAL_CONSUMPTION'])), max(max(Y_test_year_month_day['TOTAL_CONSUMPTION']),max(Y_pred_year_month_day['TOTAL_CONSUMPTION']))])
plt.xticks(())
plt.yticks(())
plt.show()

# Ploting comparison for a whole month 
fig2 = plt.figure("Figure 2")
plt.plot(X_test.index, Y_test_df['TOTAL_CONSUMPTION'], color="black",  label='Actual')
plt.plot(X_test.index, Y_pred_df['TOTAL_CONSUMPTION'], color="blue",  label='Prediction')
plt.title("Comparison between Actual and Predicted Power Consumption for entire Test Set")
plt.xlabel("Index")
plt.ylabel("Power Consumption [kW]")
plt.legend()
plt.axes([0, 53000, min(min(Y_test_df['TOTAL_CONSUMPTION']),min(Y_pred_df['TOTAL_CONSUMPTION'])), max(max(Y_test_df['TOTAL_CONSUMPTION']),max(Y_pred_df['TOTAL_CONSUMPTION']))])
plt.xticks(())
plt.yticks(())
plt.show()