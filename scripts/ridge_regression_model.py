### First Regression Model using Linear Regression from Scikit Learn ###

import os
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import SplineTransformer

## Import saved CSV into script as dataframes
data_dir = "./data"
temp_dir = "./temp"
df_directory = os.path.join(data_dir, "Data_Frames")
X = pd.read_csv(os.path.join(df_directory, "X_all.csv"))
Y = pd.read_csv(os.path.join(df_directory, "Y_all.csv"))

# Create pipeline containing linear regression model and standard scalar
pipe = make_pipeline(SplineTransformer(n_knots=5, degree=4), linear_model.Ridge(alpha=1.6667))
#pipe = make_pipeline(linear_model.Ridge(alpha=5.0))
#mod = linear_model.LinearRegression()

# Train model - Extract only January from X dataframe
X_train, X_test, Y_train, Y_test = train_test_split(X, Y['TOTAL_CONSUMPTION'], test_size=1/(6*12), shuffle=False)
X_train.to_csv(os.path.join(temp_dir,'X_train.csv'), index=False) 
Y_train.to_csv(os.path.join(temp_dir,'Y_train.csv'), index=False) 
X_test.to_csv(os.path.join(temp_dir,'X_test.csv'), index=False) 
Y_test.to_csv(os.path.join(temp_dir,'Y_test.csv'), index=False) 
pipe.fit(X_train, Y_train)

# Fit model
Y_pred = pipe.predict(X_test)

# Modify Y dataframes
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


# Evaluate Model
mape = mean_absolute_percentage_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("## Model Evaluation ##")
print("MAPE: ", str(mape))
print("MAE: " , str(mae))
print("R^2 ", str(r2))



# Plotting
year_plot = 2023
month_plot = 12
day_plot = 6

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


X_test_year_month_day = X_test[X_test['YEAR'] == year_plot]
X_test_year_month_day = X_test_year_month_day[X_test_year_month_day['MONTH'] == month_plot]
X_test_year_month_day = X_test_year_month_day[X_test_year_month_day['DAY'] == day_plot]

Y_test_year_month_day = Y_test_df[Y_test_df['YEAR'] == year_plot]
Y_test_year_month_day = Y_test_year_month_day[Y_test_year_month_day['MONTH'] == month_plot]
Y_test_year_month_day = Y_test_year_month_day[Y_test_year_month_day['DAY'] == day_plot]

Y_pred_year_month_day = Y_pred_df[Y_pred_df['YEAR'] == year_plot]
Y_pred_year_month_day = Y_pred_year_month_day[Y_pred_year_month_day['MONTH'] == month_plot]
Y_pred_year_month_day = Y_pred_year_month_day[Y_pred_year_month_day['DAY'] == day_plot]

# Plotting comparison for randomly selected day
fig1 = plt.figure("Figure 1")
plt.plot(X_test_year_month_day['HOUR'], Y_test_year_month_day['TOTAL_CONSUMPTION'], color="black",  label='Actual')
plt.plot(X_test_year_month_day['HOUR'], Y_pred_year_month_day['TOTAL_CONSUMPTION'], color="blue", linewidth=3, label='Predicted')
plt.xlabel("Hour")
plt.ylabel("Power Demand [kW]")
plt.title("Comparison between Actual and Prediction for " + str(month_plot) + "/" + str(day_plot) + "/" + str(year_plot))
plt.legend()
plt.axes([0, max(X_test_year_month_day['HOUR'])+1, min(min(Y_test_year_month_day['TOTAL_CONSUMPTION']),min(Y_pred_year_month_day['TOTAL_CONSUMPTION'])), max(max(Y_test_year_month_day['TOTAL_CONSUMPTION']),max(Y_pred_year_month_day['TOTAL_CONSUMPTION']))])
plt.xticks(())
plt.yticks(())
plt.show(block=False)

# Ploting comparison for a whole month 
fig2 = plt.figure("Figure 2")
plt.plot(X_test.index, Y_test_df['TOTAL_CONSUMPTION'], color="black",  label='Actual')
plt.plot(X_test.index, Y_pred_df['TOTAL_CONSUMPTION'], color="blue",  label='Prediction')
plt.title("Comparison between Actual and Prediction for " + str(month_plot) + "/" + str(year_plot))
plt.xlabel("Index")
plt.ylabel("Power Demand [kW]")
plt.legend()
plt.axes([0, 53000, min(min(Y_test_df['TOTAL_CONSUMPTION']),min(Y_pred_df['TOTAL_CONSUMPTION'])), max(max(Y_test_df['TOTAL_CONSUMPTION']),max(Y_pred_df['TOTAL_CONSUMPTION']))])
plt.xticks(())
plt.yticks(())
plt.show()