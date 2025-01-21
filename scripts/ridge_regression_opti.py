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
X = pd.read_csv(os.path.join(df_directory, "weather_data_L9G_20180101_20231231_V11_24lags.csv")).drop(["Date/Time (LST)"],axis=1)
Y = pd.read_csv(os.path.join(df_directory, "power_data_L9G_20180101_20231231_V11_24lags.csv"))
# df_directory = os.path.join(data_dir, "X_Y_Inputs")
# X = pd.read_csv(os.path.join(df_directory, "X_df_L9G_orig.csv")).drop(["Rel Hum (%)", "Wind Spd (km/h)", "DATE"],axis=1)
# Y = pd.read_csv(os.path.join(df_directory, "Y_df_L9G_orig.csv"))

# Parameters to Test
n_knots = [2,3,4,5,6,7,8,9,10]
degrees = [2,3,4,5,6,7]
knots = ['uniform', 'quantile']
alphas = np.linspace(1.0, 4.0, num=20)

for alpha_now in alphas:
    for knot in knots:
        for n_knot_value  in n_knots:
            for degree_value in degrees:
                # Create pipeline containing linear regression model and standard scalar
                pipe = make_pipeline(SplineTransformer(n_knots=n_knot_value, degree=degree_value, knots=knot), linear_model.Ridge(alpha=alpha_now))
                #mod = linear_model.LinearRegression()

                # Train model - Extract only January from X dataframe
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y['TOTAL_CONSUMPTION'], test_size=1/(5), shuffle=False)
                # X_train.to_csv(os.path.join(temp_dir,'X_train.csv'), index=False) 
                # Y_train.to_csv(os.path.join(temp_dir,'Y_train.csv'), index=False) 
                # X_test.to_csv(os.path.join(temp_dir,'X_test.csv'), index=False) 
                # Y_test.to_csv(os.path.join(temp_dir,'Y_test.csv'), index=False) 
                pipe.fit(X_train, Y_train)

                # Fit model
                Y_pred = pipe.predict(X_test)

                # Modify Y dataframes
                Y_pred_df = pd.DataFrame(Y_pred, columns=['TOTAL_CONSUMPTION'], index = Y_test.index)
                Y_test_df = pd.DataFrame(Y_test, columns=['TOTAL_CONSUMPTION'])

                # Evaluate Model
                mape = mean_absolute_percentage_error(Y_test, Y_pred)
                mae = mean_absolute_error(Y_test, Y_pred)
                r2 = r2_score(Y_test, Y_pred)

                # Print out result to STDOUT (pipe output to file to examine later)
                print("### Model Evaluation - n_knots:  " + str(n_knot_value) + " degree: " + str(degree_value) +  " Knot:" + knot + " Alpha:" + str(alpha_now) + " ###")
                print("MAPE: ", str(mape))
                print("MAE: " , str(mae))
                print("R^2 ", str(r2) + "\n")