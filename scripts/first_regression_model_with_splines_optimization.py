# Script used to find optimal values for splines to get best possible fit for LINEAR REGRESSION model

# Set Up
import os
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import SplineTransformer

## Import saved CSV into script as dataframes
data_dir = "./data"
temp_dir = "./temp"
df_directory = os.path.join(data_dir, "Data_Frames")
X = pd.read_csv(os.path.join(df_directory, "X_all.csv"))
Y = pd.read_csv(os.path.join(df_directory, "Y_all.csv"))

# Parameters to Test
n_knots = [2,3,4,5,6,7,8,9,10]
degrees = [2,3,4,5,6]
knots = ['uniform', 'quantile']

# Loop to check each combination of spline parameters
for knot in knots:
    for n_knot_value  in n_knots:
        for degree_value in degrees:
            
            # Create pipeline containing linear regression model and standard scalar
            pipe = make_pipeline(SplineTransformer(n_knots=n_knot_value, degree=degree_value, knots=knot), linear_model.LinearRegression())
            #mod = linear_model.LinearRegression()

            # Train model - Extract only January from X dataframe
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y['TOTAL_CONSUMPTION'], test_size=1/(6*12), shuffle=False)
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

            # Adding dates to Output Prediction Dataframe
            Y_pred_df['YEAR'] = X_test['YEAR']
            Y_pred_df['MONTH'] = X_test['MONTH']
            Y_pred_df['DAY'] = X_test['DAY']
            Y_pred_df['HOUR'] = X_test['HOUR']

            # Adding dates to Output Test Dataframe
            Y_test_df['YEAR'] = X_test['YEAR']
            Y_test_df['MONTH'] = X_test['MONTH']
            Y_test_df['DAY'] = X_test['DAY']
            Y_test_df['HOUR'] = X_test['HOUR']


            # Evaluate Model
            mape = mean_absolute_percentage_error(Y_test, Y_pred)
            mae = mean_absolute_error(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)

            # Print out result to STDOUT (pipe output to file to examine later)
            print("### Model Evaluation - n_knots:  " + str(n_knot_value) + " degree: " + str(degree_value) +  " Knot:" + knot + " ###")
            print("MAPE: ", str(mape))
            print("MAE: " , str(mae))
            print("R^2 ", str(r2) + "\n")
