### First Regression Model using Linear Regression from Scikit Learn ###

import os
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## Import saved CSV into script as dataframes
data_dir = "./data"
temp_dir = "./temp"
df_directory = os.path.join(data_dir, "Data_Frames")
X = pd.read_csv(os.path.join(df_directory, "X_all.csv"))
Y = pd.read_csv(os.path.join(df_directory, "Y_all.csv"))

# Create regression model
mod = linear_model.LinearRegression()

# Train model - Extract only January from X dataframe
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/(6*12), shuffle=False)
X_train.to_csv(os.path.join(temp_dir,'X_train.csv'), index=False) 
Y_train.to_csv(os.path.join(temp_dir,'Y_train.csv'), index=False) 
X_test.to_csv(os.path.join(temp_dir,'X_test.csv'), index=False) 
Y_test.to_csv(os.path.join(temp_dir,'Y_test.csv'), index=False) 
mod.fit(X_train, Y_train)

# Fit model
Y_pred = mod.predict(X_test)


# Plot Models 
plt.plot(X_test.index.values.tolist(), Y_test, color="black")
plt.plot(X_test.index.values.tolist(), Y_pred, color="blue", linewidth=3)
plt.xlabel("Index")
plt.ylabel("Power Demand")
plt.xticks(())
plt.yticks(())

plt.show()
