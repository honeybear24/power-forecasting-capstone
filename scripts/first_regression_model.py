### First Regression Model using Linear Regression from Scikit Learn ###

import os
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

## Import saved CSV into script as dataframes
data_dir = "./data"
df_directory = os.path.join(data_dir, "Data_Frames")
X = pd.read_csv(os.path.join(df_directory, "X_all_interpolated.csv"))
Y = pd.read_csv(os.path.join(df_directory, "Y_all.csv"))

# Create regression model
mod = linear_model.LinearRegression()

# Train model
mod.fit(X, Y)

# Fit model
Y_pred = mod.predict(X)


# Plot Models 
plt.scatter(X.index.values.tolist(), Y, color="black")
plt.plot(X.index.values.tolist(), Y_pred, color="blue", linewidth=3)
plt.xlabel("Index")
plt.ylabel("Power Demand")
plt.xticks(())
plt.yticks(())

plt.show()
