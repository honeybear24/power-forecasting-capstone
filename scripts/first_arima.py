import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

## Import saved CSV into script as dataframes
data_dir = "./data"
temp_dir = "./temp"
df_directory = os.path.join(data_dir, "Data_Frames")
Y = pd.read_csv(os.path.join(df_directory, "Y_all.csv"))

# # Plot the time series
# plt.plot(Y['TOTAL_CONSUMPTION'])
# plt.title('Power Consumption')cle
# plt.xlabel('Hours (Hour 1 = Jan 1, 2018 @ 1AM)')
# plt.ylabel('Power [kW]')
# plt.show()

# Print first few lines
Y_new = Y.drop(["YEAR", "MONTH", "DAY", "HOUR"], axis=1)
print(Y_new.head())

# ETS Decomposition 
result = seasonal_decompose(Y_new['TOTAL_CONSUMPTION'], model ='multiplicative', period=8764) 
  
# ETS plot  
result.plot() 