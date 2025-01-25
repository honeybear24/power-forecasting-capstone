#Imports
import pandas as pd
import os
import numpy as np
import joblib  # To load the scaler
from sklearn import preprocessing

from pyhelpers.store import save_fig, save_svg_as_emf
import subprocess
import  aspose.cells 

#%% Student directory
hanad_run = ["./data", 1]
clover_run = ["./data", 2]
joseph_laptop_run = ["C:\\Users\\sposa\\Documents\\GitHub\\power-forecasting-capstone\\data", 3]
joseph_pc_run = ["D:\\Users\\Joseph\\Documents\\GitHub\\power-forecasting-capstone\\data", 3]
janna_run = ["../data", 4]

###############################################################################
############### MAKE SURE TO CHANGE BEFORE RUNNING CODE #######################
###############################################################################
# Paste student name_run for whoever is running the code
run_student = janna_run
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

months_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#%% Input files (PRE-PROCESSING)
dirs_inputs = run_student[0]


# Load the scaler
#dirs_inputs = "../data"
scaler_y_filename = r"C:\Users\janna\OneDrive - Toronto Metropolitan University (TMU)\power-forecasting-capstone\data\X_Y_Inputs\min_max_scaler_y.pkl"

#I swear this worked when my whole code was in one file but it certainly doesn't now
#scaler_y_filename = os.path.join(dirs_inputs, "X_Y_Inputs", "min_max_scaler_y.pkl")
loaded_scaler_y = joblib.load(scaler_y_filename)


# Load the normalized DataFrame from CSV
normalized_y_filename = r"C:\Users\janna\OneDrive - Toronto Metropolitan University (TMU)\power-forecasting-capstone\data\X_Y_Inputs\Y_df_normalized.csv"

#I swear this worked when my whole code was in one file but it certainly doesn't now
#normalized_y_filename = os.path.join(dirs_inputs, "X_Y_Inputs", "Y_df_normalized.csv")
Y_df_normalized = pd.read_csv(normalized_y_filename)

# Denormalize the TOTAL_CONSUMPTION column. CHANGE Y_df_normalized TO YOUR Y_PRED.
denormalized_y = loaded_scaler_y.inverse_transform(Y_df_normalized[['TOTAL_CONSUMPTION']])

# Create a new DataFrame for the denormalized values
denormalized_y_df = pd.DataFrame(denormalized_y, columns=['TOTAL_CONSUMPTION'])

# Path for Y_df_denormalized
dirs_dataframes = os.path.join(dirs_inputs, "X_Y_Inputs")

# Ensure the directory exists
os.makedirs(dirs_dataframes, exist_ok=True)

denormalized_y_output_string = os.path.join(dirs_dataframes, "Y_df_denormalized.csv")

# Save only the TOTAL_CONSUMPTION column to CSV
denormalized_y_df.to_csv(denormalized_y_output_string, index=False)