# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:18:11 2025

@author: sposa
"""

# from tkinter import *

# root = Tk()

# # Creating a Label Widget
# myLabel = Label(root, text = "POWER SYSTEM FORECASTING")
# myLabel2 = Label(root, text = "BY: Hanad Mohmaud, Clover K. Joseph, Joseph Sposato, and Janna Wong.")

# # Pushing it onto the screen
# myLabel.grid(row = 0, column =0)
# myLabel2.grid(row = 1, column =1)

# root.mainloop()

import customtkinter
import os
from PIL import Image

import pandas as pd
import datetime
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import canada_holiday


class App(customtkinter.CTk):
    def __init__(self):  
        #%% Code for Initalization of GUI application
        
        months_name = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        
        days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
        
        
        # Path for all the graphs that will be shown
        global image_path
        image_path = os.path.join(dirs_inputs, "Model_Plots")    
        
        year_chosen_option_menu = ""
        month_chosen_option_menu = ""
        day_chosen_option_menu = ""
        
        
        
        
        super().__init__()

        
        self.title("Power System Forecasting.py")
        self.geometry("1024x768")

        # Set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)


        ###############################################################################
        # Create Start Frame (all code for desired frame is in here)
        ###############################################################################
        self.start_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.start_frame.grid(row=0, column=0, sticky="nsew")
        
        
        self.start_button = customtkinter.CTkButton(self.start_frame, text="Start", command=self.start_button_event)
        self.start_button.grid(row=0, column=0)
        
        
        
        
        
        
        
        
        
        
        
        
        
        ###############################################################################
        # Create Navigation Frame (all code for desired frame is in here)
        ###############################################################################
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid_rowconfigure(6, weight=1)
        
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="Power System Forecasting", 
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                    anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.model_1_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Model 1",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                       anchor="w", command=self.model_1_button_event)
        self.model_1_button.grid(row=2, column=0, sticky="ew")

        self.model_2_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Model 2",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.model_2_button_event)
        self.model_2_button.grid(row=3, column=0, sticky="ew")
        
        self.model_3_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Model 3",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                       anchor="w", command=self.model_3_button_event)
        self.model_3_button.grid(row=4, column=0, sticky="ew")

        self.model_4_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Model 4",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.model_4_button_event)
        self.model_4_button.grid(row=5, column=0, sticky="ew")

        ###############################################################################
        # Create Home Frame (all code for desired frame is in here)
        ###############################################################################
        # Create frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure((0), weight=1)
        
        # Create title and description
        self.home_frame_Label_Title = customtkinter.CTkLabel(self.home_frame, text="Welcome to Power Forecasting!", font=customtkinter.CTkFont(size=30, weight="bold"))
        self.home_frame_Label_Title.grid(row=0, column=0, padx=20, pady=20, sticky="ew", columnspan=5)
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Please Select the corresponding features below.", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.home_frame_Label_Selection.grid(row=1, column=0, padx=20, pady=20, sticky="ew", columnspan=5)
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Postal Code", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.home_frame_Label_Selection.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="w")
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Year", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.home_frame_Label_Selection.grid(row=2, column=1, padx=20, pady=(0, 20), sticky="w")
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Month", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.home_frame_Label_Selection.grid(row=2, column=2, padx=20, pady=(0, 20), sticky="w")
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Day", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.home_frame_Label_Selection.grid(row=2, column=3, padx=20, pady=(0, 20), sticky="w")
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Number of Days", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.home_frame_Label_Selection.grid(row=2, column=4, padx=20, pady=(0, 20), sticky="w")
        
        
        
        # Create drop down menus
        # FSA
        self.home_frame_fsa_option_menu = customtkinter.CTkOptionMenu(self.home_frame, values=["L9G", "L7G", "L8G", "L6G"], command = self.fsa_option_menu_event)
        self.home_frame_fsa_option_menu.set("L9G")
        self.home_frame_fsa_option_menu.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="w")
        
        
        # Year
        self.home_frame_year_option_menu = customtkinter.CTkOptionMenu(self.home_frame, values=["2024"], command = self.year_option_menu_event)
        self.home_frame_year_option_menu.set("2024")
        self.home_frame_year_option_menu.grid(row=3, column=1, padx=20, pady=(0, 20), sticky="w")
        
        # Month
        self.home_frame_month_option_menu = customtkinter.CTkOptionMenu(self.home_frame, values = months_name, command = self.month_option_menu_event)
        self.home_frame_month_option_menu.set("January")
        self.home_frame_month_option_menu.grid(row=3, column=2, padx=20, pady=(0, 20), sticky="w")
        
        # Day
        self.home_frame_day_option_menu = customtkinter.CTkOptionMenu(self.home_frame, values = days, command = self.day_option_menu_event)
        self.home_frame_day_option_menu.set("01")
        self.home_frame_day_option_menu.grid(row=3, column=3, padx=20, pady=(0, 20), sticky="w")
        
        # Number of Days
        self.home_frame_number_of_days_option_menu = customtkinter.CTkOptionMenu(self.home_frame, values=["1", "2", "3", "4", "5", "6"], command = self.number_of_days_option_menu_event)
        self.home_frame_number_of_days_option_menu.set("1")
        self.home_frame_number_of_days_option_menu.grid(row=3, column=4, padx=20, pady=(0, 20), sticky="w")
        
        
        # Create Generate Models Button
        # Generate Models
        self.generate_models_button = customtkinter.CTkButton(self.home_frame, corner_radius=0, height=40, border_spacing=10, text="Generate Models",
                                                      text_color=("gray10", "gray90"),
                                                      anchor="w", command=self.generate_models_button_event)
        self.generate_models_button.grid(row=4, column=0, sticky="ew")
        
        ###############################################################################        
        # Create second frame (Model 1) (all code for desired frame is in here)
        ###############################################################################
        self.model_1_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        
        

        ###############################################################################
        # Create third frame (Model 2) (all code for desired frame is in here)
        ###############################################################################
        self.model_2_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        
        
        ###############################################################################
        # Create fourth frame (Model 3) (all code for desired frame is in here)
        ###############################################################################
        self.model_3_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        
        
        
        
        ###############################################################################
        # Create fifth frame (Model 4) (all code for desired frame is in here)
        ###############################################################################
        self.model_4_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        ###############################################################################
        # Select default frame
        ###############################################################################
        self.select_frame_by_name("Start")

    ###############################################################################
    # Function to select different frames
    ###############################################################################
    def select_frame_by_name(self, name):
        # set button color for selected button
        self.start_button.configure(fg_color=("gray75", "gray25") if name == "Start" else "transparent")
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "Home" else "transparent")
        self.model_1_button.configure(fg_color=("gray75", "gray25") if name == "Model 1" else "transparent")
        self.model_2_button.configure(fg_color=("gray75", "gray25") if name == "Model 2" else "transparent")
        self.model_3_button.configure(fg_color=("gray75", "gray25") if name == "Model 3" else "transparent")
        self.model_4_button.configure(fg_color=("gray75", "gray25") if name == "Model 4" else "transparent")

        # show selected frame
        if name == "Start":
            self.start_frame.grid(row=0, column=1, sticky="nsew")
            self.navigation_frame.grid_forget()
        else:
            self.navigation_frame.grid(row=0, column=0, sticky="nsew")
            self.start_frame.grid_forget()
        if name == "Home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "Model 1":
            self.model_1_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.model_1_frame.grid_forget()
        if name == "Model 2":
            self.model_2_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.model_2_frame.grid_forget()
        if name == "Model 3":
            self.model_3_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.model_3_frame.grid_forget()
        if name == "Model 4":
            self.model_4_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.model_4_frame.grid_forget()

    ###############################################################################
    # Function when selecting buttons
    ###############################################################################
    def start_button_event(self):
        self.select_frame_by_name("Home")
    
    def home_button_event(self):
        self.select_frame_by_name("Home")

    def model_1_button_event(self):
        self.select_frame_by_name("Model 1")

    def model_2_button_event(self):
        self.select_frame_by_name("Model 2")
        
    def model_3_button_event(self):
        self.select_frame_by_name("Model 3")

    def model_4_button_event(self):
        self.select_frame_by_name("Model 4")
        
    def generate_models_button_event(self):
        
        months_name = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        
        try:
            fsa_chosen = fsa_chosen_option_menu
        except NameError:
            fsa_chosen = "L9G"
            
        try:
            year = year_chosen_option_menu
        except NameError:
            year = "2024"
            
        try:
            month = months[months_name.index(month_chosen_option_menu)]
        except NameError:
            month = "01"
            
        try:
            day = day_chosen_option_menu
        except NameError:
            day = "01"

        dirs_hourly_consumption_demand = os.path.join(dirs_inputs, "Hourly_Demand_Data")

        ###############################################################################
        # Dictionary for reading in hourly consumption by FSA
        ###############################################################################
        # FSA -> Year -> Month -> Value
        hourly_consumption_data_dic_by_month = pd.DataFrame()
        
        # Initialize dataframes to be used
        hourly_data_date = pd.DataFrame()
        hourly_data_res = pd.DataFrame()
        hourly_data_res_fsa = pd.DataFrame()
        hourly_data_hour_sum = pd.DataFrame()
        
        hourly_data_string = "PUB_HourlyConsumptionByFSA_"+year+month+"_v1.csv"
        
        # Use try and catch if problems reading input data
        try:
            # Not cooked yet, we are going to let it COOK below
            file_path = os.path.join(dirs_hourly_consumption_demand, hourly_data_string)
            hourly_data_raw = pd.read_csv(file_path, skiprows=3, header = 0, usecols= ['FSA', 'DATE', 'HOUR', 'CUSTOMER_TYPE', 'TOTAL_CONSUMPTION'])
        except ValueError: # skiprows=x does not match the "normal sequence" of 3. For example, 2023 08 data had a different skip_row value
            hourly_data_raw = pd.read_csv(file_path, skiprows=7, header = 0, usecols= ['FSA', 'DATE', 'HOUR', 'CUSTOMER_TYPE', 'TOTAL_CONSUMPTION'])
       
        # Convert Date into year, month, day
        hourly_data_fix_date = hourly_data_raw
        hourly_data_fix_date['DATE'] = pd.to_datetime(hourly_data_raw['DATE'])
        hourly_data_fix_date['YEAR'] = hourly_data_fix_date['DATE'].dt.year
        hourly_data_fix_date['MONTH'] = hourly_data_fix_date['DATE'].dt.month
        hourly_data_fix_date['DAY'] = hourly_data_fix_date['DATE'].dt.day
        
        # Filter out only residential data
        hourly_data_res = hourly_data_fix_date.loc[hourly_data_fix_date['CUSTOMER_TYPE'] == "Residential"].reset_index(drop=True)
        
        # Then filter out by the fsa
        hourly_data_res_fsa = hourly_data_res.loc[hourly_data_res['FSA'] == fsa_chosen].reset_index(drop=True)
        
        # Take the sum if fsa has more than 1 date (this is because there are different pay codes in residential loads)
        hourly_data_hour_sum = hourly_data_res_fsa.groupby(["FSA", "CUSTOMER_TYPE", "YEAR", "MONTH", "DAY", "HOUR", "DATE"]).TOTAL_CONSUMPTION.sum().reset_index()
        
        
        hourly_consumption_data_dic_by_month = hourly_data_hour_sum
        
        
        print(hourly_data_string)
        
        # Plot Day
        hourly_data_month_day = hourly_consumption_data_dic_by_month[hourly_consumption_data_dic_by_month['DAY'] == int(day)]

        plot = plt.plot(hourly_data_month_day["HOUR"], hourly_data_month_day["TOTAL_CONSUMPTION"], 'o-')

        plt.title(year + "/" + month + "/" + day)
        plt.xlabel("HOUR")
        plt.ylabel("CONSUMPTION in KW")
        plot_svg =  os.path.join(image_path, year + "_" + month + "_" + day + "_Actual_Graph.png")
        plt.savefig(plot_svg)

        plt.show()
        
        
        self.model_1_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, year + "_" + month + "_" + day + "_Actual_Graph.png")), size=(400, 400))
        
        self.model_1_frame_image_label = customtkinter.CTkLabel(self.model_1_frame, text="", image=self.model_1_image)
        self.model_1_frame_image_label.grid(row=0, column=0, padx=20, pady=10)
        
        os.remove(plot_svg)
        
        
        
    ###############################################################################
    # Function when using dropdown menus
    ###############################################################################
    def fsa_option_menu_event(self, choice):
        global fsa_chosen_option_menu
        fsa_chosen_option_menu = choice
        print("optionmenu dropdown clicked:", choice)
        
    def year_option_menu_event(self, choice):
        global year_chosen_option_menu
        year_chosen_option_menu = choice
        print("optionmenu dropdown clicked:", choice)
        
    def month_option_menu_event(self, choice):
        global month_chosen_option_menu
        month_chosen_option_menu = choice
        print("optionmenu dropdown clicked:", choice)

    def day_option_menu_event(self, choice):
        global day_chosen_option_menu
        day_chosen_option_menu = choice
        print("optionmenu dropdown clicked:", choice)
        
    def number_of_days_option_menu_event(self, choice):
        global number_of_days_chosen_option_menu
        number_of_days_chosen_option_menu = choice
        print("optionmenu dropdown clicked:", choice)

if __name__ == "__main__":
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
        
    dirs_inputs = run_student[0]

    #%% Collect actual hourly consumption data that will be used for the model.

    
    
    
    
    #%% Run GUI
    app = App()
    app.mainloop()