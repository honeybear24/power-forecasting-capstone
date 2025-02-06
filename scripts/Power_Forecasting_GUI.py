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

import realDataGetter
import asyncio  # Import asyncio library for async operations
import aiohttp # Import aiohttp library for making HTTP requests
import nest_asyncio # Allows for asyncio to be nested

import customtkinter
from CTkTable import *
import tkinter as Tk
from tkcalendar import Calendar
import os
import glob
import PIL
from PIL import Image

import pandas as pd
import datetime
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import canada_holiday

import joblib

class ScrollableCheckBoxFrame(customtkinter.CTkScrollableFrame):
    #%% Code for Initalization of scollable frame
    def __init__(self, master, item_list, command=None, **kwargs):
        super().__init__(master, **kwargs)

        self.command = command
        self.checkbox_list = []
        for i, item in enumerate(item_list):
            self.add_item(item)

    def add_item(self, item):
        
        checkbox = customtkinter.CTkCheckBox(self, text=item, variable = customtkinter.IntVar(value = 0), onvalue = 1, offvalue = 0)
        if self.command is not None:
            checkbox.configure(command=self.command)
        checkbox.grid(row=len(self.checkbox_list), column=0, pady=(0, 10), sticky = "w")
        self.checkbox_list.append(checkbox)

    def remove_item(self, item):
        for checkbox in self.checkbox_list:
            if item == checkbox.cget("text"):
                checkbox.destroy()
                self.checkbox_list.remove(checkbox)
                return

    def get_checked_items(self):
        return [checkbox.cget("text") for checkbox in self.checkbox_list if checkbox.get() == 1]




class App(customtkinter.CTk):  
    def __init__(self):  
        #%% Code for Initalization of GUI application
        
        ###############################################################################
        # All file paths
        ###############################################################################
        
        # Paths for all the graphs that will be shown
        global image_path, background_images_path, model_path, x_y_input_path, hourly_demand 
        image_path = os.path.join(dirs_inputs, "Model_Plots") 
        model_path = os.path.join(dirs_inputs, "Saved_Models")    
        x_y_input_path = os.path.join(dirs_inputs, "X_Y_Inputs")   
        background_images_path = os.path.join(dirs_inputs, "GUI_Background_Images") 
        hourly_demand = os.path.join(dirs_inputs, "raw_data")
        
        year_chosen_option_menu = ""
        month_chosen_option_menu = ""
        day_chosen_option_menu = ""
        
        
        
        customtkinter.set_default_color_theme("blue") # change the colour theme of the application
        super().__init__()

        
        self.title("Power System Forecasting.py")
        self.geometry("1024x768")

        # Set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Path for all the background images
        start_menu_image_path = os.path.join(background_images_path, "Start_Menu_Page.png")
        
        # Create background image
        image = PIL.Image.open(start_menu_image_path)
        background_image = customtkinter.CTkImage(image, size=(1920, 1080))
        
        

        ###############################################################################
        # Create Start Frame (all code for desired frame is in here)
        ###############################################################################
        self.start_frame = customtkinter.CTkFrame(self, corner_radius=0)
        
         # Create background label for start frame
        self.background_label = customtkinter.CTkLabel(self.start_frame, 
                                                     image=background_image,
                                                     text="")  # Empty text
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        customtkinter.set_appearance_mode("dark")
        
        
        self.start_frame.grid(row=0, column=0, sticky="nsew")
        
        
        self.start_frame.grid(row=0, column=0, sticky="nsew")
        
        self.start_frame_Label_Title = customtkinter.CTkLabel(self.start_frame, text="Power Forecasting! ", 
            font=customtkinter.CTkFont(family="Roboto Condensed", size=80, slant="italic"),
            bg_color='#220549', text_color=("white", "white"))
        self.start_frame_Label_Title.place(relx=0.55, rely=0.2, anchor='n')
        
        # # Start the fade-in animation
        # self.alpha = 0
        # self.fading_in = True
        # self.animate_fade()
    
    
        
        # # Convert alpha to hex color
        # color_value = int(self.alpha * 255)
        # hex_color = f'#{color_value:02x}{color_value:02x}{color_value:02x}'
        
        # self.start_frame_Label_Title.configure(text_color=hex_color)
        # self.start_frame.after(5, self.animate_fade)

        self.start_frame_Label_Text =   customtkinter.CTkLabel(self.start_frame, text="Predicting the power demand of tomorrow ", 
            font=customtkinter.CTkFont(family="Roboto Flex", size=50, slant="italic"),
            bg_color='#220549', text_color=("white"))
        self.start_frame_Label_Text.place(relx=0.55, rely=0.45, anchor='center')

        

        my_font = customtkinter.CTkFont(family="Roboto", size=40, 
	weight="bold", slant="italic", underline=False, overstrike=False) #font to be used for titles       
        self.start_button = customtkinter.CTkButton(self.start_frame, text="Start ", command=self.start_button_event, height=85, width=250, font=my_font, corner_radius=50,bg_color='#220549',fg_color="#4B0082")
        self.start_button.place(relx=0.60, rely=0.78, anchor='se')
    
        
        # other color290753
        
        
        ###############################################################################
        # Create Navigation Frame (all code for desired frame is in here)
        ###############################################################################
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid_rowconfigure(6, weight=1)
        
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="        Power System Forecasting", 
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
            
        
        # Add hamburger menu button
        self.hamburger_button = customtkinter.CTkButton(self.navigation_frame, text="☰", width=40, height=40, command=self.toggle_navigation)
        self.hamburger_button.place(x=10, y=10)  # Adjust position as needed

        self.navigation_visible = False  # Track the visibility of the navigation frame

        # Initially hide the navigation frame
        self.toggle_navigation()
        
        
        
        
        # self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="        Power System Forecasting",
        #                                                      compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        # self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # self.start_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Start",
        #                                             fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
        #                                             anchor="w", command=self.start_button_event)
        # self.start_button.grid(row=1, column=0, sticky="ew")

        # self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",
        #                                            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
        #                                            anchor="w", command=self.home_button_event)
        # self.home_button.grid(row=2, column=0, sticky="ew")

        # self.model_1_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Model 1",
        #                                               fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
        #                                               anchor="w", command=self.model_1_button_event)
        # self.model_1_button.grid(row=3, column=0, sticky="ew")

        # self.model_2_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Model 2",
        #                                               fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
        #                                               anchor="w", command=self.model_2_button_event)
        # self.model_2_button.grid(row=4, column=0, sticky="ew")

        # self.model_3_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Model 3",
        #                                               fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
        #                                               anchor="w", command=self.model_3_button_event)
        # self.model_3_button.grid(row=5, column=0, sticky="ew")

        # self.model_4_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Model 4",
        #                                               fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
        #                                               anchor="w", command=self.model_4_button_event)
        # self.model_4_button.grid(row=6, column=0, sticky="ew")

        
        
        
        
        ###############################################################################
        # Create Home Frame (all code for desired frame is in here)
        ###############################################################################
        # Create frame
        home_menu_image_path = os.path.join(background_images_path, "Home_Page.png")
        # Create background image
        image = PIL.Image.open(home_menu_image_path)
        background_image_home = customtkinter.CTkImage(image, size=(1920, 1080))
        
    
        
        # Create frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure((0), weight=1)
        
        self.background_label = customtkinter.CTkLabel(self.home_frame,
                                                     image=background_image_home,
                                                     text="")  # Empty text
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.home_frame.grid(row=0, column=0, sticky="nsew")

        
        # Create title and description
        self.home_frame_Label_Title = customtkinter.CTkLabel(self.home_frame, text="Welcome to Power Forecasting!", font=customtkinter.CTkFont(family="Roboto Flex", size=50, slant="italic"), 
                                                             bg_color='#140034', text_color=("white"))
        self.home_frame_Label_Title.grid(row=0, column=0, padx=20, pady=20, sticky="ew", columnspan=3)
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Please Select the corresponding features below.", font=customtkinter.CTkFont(family="Roboto Flex", size=40),
            bg_color='#140034', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=1, column=0, padx=20, pady=20, sticky="ew", columnspan=3)
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Postal Code", font=customtkinter.CTkFont(size=20, weight="bold"), bg_color='#140034', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="w")
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Start Date", font=customtkinter.CTkFont(size=20, weight="bold"), bg_color='#140034', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=2, column=1, padx=20, pady=(0, 20), sticky="w")
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Number of Days", font=customtkinter.CTkFont(size=20, weight="bold"), bg_color='#140034', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=2, column=2, padx=20, pady=(0, 20), sticky="w")
        
        

        

        # Create drop down menus
        # FSA
        self.home_frame_fsa_option_menu = customtkinter.CTkOptionMenu(self.home_frame, values=["L9G", "L7G", "L8G", "L6G"], command = self.fsa_option_menu_event)
        self.home_frame_fsa_option_menu.set("L9G")
        self.home_frame_fsa_option_menu.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="w")
        
        # Start date
        # Add calendar frame for Date
        self.calendar_frame = customtkinter.CTkFrame(self.home_frame, corner_radius=0,fg_color='#140034')
        self.calendar_frame.grid(row=3, column=1, padx=20, pady=(0, 20), sticky="w")

        # Set minimum date
        min_date = date(2023, 1, 1)
        
        # Set maximum date
        max_date = date(2023, 3, 31)
        
        self.calendar = Calendar(self.calendar_frame, selectmode='day', year=2023, month=1, day=1, mindate = min_date, maxdate = max_date)
        self.calendar.pack(pady=20, padx=20)
        #self.calendar.bind("<<CalendarSelected>>", self.print_calendar_size)
        
        # Number of Days
        self.home_frame_number_of_days_option_menu = customtkinter.CTkOptionMenu(self.home_frame, values=["1", "2", "3"], command = self.number_of_days_option_menu_event)
        self.home_frame_number_of_days_option_menu.set("1")
        self.home_frame_number_of_days_option_menu.grid(row=3, column=2, padx=20, pady=(0, 20), sticky="w")
        
        # Create Generate Models Button
        # Generate Models
        self.generate_models_button = customtkinter.CTkButton(self.home_frame, corner_radius=0, height=40, border_spacing=10, text="Generate Models",
                                                      text_color=("gray10", "gray90"),
                                                      anchor="w", command=self.generate_models_button_event)
        self.generate_models_button.grid(row=4, column=1, sticky="ew")
        
        # Create search bar for FSA
        self.fsa_search_bar = customtkinter.CTkEntry(self.home_frame, placeholder_text ="Please enter first three digits of postal code.")
        
        self.fsa_search_bar.grid(row=5, column=0, pady=20, padx=20, sticky="nsew")

        # Create scrollable check box of features
        column_names = pd.read_csv(os.path.join(x_y_input_path, "X_transformed_with_origCalVariables.csv"), nrows = 0)
        self.scrollable_features_checkbox_frame = ScrollableCheckBoxFrame(self.home_frame, width=200, command=self.features_checkbox_event,
                                                         item_list=column_names)
        self.scrollable_features_checkbox_frame.grid(row=5, column=1, pady=20, padx=20, sticky="ew")
        
        
        # Create Train button
        self.train_models_button = customtkinter.CTkButton(self.home_frame, corner_radius=0, height=40, border_spacing=10, text="Train Models",
                                                      text_color=("gray10", "gray90"),
                                                      anchor="w", command=self.train_models_button_event)
        self.train_models_button.grid(row=5, column=2, sticky="ew")
        
        # Add hamburger menu button
        self.hamburger_button = customtkinter.CTkButton(self.home_frame, text="☰", width=40, height=40, command=self.toggle_navigation)
        self.hamburger_button.place(x=10, y=10)  # Adjust position as needed
        
        
        
        
        
        
        #CLOVER###################

        # THIS IS CLOVER'S METHOD FOR POSITIONING
        # FSA
        # self.home_frame_fsa_option_menu.place(relx=0.28, rely=0.60, anchor='center')
        
        
        ##########################
        


        
        
        
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

        
        # Function to plot the model figures
        def plot_figures_model(self, hourly_data_month_day_saved, Y_pred_denorm_saved_df_saved, hourly_data_month_day_error, model_frame):
            
            # Plot models on same graph
            fig, ax = plt.subplots(figsize = (15, 5))
            
            plt.plot(hourly_data_month_day_saved["HOUR_NEW"], hourly_data_month_day_saved["TOTAL_CONSUMPTION"], 'o-', label = "Actual Consumption", color = "pink")
            plt.plot(hourly_data_month_day_saved["HOUR_NEW"], Y_pred_denorm_saved_df_saved["TOTAL_CONSUMPTION"], 'o-', label = "Predicted Consumption", color = "purple")
            plt.title(title)       
            plt.xlabel("HOUR")
            plt.ylabel("CONSUMPTION in KW")
            plt.legend(loc = "upper left")
            plot_svg =  os.path.join(image_path, "Predicted_Actual_Graph.png")
            plt.savefig(plot_svg)
            plt.close()
            
            # Positining of Figure
            self.model_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "Predicted_Actual_Graph.png")), size=(1200, 400))
            
            self.model_frame_image_label = customtkinter.CTkLabel(model_frame, text="", image=self.model_image)
            self.model_frame_image_label.grid(row=0, column=0, padx=20, pady=20, sticky = "ew")
            
            # Display Table of Error
            hourly_data_month_day_saved_table_tp = hourly_data_month_day_error.transpose()
            
            
            # Positining of Figure
            self.model_table = CTkTable(model_frame, width=1, height=1, values=hourly_data_month_day_saved_table_tp.values.tolist(), anchor="w", font = customtkinter.CTkFont(family="Roboto Flex", size=10))
            self.model_table.grid(row=1, column=0, padx=20, pady=20, sticky = "ew")
            
            
            
            
            #os.remove(plot_svg)
            
            
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        
        
        selected_date = self.calendar.get_date()

        selected_date_datetime = datetime.strptime(selected_date, '%m/%d/%y')
        
        for widget in self.model_1_frame.winfo_children():
            if (widget.winfo_exists()):
                widget.destroy()  # deleting widget
        
        
        
        print(selected_date_datetime.year)
        print(selected_date_datetime.month)
        print(selected_date_datetime.day)

        try:
            fsa_chosen = fsa_chosen_option_menu
        except NameError:
            fsa_chosen = "L9G"
        
        try:
            num_of_days = int(number_of_days_chosen_option_menu)
        except NameError:
            num_of_days = 1

        dirs_hourly_consumption_demand = os.path.join(dirs_inputs, "Hourly_Demand_Data")
        
        ###############################################################################
        # Import Models
        ###############################################################################

        # Load model from gui_pickup folder using joblib
        pipe_saved = joblib.load(os.path.join(model_path, "ridge_regression_model.pkl"))
        

        
        hourly_data_month_day_saved = pd.DataFrame(columns = ['HOUR', 'TOTAL_CONSUMPTION'])
        Y_pred_denorm_saved_df_saved = pd.DataFrame(columns = ['TOTAL_CONSUMPTION'])
        hourly_data_month_day_error = pd.DataFrame()
        
        
        for day_num in range (num_of_days):
            
            # new date
            new_date = selected_date_datetime + timedelta(days=day_num) 
            print(new_date) 
            try:
                year = str(new_date.year)
            except NameError:
                year = "2024"
                
            try:
                month = months[(new_date.month-1)]
            except NameError:
                month = "01"
                
            try:
                day = str(new_date.day)
            except NameError:
                day = "01"
            
            # Import saved CSV into script as dataframes
            X_test = pd.read_csv(os.path.join(x_y_input_path, "X_transformed_with_origCalVariables.csv"))
            X_test = X_test[X_test["Year"] == int(year)]
            X_test = X_test[X_test["Month"] == int(month)]
            X_test = X_test[X_test["Day"] == int(day)]
            
            # Predict using loaded model
            Y_pred_saved = pipe_saved.predict(X_test)
            
            # Ensure Y_pred and Y_test are reshaped correctly
            Y_pred_saved = Y_pred_saved.reshape(-1, 1)
            
            # Denormalize Y_pred and Y_test with min_max_scaler_y.pkl using joblib
            scaler_path = os.path.join(model_path, "min_max_scaler_y.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            scaler = joblib.load(scaler_path)
            
            # Denormalize Y_pred_saved and Y_test with min_max_scaler.pkl
            Y_pred_denorm_saved = scaler.inverse_transform(Y_pred_saved)
            Y_pred_denorm_saved_df = pd.DataFrame(Y_pred_denorm_saved, columns=['TOTAL_CONSUMPTION'])
            
            
            
            ###############################################################################
            # Dictionary for reading in hourly consumption by FSA
            ###############################################################################
            # FSA -> Year -> Month -> Value
            hourly_consumption_data_dic_by_month = pd.DataFrame()
            
            # Initialize dataframes to be used
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
            hourly_data_month_day = hourly_consumption_data_dic_by_month[hourly_consumption_data_dic_by_month['DAY'] == int(day)]


            
            
            # Append next day to another dataframe to plot on same figure
            if (day_num == 0):
                title = "Hourly Power Consumption Beginning: " + year + "/" + month + "/" + day
                hourly_data_month_day_saved = hourly_data_month_day[["HOUR", "TOTAL_CONSUMPTION"]].copy()
                Y_pred_denorm_saved_df_saved = Y_pred_denorm_saved_df.copy()
                hourly_data_month_day_error["HOUR_NEW"] = hourly_data_month_day.index + 1
                hourly_data_month_day_error = hourly_data_month_day_error.rename(columns={"HOUR_NEW": "Hour"})
            else:
                hourly_data_month_day_saved = pd.concat([hourly_data_month_day_saved, hourly_data_month_day], axis=0, ignore_index=True)
                Y_pred_denorm_saved_df_saved = pd.concat([Y_pred_denorm_saved_df_saved, Y_pred_denorm_saved_df], axis=0, ignore_index=True)
            
            
            # Find Error
            hourly_data_month_day_error["Actual Consumption: Day " + str(day_num+1) + " (KW)"] = hourly_data_month_day["TOTAL_CONSUMPTION"].reset_index(drop = True)
            hourly_data_month_day_error["Predicted Consumption: Day " + str(day_num+1) + " (KW)"] = Y_pred_denorm_saved_df["TOTAL_CONSUMPTION"].reset_index(drop = True)
            hourly_data_month_day_error["Error: Day " + str(day_num+1) + " (KW)"] = abs(hourly_data_month_day["TOTAL_CONSUMPTION"].reset_index(drop = True) -  Y_pred_denorm_saved_df["TOTAL_CONSUMPTION"].reset_index(drop = True))
            hourly_data_month_day_error = hourly_data_month_day_error.round(decimals = 1)
            
            
       
        hourly_data_month_day_error_columns = pd.DataFrame([hourly_data_month_day_error.columns], columns = hourly_data_month_day_error.columns)
        hourly_data_month_day_error = pd.concat([hourly_data_month_day_error_columns, hourly_data_month_day_error.iloc[0:]]).reset_index(drop=True)

        hourly_data_month_day_saved["HOUR_NEW"] = hourly_data_month_day_saved.index + 1
        hourly_data_month_day_saved["HOUR_NEW"] = hourly_data_month_day_saved["HOUR_NEW"].astype(int)
        

        plot_figures_model(self, hourly_data_month_day_saved, Y_pred_denorm_saved_df_saved, hourly_data_month_day_error, self.model_1_frame)
    
        
    def train_models_button_event(self):
        nest_asyncio.apply() # Apply nest_asyncio to allow for nested asyncio operations
        
        fsa_typed = self.fsa_search_bar.get()
        
        print(fsa_typed)
        print("checkbox frame modified: ", selected_features)
        
        
        # Set Up data direcotry path for data collection
        data_path = run_student[0]
        target_dir = data_path + "/raw_data"
        

        ### Calling Data ###
        # Choose FSA for data collection + Get latitude and longitude of chosen fsa
        fsa = "M2M"
        lat = realDataGetter.fsa_map[fsa_typed]["lat"]
        lon = realDataGetter.fsa_map[fsa_typed]["lon"]

        # Choose date range for data collection
        start_year = 2018
        start_month = 1
        start_day = 1
        start_hour = 0

        end_year = 2023
        end_month = 12
        end_day = 31
        end_hour = 23

        # Making datetime objects for start and end dates
        start_date = datetime(start_year, start_month, start_day, start_hour,0,0)
        end_date = datetime(end_year, end_month, end_day, end_hour,0,0)

        # Collect data - Using asynchronous functions
        weather_data, power_data = asyncio.run(realDataGetter.get_data_for_time_range(dirs_inputs, start_date, end_date, fsa_typed, lat, lon))

        weather_data.to_csv(f'{hourly_demand}/weather_data_{fsa_typed}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_prevPowerAnal.csv', index=False)
        power_data.to_csv(f'{hourly_demand}/power_data_{fsa_typed}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_prevPowerAnal.csv', index=False)

        
        
        
                
        
        
        
        
    ###############################################################################
    # Functions when using dropdown menus
    ###############################################################################
    def fsa_option_menu_event(self, choice):
        global fsa_chosen_option_menu
        fsa_chosen_option_menu = choice
        #print("optionmenu dropdown clicked:", choice)
        
    def year_option_menu_event(self, choice):
        global year_chosen_option_menu
        year_chosen_option_menu = choice
        #print("optionmenu dropdown clicked:", choice)
        
    def month_option_menu_event(self, choice):
        global month_chosen_option_menu
        month_chosen_option_menu = choice
        #print("optionmenu dropdown clicked:", choice)

    def day_option_menu_event(self, choice):
        global day_chosen_option_menu
        day_chosen_option_menu = choice
        #print("optionmenu dropdown clicked:", choice)
        
    def number_of_days_option_menu_event(self, choice):
        global number_of_days_chosen_option_menu
        number_of_days_chosen_option_menu = choice
        #print("optionmenu dropdown clicked:", choice)
    
    def features_checkbox_event(self):
        global selected_features
        selected_features = self.scrollable_features_checkbox_frame.get_checked_items()
        #print("checkbox frame modified: ", selected_features)
    
    
    ###############################################################################
    # Function to animate text
    ###############################################################################
    def animate_fade(self):
        """Animate the fade in/out effect"""
        if self.fading_in:
            self.alpha += 0.009
            if self.alpha >= 1:
                self.fading_in = False
                self.start_frame.after(200000000, self.animate_fade)  # Wait 2 seconds before fading out
                return
        else:
            self.alpha -= 0.05
            if self.alpha <= 0:
                self.fading_in = True
                self.start_frame.after(100, self.animate_fade)  # Wait 1 second before fading in again
                return
            
    def print_calendar_size(self, event=None):
        bbox = self.calendar.bbox("1.0")
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        print(f"Calendar size - Width: {width}, Height: {height}")
        
    ###############################################################################
    # Function for Hamburger menu!
    ###############################################################################
    def toggle_navigation(self):
        if self.navigation_visible:
            self.navigation_frame.grid_forget()
        else:
            self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_visible = not self.navigation_visible
    
    
    
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
    run_student = joseph_laptop_run
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