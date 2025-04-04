# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:18:11 2025

@author: BV05 Power Forecasting
"""

import warnings
# Do not print any warnings to screen
warnings.filterwarnings('ignore')

import Power_Forecasting_dataCollectionAndPreprocessingFlow
import Power_Forecasting_KNN_Saver
import Power_Forecasting_LR_Saver
import Power_Forecasting_XGB_Saver
import Power_Forecasting_CNN_Saver
try:
    import Power_Forecasting_Corsair_RGB
    from pyrgbdev import Corsair
except:
    pass
import asyncio
import aiohttp 
import nest_asyncio 
import customtkinter
from CTkTable import *
import tkinter as Tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkcalendar import Calendar
import os
import glob
import PIL
from PIL import Image
import pandas as pd
import datetime
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xlsxwriter
import numpy as np
import canada_holiday
from keras import models
import time
import threading
import joblib
import gc
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


#%%  Code for Initalization of scollable frame
class ScrollableCheckBoxFrame(customtkinter.CTkScrollableFrame):

    def __init__(self, master, item_list, command=None, **kwargs):
        super().__init__(master, **kwargs)

        self.command = command
        self.checkbox_list = []
        for i, item in enumerate(item_list):
            self.add_item(item)

    def add_item(self, item):
        
        checkbox = customtkinter.CTkCheckBox(self, text=item, variable = customtkinter.IntVar(value = 1), onvalue = 1, offvalue = 0, checkmark_color="#14206d", font = customtkinter.CTkFont(family="Roboto Condensed", size=16))
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

#%% Code for Initalization of GUI application
class App(customtkinter.CTk):  
    def __init__(self):  
        ###############################################################################
        # All file paths
        ###############################################################################
        # Global Variables
        global save_results_dic
        save_results_dic = {}
        global model_names_list, selected_models, options_list, fsa_predict_list
        global selected_features
        
        # All file path locations
        global image_path, background_images_path, saved_model_path, x_y_input_path, power_weather_data_path, input_excel_path, output_results_path
        image_path = os.path.join(dirs_inputs, "Model_Plots") 
        saved_model_path = os.path.join(dirs_inputs, "Saved_Models")    
        x_y_input_path = os.path.join(dirs_inputs, "X_Y_Inputs")   
        background_images_path = os.path.join(dirs_inputs, "GUI_Background_Images") 
        power_weather_data_path = os.path.join(dirs_inputs, "Power_Weather_Data")
        input_excel_path = os.path.join(dirs_inputs, "Input_Data_Excel")
        output_results_path = os.path.join(dirs_inputs, "Output_Results")
        fsa_conversion_path = os.path.join(dirs_inputs, "Conversion_FSA_LatLong")
        
        
        ###############################################################################
        # Initialize GUI
        ###############################################################################
            
        customtkinter.set_default_color_theme("blue") # change the colour theme of the application
        super().__init__()    
        self.title("Power Forecasting")
        self.after(1, self.wm_state, 'zoomed') 
        
        
        # Set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Path for Icon
        icon_path = os.path.join(background_images_path, "Power_Forecasting_Icon.ico")
        
        # Set Icon for GUI
        self.iconbitmap(icon_path)
        
        # Path for start menu background
        start_menu_image_path = os.path.join(background_images_path, "Start_Menu_Page.png")
        
        # Create background image
        image = PIL.Image.open(start_menu_image_path)
        background_image = customtkinter.CTkImage(image, size=(1920, 1080))
        
        # Initialize Corsair RGB
        try:
            global rgb_lights    
            rgb_lights = Corsair.sdk()
        except:
            pass
        
        # Get FSA Map
        fsa_map_path = os.path.join(fsa_conversion_path, "ontario_fsas.csv")
        global fsa_map
        fsa_map = Power_Forecasting_dataCollectionAndPreprocessingFlow.setup_fsa_map(fsa_map_path)
        
        global cnn_days_back
        # Get X dates behind for CNN model Lags
        cnn_days_back = 2
        
        
        ###############################################################################
        # Create Start Frame (all code for desired frame is in here)
        ###############################################################################
        
        self.start_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.start_frame.grid_columnconfigure(0, weight=2)
        self.start_frame.grid_columnconfigure(1, weight=2)
        self.start_frame.grid_columnconfigure(2, weight=1)
        self.start_frame.grid_columnconfigure(3, weight=1)
        self.start_frame.grid_columnconfigure(4, weight=2)
        self.start_frame.grid_columnconfigure(5, weight=2)
        self.start_frame.grid_rowconfigure(0, weight=1)
        
        # Create background label for start frame
        self.background_label = customtkinter.CTkLabel(self.start_frame, 
                                                     image=background_image,
                                                     text="")  # Empty text
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        customtkinter.set_appearance_mode("dark")
        
        # Create desired font
        my_font = customtkinter.CTkFont(family="Roboto", size=40, 
                                        weight="bold", slant="italic", underline=False, overstrike=False) #font to be used for titles       
        
        # Create start button to go to main menu
        self.start_button = customtkinter.CTkButton(self.start_frame, text="Start ", command=self.start_button_event, height=85, width=250, font=my_font, corner_radius=40, bg_color="#0f0f39", hover_color="#560067")
        self.start_button.grid(row = 0, column = 2, padx = 100, pady = (0, 175), columnspan = 2, sticky = "sew")
    
        
        ###############################################################################
        # Create Navigation Frame (all code for desired frame is in here)
        ###############################################################################
        
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0, bg_color="#34495E",fg_color="#05122d", height = 50)
        self.navigation_frame.grid_rowconfigure(7, weight=1)
        
        global my_text_font
        my_text_font = customtkinter.CTkFont(family="Roboto Condensed", size=16)
        
        # Create all labels and buttons on navigation frame
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="        BV05 -- Power Forecasting", 
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                    anchor="w", font = my_text_font, command=self.home_button_event)
        self.home_button.grid(row=1, column=0, padx = (0, 2), sticky="ew")

        self.model_1_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Linear Regression",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                       anchor="w", font = my_text_font, command=self.model_1_button_event)
        self.model_1_button.grid(row=2, column=0, padx = (0, 2), sticky="ew")

        self.model_2_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="X Gradient Boost",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", font = my_text_font, command=self.model_2_button_event)
        self.model_2_button.grid(row=3, column=0, padx = (0, 2), sticky="ew")
        
        self.model_3_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="K-Nearest Neighbors",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                       anchor="w", font = my_text_font, command=self.model_3_button_event)
        self.model_3_button.grid(row=4, column=0, padx = (0, 2), sticky="ew")

        self.model_4_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Convolutional Neural Network",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", font = my_text_font, command=self.model_4_button_event)
        self.model_4_button.grid(row=5, column=0, padx = (0, 2), sticky="ew")
        
        self.summary_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Summary",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", font = my_text_font, command=self.summary_button_event)
        self.summary_button.grid(row=6, column=0, padx = (0, 2), sticky="ew")
             
        # Track the visibility of the navigation frame
        self.navigation_visible = False  

        # Initially hide the navigation frame
        self.toggle_navigation()


        ###############################################################################
        # Create Home Frame (all code for desired frame is in here)
        ###############################################################################
        
        ###############################################################################
        # Initial Setup
        
        # Create frame
        home_menu_image_path = os.path.join(background_images_path, "Home_Page.png")
        
        # Create home background image
        image = PIL.Image.open(home_menu_image_path)
        background_image_home = customtkinter.CTkImage(image, size=(1920, 1080))
        
        # Create frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=10)
        self.home_frame.grid_columnconfigure(1, weight=1)
        self.home_frame.grid_columnconfigure(2, weight=1)
        self.home_frame.grid_columnconfigure(3, weight=1)
        self.home_frame.grid_columnconfigure(4, weight=1)
        self.home_frame.grid_columnconfigure(5, weight=10)


        # Insert background image
        self.background_label = customtkinter.CTkLabel(self.home_frame,
                                                     image=background_image_home,
                                                     text="")  # Empty text
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Fonts
        global my_button_font, padding_x, padding_x_option1, padding_x_option2, padding_x_option3
        my_text_font = customtkinter.CTkFont(family="Roboto Condensed", size=16)
        my_button_font = customtkinter.CTkFont(family="Roboto Condensed", size=18, weight="bold")
        my_title_font = customtkinter.CTkFont(family="Roboto Condensed", size=30)
        my_textbox_font = customtkinter.CTkFont(family="Roboto Condensed", size=18)
        
        # Set X padding for all frames
        padding_x = 10
        padding_x_option1 = 0
        padding_x_option2 = 20
        padding_x_option3 = 30
        pad_home = 50
        
        # Create main title and description
        self.home_frame_Label_Title = customtkinter.CTkLabel(self.home_frame, text="Welcome to Power Forecasting!", font=customtkinter.CTkFont(family="RobotoCondensed-ExtraBoldItalic", size=50, weight="bold", slant = "italic"), 
                                                             bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Title.grid(row=0, column=0, padx = padding_x, pady = 20, columnspan=6)
        
        # Get FSA List
        fsa_predict_list = ["No Models"]
        for fsa_str in os.listdir(saved_model_path):
            if ("power_scaler" in fsa_str):
                if ("power_scaler_Input" not in fsa_str):
                    fsa_str = os.path.basename(fsa_str)
                    fsa_str = os.path.splitext(fsa_str)
                    fsa_str = fsa_str[0]
                    fsa_str = fsa_str[-3:]
                    if "No Models" in fsa_predict_list:
                        fsa_predict_list.remove("No Models")
                    fsa_predict_list.append(fsa_str)
        fsa_predict_list = list(dict.fromkeys(fsa_predict_list))
                    
        # Create selected option frames
        self.option1_frame = customtkinter.CTkFrame(self.home_frame, fg_color = '#05122d', bg_color = '#05122d')
        self.option2_frame = customtkinter.CTkFrame(self.home_frame, fg_color = '#05122d',bg_color = '#05122d')
        self.option3_frame = customtkinter.CTkFrame(self.home_frame, fg_color = '#05122d', bg_color = '#05122d')
        for frame in [self.option1_frame, self.option2_frame, self.option3_frame]:
            frame.grid(row=4, column=0, rowspan=10, columnspan=6, sticky="nsew", padx=200, pady=60)
            frame.grid_columnconfigure(0,weight=5)
            frame.grid_columnconfigure(1,weight=1)
            frame.grid_columnconfigure(2,weight=1)
            frame.grid_columnconfigure(3,weight=1)
            frame.grid_columnconfigure(4,weight=5)
            
            frame.rowconfigure(0,weight=1)
            frame.rowconfigure(1,weight=1)
            frame.rowconfigure(2,weight=1)
            frame.rowconfigure(3,weight=1)
            frame.rowconfigure(4,weight=1)
            frame.rowconfigure(5,weight=1)
            frame.rowconfigure(6,weight=1)
            frame.grid_remove()
        
        # Create titles for selecting models, features, and option
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Select models to train and forecast, and corresponding features.", font=my_title_font,
            bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=1, column=0, padx = padding_x, pady = (40, 10), columnspan=6)
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Select Models to Train and Forecast", font=customtkinter.CTkFont(family="Roboto Flex", size=20, weight="bold"), bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=2, column=1, padx = padding_x,  pady = (0,1), sticky = "ew")
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Select Training Features", font=customtkinter.CTkFont(family="Roboto Flex", size=20, weight="bold"), bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=2, column=2, padx = padding_x, pady = (0,1), sticky = "ew")

        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Select CNN Hyperparameters", font=customtkinter.CTkFont(family="Roboto Flex", size=20, weight="bold"), bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=2, column=3, padx = padding_x, pady = (0,1), sticky = "ew")

        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.home_frame, text="Select Option", font=customtkinter.CTkFont(family="Roboto Flex", size=20, weight="bold"), bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=2, column=4, padx = padding_x, pady = (0,1), sticky = "ew")
        
        # Create scrollable check box of models
        model_names_list = ["Linear Regression", "X Gradient Boost", "K-Nearest Neighbors", "Convolutional Neural Network"]
        selected_models = model_names_list
        self.scrollable_models_checkbox_frame = ScrollableCheckBoxFrame(self.home_frame, height = 130, width=130, command=self.models_checkbox_event,
                                                         item_list=model_names_list, 
                                                         fg_color="#05122d",
                                                         bg_color= "#05122d")
        self.scrollable_models_checkbox_frame.grid(row=3, column=1, padx = padding_x, pady = (10, 0), sticky = "ew")
        self.scrollable_models_checkbox_frame._scrollbar.configure(height=0)
        
        # Create scrollable check box of features 
        column_names = pd.read_csv(os.path.join(x_y_input_path, "Features_Column_Template.csv"), nrows = 0)
        selected_features = column_names.columns.tolist()
        
        # Create 3 digit subset of string for saving models
        global selected_features_3_digits
        selected_features_3_digits = []
        for selected_features_str in selected_features:
            if selected_features_str == "Wind Speed":
                selected_features_3_digits.append("Wsd")
            elif selected_features_str == "Windchill":
                selected_features_3_digits.append("Wch")
            else:
                selected_features_3_digits.append(selected_features_str[:3])
                
        self.scrollable_features_checkbox_frame = ScrollableCheckBoxFrame(self.home_frame, height = 130, width=150, command=self.features_checkbox_event,
                                                         item_list=column_names.columns, 
                                                         fg_color="#05122d",
                                                         bg_color= "#05122d")
        self.scrollable_features_checkbox_frame.grid(row=3, column=2, padx = padding_x, pady = (10, 0), sticky = "ew")
        self.scrollable_features_checkbox_frame._scrollbar.configure(height=0)
        
        # Create open excel file button for CNN parameters
        self.upload_open_cnn_param_button = customtkinter.CTkButton(self.home_frame, corner_radius=20, height=40, border_spacing=10, text="Modify and Save Hyperparameters",
                                                      bg_color="#05122d",
                                                      fg_color="#14206d",
                                                      hover_color="#560067",
                                                      text_color=("gray10", "gray90"),
                                                      font = my_button_font,
                                                      anchor="center", command=self.upload_open_cnn_param_button_event)
        self.upload_open_cnn_param_button.grid(row=3, column=3, padx = padding_x, pady = (10, 0), sticky = "new")
        
        # Create drop down menu for options
        options_list = ["Forecast Ontario Models", "Train Ontario Models", "Train and Forecast Provided Dataset Models"]
        self.options_dropdown_menu = customtkinter.CTkOptionMenu(self.home_frame, values=options_list, command=self.show_frame_based_on_option,
            fg_color="#14206d", button_color="#14206d", dropdown_fg_color="#05122d", bg_color="#05122d", font=my_text_font)
        self.options_dropdown_menu.set(options_list[0])
        self.options_dropdown_menu.grid(row=3, column=4, padx=padding_x, pady = (10, 0), sticky ='new')
        
        self.show_frame_based_on_option(options_list[0])
        
        ###############################################################################
        # Create Option 1 (Predict Ontario Models) Widgets
        
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.option1_frame, text="Forecast using saved Ontario located models.", font=my_title_font,
            bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=0, column=1, padx = padding_x_option1, pady = (10, 40), columnspan=3)
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.option1_frame, text="FSA", font=customtkinter.CTkFont(family="Roboto Flex", size=20, weight="bold"), bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=1, column=2, padx = padding_x_option1, sticky = "ew")

        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.option1_frame, text="Number of Days", font=customtkinter.CTkFont(family="Roboto Flex", size=20, weight="bold"), bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=3, column=2, padx = padding_x_option1, sticky = "ew")
        
        # FSA (LOOK IN OPTION 1 FUNCTION)
        
        # Calendar
        # FOR buttons on the calendar
        style = ttk.Style()
        style.theme_use('default')

        #Configure button styles for the calendar
        style.configure('TButton', background='#14206d', foreground='#ffffff')  # Blue background, white text
        style.map('TButton',
          background=[('active', '#8A2BE2')],  # Violet background when the button is active (hovered)
          foreground=[('active', '#ffffff')],  # White text when the button is active (hovered)
          relief=[('pressed', 'groove'), ('!pressed', 'ridge')])

        # Set minimum date
        min_date = date(2024, 11, 1)
        
        # Set maximum date
        max_date = date(2024, 11, 28)
        
        self.calendar = Calendar(self.option1_frame, selectmode='day', year=min_date.year, month=min_date.month, day=min_date.day, mindate = min_date, maxdate = max_date,
                   background='#14206d',  # Dark Blue Background
                    foreground='#FFFFFF',  # White Text
                    selectbackground='#05122d',  # Turquoise for Selected Date
                    selectforeground='#FFFFFF',  # White Text for Selected Date    
                    othermonthbackground='#BDC3C7',  # Light Gray for Other Month's Days
                    othermonthwebackground='#95A5A6',  # Darker Gray for Other Month's Weekends
                    othermonthforeground='#7F8C8D',  # Medium Gray Text for Other Month's Days
                    hover_color=("gray70", "gray30"),
                    othermonthweforeground='#7F8C8D')
        self.calendar.grid(row=1, column=1, padx = padding_x_option1, sticky = "nsew", rowspan = 4)
        
        # Number of Days
        self.home_frame_number_of_days_option_menu = customtkinter.CTkOptionMenu(self.option1_frame, values=["1", "2", "3"], command = self.number_of_days_option_menu_event,
            fg_color="#14206d",
            button_color="#14206d",
            dropdown_fg_color="#05122d",
            bg_color="#05122d")
        self.home_frame_number_of_days_option_menu.set("1")
        self.home_frame_number_of_days_option_menu.grid(row=4, column=2, padx = padding_x_option1, sticky ='n')

        
        
        # Create Show Detailed Table Check Box
        self.detailed_table_checkbox_var = customtkinter.IntVar(value = 0)
        self.detailed_table_checkbox = customtkinter.CTkCheckBox(self.option1_frame, text="Show Detailed Table",
                                                      text_color=("gray10", "gray90"), variable = self.detailed_table_checkbox_var, onvalue = 1, offvalue = 0, 
                                                      checkmark_color="#14206d",  
                                                      bg_color= "#05122d",
                                                      command = self.show_table_checkbox_event)
        self.detailed_table_checkbox.grid(row=3, column=3, padx = padding_x_option1, pady = (20, 0), sticky = "new")
        
        # Create Generate Forecasts Button
        self.generate_models_button = customtkinter.CTkButton(self.option1_frame, corner_radius=20, height=40, border_spacing=10, text="Generate Forecasts",
                                                      text_color=("gray10", "gray90"),
                                                      fg_color="#14206d",  
                                                      hover_color="#560067", 
                                                      bg_color= "#05122d",
                                                      anchor="center", command=self.predict_models_button_event, font=my_button_font)
        self.generate_models_button.grid(row=2, column=3, padx = padding_x_option1, sticky = "ew")
        
        # Create textbox 1 for guide
        
        # Create title for textbox
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.option1_frame, text="Guide:", font=customtkinter.CTkFont(family="Roboto Flex", size=20, weight="bold"), bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=5, column=1, padx = padding_x_option1, pady = (20, 0), columnspan=3, sticky = "w")
        
        
        # Create textbox
        self.textbox_1 = customtkinter.CTkTextbox(self.option1_frame, height = 100, 
                                                  font=my_textbox_font,
                                                  fg_color = '#05122d')
        self.textbox_1.grid(row=6, column=1, columnspan=3, padx = padding_x_option1,  sticky = "ew")
        
        # Last line is first line in GUI
        self.textbox_1.insert("0.0", "Once complete, forecast for each model will appear. If model is selected and forecast does \nnot show up, then there is no saved model for this FSA and combination of input features. To\nsee forecast of selected model, please train desired model.")
        self.textbox_1.insert("0.0", " \n")
        self.textbox_1.insert("0.0", "Loading screen may be unresponsive while forecasting. This is meant to happen.\n")
        self.textbox_1.insert("0.0", "\n")
        self.textbox_1.insert("0.0", "FSA stands for Forward Sortation Area (i.e. first three characters of postal code).\n")
        
        ###############################################################################
        # Create Option 2 (Train Ontario Models) Widgets
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.option2_frame, text="Train models with ANY postal code in Ontario.", font=my_title_font,
            bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=0, column=1, padx = padding_x_option2, pady = (10, 10), columnspan=3)
   
        # Create search bar for FSA
        self.fsa_search_bar = customtkinter.CTkEntry(self.option2_frame, placeholder_text ="Enter first three characters of postal code (ex. 'LOH', 'M5B', etc.)",
                                fg_color="#14206d",  # Foreground color (entry background)
                              bg_color="#05122d",  # Background color (frame background)
                              text_color="#ffffff",  # Text color
                              font=my_text_font)
        self.fsa_search_bar.grid(row=1, column=2, padx = padding_x_option2, pady = (10, 10), sticky = "new")
        
        # Create Train button
        self.train_models_button = customtkinter.CTkButton(self.option2_frame, corner_radius=20, height=40, border_spacing=10, text="Train Models",
                                                      fg_color="#14206d",  
                                                      bg_color= "#05122d",hover_color="#560067",
                                                      text_color=("gray10", "gray90"),
                                                      font = my_button_font,
                                                      anchor="center", command=self.train_models_button_event)
        self.train_models_button.grid(row=2, column=2, padx = padding_x_option2, pady = (10, 10), sticky = "new")
        
        # Create progress bar for training Ontario data
        self.progress_bar_train_ontario = customtkinter.CTkProgressBar(self.option2_frame, width=300, fg_color="#14206d")
        self.progress_bar_train_ontario.grid(row=3, column=2, padx = padding_x_option2, pady=0, sticky="new")
        self.progress_bar_train_ontario.set(0)  # Initialize the progress bar to 0
        
        # Create textbox 2 for guide
        
        # Create title for textbox
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.option2_frame, text="Guide:", font=customtkinter.CTkFont(family="Roboto Flex", size=20, weight="bold"), bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=4, column=1, padx = padding_x_option1, columnspan = 3, pady = (20, 0), sticky = "w")
        
        
        # Create textbox
        self.textbox_2 = customtkinter.CTkTextbox(self.option2_frame, height = 200, 
                                                  font=my_textbox_font,
                                                  fg_color = '#05122d')
        self.textbox_2.grid(row=5, column=1, columnspan=3, padx = padding_x_option1,  sticky = "ew")
        
        # Last line is first line in GUI
        self.textbox_2.insert("0.0", "            -May be a problem with Environment Canada Weather API server. Please close program\n             and try again.")
        self.textbox_2.insert("0.0", " \n")
        self.textbox_2.insert("0.0", "            -FSA is not registered within the program (FSA in Ontario changes every few years).\n")
        self.textbox_2.insert("0.0", " \n")
        self.textbox_2.insert("0.0", "    2. If typed FSA is valid there are two cases:\n")
        self.textbox_2.insert("0.0", " \n")
        self.textbox_2.insert("0.0", "    1. Ensure Correct FSA.\n")
        self.textbox_2.insert("0.0", " \n")
        self.textbox_2.insert("0.0", "If models fail to train (loading bar is not filled):\n")
        self.textbox_2.insert("0.0", " \n")
        self.textbox_2.insert("0.0", "Loading screen may be unresponsive while training. This is meant to happen.\n")
        
        ###############################################################################
        # Create Option 3 (Train and Forecaste Provided Dataset Models) Widgets
        
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.option3_frame, text="Train and forecast models with ANY input dataset.", font=my_title_font,
            bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=0, column=1, padx = padding_x_option3, pady = (10, 10), columnspan=3)
        
        # Open excel file template
        self.open_file_button = customtkinter.CTkButton(self.option3_frame, corner_radius=20, height=40, border_spacing=10, text="Open Template",
                                                      bg_color='#05122d',
                                                      fg_color="#4B0082",
                                                      hover_color="#560067",
                                                      text_color=("gray10", "gray90"),
                                                      font = my_button_font,
                                                      anchor="center", command=self.open_file_button_event)
        self.open_file_button.grid(row=1, column=1, padx = padding_x_option3, pady = 20, sticky = "new")
        
        # Upload excel file template
        self.upload_file_button = customtkinter.CTkButton(self.option3_frame, corner_radius=20, height=40, border_spacing=10, text="Upload Saved File",
                                                      bg_color='#05122d',
                                                      fg_color="#4B0082",
                                                      hover_color="#560067",
                                                      text_color=("gray10", "gray90"),
                                                      font = my_button_font,
                                                      anchor="center", command=self.upload_file_button_event)
        self.upload_file_button.grid(row=1, column=3, padx = padding_x_option3, pady = 20, sticky = "new")
        
        # Create Train button
        self.train_models_button_any = customtkinter.CTkButton(self.option3_frame, corner_radius=20, height=40, border_spacing=10, text="Train Models",
                                                      fg_color="#14206d",
                                                      bg_color= "#05122d",
                                                      hover_color="#560067",
                                                      text_color=("gray10", "gray90"),
                                                      font = my_button_font,
                                                      anchor="center", command=self.train_input_excel_models_button_event)
        self.train_models_button_any.grid(row=2, column=1, padx = padding_x_option3, pady = 20, sticky = "sew")
        
        # Create progress bar for training Ontario data
        self.progress_bar_train_any = customtkinter.CTkProgressBar(self.option3_frame, width=200, fg_color="#14206d")
        self.progress_bar_train_any.grid(row=3, column=1, padx = padding_x_option3, pady=0, sticky="new")
        self.progress_bar_train_any.set(0)  # Initialize the progress bar to 0
        
        # Create Show Detailed Table Check Box
        self.detailed_table_checkbox_var_excel = customtkinter.IntVar(value = 0)
        self.detailed_table_checkbox_excel = customtkinter.CTkCheckBox(self.option3_frame, text="Show Detailed Table",
                                                      text_color=("gray10", "gray90"), variable = self.detailed_table_checkbox_var_excel, onvalue = 1, offvalue = 0, 
                                                      checkmark_color="#14206d",  
                                                      bg_color= "#05122d",
                                                      command = self.show_table_checkbox_event)
        self.detailed_table_checkbox_excel.grid(row=3, column=3, padx = padding_x_option3, pady = 0, sticky = "new")
        
        # Create Predict button
        self.predict_models_button = customtkinter.CTkButton(self.option3_frame, corner_radius=20, height=40, border_spacing=10, text="Forecast Models",
                                                      fg_color="#14206d",
                                                      bg_color= "#05122d",
                                                      hover_color="#560067",
                                                      text_color=("gray10", "gray90"),
                                                      font = my_button_font,
                                                      anchor="center", command=self.predict_input_excel_models_button_event)
        self.predict_models_button.grid(row=2, column=3, padx = padding_x_option3, pady = 20, sticky = "sew")
 
        # Create textbox 3 for guide
        
        # Create title for textbox
        self.home_frame_Label_Selection = customtkinter.CTkLabel(self.option3_frame, text="Guide:", font=customtkinter.CTkFont(family="Roboto Flex", size=20, weight="bold"), bg_color='#05122d', text_color=("white"))
        self.home_frame_Label_Selection.grid(row=4, column=1, padx = padding_x_option1, columnspan = 3, pady = (20, 0), sticky = "w")
        
        
        # Create textbox
        self.textbox_3 = customtkinter.CTkTextbox(self.option3_frame, height = 175, 
                                                  font=my_textbox_font,
                                                  fg_color = '#05122d')
        self.textbox_3.grid(row=5, column=1, columnspan=3, padx = padding_x_option1,  sticky = "ew")
        
        # Last line is first line in GUI
        self.textbox_3.insert("0.0", "Loading screen may be unresponsive while training or predicting. This is meant to happen.")
        self.textbox_3.insert("0.0", " \n")
        self.textbox_3.insert("0.0", "5. Click “Forecast Models” button to generate forecasts with saved models.\n")
        self.textbox_3.insert("0.0", " \n")
        self.textbox_3.insert("0.0", "4. Click “Train Models” button to train selected models with selected features.\n")
        self.textbox_3.insert("0.0", " \n")
        self.textbox_3.insert("0.0", "3. Upload the saved file.\n")
        self.textbox_3.insert("0.0", " \n")
        self.textbox_3.insert("0.0", "2. After inputting all data, save the file.\n")
        self.textbox_3.insert("0.0", " \n")
        self.textbox_3.insert("0.0", "1. Open template and navigate to “README” sheet for instructions.\n")
        ###############################################################################        
        # Create second frame (Model 1) (all code for desired frame is in here)
        ###############################################################################
        
        # Create frame
        model_menu_image_path = os.path.join(background_images_path, "Home_Page.png")
        
        # Create home background image
        image = PIL.Image.open(model_menu_image_path)
        background_image_model = customtkinter.CTkImage(image, size=(1920, 1080))
        
        
        self.model_1_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.model_1_frame.grid_columnconfigure(0, weight=1)
        self.model_1_frame.grid_columnconfigure(1, weight=1)
        self.model_1_frame.grid_rowconfigure(0, weight=1)
        self.model_1_frame.grid_rowconfigure(1, weight=1)
        self.model_1_frame.grid_rowconfigure(2, weight=1)
        self.model_1_frame.grid_rowconfigure(3, weight=1)
        self.model_1_frame.grid_rowconfigure(4, weight=1)
        
        self.background_label = customtkinter.CTkLabel(self.model_1_frame,
                                                     image=background_image_model,
                                                     text="")  # Empty text
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        ###############################################################################
        # Create third frame (Model 2) (all code for desired frame is in here)
        ###############################################################################
        self.model_2_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.model_2_frame.grid_columnconfigure(0, weight=1)
        self.model_2_frame.grid_columnconfigure(1, weight=1)
        self.model_2_frame.grid_rowconfigure(0, weight=1)
        self.model_2_frame.grid_rowconfigure(1, weight=1)
        self.model_2_frame.grid_rowconfigure(2, weight=1)
        self.model_2_frame.grid_rowconfigure(3, weight=1)
        self.model_2_frame.grid_rowconfigure(4, weight=1)
        
        
        self.background_label = customtkinter.CTkLabel(self.model_2_frame,
                                                     image=background_image_model,
                                                     text="")  # Empty text
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

     
        ###############################################################################
        # Create fourth frame (Model 3) (all code for desired frame is in here)
        ###############################################################################
        self.model_3_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.model_3_frame.grid_columnconfigure(0, weight=1)
        self.model_3_frame.grid_columnconfigure(1, weight=1)
        self.model_3_frame.grid_rowconfigure(0, weight=1)
        self.model_3_frame.grid_rowconfigure(1, weight=1)
        self.model_3_frame.grid_rowconfigure(2, weight=1)
        self.model_3_frame.grid_rowconfigure(3, weight=1)
        self.model_3_frame.grid_rowconfigure(4, weight=1)
        
        self.background_label = customtkinter.CTkLabel(self.model_3_frame,
                                                     image=background_image_model,
                                                     text="")  # Empty text
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)


        ###############################################################################
        # Create fifth frame (Model 4) (all code for desired frame is in here)
        ###############################################################################
        self.model_4_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.model_4_frame.grid_columnconfigure(0, weight=1)
        self.model_4_frame.grid_columnconfigure(1, weight=1)
        self.model_4_frame.grid_rowconfigure(0, weight=1)
        self.model_4_frame.grid_rowconfigure(1, weight=1)
        self.model_4_frame.grid_rowconfigure(2, weight=1)
        self.model_4_frame.grid_rowconfigure(3, weight=1)
        self.model_4_frame.grid_rowconfigure(4, weight=1)
        
        self.background_label = customtkinter.CTkLabel(self.model_4_frame,
                                                     image=background_image_model,
                                                     text="")  # Empty text
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        ###############################################################################
        # Create sixth frame (summary) (all code for desired frame is in here)
        ###############################################################################
        self.summary_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.summary_frame.grid_columnconfigure(0, weight=1)
        self.summary_frame.grid_rowconfigure(0, weight=1)
        self.summary_frame.grid_rowconfigure(1, weight=1)
        self.summary_frame.grid_rowconfigure(2, weight=1)
        
        self.background_label = customtkinter.CTkLabel(self.summary_frame,
                                                     image=background_image_model,
                                                     text="")  # Empty text
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        
        ###############################################################################
        # Select default frame
        ###############################################################################
        self.select_frame_by_name("Start")
        
    
    ###############################################################################
    # Function to select small frames (option menu frames) on the home menu
    ###############################################################################     
    def show_frame_based_on_option(self, option):
        if option == options_list[0]:
            self.option1_frame.grid()
            self.option2_frame.grid_remove()
            self.option3_frame.grid_remove()
            
            # Update FSA List for predicting
            self.home_frame_fsa_option_menu = customtkinter.CTkOptionMenu(self.option1_frame, values=fsa_predict_list, command = self.fsa_option_menu_event,
             fg_color="#14206d",button_color="#14206d",
             dropdown_fg_color="#05122d", 
             bg_color="#05122d", 
             font=my_text_font)
            if fsa_predict_list:
                self.home_frame_fsa_option_menu.set(fsa_predict_list[0])
            self.home_frame_fsa_option_menu.grid(row=2, column=2, padx = padding_x_option1, pady= 5,sticky = "n")
            
        elif option == options_list[1]:
            self.option1_frame.grid_remove()
            self.option2_frame.grid()
            self.option3_frame.grid_remove()
            self.home_frame_fsa_option_menu.grid_forget()
        elif option == options_list[2]:
            self.option1_frame.grid_remove()
            self.option2_frame.grid_remove()
            self.option3_frame.grid()
            self.home_frame_fsa_option_menu.grid_forget()
    
    
    ###############################################################################
    # Function to select different frames
    ###############################################################################
    def select_frame_by_name(self, name):
        # set button color for selected button
        self.start_button.configure(fg_color=("gray75", "gray25") if name == "Start" else "transparent")
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "Home" else "transparent")
        self.model_1_button.configure(fg_color=("gray75", "gray25") if name == "Model: Linear Regression" else "transparent")
        self.model_2_button.configure(fg_color=("gray75", "gray25") if name == "Model: X Gradient Boost" else "transparent")
        self.model_3_button.configure(fg_color=("gray75", "gray25") if name == "Model: K-Nearest Neighbors" else "transparent")
        self.model_4_button.configure(fg_color=("gray75", "gray25") if name == "Model: Convolutional Neural Network" else "transparent")
        self.summary_button.configure(fg_color=("gray75", "gray25") if name == "Summary" else "transparent")

        # show selected frame
        if name == "Start":
            self.start_frame.grid(row=0, column=1, sticky="nsew")
            self.navigation_frame.grid_forget()
        else:
            self.navigation_frame.grid(row=0, column=0, sticky="nsew")
            self.start_frame.grid_forget()
        if name == "Home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
            self.navigation_frame.grid_forget()
            self.navigation_visible = False
        else:
            self.home_frame.grid_forget()
        if name == "Model: Linear Regression":
            self.model_1_frame.grid(row=0, column=1, sticky="nsew")
            self.navigation_frame.grid_forget()
            self.navigation_visible = False
        else:
            self.model_1_frame.grid_forget()
        if name == "Model: X Gradient Boost":
            self.model_2_frame.grid(row=0, column=1, sticky="nsew")
            self.navigation_frame.grid_forget()
            self.navigation_visible = False
        else:
            self.model_2_frame.grid_forget()
        if name == "Model: K-Nearest Neighbors":
            self.model_3_frame.grid(row=0, column=1, sticky="nsew")
            self.navigation_frame.grid_forget()
            self.navigation_visible = False
        else:
            self.model_3_frame.grid_forget()
        if name == "Model: Convolutional Neural Network":
            self.model_4_frame.grid(row=0, column=1, sticky="nsew")
            self.navigation_frame.grid_forget()
            self.navigation_visible = False
        else:
            self.model_4_frame.grid_forget()
        if name == "Summary":
            self.summary_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.summary_frame.grid_forget()

    ###############################################################################
    # Functions when selecting buttons
    ###############################################################################
    def start_button_event(self):
        self.hamburger_button = customtkinter.CTkButton(self, text="☰", width=40, height=40, command=self.toggle_navigation, fg_color="#14206d")
        self.hamburger_button.place(x=10, y=10)
        self.select_frame_by_name("Home")
    
    def home_button_event(self):
        self.select_frame_by_name("Home")

    def model_1_button_event(self):
        self.select_frame_by_name("Model: Linear Regression")
        self.navigation_frame.grid_forget()
        self.navigation_visible = True
        self.toggle_navigation()

    def model_2_button_event(self):
        self.select_frame_by_name("Model: X Gradient Boost")
        self.navigation_frame.grid_forget()
        self.navigation_visible = True
        self.toggle_navigation()
        
    def model_3_button_event(self):
        self.select_frame_by_name("Model: K-Nearest Neighbors")
        self.navigation_frame.grid_forget()
        self.navigation_visible = True
        self.toggle_navigation()

    def model_4_button_event(self):
        self.select_frame_by_name("Model: Convolutional Neural Network")
        self.navigation_frame.grid_forget()
        self.navigation_visible = True
        self.toggle_navigation()
        
    def summary_button_event(self):
        self.select_frame_by_name("Summary")
        self.navigation_visible = False
        self.toggle_navigation()
       
    def predict_models_button_event(self):
        try:
            # Function to plot the model figures
            def plot_figures_model(self, hourly_data_month_day_saved, Y_pred_denorm_saved_df_saved, metrics_values, hourly_data_month_day_error, model_frame, model_event_next, model_event_back, model_name):
            
                
                if (model_frame != self.summary_frame):
                    model_menu_image_path = os.path.join(background_images_path, "Model_Page.png")
                else:
                    model_menu_image_path = os.path.join(background_images_path, "Home_Page.png")
                # Create background image
                image = PIL.Image.open(model_menu_image_path)
                background_image_model = customtkinter.CTkImage(image, size=(1920, 1080))
                
                
                self.background_label = customtkinter.CTkLabel(model_frame,
                                                             image=background_image_model,
                                                             text="")  # Empty text
                self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
             
                my_title_font = customtkinter.CTkFont(family="RobotoCondensed-ExtraBoldItalic", size=50, weight="bold", slant = "italic")
                self.model_frame_Label_Title = customtkinter.CTkLabel(model_frame, text=model_name, font=my_title_font, 
                                                                     bg_color='#05122d', text_color=("white"))
                self.model_frame_Label_Title.grid(row=0, column=0, padx=20, pady=(40, 10), columnspan=2, sticky = "n")  
    
                
                if (model_frame != self.summary_frame):
                    self.next_button = customtkinter.CTkButton(model_frame, text="→", command=model_event_next, height=40, width=45, font=customtkinter.CTkFont(family="Roboto Flex", size= 30), corner_radius=40, bg_color='#05122d',fg_color="#4B0082")
                    self.next_button.grid(row=4, column=1, padx=20, pady=20,  sticky = "se")  
    
                    if (model_frame != self.model_1_frame):
                        self.next_button = customtkinter.CTkButton(model_frame, text="←", command=model_event_back, height=40, width=45, font=customtkinter.CTkFont(family="Roboto Flex", size= 30), corner_radius=40, bg_color='#05122d',fg_color="#4B0082")
                        self.next_button.grid(row=4, column=0, padx=20, pady=20,  sticky = "sw") 
                
                
                # Plot models on same graph
                fig, ax = plt.subplots(figsize = (15, 5))
                fig.patch.set_facecolor('#05122d')  # Set the figure background color
                ax.set_facecolor('#05122d')  # Set the axes background color
                
                ax.plot(hourly_data_month_day_saved["DATE"], hourly_data_month_day_saved["TOTAL_CONSUMPTION"], 'o-', label = "Actual Consumption", color = "pink")
                if (model_frame == self.summary_frame): 
                    for model_name in selected_models:
                        if model_name == "K-Nearest Neighbors":
                            color_name = "purple"
                            model_name = "KNN"
                        if model_name == "Convolutional Neural Network":
                            color_name = "indigo"
                            model_name = "CNN" 
                        if model_name == "Linear Regression":
                            color_name = "violet"
                            model_name = "LR"
                        if model_name == "X Gradient Boost":
                            color_name = "darkviolet"
                            model_name = "XGB" 
                        try:
                            ax.plot(hourly_data_month_day_saved["DATE"], Y_pred_denorm_saved_df_saved[model_name]["FORECASTED CONSUMPTION (MW)"], 'o-', label = model_name + " Forecast", color = color_name)
                            ax.legend(loc = "best", facecolor='#34495E', edgecolor='pink', labelcolor='white')
                        except:
                            continue
                else:
                    ax.plot(hourly_data_month_day_saved["DATE"], Y_pred_denorm_saved_df_saved["TOTAL_CONSUMPTION"], 'o-', label = "Forecasted Consumption", color = "purple")
                    ax.legend(loc = "upper left", facecolor='#34495E', edgecolor='pink', labelcolor='white')
         
                
                ax.set_title(title, color="white")     
                
                ax.set_xlabel("Hour", color = "white")
                ax.set_ylabel("Consumption [MW]",color = "white")
            
                # Customize the x and y axis lines and text color
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.spines['top'].set_color('#05122d')
                ax.spines['right'].set_color('#05122d')
    
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d, %H:%M'))
                plot_svg =  os.path.join(image_path, "Predicted_Actual_Graph.png")
                plt.savefig(plot_svg)
                plt.close()
                
                # Positining of Figure
                self.model_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "Predicted_Actual_Graph.png")), size=(1500, 500))
                
                self.model_frame_image_label = customtkinter.CTkLabel(model_frame, text="", image=self.model_image)
                self.model_frame_image_label.grid(row=1, column=0, padx=20, pady=10, columnspan=2)
                
                if (model_frame != self.summary_frame):
                    # Display Table of Error
                    hourly_data_month_day_saved_table_tp = hourly_data_month_day_error.transpose()
                    
                    # Positining of model table
                    self.model_table = CTkTable(model_frame, width=1, height=1, values=hourly_data_month_day_saved_table_tp.values.tolist(), 
                                fg_color='#05122d',       # Foreground color (table background)
                                bg_color='#05122d',       # Background color (frame background)
                                text_color='white',     # Text color
                                header_color='#560067',
                                hover_color=("gray70", "gray30"), anchor="w",
                                 font = customtkinter.CTkFont(family="Roboto Condensed", size=12))
                    self.model_table.grid(row=2, column=0, padx=20, pady=10, columnspan=2)
                
                
                    # Positining of metrix table
                    self.metrix_table = CTkTable(model_frame, width=1, height=1, values=metrics_values.values.tolist(), 
                                fg_color='#05122d',       # Foreground color (table background)
                                bg_color='#05122d',       # Background color (frame background)
                                text_color='white',     # Text color
                                header_color='#560067',
                                hover_color=("gray70", "gray30"), anchor="w",
                                 font = customtkinter.CTkFont(family="Roboto Condensed", size=12))
                    self.metrix_table.grid(row=3, column=0, padx=20, pady=10, columnspan=2)
                else:
                    
                    # Create frame for save results and back to start menu buttons
                    self.button_summary_frame = customtkinter.CTkFrame(self.summary_frame, fg_color = '#05122d', bg_color = '#05122d')
                    self.button_summary_frame.grid(row=2, column=0, columnspan=5, sticky="nsew", padx=200, pady=80)
                    self.button_summary_frame.grid_columnconfigure(0,weight=5)
                    self.button_summary_frame.grid_columnconfigure(1,weight=1)
                    self.button_summary_frame.grid_columnconfigure(2,weight=1)
                    self.button_summary_frame.grid_columnconfigure(3,weight=1)
                    self.button_summary_frame.grid_columnconfigure(4,weight=5)
                    self.button_summary_frame.rowconfigure(0,weight=1)
                    self.button_summary_frame.rowconfigure(1,weight=1)
                
                    # Save Results
                    self.save_results_button = customtkinter.CTkButton(self.button_summary_frame, corner_radius=20, height=40, border_spacing=10, text="Save Results (Most Recent Run)",
                                                                  bg_color='#05122d',
                                                                  fg_color="#4B0082",
                                                                  hover_color="#560067",
                                                                  text_color=("gray10", "gray90"),
                                                                  font = my_button_font,
                                                                  anchor="center", command=self.save_results_button_event)
                    self.save_results_button.grid(row=0, column=1, padx = padding_x, sticky = "ew")
                    
                    
                    # Create search bar for Email
                    self.email_search_bar = customtkinter.CTkEntry(self.button_summary_frame, placeholder_text ="Enter Email(s) (ex. john.doe@gmail.com, ...)",
                                            fg_color="#14206d",  # Foreground color (entry background)
                                          bg_color="#05122d",  # Background color (frame background)
                                          text_color="#ffffff",  # Text color
                                          font=my_text_font)
                    self.email_search_bar.grid(row=0, column=2, padx = padding_x, sticky = "ew")
                    
                    # Create Email button
                    self.email_button = customtkinter.CTkButton(self.button_summary_frame, corner_radius=20, height=40, border_spacing=10, text="Send Saved Results to Email",
                                                                  fg_color="#4B0082",  
                                                                  bg_color= '#05122d',hover_color="#560067",
                                                                  text_color=("gray10", "gray90"),
                                                                  font = my_button_font,
                                                                  anchor="center", command=self.send_email_button_event)
                    self.email_button.grid(row=1, column=2, padx = padding_x, sticky = "new")
                    
                    
                    
                    
                    self.restart_program_button = customtkinter.CTkButton(self.button_summary_frame, corner_radius=20, height=40, border_spacing=10, text="Exit Back to Start Menu",
                                                                  bg_color='#05122d',
                                                                  fg_color="#4B0082",
                                                                  hover_color="#560067",
                                                                  text_color=("gray10", "gray90"),
                                                                  font = my_button_font,
                                                                  anchor="center", command=self.restart_program_button_event)
                    self.restart_program_button.grid(row=0, column=3, padx = padding_x, sticky = "ew")
     
                
                
            # Function to display when no model is saved
            def plot_no_model(self, model_frame, model_event_next, model_event_back, model_name):
                
                
                if (model_frame != self.summary_frame):
                    model_menu_image_path = os.path.join(background_images_path, "Model_Page.png")
                else:
                    model_menu_image_path = os.path.join(background_images_path, "Home_Page.png")
                    
                # Create background image
                image = PIL.Image.open(model_menu_image_path)
                background_image_model = customtkinter.CTkImage(image, size=(1920, 1080))
                
                
                self.background_label = customtkinter.CTkLabel(model_frame,
                                                             image=background_image_model,
                                                             text="")  # Empty text
                self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
             
    
                self.model_frame_Label_Title = customtkinter.CTkLabel(model_frame, text=model_name, font=customtkinter.CTkFont(family="RobotoCondensed-ExtraBoldItalic", size=50, weight="bold", slant = "italic"), 
                                                                     bg_color='#05122d', text_color=("white"))
                self.model_frame_Label_Title.grid(row=0, column=0, padx=20, pady=(40, 10), columnspan=2, sticky = "n")  
                
                if (model_frame != self.summary_frame):
                    self.next_button = customtkinter.CTkButton(model_frame, text="→", command=model_event_next, height=40, width=45, font=customtkinter.CTkFont(family="Roboto Flex", size= 30), corner_radius=40, bg_color='#05122d',fg_color="#4B0082")
                    self.next_button.grid(row=4, column=1, padx=20, pady=20,  sticky = "se")  
                    
                    
                    if (model_frame != self.model_1_frame):
                        self.next_button = customtkinter.CTkButton(model_frame, text="←", command=model_event_back, height=40, width=45, font=customtkinter.CTkFont(family="Roboto Flex", size= 30), corner_radius=40, bg_color='#05122d',fg_color="#4B0082")
                        self.next_button.grid(row=4, column=0, padx=20, pady=20,  sticky = "sw") 
                else:
                    # Create frame for save results and back to start menu buttons
                    self.button_summary_frame = customtkinter.CTkFrame(self.summary_frame, fg_color = '#05122d', bg_color = '#05122d')
                    self.button_summary_frame.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=200, pady=80)
                    self.button_summary_frame.grid_columnconfigure(0,weight=2)
                    self.button_summary_frame.grid_columnconfigure(1,weight=1)
                    self.button_summary_frame.grid_columnconfigure(2,weight=2)
                    self.button_summary_frame.rowconfigure(0,weight=1)
                    
                    self.restart_program_button = customtkinter.CTkButton(self.button_summary_frame, corner_radius=20, height=40, border_spacing=10, text="Exit Back to Start Menu",
                                                                  bg_color='#05122d', 
                                                                  fg_color="#4B0082",
                                                                  hover_color="#560067",
                                                                  text_color=("gray10", "gray90"),
                                                                  font = my_button_font,
                                                                  anchor="center", command=self.restart_program_button_event)
                    self.restart_program_button.grid(row=0, column=1, sticky = "new")
            print("Forecasting Ontario Model(s)...")
            
            # Begin RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.connect(rgb_lights)
            except:
                pass
            
            try:
                Power_Forecasting_Corsair_RGB.waiting(rgb_lights)
            except:
                pass
            
            months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            
            
            selected_date = self.calendar.get_date()
    
            selected_date_datetime = datetime.strptime(selected_date, '%m/%d/%y')
            
            for widget in self.model_1_frame.winfo_children():
                if (widget.winfo_exists()):
                    widget.destroy()  # deleting widget
            
    
            try:
                fsa_chosen = fsa_chosen_option_menu
            except NameError:
                fsa_chosen = fsa_predict_list[0]
            
            try:
                num_of_days = int(number_of_days_chosen_option_menu)
            except NameError:
                num_of_days = 1
    
            dirs_hourly_consumption_demand = os.path.join(dirs_inputs, "Hourly_Demand_Data")
        
            
    
            # Define dataframes
            hourly_data_month_day_saved = pd.DataFrame(columns = ['HOUR', 'TOTAL_CONSUMPTION'])
            
            # Get one date behind so that lags can be propoerly incorporated
            new_date = selected_date_datetime + timedelta(num_of_days-1) 
            old_date = selected_date_datetime - timedelta(days=1)
            old_year = str(old_date.year)
            old_month = months[(old_date.month-1)]
            old_day = str(old_date.day)
            
            
            old_date_cnn = selected_date_datetime - timedelta(days=cnn_days_back)
            old_year_cnn = str(old_date_cnn.year)
            old_month_cnn = months[(old_date_cnn.month-1)]
            old_day_cnn = str(old_date_cnn.day)
            
            year = str(new_date.year)
            month = months[(new_date.month-1)]
            day = str(new_date.day)
    
    
            ### Calling Data ###
            # Choose FSA for data collection + Get latitude and longitude of chosen fsa
            lat = fsa_map[fsa_chosen]["lat"]
            lon = fsa_map[fsa_chosen]["lon"]
    
            # Choose date range for data collection
            # All models
            start_year = int(old_year)
            start_month = int(old_month)
            start_day = int(old_day)
            start_hour = 0
            
            # CNN Model
            start_year_cnn = int(old_year_cnn)
            start_month_cnn = int(old_month_cnn)
            start_day_cnn = int(old_day_cnn)
            start_hour_cnn = 0
            
            # End date
            end_year = int(year)
            end_month = int(month)
            end_day = int(day)
            end_hour = 23
    
            # Making datetime objects for start and end dates
            start_date = datetime(start_year, start_month, start_day, start_hour,0,0)
            start_date_cnn = datetime(start_year_cnn, start_month_cnn, start_day_cnn, start_hour_cnn,0,0)
            end_date = datetime(end_year, end_month, end_day, end_hour,0,0)
    
            # # Collect data - Using asynchronous functions
            # 
            weather_data, dummy_hourly_data_month_day = asyncio.run(Power_Forecasting_dataCollectionAndPreprocessingFlow.get_data_for_time_range(dirs_inputs, start_date, end_date, fsa_chosen, lat, lon, fsa_map))
            
            # CNN weather data
            weather_data_cnn, dummy_hourly_data_month_day = asyncio.run(Power_Forecasting_dataCollectionAndPreprocessingFlow.get_data_for_time_range(dirs_inputs, start_date_cnn, end_date, fsa_chosen, lat, lon, fsa_map))
            
            dummy_hourly_data_month_day = dummy_hourly_data_month_day.reset_index(drop = True) 
            weather_data = weather_data.reset_index(drop = True)
            
            
            print("    Data is collected.")
            
    
            index_first_day = weather_data[(weather_data['Day'] == start_day)].index
            
            dummy_hourly_data_month_day = dummy_hourly_data_month_day.drop(index_first_day, axis='index', inplace = False).reset_index(drop=True)
            weather_data = weather_data.drop(index_first_day, axis='index', inplace = False).reset_index(drop=True)
            
            # Drop temporary year month day hour columns
            weather_data = weather_data.drop(columns=['Year', 'Month', 'Day', 'Hour'])
            # Drop temporary year month day hour columns
            weather_data_cnn = weather_data_cnn.drop(columns=['Year', 'Month', 'Day', 'Hour'])
            
            # Open weather scaler
            scaler_path = os.path.join(saved_model_path, "weather_scaler_"+fsa_chosen+".pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            weather_scaler = joblib.load(scaler_path)
            
            # Normalize Weather
            norm_weather_data = weather_scaler.transform(weather_data)
            norm_weather_data = pd.DataFrame(norm_weather_data, columns = weather_data.columns)
            
            # Normalize CNN Weather
            norm_weather_data_cnn = weather_scaler.transform(weather_data_cnn)
            norm_weather_data_cnn = pd.DataFrame(norm_weather_data_cnn, columns = weather_data_cnn.columns)
            
            
            ###############################################################################
            # Import and predict Models
            ###############################################################################
            
            total_features = []
            # Convert year, month, day, hour to boolean values
            # Day range 1 to 31 (subtract last day for 0 condition)
            days_range = [*range(1, 31)]
            # Hour range 0 to 23 (subtract last hour for 0 condition)
            hours_range = [*range(0, 23)]
            # Hour range 0 to 23 (subtract last month for 0 condition)
            months_range = [*range(1, 12)]
            # Year range depending on weather data columns
            years_range = weather_data.columns
            
            for year in years_range:
                if ("Year_" in year):
                    total_features.append(year)
            for month in months_range:
                total_features.append("Month_" + str(month))
    
            for day in days_range:
                total_features.append("Day_" + str(day))
            
            for hour in hours_range:
                total_features.append("Hour_" + str(hour))
                         
            for feature in selected_features:
                total_features.append(feature)
                
            total_features_temp = total_features.copy()
            for lag in range (1, 24):
                for feature in total_features_temp:
                    if (feature == "Weekend" or feature == "Season" or feature == "Holiday (Ontario)" or ("Year_" in feature) or ("Month_" in feature) or ("Day_" in feature)):
                        continue
                    else:
                        total_features.append(feature+"_Lag_"+str(lag))
            
            # Dictionary for model prediction dataframes
            # model -> Value
            Y_pred_denorm_saved_df = {}
            
            for model_name in selected_models:
                if model_name == "K-Nearest Neighbors":
                    model_name = "KNN"
                if model_name == "Convolutional Neural Network":
                    model_name = "CNN" 
                if model_name == "Linear Regression":
                    model_name = "LR"
                if model_name == "X Gradient Boost":
                    model_name = "XGB" 
                
                if model_name == "CNN":
                    X_test = norm_weather_data_cnn[total_features]
                else:
                    X_test = norm_weather_data[total_features]
                    
                # Load model from gui_pickup folder using joblib
                try:
                    if model_name == "CNN":
                        pipe_saved = models.load_model(os.path.join(saved_model_path, (model_name+"_"+fsa_chosen+"_Model_" + "_".join(selected_features_3_digits) + ".keras")))
                    else:
                        pipe_saved = joblib.load(os.path.join(saved_model_path, (model_name+"_"+fsa_chosen+"_Model_" + "_".join(selected_features_3_digits) + ".pkl"))) 
                except:
                    continue
                
                if (model_name == "CNN"):
                    
                    for feature in X_test.columns:
                      if (feature == "Weekend" or feature == "Season" or feature == "Holiday (Ontario)" or ("Year_" in feature) or ("Month_" in feature) or ("Day_" in feature) or ("Hour_" in feature)):
                        X_test[feature].astype('bool')
                      if ("Lag" in feature):
                        X_test = X_test.drop(columns = [feature])
                    
                    # Create dataframe without humidity and speed
                    data = X_test
                    
                    #window_size = 168  # Last 168 hours (one week)
                    window_size = 24*cnn_days_back # Last 24 hours (one day)
                    forecast_horizon = 24  # Next 24 hours
                    
                    
                    # Create input-output pairs using a sliding window
                    Y_pred_saved = pd.DataFrame(columns=['TOTAL_CONSUMPTION'])
                    for day in range(num_of_days):
                        X_data = []
    
                        X_data.append(data.iloc[day*24:day*24 + window_size + forecast_horizon].values)  # Collect 24 hours of feature data
                        
                        # Convert to numpy arrays
                        X_data = np.array(X_data, dtype=np.float16)
                        X_data = np.expand_dims(X_data, axis=-1)
        
                        # Predict using loaded model
                        Y_pred_cnn = pipe_saved.predict(X_data)
                        Y_pred_cnn = pd.DataFrame(Y_pred_cnn)
                        
                        # Ensure Y_pred and Y_test are reshaped correctly
                        Y_pred_cnn = Y_pred_cnn.values.reshape(-1, 1)
                        
                        Y_pred_cnn = pd.DataFrame(Y_pred_cnn, columns=['TOTAL_CONSUMPTION'])
    
                        Y_pred_saved = pd.concat([Y_pred_saved, Y_pred_cnn], axis=0, ignore_index=True)
                    
    
                else:
                    # Predict using loaded model
                    Y_pred_saved = pipe_saved.predict(X_test)
                    
                    # Ensure Y_pred and Y_test are reshaped correctly
                    Y_pred_saved = Y_pred_saved.reshape(-1, 1)
                    
                # Denormalize Y_pred and Y_test with min_max_scaler_y.pkl using joblib
                scaler_path = os.path.join(saved_model_path, "power_scaler_"+fsa_chosen+".pkl")
                if not os.path.exists(scaler_path):
                    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
                power_scaler = joblib.load(scaler_path)
                
                # Inverse Transform power scaler
                Y_pred_denorm_saved = power_scaler.inverse_transform(Y_pred_saved)
    
                # Save power scaler to dictionary of models
                Y_pred_denorm_saved_df[model_name] = pd.DataFrame(Y_pred_denorm_saved, columns=['TOTAL_CONSUMPTION'])
                
                # Convert to MW
                Y_pred_denorm_saved_df[model_name] = Y_pred_denorm_saved_df[model_name]*0.001
                
                # Convert to float64
                Y_pred_denorm_saved_df[model_name] = Y_pred_denorm_saved_df[model_name].astype(float)
                
                # Round to 4 decimal places
                Y_pred_denorm_saved_df[model_name] = Y_pred_denorm_saved_df[model_name].round(decimals = 4)
                print("    " + model_name + " - Forecast Generated.")
            
            # Dictionary for model prediction dataframes error 
            # model -> Value
            hourly_data_month_day_error = {}
            hourly_data_month_day_error_df = pd.DataFrame()
            metrics_values = {}
            save_results_dic.clear()
            
            for day_num in range (num_of_days):
                
                # new date
                new_date = selected_date_datetime + timedelta(days=day_num) 
        
                year = str(new_date.year)
                month = months[(new_date.month-1)]
                day = str(new_date.day)
    
                ###############################################################################
                # Dictionary for reading in hourly consumption by FSA
                ###############################################################################
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
                
                # Convert to MW
                hourly_data_month_day.loc[:, "TOTAL_CONSUMPTION"] = hourly_data_month_day["TOTAL_CONSUMPTION"]*0.001
                
                # Add column for date and time
                hourly_data_month_day.loc[:, "HOUR"] = hourly_data_month_day["HOUR"] - 1
                hourly_data_month_day.loc[:, "DATE"] = pd.to_datetime(hourly_data_month_day[["YEAR", "MONTH", "DAY","HOUR"]])
                
                # Append next day to another dataframe to plot on same figure
                if (day_num == 0):
                    title = "Hourly Power Consumption"
                    hourly_data_month_day_saved = hourly_data_month_day[["YEAR", "MONTH", "DAY", "HOUR", "DATE", "TOTAL_CONSUMPTION"]].copy()
                else:
                    hourly_data_month_day_saved = pd.concat([hourly_data_month_day_saved, hourly_data_month_day], axis=0, ignore_index=True)
    
                for model_name in selected_models:
                    if model_name == "K-Nearest Neighbors":
                        model_name = "KNN"
                    if model_name == "Convolutional Neural Network":
                        model_name = "CNN" 
                    if model_name == "Linear Regression":
                        model_name = "LR"
                    if model_name == "X Gradient Boost":
                        model_name = "XGB" 
                    try:
                        if (day_num == 0):
                            hourly_data_month_day_error_df["HOUR_NEW"] = (hourly_data_month_day.index%24)
                            hourly_data_month_day_error[model_name] = hourly_data_month_day_error_df
                            hourly_data_month_day_error[model_name] = hourly_data_month_day_error[model_name].rename(columns={"HOUR_NEW": "Hour"})
                           
                        Y_pred_denorm_saved_df_day = Y_pred_denorm_saved_df[model_name].iloc[(24*day_num):(24*day_num + 24)]
                        # Find Error
                        if (self.detailed_table_checkbox_var.get() == 1):
                            hourly_data_month_day_error[model_name]["Actual Consumption: Day " + str(day_num+1) + " (MW)"] = hourly_data_month_day["TOTAL_CONSUMPTION"].reset_index(drop = True)
                            hourly_data_month_day_error[model_name]["Forecasted Consumption: Day " + str(day_num+1) + " (MW)"] = Y_pred_denorm_saved_df_day["TOTAL_CONSUMPTION"].reset_index(drop = True)
                        hourly_data_month_day_error[model_name]["Error: Day " + str(day_num+1) + " (%)"] = 100*abs(Y_pred_denorm_saved_df_day["TOTAL_CONSUMPTION"].reset_index(drop = True) - hourly_data_month_day["TOTAL_CONSUMPTION"].reset_index(drop = True))/hourly_data_month_day["TOTAL_CONSUMPTION"].reset_index(drop = True)
                        hourly_data_month_day_error[model_name] = hourly_data_month_day_error[model_name].round(decimals = 4)
     
                    except:
                        continue
            
            
            for model_name in selected_models:
                if model_name == "K-Nearest Neighbors":
                    model_name = "KNN"
                if model_name == "Convolutional Neural Network":
                    model_name = "CNN" 
                if model_name == "Linear Regression":
                    model_name = "LR"
                if model_name == "X Gradient Boost":
                    model_name = "XGB" 
                try:    
                    Y_pred_denorm_saved_df[model_name] = Y_pred_denorm_saved_df[model_name].reset_index(drop = True)
                    hourly_data_month_day_saved = hourly_data_month_day_saved.reset_index(drop = True)
                    
                    # save_results_dic[model_name] = hourly_data_month_day_saved[["YEAR", "MONTH", "DAY", "HOUR", "TOTAL_CONSUMPTION"]]
                    # save_results_dic[model_name].columns.values[4] = "ACTUAL CONSUMPTION (MW)"
                    # save_results_dic[model_name]["FORECASTED CONSUMPTION (MW)"] = Y_pred_denorm_saved_df[model_name]
                    
                    save_results_dic[model_name] = pd.concat([hourly_data_month_day_saved[["YEAR", "MONTH", "DAY", "HOUR", "TOTAL_CONSUMPTION"]], Y_pred_denorm_saved_df[model_name]], axis=1)
                    save_results_dic[model_name]["HOUR"] = save_results_dic[model_name]["HOUR"] + 1
                    save_results_dic[model_name].columns.values[4] = "ACTUAL CONSUMPTION (MW)"
                    save_results_dic[model_name].columns.values[5] = "FORECASTED CONSUMPTION (MW)"
                    
                    metrics_model_path = os.path.join(saved_model_path, model_name+"_"+fsa_chosen+"_Metrics_" + "_".join(selected_features_3_digits) + ".csv") 
                    metrics_values[model_name] = pd.read_csv(metrics_model_path, header=0)
                    metrics_values[model_name] = metrics_values[model_name].round(decimals = 4)
                    
                    save_results_dic[model_name] = pd.concat([save_results_dic[model_name],  metrics_values[model_name][["MAPE (%)", "MAE (MW)", "r2", "MSE (MW Squared)", "RMSE (MW)"]]], axis=1).fillna("")
                    
                    hourly_data_month_day_error_columns = pd.DataFrame([hourly_data_month_day_error[model_name].columns], columns = hourly_data_month_day_error[model_name].columns)
                    metrics_values_columns = pd.DataFrame([metrics_values[model_name].columns], columns = metrics_values[model_name].columns)
                    
                    hourly_data_month_day_error[model_name] = pd.concat([hourly_data_month_day_error_columns, hourly_data_month_day_error[model_name].iloc[0:]]).reset_index(drop=True)
                    metrics_values[model_name] = pd.concat([metrics_values_columns, metrics_values[model_name].iloc[0:]]).reset_index(drop=True)
                except:
                    continue
            
            count_no_models = 0
            try:
                plot_figures_model(self, hourly_data_month_day_saved, Y_pred_denorm_saved_df["LR"], metrics_values["LR"], hourly_data_month_day_error["LR"], self.model_1_frame, self.model_2_button_event, "N/A", model_names_list[0])
            except:
                plot_no_model(self, self.model_1_frame, self.model_2_button_event, "N/A", "No Saved Model For " + model_names_list[0])
                count_no_models = count_no_models + 1
                
            try:
                plot_figures_model(self, hourly_data_month_day_saved, Y_pred_denorm_saved_df["XGB"], metrics_values["XGB"], hourly_data_month_day_error["XGB"], self.model_2_frame, self.model_3_button_event, self.model_1_button_event, model_names_list[1])        
            except:
                plot_no_model(self, self.model_2_frame, self.model_3_button_event, self.model_1_button_event, "No Saved Model For " + model_names_list[1])
                count_no_models = count_no_models + 1
                
            try:
                plot_figures_model(self, hourly_data_month_day_saved, Y_pred_denorm_saved_df["KNN"], metrics_values["KNN"], hourly_data_month_day_error["KNN"], self.model_3_frame, self.model_4_button_event, self.model_2_button_event, model_names_list[2])
            except:
                plot_no_model(self, self.model_3_frame, self.model_4_button_event, self.model_2_button_event, "No Saved Model For " + model_names_list[2])
                count_no_models = count_no_models + 1
                
            try:
                plot_figures_model(self, hourly_data_month_day_saved, Y_pred_denorm_saved_df["CNN"], metrics_values["CNN"], hourly_data_month_day_error["CNN"], self.model_4_frame, self.summary_button_event, self.model_3_button_event, model_names_list[3])
            except:
                plot_no_model(self, self.model_4_frame, self.summary_button_event, self.model_3_button_event, "No Saved Model For " + model_names_list[3])
                count_no_models = count_no_models + 1
                
            try:
                if (count_no_models == 4):
                    plot_no_model(self, self.summary_frame, "N/A","N/A", "No Saved Models")
                else:
                    plot_figures_model(self, hourly_data_month_day_saved, save_results_dic, "N/A", "N/A", self.summary_frame, "N/A", "N/A", "Summary of All Models")
            except:
                plot_no_model(self, self.summary_frame, "N/A","N/A", "No Saved Models")
            
            print("Done Forecating!")
            print("\n")
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.done_waiting(rgb_lights)
            except:
                pass
            
            count_no_models = 0
            self.select_frame_by_name("Model: Linear Regression")
            
        except Exception as error:
            print("An exception occurred:", error)
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.error(rgb_lights)
            except:
                pass
        
        
    def train_models_button_event(self):
        try:
            print("Training Ontario Models...")
            # Reset progress bar to 0 and make color of button the hover color
            self.train_models_button.grid_forget()
            self.train_models_button = customtkinter.CTkButton(self.option2_frame, corner_radius=20, height=40, border_spacing=10, text="Train Models",
                                                          fg_color="#560067",  
                                                          bg_color= "#05122d",
                                                          text_color=("gray10", "gray90"),
                                                          font = my_button_font,
                                                          anchor="center", command=self.train_models_button_event)
            self.train_models_button.grid(row=2, column=2, padx = padding_x_option2, pady = (10, 10), sticky = "new")
            
            self.progress_bar_train_ontario.grid_forget()
            self.progress_bar_train_ontario = customtkinter.CTkProgressBar(self.option2_frame, width=300, fg_color="#14206d")
            self.progress_bar_train_ontario.grid(row=3, column=2, padx = padding_x_option2, pady=0, sticky="new")
            self.progress_bar_train_ontario.set(0)  # Initialize the progress bar to 0
            self.update_idletasks()
            
            # Begin RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.connect(rgb_lights)
            except:
                pass
            
            try:
                Power_Forecasting_Corsair_RGB.waiting(rgb_lights)
            except:
                pass
    
            nest_asyncio.apply() # Apply nest_asyncio to allow for nested asyncio operations
            
            fsa_typed = self.fsa_search_bar.get()
            
            ### Calling Data ###
            # Choose FSA for data collection + Get latitude and longitude of chosen fsa
            lat = fsa_map[fsa_typed]["lat"]
            lon = fsa_map[fsa_typed]["lon"]
    
            # Choose date range for data collection
            start_year = 2018
            start_month = 1
            start_day = 1
            start_hour = 0
    
            end_year = 2024
            end_month = 11
            end_day = 30
            end_hour = 23
    
            # Making datetime objects for start and end dates
            start_date = datetime(start_year, start_month, start_day, start_hour,0,0)
            end_date = datetime(end_year, end_month, end_day, end_hour,0,0)
    
            # # Collect data - Using asynchronous functions
            try:
                weather_data = pd.read_csv(f'{power_weather_data_path}/weather_data_{fsa_typed}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv')
                power_data = pd.read_csv(f'{power_weather_data_path}/power_data_{fsa_typed}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv')
            except FileNotFoundError: 
                weather_data, power_data = asyncio.run(Power_Forecasting_dataCollectionAndPreprocessingFlow.get_data_for_time_range(dirs_inputs, start_date, end_date, fsa_typed, lat, lon, fsa_map))
                weather_data.to_csv(f'{power_weather_data_path}/weather_data_{fsa_typed}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv', index=False)
                power_data.to_csv(f'{power_weather_data_path}/power_data_{fsa_typed}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv', index=False)
            
            print("    Data is collected.")
            
            
            # Normalize Data
            norm_weather_data, norm_power_data, weather_scaler, power_scaler = Power_Forecasting_dataCollectionAndPreprocessingFlow.normalize_data(weather_data, power_data)
            
            # Save Scalers
            file_path_scalar = os.path.join(saved_model_path, "power_scaler_" + fsa_typed + ".pkl")
            joblib.dump(power_scaler, file_path_scalar)
            file_path_scalar = os.path.join(saved_model_path, "weather_scaler_" + fsa_typed + ".pkl")
            joblib.dump(weather_scaler, file_path_scalar)
            
            # # Save Normalized Data to CSV
            norm_weather_data.to_csv(f'{x_y_input_path}/norm_weather_data_{fsa_typed}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv', index=False)
            norm_power_data.to_csv(f'{x_y_input_path}/norm_power_data_{fsa_typed}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv', index=False)
            
            total_features = []
            # Convert year, month, day, hour to boolean values
            # Day range 1 to 31 (subtract last day for 0 condition)
            days_range = [*range(1, 31)]
            # Hour range 0 to 23 (subtract last hour for 0 condition)
            hours_range = [*range(0, 23)]
            # Hour range 0 to 23 (subtract last month for 0 condition)
            months_range = [*range(1, 12)]
            # Year range depending on weather data columns
            years_range = weather_data.columns
            
            for year in years_range:
                if ("Year_" in year):
                    total_features.append(year)
    
            for month in months_range:
                total_features.append("Month_" + str(month))
    
            for day in days_range:
                total_features.append("Day_" + str(day))
            
            for hour in hours_range:
                total_features.append("Hour_" + str(hour))
                         
            for feature in selected_features:
                total_features.append(feature)
                
            total_features_temp = total_features.copy()
            for lag in range (1, 24):
                for feature in total_features_temp:
                    if (feature == "Weekend" or feature == "Season" or feature == "Holiday (Ontario)" or ("Year_" in feature) or ("Month_" in feature) or ("Day_" in feature)):
                        continue
                    else:
                        total_features.append(feature+"_Lag_"+str(lag))
           
            
            # Train and Save KNN Model
            for model in selected_models:
                if model == "K-Nearest Neighbors":
                    Power_Forecasting_KNN_Saver.save_knn_model(norm_weather_data[total_features], norm_power_data, power_scaler, fsa_typed, saved_model_path, selected_features_3_digits)
                    print("    "+ model +" Trained.")
                if model == "Linear Regression":
                    Power_Forecasting_LR_Saver.save_lr_model(norm_weather_data[total_features], norm_power_data, power_scaler, fsa_typed, saved_model_path, selected_features_3_digits)
                    print("    "+ model +" Trained.")
                if model == "X Gradient Boost":
                    Power_Forecasting_XGB_Saver.save_xgb_model(norm_weather_data[total_features], norm_power_data, power_scaler, fsa_typed, saved_model_path, selected_features_3_digits)
                    print("    "+ model +" Trained.")
                if model == "Convolutional Neural Network":
                    Power_Forecasting_CNN_Saver.save_cnn_model(norm_weather_data[total_features], norm_power_data, power_scaler, fsa_typed, saved_model_path, selected_features_3_digits, x_y_input_path)
                    print("    "+ model +" Trained.")
                    
            print("Done Training!")
            print("\n")
            
            # Append FSA to drop down menu list
            global fsa_predict_list
            fsa_predict_list.append(fsa_typed)
            if "No Models" in fsa_predict_list:
                fsa_predict_list.remove("No Models")
    
            fsa_predict_list = list(dict.fromkeys(fsa_predict_list))
            
            # Add normal button back
            self.train_models_button = customtkinter.CTkButton(self.option2_frame, corner_radius=20, height=40, border_spacing=10, text="Train Models",
                                                          fg_color="#14206d",  
                                                          
                                                          bg_color= "#05122d",hover_color="#560067",
                                                          text_color=("gray10", "gray90"),
                                                          font = my_button_font,
                                                          anchor="center", command=self.train_models_button_event)
            self.train_models_button.grid(row=2, column=2, padx = padding_x_option2, pady = (10, 10), sticky = "new")
                    
            #Progress Bar Function for Ontartio training dataset
            def update_progress():
                for i in range(101):
                    time.sleep(0.001)  # Simulate work being done
                    self.progress_bar_train_ontario.set(i/100)  # Update the progress bar
                    self.update_idletasks()
            # Run the update_progress function in a separate thread
            threading.Thread(target=update_progress).start()
    
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.done_waiting(rgb_lights)
            except:
                pass
        except Exception as error:
            print("An exception occurred:", error)
            
            # Add normal button back
            self.train_models_button = customtkinter.CTkButton(self.option2_frame, corner_radius=20, height=40, border_spacing=10, text="Train Models",
                                                          fg_color="#14206d",  
                                                          
                                                          bg_color= "#05122d",hover_color="#560067",
                                                          text_color=("gray10", "gray90"),
                                                          font = my_button_font,
                                                          anchor="center", command=self.train_models_button_event)
            self.train_models_button.grid(row=2, column=2, padx = padding_x_option2, pady = (10, 10), sticky = "new")
            
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.error(rgb_lights)
            except:
                pass
            
        
    def train_input_excel_models_button_event(self):
        try:
            print("Training Provided Dataset Models...")
            # Reset progress bar to 0 and make color of button the hover colnor
            self.train_models_button_any.grid_forget()
            self.train_models_button_any = customtkinter.CTkButton(self.option3_frame, corner_radius=20, height=40, border_spacing=10, text="Train Models",
                                                          fg_color="#560067",
                                                          bg_color= "#05122d",
                                                          text_color=("gray10", "gray90"),
                                                          font = my_button_font,
                                                          anchor="center", command=self.train_input_excel_models_button_event)
            self.train_models_button_any.grid(row=2, column=1, padx = padding_x_option3, pady = 20, sticky = "sew")
            
            self.progress_bar_train_any.grid_forget()
            self.progress_bar_train_any = customtkinter.CTkProgressBar(self.option3_frame, width=200, fg_color="#14206d")
            self.progress_bar_train_any.grid(row=3, column=1, padx = padding_x_option3, pady=0, sticky="new")
            self.progress_bar_train_any.set(0)  # Initialize the progress bar to 0
            self.update_idletasks()
            
            # Begin RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.connect(rgb_lights)
            except:
                pass
            
            try:
                Power_Forecasting_Corsair_RGB.waiting(rgb_lights)
            except:
                pass
            
            weather_data =  pd.read_excel(input_data_filename, sheet_name = "Weather_Information", header = 0)
            weather_data = Power_Forecasting_dataCollectionAndPreprocessingFlow.add_calendar_columns(weather_data)
            weather_data = Power_Forecasting_dataCollectionAndPreprocessingFlow.add_lags_to_weather_data(weather_data, 23)
            
            
            power_data = pd.read_excel(input_data_filename, sheet_name = "Power_Consumption", header = 0)
            power_data = power_data.rename(columns={"Power Consumption": "TOTAL_CONSUMPTION"})
            power_data = power_data.rename(columns={"Year": "YEAR"})
            power_data = power_data.rename(columns={"Month": "MONTH"})
            power_data = power_data.rename(columns={"Day": "DAY"})
            power_data = power_data.rename(columns={"Hour": "HOUR"})
            
            # Normalize Data
            norm_weather_data, norm_power_data, weather_scaler, power_scaler = Power_Forecasting_dataCollectionAndPreprocessingFlow.normalize_data(weather_data, power_data)
            
            # Save Scaler
            file_path_scalar = os.path.join(saved_model_path, "power_scaler_" + input_data_basename + ".pkl")
            joblib.dump(power_scaler, file_path_scalar)
            
            file_path_scalar = os.path.join(saved_model_path, "weather_scaler_" + input_data_basename + ".pkl")
            joblib.dump(weather_scaler, file_path_scalar)
            
            total_features = []
            # Convert year, month, day, hour to boolean values
            # Day range 1 to 31 (subtract last day for 0 condition)
            days = [*range(1, 31)]
            # Hour range 0 to 23 (subtract last hour for 0 condition)
            hours = [*range(0, 23)]
            # Hour range 0 to 23 (subtract last month for 0 condition)
            months = [*range(1, 12)]
            
            # Year range depending on weather data columns
            years_range = weather_data.columns
            
            for year in years_range:
                if ("Year_" in year):
                    total_features.append(year)
    
            for month in months:
                total_features.append("Month_" + str(month))
    
            for day in days:
                total_features.append("Day_" + str(day))
            
            for hour in hours:
                total_features.append("Hour_" + str(hour))
                         
            for feature in selected_features:
                total_features.append(feature)
                
            total_features_temp = total_features.copy()
            for lag in range (1, 24):
                for feature in total_features_temp:
                    if (feature == "Weekend" or feature == "Season" or feature == "Holiday (Ontario)" or ("Year_" in feature) or ("Month_" in feature) or ("Day_" in feature)):
                        continue
                    else:
                        total_features.append(feature+"_Lag_"+str(lag))
            
            # Train and Save Models
            for model in selected_models:
                if model == "K-Nearest Neighbors":
                    Power_Forecasting_KNN_Saver.save_knn_model(norm_weather_data[total_features], norm_power_data, power_scaler, input_data_basename, saved_model_path, selected_features_3_digits)
                    print("    "+ model +" Trained.")
                if model == "Linear Regression":
                    Power_Forecasting_LR_Saver.save_lr_model(norm_weather_data[total_features], norm_power_data, power_scaler, input_data_basename, saved_model_path, selected_features_3_digits)
                    print("    "+ model +" Trained.")
                if model == "X Gradient Boost":
                    Power_Forecasting_XGB_Saver.save_xgb_model(norm_weather_data[total_features], norm_power_data, power_scaler, input_data_basename, saved_model_path, selected_features_3_digits)   
                    print("    "+ model +" Trained.")
                if model == "Convolutional Neural Network":
                    Power_Forecasting_CNN_Saver.save_cnn_model(norm_weather_data[total_features], norm_power_data, power_scaler, input_data_basename, saved_model_path, selected_features_3_digits, x_y_input_path)
                    print("    "+ model +" Trained.")
                    
            print("Done Training!")
            print("\n")
            
            
            # Add normal button back
            self.train_models_button_any = customtkinter.CTkButton(self.option3_frame, corner_radius=20, height=40, border_spacing=10, text="Train Models",
                                                          fg_color="#14206d",
                                                          bg_color= "#05122d",
                                                          hover_color="#560067",
                                                          text_color=("gray10", "gray90"),
                                                          font = my_button_font,
                                                          anchor="center", command=self.train_input_excel_models_button_event)
            self.train_models_button_any.grid(row=2, column=1, padx = padding_x_option3, pady = 20, sticky = "sew")
            
            #Progress Bar Function for any training dataset
            def update_progress():
                for i in range(101):
                    time.sleep(0.001)  # Simulate work being done
                    self.progress_bar_train_any.set(i/100)  # Update the progress bar
                    self.update_idletasks()
            # Run the update_progress function in a separate thread
            threading.Thread(target=update_progress).start()
            
            
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.done_waiting(rgb_lights)
            except:
                pass
        except Exception as error:
            print("An exception occurred:", error)
            
            # Add normal button back
            self.train_models_button_any = customtkinter.CTkButton(self.option3_frame, corner_radius=20, height=40, border_spacing=10, text="Train Models",
                                                          fg_color="#14206d",
                                                          bg_color= "#05122d",
                                                          hover_color="#560067",
                                                          text_color=("gray10", "gray90"),
                                                          font = my_button_font,
                                                          anchor="center", command=self.train_input_excel_models_button_event)
            self.train_models_button_any.grid(row=2, column=1, padx = padding_x_option3, pady = 20, sticky = "sew")
            
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.error(rgb_lights)
            except:
                pass
        
        
    def predict_input_excel_models_button_event(self):
        try:
            # Function to plot the model figures
            def plot_figures_model_2(self, weather_data, Y_pred_denorm_saved_df_saved, metrics_values, table_values, model_frame, model_event_next, model_event_back, model_name):
                    
                if (model_frame != self.summary_frame):
                    model_menu_image_path = os.path.join(background_images_path, "Model_Page.png")
                else:
                    model_menu_image_path = os.path.join(background_images_path, "Home_Page.png")
                # Create background image
                image = PIL.Image.open(model_menu_image_path)
                background_image_model = customtkinter.CTkImage(image, size=(1920, 1080))
                
                
                self.background_label = customtkinter.CTkLabel(model_frame,
                                                             image=background_image_model,
                                                             text="")  # Empty text
                self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
             
                my_title_font = customtkinter.CTkFont(family="RobotoCondensed-ExtraBoldItalic", size=50, weight="bold", slant = "italic")
                self.model_frame_Label_Title = customtkinter.CTkLabel(model_frame, text=model_name, font=my_title_font, 
                                                                     bg_color='#05122d', text_color=("white"))
                self.model_frame_Label_Title.grid(row=0, column=0, padx=20, pady=(40, 10), columnspan=2, sticky = "n")  
                if (model_frame != self.summary_frame): 
                    
                    self.next_button = customtkinter.CTkButton(model_frame, text="→", command=model_event_next, height=40, width=45, font=customtkinter.CTkFont(family="Roboto Flex", size= 30), corner_radius=40, bg_color='#05122d',fg_color="#4B0082")
                    self.next_button.grid(row=4, column=1, padx=20, pady=20,  sticky = "se")  
                    
                    if (model_frame != self.model_1_frame):
                        self.next_button = customtkinter.CTkButton(model_frame, text="←", command=model_event_back, height=40, width=45, font=customtkinter.CTkFont(family="Roboto Flex", size= 30), corner_radius=40, bg_color='#05122d',fg_color="#4B0082")
                        self.next_button.grid(row=4, column=0, padx=20, pady=20,  sticky = "sw") 
                
                
                # Plot models on same graph
                fig, ax = plt.subplots(figsize = (15, 5))
                fig.patch.set_facecolor('#05122d')  # Set the figure background color
                ax.set_facecolor('#05122d')  # Set the axes background color
                
                
                if (model_frame == self.summary_frame): 
                    for model_name in selected_models:
                        if model_name == "K-Nearest Neighbors":
                            color_name = "purple"
                            model_name = "KNN"
                        if model_name == "Convolutional Neural Network":
                            color_name = "indigo"
                            model_name = "CNN" 
                        if model_name == "Linear Regression":
                            color_name = "violet"
                            model_name = "LR"
                        if model_name == "X Gradient Boost":
                            color_name = "darkviolet"
                            model_name = "XGB" 
                        try:
                            ax.plot(weather_data["DATE"], Y_pred_denorm_saved_df_saved[model_name]["FORECASTED CONSUMPTION (MW)"], 'o-', label = model_name + " Forecast", color = color_name)
                            ax.legend(loc = "best", facecolor='#34495E', edgecolor='pink', labelcolor='white')
                        except:
                            continue
                else:
                    ax.plot(weather_data["DATE"], Y_pred_denorm_saved_df_saved["TOTAL_CONSUMPTION"], 'o-', label = "Forecasted Consumption", color = "purple")
                    ax.legend(loc = "upper left", facecolor='#34495E', edgecolor='pink', labelcolor='white')
                
                ax.set_title(title, color="white")     
                ax.set_xlabel("Hour", color = "white")
                ax.set_ylabel("Consumption [MW]",color = "white")
                
                
                # Customize the x and y axis lines and text color
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.spines['top'].set_color('#05122d')
                ax.spines['right'].set_color('#05122d')
    
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d, %H:%M'))
                plot_svg =  os.path.join(image_path, "Predicted_Actual_Graph.png")
                plt.savefig(plot_svg)
                plt.close()
                
                # Positining of Figure
                self.model_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "Predicted_Actual_Graph.png")), size=(1500, 500))
                
                self.model_frame_image_label = customtkinter.CTkLabel(model_frame, text="", image=self.model_image)
                self.model_frame_image_label.grid(row=1, column=0, padx=20, pady=10, columnspan=2)
                
                if (model_frame != self.summary_frame): 
                    # Display Table of Error
                    table_values_tp = table_values.transpose()
                    
                    # Do not show table if user did not click "Show Detailed Table" button
                    try:
                        self.model_table.grid_forget()
                    except:
                        pass
                    if (self.detailed_table_checkbox_var_excel.get() == 1):
                        # Positining of Figure
                        self.model_table = CTkTable(model_frame, width=1, height=1, values=table_values_tp.values.tolist(), 
                                    fg_color='#05122d',       # Foreground color (table background)
                                    bg_color='#05122d',       # Background color (frame background)
                                    text_color='white',     # Text color
                                    header_color='#560067',
                                    hover_color=("gray70", "gray30"), anchor="w",
                                     font = customtkinter.CTkFont(family="Roboto Condensed", size=12))
                        self.model_table.grid(row=2, column=0, padx=20, pady=10, columnspan=2)
                    
                    # Positining of Figure
                    self.metrix_table = CTkTable(model_frame, width=1, height=1, values=metrics_values.values.tolist(), 
                                fg_color='#05122d',       # Foreground color (table background)
                                bg_color='#05122d',       # Background color (frame background)
                                text_color='white',     # Text color
                                header_color='#560067',
                                hover_color=("gray70", "gray30"), anchor="w",
                                 font = customtkinter.CTkFont(family="Roboto Condensed", size=12))
                    self.metrix_table.grid(row=3, column=0, padx=20, pady=10, columnspan=2)
                else:
                    # Create frame for save results and back to start menu buttons
                    self.button_summary_frame = customtkinter.CTkFrame(self.summary_frame, fg_color = '#05122d', bg_color = '#05122d')
                    self.button_summary_frame.grid(row=2, column=0, columnspan=5, sticky="nsew", padx=200, pady=80)
                    self.button_summary_frame.grid_columnconfigure(0,weight=5)
                    self.button_summary_frame.grid_columnconfigure(1,weight=1)
                    self.button_summary_frame.grid_columnconfigure(2,weight=1)
                    self.button_summary_frame.grid_columnconfigure(3,weight=1)
                    self.button_summary_frame.grid_columnconfigure(4,weight=5)
                    self.button_summary_frame.rowconfigure(0,weight=1)
                    self.button_summary_frame.rowconfigure(1,weight=1)
                
                    # Save Results
                    self.save_results_button = customtkinter.CTkButton(self.button_summary_frame, corner_radius=20, height=40, border_spacing=10, text="Save Results (Most Recent Run)",
                                                                  bg_color='#05122d',
                                                                  fg_color="#4B0082",
                                                                  hover_color="#560067",
                                                                  text_color=("gray10", "gray90"),
                                                                  font = my_button_font,
                                                                  anchor="center", command=self.save_results_button_event)
                    self.save_results_button.grid(row=0, column=1, padx = padding_x, sticky = "ew")
                    
                    
                    # Create search bar for Email
                    self.email_search_bar = customtkinter.CTkEntry(self.button_summary_frame, placeholder_text ="Enter Email(s) (ex. john.doe@gmail.com, ...)",
                                            fg_color="#14206d",  # Foreground color (entry background)
                                          bg_color="#05122d",  # Background color (frame background)
                                          text_color="#ffffff",  # Text color
                                          font=my_text_font)
                    self.email_search_bar.grid(row=0, column=2, padx = padding_x, sticky = "ew")
                    
                    # Create Email button
                    self.email_button = customtkinter.CTkButton(self.button_summary_frame, corner_radius=20, height=40, border_spacing=10, text="Send Saved Results to Email",
                                                                  fg_color="#4B0082",  
                                                                  bg_color= '#05122d',hover_color="#560067",
                                                                  text_color=("gray10", "gray90"),
                                                                  font = my_button_font,
                                                                  anchor="center", command=self.send_email_button_event)
                    self.email_button.grid(row=1, column=2, padx = padding_x, sticky = "new")
                    
                    
                    
                    
                    self.restart_program_button = customtkinter.CTkButton(self.button_summary_frame, corner_radius=20, height=40, border_spacing=10, text="Exit Back to Start Menu",
                                                                  bg_color='#05122d',
                                                                  fg_color="#4B0082",
                                                                  hover_color="#560067",
                                                                  text_color=("gray10", "gray90"),
                                                                  font = my_button_font,
                                                                  anchor="center", command=self.restart_program_button_event)
                    self.restart_program_button.grid(row=0, column=3, padx = padding_x, sticky = "ew")
                
                
            # Function to display when no model is saved
            def plot_no_model_2(self, model_frame, model_event_next, model_event_back, model_name):
                
                
                if (model_frame != self.summary_frame):
                    model_menu_image_path = os.path.join(background_images_path, "Model_Page.png")
                else:
                    model_menu_image_path = os.path.join(background_images_path, "Home_Page.png")
                # Create background image
                image = PIL.Image.open(model_menu_image_path)
                background_image_model = customtkinter.CTkImage(image, size=(1920, 1080))
                
                
                self.background_label = customtkinter.CTkLabel(model_frame,
                                                             image=background_image_model,
                                                             text="")  # Empty text
                self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
             
    
                self.model_frame_Label_Title = customtkinter.CTkLabel(model_frame, text=model_name, font=customtkinter.CTkFont(family="RobotoCondensed-ExtraBoldItalic", size=50, weight="bold", slant = "italic"), 
                                                                     bg_color='#05122d', text_color=("white"))
                self.model_frame_Label_Title.grid(row=0, column=0, padx=20, pady=(40, 10), columnspan=2, sticky = "n")  
                
                if (model_frame != self.summary_frame): 
                    self.next_button = customtkinter.CTkButton(model_frame, text="→", command=model_event_next, height=40, width=45, font=customtkinter.CTkFont(family="Roboto Flex", size= 30), corner_radius=40, bg_color='#05122d',fg_color="#4B0082")
                    self.next_button.grid(row=4, column=1, padx=20, pady=20,  sticky = "se")  
                    
                    if (model_frame != self.model_1_frame):
                        self.next_button = customtkinter.CTkButton(model_frame, text="←", command=model_event_back, height=40, width=45, font=customtkinter.CTkFont(family="Roboto Flex", size= 30), corner_radius=40, bg_color='#05122d',fg_color="#4B0082")
                        self.next_button.grid(row=4, column=0, padx=20, pady=20,  sticky = "sw") 
                else:
                    # Create frame for save results and back to start menu buttons
                    self.button_summary_frame = customtkinter.CTkFrame(self.summary_frame, fg_color = '#05122d', bg_color = '#05122d')
                    self.button_summary_frame.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=200, pady=80)
                    self.button_summary_frame.grid_columnconfigure(0,weight=2)
                    self.button_summary_frame.grid_columnconfigure(1,weight=1)
                    self.button_summary_frame.grid_columnconfigure(2,weight=2)
                    self.button_summary_frame.rowconfigure(0,weight=1)
                    
                    self.restart_program_button = customtkinter.CTkButton(self.button_summary_frame, corner_radius=20, height=40, border_spacing=10, text="Exit Back to Start Menu",
                                                                  bg_color='#05122d', 
                                                                  fg_color="#4B0082",
                                                                  hover_color="#560067",
                                                                  text_color=("gray10", "gray90"),
                                                                  font = my_button_font,
                                                                  anchor="center", command=self.restart_program_button_event)
                    self.restart_program_button.grid(row=0, column=1, sticky = "new")
            
            print("Forecasting Predicted Datset Models.")
            # Begin RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.connect(rgb_lights)
            except:
                pass
            
            try:
                Power_Forecasting_Corsair_RGB.waiting(rgb_lights)
            except:
                pass
            
            weather_data =  pd.read_excel(input_data_filename, sheet_name = "Weather_Forecast", header = 0)
            weather_data = Power_Forecasting_dataCollectionAndPreprocessingFlow.add_calendar_columns(weather_data)
            weather_data = Power_Forecasting_dataCollectionAndPreprocessingFlow.add_lags_to_weather_data(weather_data, 23)
            weather_data_cnn = weather_data.copy()
            
    
            # Remove X days because of lags
            weather_data = weather_data.reset_index(drop = True)
            
            for day in range(cnn_days_back):
                index_remove_day = weather_data[(weather_data['Day'] == weather_data["Day"].iloc[0])].index     
                weather_data = weather_data.drop(index_remove_day, axis='index', inplace = False).reset_index(drop=True)
        
            # Open weather scaler
            scaler_path = os.path.join(saved_model_path, "weather_scaler_"+input_data_basename+".pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            weather_scaler = joblib.load(scaler_path)
            
            
            # Drop temporary year month day hour columns
            weather_data_dropped = weather_data.drop(columns=['Year', 'Month', 'Day', 'Hour'])
            # Drop temporary year month day hour columns
            weather_data_cnn_dropped = weather_data_cnn.drop(columns=['Year', 'Month', 'Day', 'Hour'])
            
            # Normalize Weather
            norm_weather_data = weather_scaler.transform(weather_data_dropped)
            norm_weather_data = pd.DataFrame(norm_weather_data, columns = weather_data_dropped.columns)
            
            # Normalize CNN Weather
            norm_weather_data_cnn = weather_scaler.transform(weather_data_cnn_dropped)
            norm_weather_data_cnn = pd.DataFrame(norm_weather_data_cnn, columns = weather_data_cnn_dropped.columns)
            
      
            ###############################################################################
            # Import and predict Models
            ###############################################################################
            num_of_days = weather_data["Day"].nunique()
            
            total_features = []
            # Convert year, month, day, hour to boolean values
            # Day range 1 to 31 (subtract last day for 0 condition)
            days = [*range(1, 31)]
            # Hour range 0 to 23 (subtract last hour for 0 condition)
            hours = [*range(0, 23)]
            # Hour range 0 to 23 (subtract last month for 0 condition)
            months = [*range(1, 12)]
            
            # Year range depending on weather data columns
            years_range = weather_data.columns
            
            for year in years_range:
                if ("Year_" in year):
                    total_features.append(year)
    
            for month in months:
                total_features.append("Month_" + str(month))
    
            for day in days:
                total_features.append("Day_" + str(day))
            
            for hour in hours:
                total_features.append("Hour_" + str(hour))
                         
            for feature in selected_features:
                total_features.append(feature)
                
            total_features_temp = total_features.copy()
            for lag in range (1, 24):
                for feature in total_features_temp:
                    if (feature == "Weekend" or feature == "Season" or feature == "Holiday (Ontario)" or ("Year_" in feature) or ("Month_" in feature) or ("Day_" in feature)):
                        continue
                    else:
                        total_features.append(feature+"_Lag_"+str(lag))
            
            
            
            # Dictionary for model prediction dataframes error 
            # model -> Value
            Y_pred_denorm_saved_df = {}
            
            for model_name in selected_models:
                if model_name == "K-Nearest Neighbors":
                    model_name = "KNN"
                if model_name == "Convolutional Neural Network":
                    model_name = "CNN" 
                if model_name == "Linear Regression":
                    model_name = "LR"
                if model_name == "X Gradient Boost":
                    model_name = "XGB" 
                
                # Import saved CSV into script as dataframes
                if model_name == "CNN":
                    X_test = norm_weather_data_cnn[total_features]
                else:
                    X_test = norm_weather_data[total_features]
                
                # Load model from gui_pickup folder using joblib
                try:
                    if model_name == "CNN":
                        pipe_saved = models.load_model(os.path.join(saved_model_path, (model_name+"_"+input_data_basename+"_Model_" + "_".join(selected_features_3_digits) + ".keras")))
                    else:
                        pipe_saved = joblib.load(os.path.join(saved_model_path, (model_name+"_"+input_data_basename+"_Model_" + "_".join(selected_features_3_digits) + ".pkl")))
                except:
                    continue
    
                if (model_name == "CNN"):
                    
                    for feature in X_test.columns:
                      if (feature == "Weekend" or feature == "Season" or feature == "Holiday (Ontario)" or ("Year_" in feature) or ("Month_" in feature) or ("Day_" in feature) or ("Hour_" in feature)):
                        X_test[feature].astype('bool')
                      if ("Lag" in feature):
                        X_test = X_test.drop(columns = [feature])
                    
                    # Create dataframe without humidity and speed
                    data = X_test
                    
                    #window_size = 168  # Last 168 hours (one week)
                    window_size = 24*cnn_days_back  # Last 24 hours (one day)
                    forecast_horizon = 24  # Next 24 hours
                    
                    
                    # Create input-output pairs using a sliding window
                    Y_pred_saved = pd.DataFrame(columns=['TOTAL_CONSUMPTION'])
                    for day in range(num_of_days):
                        X_data = []
    
                        X_data.append(data.iloc[day*24:day*24 + window_size + forecast_horizon].values)  # Collect 24 hours of feature data
                        
                        # Convert to numpy arrays
                        X_data = np.array(X_data, dtype=np.float16)
                        X_data = np.expand_dims(X_data, axis=-1)
        
                        # Predict using loaded model
                        Y_pred_cnn = pipe_saved.predict(X_data)
                        Y_pred_cnn = pd.DataFrame(Y_pred_cnn)
                        
                        # Ensure Y_pred and Y_test are reshaped correctly
                        Y_pred_cnn = Y_pred_cnn.values.reshape(-1, 1)
                        
                        Y_pred_cnn = pd.DataFrame(Y_pred_cnn, columns=['TOTAL_CONSUMPTION'])
                        
                        Y_pred_saved = pd.concat([Y_pred_saved, Y_pred_cnn], axis=0, ignore_index=True)
    
                else:
                    # Predict using loaded model
                    Y_pred_saved = pipe_saved.predict(X_test)
                    
                    # Ensure Y_pred and Y_test are reshaped correctly
                    Y_pred_saved = Y_pred_saved.reshape(-1, 1)
                
                
                # Denormalize Y_pred and Y_test with min_max_scaler_y.pkl using joblib
                scaler_path = os.path.join(saved_model_path, "power_scaler_"+input_data_basename+".pkl")
                if not os.path.exists(scaler_path):
                    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
                power_scaler = joblib.load(scaler_path)
                
                # Inverse Transform power scaler
                Y_pred_denorm_saved = power_scaler.inverse_transform(Y_pred_saved)
    
                # Save power scaler to dictionary of models
                Y_pred_denorm_saved_df[model_name] = pd.DataFrame(Y_pred_denorm_saved, columns=['TOTAL_CONSUMPTION'])
    
                # Convert to MW
                Y_pred_denorm_saved_df[model_name] = Y_pred_denorm_saved_df[model_name]*0.001
                
                # Convert to float64
                Y_pred_denorm_saved_df[model_name] = Y_pred_denorm_saved_df[model_name].astype(float)
                
                # Round to 4 decimal places
                Y_pred_denorm_saved_df[model_name] = Y_pred_denorm_saved_df[model_name].round(decimals = 4)
                
                print("    " + model_name + " - Forecast Generated.")
                
            # Dictionary for model prediction dataframes error 
            # model -> Value
            table_values = {}
            metrics_values = {}
            save_results_dic.clear()
            table_values_df = pd.DataFrame() 
            
            weather_data["DATE"] = pd.to_datetime(weather_data[["Year", "Month", "Day","Hour"]])
    
            
            
            year = str(weather_data["Year"].iloc[0])
            month = str(weather_data["Month"].iloc[0])
            day = str(weather_data["Day"].iloc[0])
            
            
            
            for day_num in range (num_of_days):
                # Append next day to another dataframe to plot on same figure
                if (day_num == 0):
                    title = "Hourly Power Consumption"
        
                for model_name in selected_models:
                    if model_name == "K-Nearest Neighbors":
                        model_name = "KNN"
                    if model_name == "Convolutional Neural Network":
                        model_name = "CNN" 
                    if model_name == "Linear Regression":
                        model_name = "LR"
                    if model_name == "X Gradient Boost":
                        model_name = "XGB" 
    
                    try:
                    
                        Y_pred_denorm_saved_df_day = Y_pred_denorm_saved_df[model_name].iloc[(24*day_num):(24*day_num + 24)]
                            
                        if (day_num == 0):
                            table_values_df["Hour"] = (Y_pred_denorm_saved_df_day.index%24)
                            table_values[model_name] = table_values_df
                           
                        table_values[model_name]["Forecasted Consumption: Day " + str(day_num+1) + " (MW)"] = Y_pred_denorm_saved_df_day["TOTAL_CONSUMPTION"].reset_index(drop = True)
                        table_values[model_name] = table_values[model_name].round(decimals = 4)
    
                    except:
                        continue 
    
            for model_name in selected_models:
                if model_name == "K-Nearest Neighbors":
                    model_name = "KNN"
                if model_name == "Convolutional Neural Network":
                    model_name = "CNN" 
                if model_name == "Linear Regression":
                    model_name = "LR"
                if model_name == "X Gradient Boost":
                    model_name = "XGB" 
                try:  
                    Y_pred_denorm_saved_df[model_name] = Y_pred_denorm_saved_df[model_name].reset_index(drop = True)
                    weather_data = weather_data.reset_index(drop = True)
                    
                    save_results_dic[model_name] = pd.concat([weather_data[["Year", "Month", "Day", "Hour"]], Y_pred_denorm_saved_df[model_name]], axis=1)
                    save_results_dic[model_name].columns.values[4] = "FORECASTED CONSUMPTION (MW)"
                    save_results_dic[model_name] = save_results_dic[model_name].rename(columns={"Year": "YEAR"})
                    save_results_dic[model_name] = save_results_dic[model_name].rename(columns={"Month": "MONTH"})
                    save_results_dic[model_name] = save_results_dic[model_name].rename(columns={"Day": "DAY"})
                    save_results_dic[model_name] = save_results_dic[model_name].rename(columns={"Hour": "HOUR"})
                    
                    metrics_model_path = os.path.join(saved_model_path, model_name+"_"+input_data_basename+"_Metrics_" + "_".join(selected_features_3_digits) + ".csv") 
                    metrics_values[model_name] = pd.read_csv(metrics_model_path, header=0)
                    metrics_values[model_name] = metrics_values[model_name].round(decimals = 4)
                    
                    save_results_dic[model_name] = pd.concat([save_results_dic[model_name],  metrics_values[model_name][["MAPE (%)", "MAE (MW)", "r2", "MSE (MW Squared)", "RMSE (MW)"]]], axis=1).fillna("")
                    
                    
                    
                    metrics_values_columns = pd.DataFrame([metrics_values[model_name].columns], columns = metrics_values[model_name].columns)
                    table_values_columns = pd.DataFrame([table_values[model_name].columns], columns = table_values[model_name].columns)
                    
                    metrics_values[model_name] = pd.concat([metrics_values_columns, metrics_values[model_name].iloc[0:]]).reset_index(drop=True) 
                    table_values[model_name] = pd.concat([table_values_columns, table_values[model_name].iloc[0:]]).reset_index(drop=True)
                except:
                    continue
            count_no_models = 0
            try:
                plot_figures_model_2(self, weather_data, Y_pred_denorm_saved_df["LR"], metrics_values["LR"], table_values["LR"], self.model_1_frame, self.model_2_button_event, "N/A", model_names_list[0])
            except:
                plot_no_model_2(self, self.model_1_frame, self.model_2_button_event, "N/A", "NO SAVED MODEL FOR " + model_names_list[0])
                count_no_models = count_no_models + 1
            try:
                plot_figures_model_2(self, weather_data, Y_pred_denorm_saved_df["XGB"], metrics_values["XGB"], table_values["XGB"], self.model_2_frame, self.model_3_button_event, self.model_1_button_event, model_names_list[1])
            except:
                plot_no_model_2(self, self.model_2_frame, self.model_3_button_event, self.model_1_button_event, "NO SAVED MODEL FOR " + model_names_list[1])
                count_no_models = count_no_models + 1
            
            try:
                plot_figures_model_2(self, weather_data, Y_pred_denorm_saved_df["KNN"], metrics_values["KNN"], table_values["KNN"], self.model_3_frame, self.model_4_button_event, self.model_2_button_event, model_names_list[2])
            except:
                plot_no_model_2(self, self.model_3_frame, self.model_4_button_event, self.model_2_button_event, "NO SAVED MODEL FOR " + model_names_list[2])
                count_no_models = count_no_models + 1
            
            try:
                plot_figures_model_2(self, weather_data, Y_pred_denorm_saved_df["CNN"], metrics_values["CNN"], table_values["CNN"], self.model_4_frame, self.summary_button_event, self.model_3_button_event, model_names_list[3])
            except:
                plot_no_model_2(self, self.model_4_frame, self.summary_button_event, self.model_3_button_event, "NO SAVED MODEL FOR " + model_names_list[3])
                count_no_models = count_no_models + 1
            
            try:
                if (count_no_models == 4):
                    plot_no_model_2(self, self.summary_frame, "N/A","N/A", "No Saved Models")  
                else:
                    plot_figures_model_2(self, weather_data, save_results_dic, "N/A", "N/A", self.summary_frame, "N/A", "N/A", "Summary of All Models")
                
            except:
                plot_no_model_2(self, self.summary_frame, "N/A","N/A", "No Saved Models")
            
            count_no_models = 0
            self.select_frame_by_name("Model: Linear Regression")
            print("Done Forecating!")
            print("\n")
            
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.done_waiting(rgb_lights)
            except:
                pass
        except Exception as error:
            print("An exception occurred:", error)
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.error(rgb_lights)
            except:
                pass
        
    def upload_open_cnn_param_button_event(self):
        cnn_param_template_path = os.path.join(x_y_input_path, "CNN_Hyperparameters.xlsx")
        os.startfile(cnn_param_template_path)

    def open_file_button_event(self):
        input_excel_template_path = os.path.join(input_excel_path, "Input_Data_Excel_File_Template.xlsm")
        os.startfile(input_excel_template_path)
    
    def upload_file_button_event(self):
        global input_data_filename, input_data_basename
        filetypes = (('excel files', '*.xlsm'), ('All files', '*.*'))  
        input_data_filename = fd.askopenfilename(title='Input Excel File', initialdir=input_excel_path, filetypes=filetypes)
        input_data_basename = os.path.basename(input_data_filename)
        input_data_basename_split = os.path.splitext(input_data_basename) 
        input_data_basename = input_data_basename_split[0]
        
    def save_results_button_event(self):
        try:
            blank_dataframe = pd.DataFrame(columns=['No Saved Results'])
            saved_results_path = os.path.join(output_results_path, "Output_Results.xlsx")
            writer = pd.ExcelWriter(saved_results_path, engine = "xlsxwriter")
            for model_name in model_names_list:
                if model_name == "K-Nearest Neighbors":
                    model_name = "KNN"
                if model_name == "Convolutional Neural Network":
                    model_name = "CNN" 
                if model_name == "Linear Regression":
                    model_name = "LR"
                if model_name == "X Gradient Boost":
                    model_name = "XGB" 
                try:
                    save_results_dic[model_name].to_excel(writer, sheet_name = (model_name + "_Model_Results"), index = False)
                    workbook  = writer.book
                    worksheet = writer.sheets[(model_name + "_Model_Results")]
                    worksheet.set_column('A:D', 10)
                    worksheet.set_column('E:F', 30)
                    worksheet.set_column('G:K', 20)
                except:
                    blank_dataframe.to_excel(writer, sheet_name = (model_name + "_Model_Results"), index = False)
                    workbook  = writer.book
                    worksheet = writer.sheets[(model_name + "_Model_Results")]
                    worksheet.set_column('A:A', 29)
                    continue
            writer.close()
            os.startfile(saved_results_path)
        except Exception as error:
            print("An exception occurred:", error)
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.error(rgb_lights)
            except:
                pass
    
    def send_email_button_event(self):
        try:
            
            # Begin RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.connect(rgb_lights)
            except:
                pass
            
            try:
                Power_Forecasting_Corsair_RGB.waiting(rgb_lights)
            except:
                pass
            
            receiver_email = self.email_search_bar.get()
            receiver_email = "".join(receiver_email.split())
            receiver_email = receiver_email.split(",")
            print(receiver_email)
            
            subject = "BV05 Power Forecasting - Results"
            body = "Hello,\n\nThank you for using our Power Forecasting tool. Please find the attached data file containing the detailed forecasted power consumption values for your selected area and selected machine learning models. We trust this information will be valuable to your analysis.\n\nIf you have any questions or need further assistance, please do not hesitate to contact us.\n\nSincerely,\nThe Power Forecasters\nHanad Mohamud | Clover K. Joseph | Joseph Sposato | Janna Wong\nEmail: powerforecasting@gmail.com"
            sender_email = "powerforecasting@gmail.com"
            recipient_email = receiver_email
            sender_password = "eepk rmfp cmlu lyup"
            smtp_server = 'smtp.gmail.com'
            smtp_port = 465
            
            saved_results_path = os.path.join(output_results_path, "Output_Results.xlsx")
            path_to_file = saved_results_path
            
            # MIMEMultipart() creates a container for an email message that can hold
            # different parts, like text and attachments and in next line we are
            # attaching different parts to email container like subject and others.
            message = MIMEMultipart()
            message['Subject'] = subject
            message['From'] = sender_email
            message['To'] = ', '.join(recipient_email)
            body_part = MIMEText(body)
            message.attach(body_part)
            
            # section 1 to attach file
            with open(path_to_file,'rb') as file:
                # Attach the file with filename to the email
                message.attach(MIMEApplication(file.read(), Name="Output_Results.xlsx"))
            
            # secction 2 for sending email
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
               server.login(sender_email, sender_password)
               server.sendmail(sender_email, recipient_email, message.as_string())
            print("Message Sent!")
            
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.done_waiting(rgb_lights)
            except:
                pass
        except Exception as error:
            print("An exception occurred:", error)
            # Complete RGB waiting sequence
            try:
                Power_Forecasting_Corsair_RGB.error(rgb_lights)
            except:
                pass

    def restart_program_button_event(self): 
        global restart_program
        restart_program = 1
        
        # Complete RGB waiting sequence
        try:
            Power_Forecasting_Corsair_RGB.done_waiting(rgb_lights)
        except:
            pass
        self.quit()
        
    ###############################################################################
    # Functions for home menu non button events
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
        global selected_features, selected_features_3_digits
        selected_features = self.scrollable_features_checkbox_frame.get_checked_items()
        
        # Create 3 digit subset of string for saving models
        selected_features_3_digits = []
        for selected_features_str in selected_features:
            if selected_features_str == "Wind Speed":
                selected_features_3_digits.append("Wsd")
            elif selected_features_str == "Windchill":
                selected_features_3_digits.append("Wch")
            else:
                selected_features_3_digits.append(selected_features_str[:3])
        
        #print("checkbox frame modified: ", selected_features)
        #print("checkbox frame modified: ", selected_features_3_digits)
        
    def models_checkbox_event(self):
        global selected_models
        selected_models = self.scrollable_models_checkbox_frame.get_checked_items()
        #print("checkbox frame modified: ", selected_models)
    
    def show_table_checkbox_event(self):
        self.detailed_table_checkbox_var.get()
        #print("checkbox toggled, current value:", self.detailed_table_checkbox_var.get())
    
    
    
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
    user_run = ["C:\\power-forecasting-capstone\\data", 5]

    ###############################################################################
    ############### MAKE SURE TO CHANGE BEFORE RUNNING CODE #######################
    ###############################################################################
    # Paste student name_run for whoever is running the code
    run_student = user_run
    if (run_student[1] == joseph_laptop_run[1]):
        print("JOSEPH IS RUNNING!")
    elif (run_student[1] == hanad_run[1]):
        print("HANAD IS RUNNING!")
    elif (run_student[1] == janna_run[1]):
        print("JANNA IS RUNNING!")
    elif (run_student[1] == clover_run[1]):
        print("CLOVER IS RUNNING!")
    elif (run_student[1] == user_run[1]):
        print("USER IS RUNNING!")
    else:
        print("ERROR!! NO ELIGIBLE STUDENT!")
        
    dirs_inputs = run_student[0]
    
    #%% Run GUI
    app = App()
    app.mainloop()
    restart_program = 1
    
    while restart_program == 1:
        try:
            app.destroy()
            app = App()
            app.mainloop()
        except: 
            restart_program = 0
            

        