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


class App(customtkinter.CTk):
    def __init__(self):  
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
        
        image_path = os.path.join(dirs_inputs, "Model_Plots")    
            
        super().__init__()

        self.title("Power System Forecasting.py")
        self.geometry("700x450")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(6, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="Power Forecasting", 
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


        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")
##############################################
        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)
        
        
        self.home_frame_Label = customtkinter.CTkLabel(self.home_frame, text="WELCOME TO POWER FORECASTING\nPlease Select:\n -Postal Code\n-Start Date\n-End Date")
        self.home_frame_Label.grid(row=0, column=0, padx=20, pady=20)
###############################################        
        # create second frame (Model 1)
        self.model_1_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        
        self.model_1_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "Model1.png")), size=(400, 400))
        
        self.model_1_frame_image_label = customtkinter.CTkLabel(self.model_1_frame, text="", image=self.model_1_image)
        self.model_1_frame_image_label.grid(row=0, column=0, padx=20, pady=10)

##############################################
        # create third frame (Model 2)
        self.model_2_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
       
        
##############################################
        # create fourth frame (Model 3)
        self.model_3_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        
        
##############################################
        # create fifth frame (Model 4)
        self.model_4_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # select default frame
        self.select_frame_by_name("Home")

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "Home" else "transparent")
        self.model_1_button.configure(fg_color=("gray75", "gray25") if name == "Model 1" else "transparent")
        self.model_2_button.configure(fg_color=("gray75", "gray25") if name == "Model 2" else "transparent")
        self.model_3_button.configure(fg_color=("gray75", "gray25") if name == "Model 3" else "transparent")
        self.model_4_button.configure(fg_color=("gray75", "gray25") if name == "Model 4" else "transparent")

        # show selected frame
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

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)


if __name__ == "__main__":
    app = App()
    app.mainloop()