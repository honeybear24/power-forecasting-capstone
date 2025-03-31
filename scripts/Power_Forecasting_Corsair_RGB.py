# Libraries
from pyrgbdev import Corsair
import time

# Connect to ICUE
def connect(a):
    a.connect() 
    
# Wait for code to finish
def waiting(a):  
    for g in range(0, 255, 1):
        a.set_rgb({"ALL": (255, g, 0)})
        time.sleep(0.0001)

    for r in range(255, 0, -1):
        a.set_rgb({"ALL": (r, 255, 0)})
        time.sleep(0.0001)

    for b in range(0, 255, 1):
        a.set_rgb({"ALL": (0, 255, b)})
        time.sleep(0.0001)

    for g in range(255, 0, -1):
        a.set_rgb({"ALL": (0, g, 255)})
        time.sleep(0.0001)

    for r in range(0, 255, 1):
        a.set_rgb({"ALL": (r, 0, 255)})
        time.sleep(0.0001)

    for b in range(255, 0, -1):
        a.set_rgb({"ALL": (255, 0, b)})
        time.sleep(0.0001)
    a.set_rgb({"ALL": (255, 255, 0)})
    return(a)

# Disconnect when finished
def done_waiting(a):
    a.disconnect()

# Change colour to red if an error occurs
def error(a):
    a.set_rgb({"ALL": (255, 0, 0)})
