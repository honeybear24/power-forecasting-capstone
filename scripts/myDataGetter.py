# Used to collect collect meteorlogical data from Envirnment Canada's Weather API + Power Demand data from IESO

### Imports ###
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests



### Set Up ###
# Student Directory
hanad_run = ["c:/Users/hanad/capstone_github/power-forecasting-capstone/data", 1]
clover_run = ["./data", 2]
joseph_laptop_run = ["C:\\Users\\sposa\\Documents\\GitHub\\power-forecasting-capstone\\data", 3]
joseph_pc_run = ["D:\\Users\\Joseph\\Documents\\GitHub\\power-forecasting-capstone\\data", 3]
janna_run = ["./data", 4]

# Paste student name_run for whoever is running the code
run_student = hanad_run
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
    sys.exit() # Kill script

# Set Up data direcotry path for data collection
data_path = run_student[0]
target_dir = data_path + "/raw_data"

if not os.path.exists(target_dir): # Create directory if it does not exist
    os.makedirs(target_dir)

# Set up FSA Dictionary that will map FSA names to a latitude and longitude
fsa_map = {
    "L9G": {"lat": 43.27, "lon": -79.95}, # Target FSA for Tests
    "M9M": {"lat": 43.74, "lon": -79.54}, # Jane and Finch
    "L9H": {"lat": 43.32, "lon": -79.98}, # Dundas
}


# Link - f'https://api.weather.gc.ca/collections/climate-hourly/items?bbox={bbox}&datetime={current_date.isoformat()}/{next_date.isoformat()}'
# bbox = boubding box around a given latitude and longitude, gets all data within bbox given
# datetime = start_date.isoformat() / end_date.isoformat() - gets all data within the given time frame

### Calling Data ###
# Choose FSA for data collection + Get latitude and longitude of chosen fsa
fsa = "L9G"
lat = fsa_map[fsa]["lat"]
lon = fsa_map[fsa]["lon"]
bbox = f'{lon-0.5},{lat-0.5},{lon+0.5},{lat+0.5}'

# Choose date range for data collection
start_year = 2018
start_month = 1
start_day = 1
start_hour = 0

end_year = 2023
end_month = 12
end_day = 31
end_hour = 23

start_date = datetime(start_year, start_month, start_day, start_hour,0,0) # Making datetime object for start and end dates
end_date = datetime(end_year, end_month, end_day, end_hour,0,0)
next_date = start_date + timedelta(days=1) - timedelta(hours=1) # used to datetime object for last hour of the day

## Testing out date commands
print(start_date.isoformat())
print(next_date.isoformat())
print(end_date.isoformat())




# Set up URL for data collection
url = f'https://api.weather.gc.ca/collections/climate-hourly/items?bbox={bbox}&datetime={start_date.isoformat()}/{next_date.isoformat()}'

response = requests.get(url)
print(response.status_code)
#print(response.text)
#print(response.json())
data = response.json()
#print(data['features'])

print(data['features'][0])
print(len(data['features']))
if 'features' in data and len(data['features']) > 0:
    for entry in data['features']:
        print(entry['properties'].get('STATION_NAME'))
        print(entry['properties'].get('LOCAL_DATE'))
        print(entry['properties'].get('LOCAL_HOUR'))
        print(entry['properties'].get('TEMP'))
        print(entry['properties'].get('WIND_SPEED'))
        print(entry['properties'].get('PRECIP_AMOUNT'))
        print(entry['properties'].get('RELATIVE_HUMIDITY'))
        print(entry['properties'].get('WINDCHILL'))
        print("")


