# Pull weather data from Weather Stations using Environment Canada API
# Useful link
# https://eccc-msc.github.io/open-data/msc-data/climate_obs/readme_climateobs-datamart_en/#announcements-from-the-dd_info-mailing-list
# https://dd.weather.gc.ca/climate/observations/hourly/csv/ON/

## Imports
import requests
import pandas as pd
import csv

# URLs to get data- Environment Canada
baseurl = "https://dd.weather.gc.ca/climate/observations/hourly/csv/ON/"

# Initial API Pull
r = requests.get(baseurl)
#print(r) # 200 = Sucessfull

# File Name Format - climate_hourly_ON_XXXXXXX_YYYY_P1H.csv
#    XXXXXXX = Climate Station ID
#    YYYY    = Year 

# Variables 
test_data = "c:/Users/hanad/capstone_github/power-forecasting-capstone/data/Test/"


## GOAL: Pull CSV from API and store data in Dataframe in same proper format - Testing with known weather station that is confirmed to be good
# Test CSV to pull
file_name = "climate_hourly_ON_6153193_2020_P1H.csv"

# Pull data from API
data = requests.get(baseurl + file_name)
#print(data.headers)

# Decode data and store in Dateframe
decoded_data = data.content.decode("ISO-8859-1")
cr = csv.reader(decoded_data.splitlines(), delimiter=',')
csv_data = pd.DataFrame(cr)

# Headings of df is wrong (current headings are numbers whole first row contains actual heading names), correct headings 
print("Fixing Dataframe headings!")
for column in csv_data.columns:
    print("Current: " + str(column) + " | New: " + csv_data[column][0]) # Printing out name of colum
    csv_data.rename(columns={column: csv_data[column][0]}, inplace=True)
csv_data.drop(index=0, inplace=True)

# Send out CSV data for examination
csv_data.to_csv(test_data + "test.csv", index=False)
