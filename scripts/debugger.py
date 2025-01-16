import requests
from datetime import datetime, timedelta

# Choose FSA for data collection + Get latitude and longitude of chosen fsa
fsa = "L9G"
fsa_map = {
    "L9G": {"lat": 43.17, "lon": -79.94}
}
lat = fsa_map[fsa]["lat"]
lon = fsa_map[fsa]["lon"]
bbox = f'{lon-0.1},{lat-0.1},{lon+0.1},{lat+0.1}'

# Choose date range for data collection
start_year = 2018
start_month = 1
start_day = 1
start_hour = 0

end_year = 2018
end_month = 12
end_day = 31
end_hour = 23

start_date = datetime(start_year, start_month, start_day, start_hour)
end_date = datetime(end_year, end_month, end_day, end_hour)
next_date = start_date + timedelta(days=1) - timedelta(hours=1)

# Testing out date commands
print(start_date.isoformat())
print(next_date.isoformat())
print(end_date.isoformat())

# Set up URL for data collection
url = f'https://api.weather.gc.ca/collections/climate-hourly/items?bbox={bbox}&datetime={start_date.isoformat()}Z/{next_date.isoformat()}Z'
print(f"URL: {url}")

response = requests.get(url)
print(f"Status Code: {response.status_code}")
#print(f"Response Text: {response.text}")

if response.status_code == 200:
    data = response.json()
    if 'features' in data:
        print(f"Number of features: {len(data['features'])}")
        if len(data['features']) > 0:
            for entry in data['features']:
                print("Date/Time (LST): ", entry['properties'].get('LOCAL_DATE'))
                print(entry['properties'])
        else:
            print("No features found in the response.")
    else:
        print("'features' key not found in the response.")
else:
    print("Failed to fetch data from the API.")