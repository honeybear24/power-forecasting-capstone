# Script to help Hanad leanr about API calls and how to play with data
# Link to video: https://www.youtube.com/watch?v=-oPuGc05Lxs

# In order to pull data from APIs, you will need the requests module (pip install requests if needed)
import requests
import pandas as pd

# Base URL for test API
baseurl = "https://rickandmortyapi.com/api/"
endpoint = "character"

def main_request(baseurl, endpoint, x):
    
    # Make request to API
    r = requests.get(baseurl + endpoint + f"/?page={x}")
    return r.json()

def get_pages(response):
    return response['info']['pages']

def parse_json(response):
    charlist = []
    for item in response['results']:
        char = {
            'id': item['id'],
            'name': item['name'],
            'no_ep.': len(item['episode']),
        }
        
        charlist.append(char)
    return charlist

# Extracting data from API response
mainlist = []
data = main_request(baseurl, endpoint, 1)
for x in range(1, get_pages(data)):
    print(x)
    mainlist.extend(parse_json(main_request(baseurl, endpoint, x)))

df = pd.DataFrame(mainlist)

#print(df.head(), df.tail())
df.to_csv('charlist.csv', index=False)

