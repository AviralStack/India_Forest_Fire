import pandas as pd
import requests
import time
import random
from datetime import datetime, timedelta


INPUT_FILE = "final_india_dataset.csv" 
OUTPUT_FILE = "balanced_india_dataset.csv" 

def fetch_weather(lat, lon, date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": ["temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max"],
        "timezone": "auto"
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 429:
            time.sleep(10)
            return fetch_weather(lat, lon, date)
            
        r.raise_for_status()
        data = r.json()
        daily = data.get('daily', {})
        
        return (
            daily.get('temperature_2m_max', [None])[0],
            daily.get('relative_humidity_2m_mean', [None])[0],
            daily.get('wind_speed_10m_max', [None])[0]
        )
    except:
        return None, None, None

def get_safe_date(fire_date_str):
    """
    Takes a fire date (YYYY-MM-DD) and picks a random date 
    roughly 6 months away (flipping the season) to find a 'safe' day.
    """
    fire_date = datetime.strptime(fire_date_str, "%Y-%m-%d")
    
    shift_days = random.randint(120, 240) 
    safe_date = fire_date - timedelta(days=shift_days)
    
    if safe_date.year < 2020:
        safe_date = fire_date + timedelta(days=shift_days)
        
    return safe_date.strftime("%Y-%m-%d")

print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f" Original Data: {len(df)} rows (All Fires)")

if pd.io.common.file_exists(OUTPUT_FILE):
    print(" Resuming")
    final_df = pd.read_csv(OUTPUT_FILE)
else:
    final_df = df.copy()
    final_df.to_csv(OUTPUT_FILE, index=False)

print(" Generating Non-Fire")

rows_to_add = []
total_needed = len(df)
count = 0

for index, row in df.iterrows():
 
    current_zeros = len(final_df[final_df['fire_occurred'] == 0])
    if current_zeros >= total_needed:
        break

    lat, lon, fire_date = row['latitude'], row['longitude'], row['acq_date']
    
    safe_date = get_safe_date(fire_date)
    
    temp, hum, wind = fetch_weather(lat, lon, safe_date)
    
    if temp is not None:
        print(f"[{count+1}/{total_needed}] Generated NO-FIRE at {lat},{lon} on {safe_date} ({temp}°C)")
        
        new_row = {
            'latitude': lat,
            'longitude': lon,
            'acq_date': safe_date,
            'temp_c': temp,
            'humidity': hum,
            'wind_kmh': wind,
            'fire_occurred': 0  
        }
        
        new_df = pd.DataFrame([new_row])
        new_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        count += 1
    
    time.sleep(1) 
print("Done")