import pandas as pd
import requests
import time
import os
from datetime import datetime


INPUT_FILE = "nasa_history.csv"       
OUTPUT_FILE = "final_india_dataset.csv" 
SAMPLE_SIZE = 50000  

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
    
    
    for attempt in range(3):
        try:
           
            r = requests.get(url, params=params, timeout=30)
            
            if r.status_code == 429:
               
                time.sleep(60)
                continue 
                
            r.raise_for_status()
            data = r.json()
            
            daily = data.get('daily', {})
            temp = daily.get('temperature_2m_max', [None])[0]
            hum = daily.get('relative_humidity_2m_mean', [None])[0]
            wind = daily.get('wind_speed_10m_max', [None])[0]
            
            return temp, hum, wind
            
        except Exception as e:
            
            if attempt == 2:
                print(f" Failed {lat},{lon}: {e}")
            time.sleep(2) 
            
    return None, None, None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f" ERROR: Could not find '{INPUT_FILE}'.")
        return

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    
    if 'confidence' in df.columns:
        df = df[df['confidence'] != 'l']
    
   
    if len(df) > SAMPLE_SIZE:
        print(f"Dataset too huge ({len(df)} rows). Sampling random {SAMPLE_SIZE} rows for speed...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    print(f"Processing {len(df)} rows...")

    finished_ids = set()
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        
        finished_ids = set(
            (existing['latitude'].astype(str) + "_" + 
             existing['longitude'].astype(str) + "_" + 
             existing['acq_date'].astype(str)).tolist()
        )
        print(f"Resuming... {len(finished_ids)} rows already done.")
    else:
        with open(OUTPUT_FILE, "w") as f:
            f.write("latitude,longitude,acq_date,temp_c,humidity,wind_kmh,fire_occurred\n")

    # 4. THE LOOP
    count = 0
    total = len(df)
    
    for index, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        date = row['acq_date']
        
        unique_id = f"{lat}_{lon}_{date}"
        if unique_id in finished_ids:
            continue
            
        if count % 10 == 0:
            print(f"Processing {index} ({round(count/total*100, 2)}%)...")

        temp, hum, wind = fetch_weather(lat, lon, date)
        
        if temp is not None:
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"{lat},{lon},{date},{temp},{hum},{wind},1\n")
            count += 1
        
        time.sleep(1.5) 

  

if __name__ == "__main__":
    main()