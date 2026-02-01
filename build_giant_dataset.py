import pandas as pd
import requests
import time
import os
from datetime import datetime

# --- CONFIGURATION ---
INPUT_FILE = "nasa_history.csv"       
OUTPUT_FILE = "final_india_dataset.csv" 
SAMPLE_SIZE = 50000  # <--- LIMITS DATA TO 50k RANDOM ROWS (Manageable)

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
    
    # RETRY LOGIC (3 Tries)
    for attempt in range(3):
        try:
            # INCREASED TIMEOUT to 30s
            r = requests.get(url, params=params, timeout=30)
            
            if r.status_code == 429:
                print("⏳ API Busy. Sleeping 60s...")
                time.sleep(60)
                continue # Retry
                
            r.raise_for_status()
            data = r.json()
            
            daily = data.get('daily', {})
            temp = daily.get('temperature_2m_max', [None])[0]
            hum = daily.get('relative_humidity_2m_mean', [None])[0]
            wind = daily.get('wind_speed_10m_max', [None])[0]
            
            return temp, hum, wind
            
        except Exception as e:
            # Only print error on last attempt
            if attempt == 2:
                print(f"⚠️ Failed {lat},{lon}: {e}")
            time.sleep(2) # Wait before retry
            
    return None, None, None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: Could not find '{INPUT_FILE}'.")
        return

    print(f"📂 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. FILTER LOW CONFIDENCE
    if 'confidence' in df.columns:
        df = df[df['confidence'] != 'l']
    
    # 2. SMART SAMPLING (The Time Saver)
    if len(df) > SAMPLE_SIZE:
        print(f"📉 Dataset too huge ({len(df)} rows). Sampling random {SAMPLE_SIZE} rows for speed...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    print(f"🔥 Processing {len(df)} rows...")

    # 3. RESUME CAPABILITY
    finished_ids = set()
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        # Fix: Convert to string to ensure matching works
        finished_ids = set(
            (existing['latitude'].astype(str) + "_" + 
             existing['longitude'].astype(str) + "_" + 
             existing['acq_date'].astype(str)).tolist()
        )
        print(f"✅ Resuming... {len(finished_ids)} rows already done.")
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
        
        time.sleep(1.5) # Be nice to the API

    print("\n🎉 DONE! Dataset complete.")

if __name__ == "__main__":
    main()