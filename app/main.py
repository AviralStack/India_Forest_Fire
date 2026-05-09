from fastapi import FastAPI, HTTPException, Query
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from enum import Enum

app = FastAPI(title="Global Forest Fire Risk System (V3 - with Area Type)")

OPENWEATHER_API_KEY = "c64a656e4a26f68a8c93c26695f7ebd4" 
# ---------------------


model = joblib.load("models/forest_fire_v2.pkl")
try:
    model.get_booster().feature_names = None
except:
    pass


class AreaType(str, Enum):
    forest = "Forest / Vegetation (High Fuel)"
    urban = "City / Concrete (No Fuel)"


def get_cyclical_month(month_num):
    sin = np.sin(2 * np.pi * month_num / 12)
    cos = np.cos(2 * np.pi * month_num / 12)
    return sin, cos


def get_live_weather(city: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None, None
        
    data = response.json()
    return {
        "lat": data['coord']['lat'], 
        "lon": data['coord']['lon'],
        "temp": data['main']['temp'], 
        "humidity": data['main']['humidity'], 
        "wind": data['wind']['speed'] * 3.6
    }, data['sys']['country']

@app.get("/predict_risk")
def predict_risk(
    city: str, 
    area_type: AreaType = Query(..., description="Is this a concrete city or a green area?")
):
    

    current_month = datetime.now().month
    month_sin, month_cos = get_cyclical_month(current_month)

    weather_data, country = get_live_weather(city)
    
    if weather_data is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch weather for {city}.")
    

    if area_type == AreaType.urban:
        return {
            "location": f"{city}, {country}",
            "area_type": "Urban / Concrete",
            "live_weather": {
                "temp_c": weather_data['temp'],
                "humidity": weather_data['humidity'],
                "wind_kmh": round(weather_data['wind'], 1)
            },
            "fire_probability": "0.5%",
            "risk_level": "SAFE",
            "recommended_action": "No Action Needed (No Fuel Present)"
        }

    input_data = pd.DataFrame([[

        weather_data['temp'], 
        weather_data['humidity'], 
        weather_data['wind'], 

    ]], columns=['temp_c', 'humidity', 'wind_kmh'])
    
    raw_probability = model.predict_proba(input_data.values)[0][1] * 100
    fire_probability = float(raw_probability)
    

    if fire_probability < 30:
        risk_level = "LOW"
        action = "Normal Monitoring"
    elif fire_probability < 70:
        risk_level = "MEDIUM"
        action = "Alert Standby Crews"
    else:
        risk_level = "EXTREME"
        action = "Mobilize Regional Support"
        
    return {
        "location": f"{city}, {country}",
        "area_type": "Forest / Vegetation",
        "live_weather": {
            "temp_c": weather_data['temp'],
            "humidity": weather_data['humidity'],
            "wind_kmh": round(weather_data['wind'], 1)
        },
        "fire_probability": f"{round(fire_probability, 1)}%",
        "risk_level": risk_level,
        "recommended_action": action
    }
