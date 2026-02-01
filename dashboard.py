import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np

# =========================
# SYSTEM CONFIGURATION
# =========================
st.set_page_config(
    page_title="India Forest Fire Prediction System",
    layout="centered"
)

# SECURE API KEY LOADING
try:
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
except:
    # Backup key for local testing
    API_KEY = "8f5c880ee1c0819e9db8dea1f8e4f7c6" 

# =========================
# MODEL INITIALIZATION
# =========================
@st.cache_resource
def load_inference_engine():
    try:
        model = joblib.load("models/forest_fire_v2.pkl")
        try:
            model.get_booster().feature_names = None
        except:
            pass
        return model
    except FileNotFoundError:
        st.error("Critical Error: Model file 'models/forest_fire_v2.pkl' not found.")
        return None

model = load_inference_engine()

# =========================
# DATA ACQUISITION LAYER
# =========================
def fetch_weather_telemetry(city):
    """Fetches real-time weather data from OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind": data["wind"]["speed"] * 3.6, # Convert m/s to km/h
                "country": data["sys"]["country"],
                "success": True
            }
        return {"success": False}
    except:
        return {"success": False}

# =========================
# USER INTERFACE
# =========================
st.title("India Forest Fire Prediction System")
st.markdown("### AI-Driven Environmental Risk Assessment")
st.markdown("---")

st.sidebar.header("System Configuration")

# 1. Input Source Selection
input_source = st.sidebar.radio(
    "Data Source",
    ["Live Satellite Data", "Manual Simulation"]
)

# 2. Environmental Classification
land_classification = st.sidebar.selectbox(
    "Land Classification",
    ["Forest / Vegetation", "Urban / Built Environment"]
)

# Initialize Variables
temp, humidity, wind = None, None, None
location_label = "N/A"

# =========================
# INPUT PROCESSING
# =========================

# --- MODE A: LIVE DATA ---
if input_source == "Live Satellite Data":
    target_city = st.text_input("Target City", "Shimla")
    
    if st.button("Retrieve Weather Data"):
        with st.spinner(f"Querying weather satellite for {target_city}..."):
            telemetry = fetch_weather_telemetry(target_city)
            
            if telemetry["success"]:
                temp = telemetry["temp"]
                humidity = telemetry["humidity"]
                wind = telemetry["wind"]
                location_label = f"{target_city}, {telemetry['country']}"
                
                st.success(f"Data Retrieved Successfully: {temp}°C | {humidity}% Humidity | {wind:.1f} km/h Wind")
            else:
                st.error("Error: Unable to retrieve data. Please check the city name or API connectivity.")

# --- MODE B: SIMULATION ---
else:
    sim_city = st.text_input("Simulation Target", "Jodhpur")
    
    # Standardized Seasonal Defaults for India
    month_list = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
    
    selected_month = st.selectbox("Select Season (Loads Historical Averages)", month_list, index=4) # Default to May

    # Logic for historical averages
    if selected_month in ["April", "May", "June"]: # Summer
        default_t, default_h = 42.0, 15
    elif selected_month in ["July", "August", "September"]: # Monsoon
        default_t, default_h = 30.0, 85
    elif selected_month in ["October", "November"]: # Post-Monsoon
        default_t, default_h = 28.0, 40
    else: # Winter
        default_t, default_h = 15.0, 45

    location_label = f"{sim_city} ({selected_month} Simulation)"
    
    st.info(f"Simulation Mode: Parameters initialized to typical {selected_month} conditions. Adjust below for stress testing.")

    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.number_input("Temperature (°C)", -10.0, 60.0, default_t)
    with col2:
        humidity = st.number_input("Humidity (%)", 0, 100, default_h)
    with col3:
        wind = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 15.0)

# =========================
# RISK ANALYSIS ENGINE
# =========================
if temp is not None:
    st.markdown("### Risk Analysis Report")
    
    # SCENARIO 1: Urban Area (Zero Fuel Load)
    if land_classification == "Urban / Built Environment":
        final_probability = 0.5
        risk_category = "NEGLIGIBLE"
        risk_color = "green"
        analysis_summary = f"Urban infrastructure in {location_label} presents negligible biological fuel load."
        
        # --- FIX: DEFINE VARIABLES TO PREVENT CRASH ---
        raw_probability = 0.0
        penalty_score = 0.0
    
    # SCENARIO 2: Forest Area (AI Analysis)
    else:
        # 1. AI Inference
        input_vector = pd.DataFrame([[temp, humidity, wind]], columns=["temp_c", "humidity", "wind_kmh"])
        raw_probability = model.predict_proba(input_vector.values)[0][1] * 100

        # 2. Physics-Based Post-Processing (Guardrails)
        penalty_score = 0
        dampening_factors = []

        # Humidity Guardrail
        if humidity > 50:
            penalty = (humidity - 50) * 1.5
            penalty_score += penalty
            dampening_factors.append("Elevated Humidity")
        
        # Temperature Guardrail
        if temp < 25:
            penalty = (25 - temp) * 2.5
            penalty_score += penalty
            dampening_factors.append("Low Ambient Temperature")

        # Apply Penalties
        final_probability = max(0, raw_probability - penalty_score)

        # Critical Thresholds (Hard Stops)
        if humidity > 80 or temp < 5:
            final_probability = 0.5
            dampening_factors = ["Precipitation / Freezing Conditions"]

        # 3. Risk Categorization
        if final_probability < 30:
            risk_category = "LOW"
            risk_color = "green"
        elif final_probability < 70:
            risk_category = "MODERATE"
            risk_color = "orange"
        else:
            risk_category = "EXTREME"
            risk_color = "red"

        # Generate Summary
        if dampening_factors:
            analysis_summary = f"Risk reduced due to environmental inhibitors: {', '.join(dampening_factors)}."
        else:
            analysis_summary = "Meteorological conditions align with high-probability fire patterns."

    # =========================
    # VISUALIZATION
    # =========================
    c1, c2 = st.columns([1, 2])

    with c1:
        st.metric("Fire Probability Index", f"{final_probability:.1f}%")

    with c2:
        st.subheader(f"Risk Category: :{risk_color}[{risk_category}]")
        st.write(f"**Analysis:** {analysis_summary}")

    st.progress(int(final_probability))
    
    # Technical Data Expander
    with st.expander("View Technical Telemetry"):
        st.dataframe(pd.DataFrame({
            "Parameter": ["Ambient Temp", "Rel. Humidity", "Wind Velocity", "Raw Model Output", "Physics Penalty"],
            "Value": [f"{temp} °C", f"{humidity}%", f"{wind} km/h", f"{raw_probability:.1f}%", f"-{penalty_score:.1f}%"]
        }), hide_index=True)