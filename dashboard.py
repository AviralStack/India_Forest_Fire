import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np

# =========================
# CONFIGURATION
# =========================
# SECURE METHOD: Load key from Streamlit Secrets
# If running locally, you can keep your string here, but for cloud, use st.secrets
try:
    API_KEY = st.secrets["c64a656e4a26f68a8c93c26695f7ebd4"]
except:
    API_KEY = "YOUR_RAW_KEY_FOR_LOCAL_TESTING" # Only as a backup!

st.set_page_config(
    page_title="Forest Fire Risk Assessment System",
    layout="centered"
)

# =========================
# MODEL LOADING
# =========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/forest_fire_v2.pkl")
        try:
            model.get_booster().feature_names = None
        except:
            pass
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please verify the model path.")
        return None

model = load_model()

# =========================
# WEATHER API
# =========================
def get_live_weather(city):
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind": data["wind"]["speed"] * 3.6,
                "country": data["sys"]["country"],
                "found": True
            }
        return {"found": False}
    except:
        return {"found": False}

# =========================
# UI
# =========================
st.title("Forest Fire Risk Assessment System")
st.caption("Machine Learning–Based Environmental Risk Analysis")

st.sidebar.header("Input Configuration")

input_mode = st.sidebar.radio(
    "Weather Data Source",
    ["Live Weather Data", "Manual Simulation"]
)

area_type = st.sidebar.selectbox(
    "Land Classification",
    ["Forest / Vegetation", "Urban / Built Environment"]
)

temp = humidity = wind = None
location_name = "Unknown"

# =========================
# INPUT HANDLING
# =========================
if input_mode == "Live Weather Data":
    city = st.text_input("City Name", "Shimla")
    if st.button("Fetch Weather Data"):
        with st.spinner("Retrieving weather information..."):
            weather = get_live_weather(city)
            if weather["found"]:
                temp = weather["temp"]
                humidity = weather["humidity"]
                wind = weather["wind"]
                location_name = f"{city}, {weather['country']}"
                st.success(
                    f"Temperature: {temp} °C | "
                    f"Humidity: {humidity}% | "
                    f"Wind Speed: {wind:.1f} km/h"
                )
            else:
                st.error("Unable to retrieve weather data. Check city name or API key.")

else:
    sim_city = st.text_input("Simulation City", "Jodhpur")

    months = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
    sim_month = st.selectbox("Simulation Month", months, index=4)

    if sim_month in ["April", "May", "June"]:
        def_temp, def_hum = 42.0, 15
    elif sim_month in ["July", "August", "September"]:
        def_temp, def_hum = 30.0, 85
    elif sim_month in ["October", "November"]:
        def_temp, def_hum = 28.0, 40
    else:
        def_temp, def_hum = 15.0, 45

    location_name = f"{sim_city} ({sim_month})"

    st.info("Using season-based default values. Parameters can be adjusted below.")

    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.number_input("Temperature (°C)", -10.0, 60.0, def_temp)
    with col2:
        humidity = st.number_input("Humidity (%)", 0, 100, def_hum)
    with col3:
        wind = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 15.0)

# =========================
# PREDICTION
# =========================
if temp is not None:
    st.divider()

    if area_type == "Urban / Built Environment":
        risk_prob = 0.5
        risk_label = "Negligible"
        explanation = (
            f"Urban environments in {location_name} "
            "have minimal combustible vegetation."
        )
    else:
        input_df = pd.DataFrame(
            [[temp, humidity, wind]],
            columns=["temp_c", "humidity", "wind_kmh"]
        )

        raw_prob = model.predict_proba(input_df.values)[0][1] * 100

        penalty = 0
        reasons = []

        if humidity > 50:
            penalty += (humidity - 50) * 1.5
            reasons.append("Elevated humidity")

        if temp < 25:
            penalty += (25 - temp) * 2.5
            reasons.append("Low ambient temperature")

        risk_prob = max(0, raw_prob - penalty)

        if humidity > 80 or temp < 5:
            risk_prob = 0.5
            reasons = ["Unfavorable combustion conditions"]

        if risk_prob < 30:
            risk_label = "Low"
        elif risk_prob < 70:
            risk_label = "Moderate"
        else:
            risk_label = "High"

        explanation = (
            "Risk adjusted based on environmental constraints"
            + (f" ({', '.join(reasons)})" if reasons else "")
        )

    # =========================
    # OUTPUT
    # =========================
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.metric("Estimated Fire Probability", f"{risk_prob:.1f}%")

    with col_b:
        st.subheader(f"Risk Level: {risk_label}")
        st.write(explanation)

    st.progress(int(risk_prob))
