# India Forest Fire Prediction System 

### AI-Driven Environmental Risk Assessment with Physics-Based Guardrails

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indiaforestfire.streamlit.app/)

##  Project Overview
The **India Forest Fire Prediction System** is a hybrid machine learning application designed to predict forest fire risks specifically for the Indian subcontinent. Unlike standard models that rely solely on historical patterns, this system integrates a **Physics-Based Post-Processing Layer** to enforce thermodynamic constraints.

This approach solves the common **"Winter Bias"** found in Indian datasets, where dry but cold winter days generate false positives. The system combines real-time satellite telemetry (OpenWeatherMap) with an XGBoost inference engine to provide accurate, scientifically grounded risk assessments.

##  Key Features

* **Hybrid Intelligence:** Combines an **XGBoost Classifier** with hard-coded **Thermodynamic Guardrails** to prevent physics hallucinations.
* **Real-Time Telemetry:** Fetches live weather data (Temperature, Humidity, Wind Speed) using the **OpenWeatherMap API**.
* **Dual-Mode Operation:**
    * **Live Satellite Mode:** Real-time risk assessment for any Indian city/forest.
    * **Simulation Mode:** Stress-test the model against hypothetical scenarios (e.g., Heatwaves, Monsoons).
* **Smart "Winter Bias" Correction:** Automatically dampens risk scores when ambient temperature is too low for combustion (<25°C), preventing false alarms in safe winter conditions.

##  Dataset Engineering: Real-World Data (No Kaggle)
We rejected standard, pre-packaged "Toy Datasets" (like the common UCI/Kaggle Forest Fire CSVs) which are often synthetic, small, or regionally biased towards Europe/USA.

Instead, we **engineered a custom, national-scale dataset** specifically for India:

1.  **Primary Source (NASA FIRMS):** We extracted **8,500+ real fire events** detected by the **MODIS and VIIRS satellites** over the Indian subcontinent.
2.  **Historical Weather Mapping:** For every single fire point, we back-traced the exact weather conditions (Temp, Wind, Humidity) at that specific moment using the **OpenMeteo Historical API**.
3.  **Negative Sampling:** To prevent bias, we generated 8,500+ "Safe Days" (non-fire events) spatiotemporally matched to Indian forests, creating a perfectly balanced 17,000-row dataset.

**Result:** A robust, India-specific model trained on *actual* ground realities, not synthetic simulations.

##  Tech Stack

* **Core Logic:** Python 3.12
* **Machine Learning:** XGBoost, Scikit-Learn, Joblib
* **Data Engineering:** NASA FIRMS API, OpenMeteo API, Pandas
* **Web Framework:** Streamlit (Frontend), Requests (API Handling)
* **Deployment:** Streamlit Community Cloud (Production), Docker

##  Methodology & Architecture

### 1. The Physics Guardrail System
A raw AI model often overfits to correlation (e.g., "Dry = Fire"). Our system applies a secondary logic layer:
* **Humidity Penalty:** If Humidity > 50%, risk probability is mathematically dampened.
* **Temperature Penalty:** If Temperature < 25°C, risk is reduced (Arrhenius equation logic).
* **Hard Cutoffs:** If Humidity > 80% (Rain) or Temperature < 5°C (Freezing), risk is forced to **NEGLIGIBLE**.

##  Installation & Local Run

To run this project on your local machine:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/AviralStack/India_Forest_Fire.git
    cd India_Forest_Fire
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up API Keys**
    * Create a file named `.streamlit/secrets.toml` (for local Streamlit) OR export the key in your terminal.
    * *Note: The code includes a fallback key for testing purposes.*

4.  **Run the Application**
    ```bash
    streamlit run dashboard.py
    ```

##  Docker Support

The project includes a `Dockerfile` for containerized deployment.

```bash
# Build the image
docker build -t forest-fire-ai .

# Run the container
docker run -p 80:80 forest-fire-ai
```
📈 Results
The system demonstrates high accuracy in distinguishing between "True Fire Weather" and "Safe Dry Winter Days."

<img width="875" height="250" alt="image" src="https://github.com/user-attachments/assets/3c2ce1a3-5d51-4d02-9a2c-dc67778b9821" />

