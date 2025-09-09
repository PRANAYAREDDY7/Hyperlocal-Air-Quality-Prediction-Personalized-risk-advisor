import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load your trained model (save your model using joblib in eda_and_modeling.py after training)
model = joblib.load('best_random_forest_model.joblib')

# Function to predict and give advice
def predict_aqi(input_data):
    pred = model.predict(pd.DataFrame([input_data]))[0]
    if pred <= 30:
        advice = "Air quality is good. Minimal risk."
    elif pred <= 60:
        advice = "Moderate air quality. Sensitive groups should take care."
    elif pred <= 90:
        advice = "Poor air quality. Avoid outdoor activities if you are sensitive."
    else:
        advice = "Very poor air quality. Stay indoors and use air purifiers."
    return pred, advice

st.title("Hyperlocal Air Quality Predictor & Personalized Risk Advisor")

# Input fields for features
st.header("Input Environmental Data")

PM10 = st.number_input("PM10 (ug/m3)", min_value=0.0, max_value=1000.0, value=30.0)
NO2 = st.number_input("NO2 (ug/m3)", min_value=0.0, max_value=500.0, value=20.0)
NOx = st.number_input("NOx (ppb)", min_value=0.0, max_value=500.0, value=25.0)
SO2 = st.number_input("SO2 (ug/m3)", min_value=0.0, max_value=500.0, value=15.0)
CO = st.number_input("CO (mg/m3)", min_value=0.0, max_value=50.0, value=0.5)
Ozone = st.number_input("Ozone (ug/m3)", min_value=0.0, max_value=300.0, value=40.0)
NH3 = st.number_input("NH3 (ug/m3)", min_value=0.0, max_value=200.0, value=10.0)
Temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)
RH = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
WS = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=3.0)
RF = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0)

year = st.number_input("Year", min_value=2010, max_value=2030, value=datetime.now().year)
month = st.number_input("Month", min_value=1, max_value=12, value=datetime.now().month)
day = st.number_input("Day", min_value=1, max_value=31, value=datetime.now().day)
hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
weekday = st.number_input("Weekday (0=Mon ... 6=Sun)", min_value=0, max_value=6, value=datetime.now().weekday())

input_features = {
    'PM10 (ug/m3)': PM10,
    'NO2 (ug/m3)': NO2,
    'NOx (ppb)': NOx,
    'SO2 (ug/m3)': SO2,
    'CO (mg/m3)': CO,
    'Ozone (ug/m3)': Ozone,
    'NH3 (ug/m3)': NH3,
    'Temp (degree C)': Temp,
    'RH (%)': RH,
    'WS (m/s)': WS,
    'RF (mm)': RF,
    'year': year,
    'month': month,
    'day': day,
    'hour': hour,
    'weekday': weekday,
}

if st.button('Predict AQI & Get Advice'):
    prediction, advice = predict_aqi(input_features)
    st.success(f"Predicted PM2.5 concentration: {prediction:.2f} ug/m3")
    st.info(f"Health Advice: {advice}")
