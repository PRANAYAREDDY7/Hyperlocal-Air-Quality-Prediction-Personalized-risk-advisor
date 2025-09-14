import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pretrained model
import joblib
import requests
import os

model_url = 'https://drive.google.com/uc?export=download&id=17yrvioYqSBNsnMC5AfhsHYYtxnddMj8-'


model_path = 'best_random_forest_model.joblib'

if not os.path.exists(model_path):
    st.write("Downloading model... Please wait.")
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    st.write("Model downloaded.")

model = joblib.load(model_path)

model = joblib.load('best_random_forest_model.joblib')

# Load dataset with pollution and location data
data = pd.read_csv('cleaned_air_quality_data.csv')

st.title("Telangana Air Quality Prediction & Personalized Risk Advisor")

st.markdown("""
Select your Telangana district and provide your personal info to get a tailored PM2.5 air quality prediction and health advice.
""")

# Mapping file codes to district names
location_mapping = {
    'TG001': 'Hyderabad',
    'TG002': 'Warangal',
    'TG003': 'Nizamabad',
    'TG004': 'Karimnagar',
    'TG005': 'Rangareddy',
    'TG006': 'Mahbubnagar',
    'TG007': 'Medak',
    'TG008': 'Adilabad',
    'TG009': 'Khammam',
    'TG010': 'Karimnagar',
    'TG011': 'Nalgonda',
    'TG012': 'Nagarkurnool',
    'TG013': 'Suryapet',
    'TG014': 'Bhadradri Kothagudem'
}

file_names = sorted(location_mapping.keys())
friendly_names = [location_mapping[code] for code in file_names]

# User inputs
selected_friendly_name = st.selectbox("Select your Telangana district", friendly_names)
selected_location = file_names[friendly_names.index(selected_friendly_name)]

age_group = st.selectbox("Select your age group", ['child', 'adult', 'senior'])
has_respiratory = st.checkbox("Do you have respiratory or heart conditions?")

# Filter data for location and calculate mean of numeric columns
location_data = data[data['file_name'] == selected_location]
avg_features = location_data.select_dtypes(include='number').mean()

features = np.array([[
    avg_features['PM10 (ug/m3)'],
    avg_features['NO2 (ug/m3)'],
    avg_features['NOx (ppb)'],
    avg_features['SO2 (ug/m3)'],
    avg_features['CO (mg/m3)'],
    avg_features['Ozone (ug/m3)'],
    avg_features['NH3 (ug/m3)'],
    avg_features['Temp (degree C)'],
    avg_features['RH (%)'],
    avg_features['WS (m/s)'],
    avg_features['RF (mm)'],
    avg_features['year'],
    avg_features['month'],
    avg_features['day'],
    avg_features['hour'],
    avg_features['weekday']
]])

def personalized_risk_advisory(pm25, age_group='adult', has_respiratory=False):
    if pm25 <= 12:
        if has_respiratory:
            if age_group == 'child':
                return "Air quality is good but children with respiratory or heart conditions should avoid strenuous activities."
            elif age_group == 'senior':
                return "Good air quality. Seniors with conditions may enjoy outdoor activities but stay attentive to symptoms."
            else:
                return "Good air quality. Those with heart or respiratory issues should monitor their health."
        else:
            return "Air quality is good. No precautions needed."

    elif pm25 <= 35:
        if has_respiratory:
            if age_group == 'child':
                return ("Moderate pollution. Children with respiratory or heart conditions should avoid outdoor exertion, "
                        "wear masks, and keep medications ready.")
            elif age_group == 'senior':
                return ("Moderate pollution. Seniors with heart or respiratory issues should limit outdoor exposure, "
                        "wear masks, and consult doctors if unwell.")
            else:
                return ("Moderate pollution. People with heart or respiratory issues should reduce prolonged outdoor activities and consider protective masks.")
        else:
            return "Moderate pollution. Limit prolonged exertion outdoors."

    elif pm25 <= 55:
        if has_respiratory:
            if age_group == 'child':
                return ("Unhealthy for children with conditions. Avoid outdoor play, keep medications handy, and seek doctor advice if needed.")
            elif age_group == 'senior':
                return ("Unhealthy for seniors with conditions. Stay indoors mostly, reduce physical activity, and seek doctor if symptoms worsen.")
            else:
                return ("Unhealthy for sensitive individuals. Reduce outdoor exposure, wear N95 masks, and monitor symptoms.")
        else:
            return "Unhealthy for sensitive groups; reduce outdoor exertion and wear protective masks."

    elif pm25 <= 150:
        if has_respiratory:
            if age_group == 'child':
                return ("Unhealthy air quality. Children with heart or respiratory diseases must stay indoors, avoid physical activity, "
                        "keep emergency meds accessible, and seek medical care if symptoms escalate.")
            elif age_group == 'senior':
                return ("Unhealthy air quality. Seniors with conditions should stay indoors with air filtration, avoid outdoor activity, "
                        "and monitor health closely.")
            else:
                return ("Unhealthy air quality. Vulnerable adults should minimize outdoor time, use air purifiers, and be vigilant about symptoms.")
        else:
            return "Unhealthy air quality. Reduce outdoor activities and maintain indoor air quality."

    elif pm25 <= 250:
        if has_respiratory:
            if age_group == 'child':
                return ("Very unhealthy air quality! Children with conditions must stay indoors with filtered air, avoid exposure, "
                        "and seek emergency care if needed.")
            elif age_group == 'senior':
                return ("Very unhealthy air quality! Seniors with conditions should avoid exposure completely, stay indoors with air filtration, "
                        "keep meds and emergency contacts ready.")
            else:
                return ("Very unhealthy air quality! Vulnerable adults must avoid outdoor exposure, wear masks, "
                        "and promptly consult healthcare if symptoms worsen.")
        else:
            return "Very unhealthy air quality. Avoid outdoor exposure and use indoor air purification."

    else:
        if has_respiratory:
            if age_group == 'child':
                return ("Hazardous air quality! Children with conditions must stay indoors in filtered air, have emergency plans, "
                        "and seek immediate medical help if needed.")
            elif age_group == 'senior':
                return ("Hazardous air quality! Seniors with conditions must avoid outdoor activity, use air purifiers, "
                        "keep medications ready, and seek urgent care if symptoms worsen.")
            else:
                return ("Hazardous air quality! Adults with conditions must remain indoors with filtration, avoid exposure, "
                        "and seek urgent medical attention if problems occur.")
        else:
            return "Hazardous air quality! Stay indoors with air filtration and avoid outdoor activities."

if st.button("Predict PM2.5 and Get Personalized Risk Advisory"):
    predicted_pm25 = model.predict(features)[0]
    risk_msg = personalized_risk_advisory(predicted_pm25, age_group, has_respiratory)
    st.success(f"Predicted PM2.5: {predicted_pm25:.2f} µg/m³")
    st.info(risk_msg)
