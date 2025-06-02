import streamlit as st
import joblib
import numpy as np
import os

# --- Load the model ---
model_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'random_forest_model.pkl')
model = joblib.load(model_path)

# --- Streamlit UI configuration ---
st.set_page_config(page_title="Coupon Acceptance Predictor", layout="centered")
st.title("🎯 Coupon Acceptance Prediction App")

st.markdown("""
This app predicts whether a user will accept a coupon based on contextual inputs like weather, 
time of day, age group, passenger type, and temperature (°F).
""")

# --- Input Widgets ---
temperature = st.slider('Temperature (°F)', min_value=30, max_value=110, value=70)

weather = st.selectbox("Weather condition", ['Sunny', 'Snowy', 'Rainy'])

age_group = st.selectbox("Age group", [
    'below21', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '50plus'
])

time_of_day = st.selectbox("Time of Day", ['AM', 'PM'])

passenger_type = st.selectbox("Passenger Type", ['Alone', 'Friend(s)', 'Kid(s)', 'Partner'])

# --- Feature Encoding (must match training preprocessing) ---
weather_map = {'Sunny': 0, 'Snowy': 1, 'Rainy': 2}
age_map = {
    'below21': 0, '21-25': 1, '26-30': 2, '31-35': 3,
    '36-40': 4, '41-45': 5, '46-50': 6, '50plus': 7
}
time_map = {'AM': 0, 'PM': 1}
passenger_map = {'Alone': 0, 'Friend(s)': 1, 'Kid(s)': 2, 'Partner': 3}

# --- Convert inputs to numeric values ---
input_features = np.array([[
    temperature,
    weather_map[weather],
    age_map[age_group],
    time_map[time_of_day],
    passenger_map[passenger_type]
]])

# --- Make prediction ---
if st.button("Predict Coupon Acceptance"):
    prediction = model.predict(input_features)[0]
    proba = model.predict_proba(input_features)[0][1]

    st.subheader("Prediction Result")
    st.markdown(f"**Coupon will be {'✅ ACCEPTED' if prediction == 1 else '❌ DECLINED'}**")
    st.markdown(f"**Probability of acceptance: {proba:.2%}**")
