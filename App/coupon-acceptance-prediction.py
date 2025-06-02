import streamlit as st
import joblib
import numpy as np
import os

# Model path relative to the app folder
model_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'random_forest_model.pkl')

# Loading the model
model = joblib.load(model_path)

st.set_page_config(page_title="Coupon Acceptance Predictor", layout="centered")
st.title("🌟 Coupon Acceptance Prediction App")

st.markdown("""
This app uses a machine learning model to predict whether a user will accept a coupon 
based on contextual information such as weather, time, passenger, age, and temperature.
""")

# --- User Input ---
weather_options = ['Sunny', 'Rainy', 'Snowy']
time_options = ['7AM', '10AM', '2PM', '6PM', '10PM']
passenger_options = ['Alone', 'Friends', 'Kids', 'Partner']
age_options = ['below21', '21', '26', '31', '36', '41', '46', '50plus']
temperature_options = [30, 55, 80]

weather = st.selectbox('Weather condition', weather_options)
time = st.selectbox('Time of day', time_options)
passenger = st.selectbox('Passenger type', passenger_options)
age = st.selectbox('Age group', age_options)
temperature = st.selectbox('Temperature (°F)', temperature_options)

# --- Feature Encoding ---
weather_map = {'Sunny': 0, 'Rainy': 1, 'Snowy': 2, 'Cloudy': 3}
time_map = {'7AM': 0, '10AM': 1, '2PM': 2, '6PM': 3, '10PM': 4}
passenger_map = {'Alone': 0, 'Friends': 1, 'Kids': 2, 'Partner': 3}
age_map = {'below21': 0, '21': 1, '26': 2, '31': 3, '36': 4, '41': 5, '46': 6, '50plus': 7}

weather_encoded = weather_map[weather]
time_encoded = time_map[time]
passenger_encoded = passenger_map[passenger]
age_encoded = age_map[age]

# Prepare the input array
X_input = np.array([[weather_encoded, time_encoded, passenger_encoded, age_encoded, temperature]])

# --- Prediction ---
if st.button("Predict Coupon Acceptance"):
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction Result")
    st.markdown(f"**Coupon will be {'ACCEPTED' if prediction == 1 else 'DECLINED'}**")
    st.markdown(f"Probability of acceptance: **{proba:.2%}**")
