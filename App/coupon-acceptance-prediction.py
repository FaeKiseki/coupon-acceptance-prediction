import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
model_path = '../Models/random_forest_model.pkl'
model = joblib.load(model_path)

st.set_page_config(page_title="Coupon Acceptance Predictor", layout="centered")
st.title("🌟 Coupon Acceptance Prediction App")

st.markdown("""
This app uses a machine learning model to predict whether a user will accept a coupon 
based on contextual information such as weather, time, passenger, age, and temperature.
""")

# --- User Input ---
weather = st.selectbox('Weather condition', ['Sunny', 'Rainy', 'Snowy', 'Cloudy'])
time = st.slider('Hour of the day (0-23)', 0, 23, 12)
passenger = st.selectbox('Passenger type', ['Alone', 'Friends', 'Kids'])
age = st.selectbox('Age group', ['21-30', '31-40', '41-50', '51+'])
temperature = st.slider('Temperature (°C)', -10, 40, 20)

# --- Feature Encoding ---
# These mappings must match the ones used during model training
weather_map = {'Sunny': 0, 'Rainy': 1, 'Snowy': 2, 'Cloudy': 3}
passenger_map = {'Alone': 0, 'Friends': 1, 'Kids': 2}
age_map = {'21-30': 0, '31-40': 1, '41-50': 2, '51+': 3}

weather_encoded = weather_map[weather]
passenger_encoded = passenger_map[passenger]
age_encoded = age_map[age]

# Prepare the input array
X_input = np.array([[weather_encoded, time, passenger_encoded, age_encoded, temperature]])

# --- Prediction ---
if st.button("Predict Coupon Acceptance"):
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction Result")
    st.markdown(f"**Coupon will be {'ACCEPTED' if prediction == 1 else 'DECLINED'}**")
    st.markdown(f"Probability of acceptance: **{proba:.2%}**")
