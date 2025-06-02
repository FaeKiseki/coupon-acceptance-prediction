import streamlit as st
import pickle
import os
import numpy as np

# -----------------------------
# Define path to the model file
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory where this script is located
model_path = os.path.join(BASE_DIR, '..', 'Models', 'random_forest_model.pkl')  # Path to the model file

# Load the pre-trained Random Forest model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# -----------------------------
# Streamlit app starts here
# -----------------------------

st.title("Coupon Acceptance Prediction")

# Example input features (adapt to your real feature names and expected input types)
# For example, if your model uses features like 'weather', 'passenger', 'time', 'age', 'temperature'
# you should create appropriate input widgets here to collect user inputs.

weather = st.selectbox("Weather condition", ['Sunny', 'Rainy', 'Snowy', 'Cloudy', 'Windy'])
passenger = st.selectbox("Passenger type", ['Alone', 'Friend(s)', 'Partner', 'Kids'])
time = st.slider("Time (hour of the day)", 0, 23, 12)
age = st.slider("Age", 18, 70, 30)
temperature = st.number_input("Temperature (°F)", value=70)

# You will need to preprocess these inputs exactly as you did for training.
# For example, encoding categorical variables, scaling, etc.

# Here is a placeholder function to preprocess inputs - adapt this to your real preprocessing!
def preprocess_input(weather, passenger, time, age, temperature):
    # Example: simple manual encoding (just for demonstration)
    weather_dict = {'Sunny': 0, 'Rainy': 1, 'Snowy': 2, 'Cloudy': 3, 'Windy': 4}
    passenger_dict = {'Alone': 0, 'Friend(s)': 1, 'Partner': 2, 'Kids': 3}

    weather_encoded = weather_dict.get(weather, 0)
    passenger_encoded = passenger_dict.get(passenger, 0)

    # Create feature vector as numpy array (shape = [1, n_features])
    features = np.array([[weather_encoded, passenger_encoded, time, age, temperature]])
    return features

# When user clicks this button, perform prediction
if st.button("Predict Coupon Acceptance"):
    features = preprocess_input(weather, passenger, time, age, temperature)
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0,1]  # Probability of acceptance (class 1)

    if prediction == 1:
        st.success(f"Coupon is likely to be accepted! (Confidence: {prediction_proba:.2f})")
    else:
        st.warning(f"Coupon is unlikely to be accepted. (Confidence: {1 - prediction_proba:.2f})")
