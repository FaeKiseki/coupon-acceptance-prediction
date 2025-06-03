import streamlit as st
import joblib
import pandas as pd
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
weather_options = ['Sunny', 'Snowy']
time_options = ['7AM', '10AM', '2PM', '6PM', '10PM']
passenger_options = ['Alone', 'Friend(s)', 'Kid(s)', 'Partner']
age_options = ['below21', '26', '31', '36', '41', '46', '50plus']
temperature_options = [30, 55, 80]

temperature = st.selectbox('Temperature (°F)', temperature_options)
weather = st.selectbox('Weather condition', weather_options)
time = st.selectbox('Time of day', time_options)
passenger = st.selectbox('Passenger type', passenger_options)
age = st.selectbox('Age group', age_options)

# --- One-hot encoding ---
input_dict = {
    'temperature': temperature,
    'has_children': 0,
    'toCoupon_GEQ5min': 0,
    'toCoupon_GEQ15min': 0,
    'toCoupon_GEQ25min': 1,
    'direction_same': 1,
    'direction_opp': 0,
    'destination_No Urgent Place': 0,
    'destination_Work': 1
}

# Encode passenger
for val in ['Friend(s)', 'Kid(s)', 'Partner']:
    input_dict[f'passanger_{val}'] = int(passenger == val)

# Encode weather
for val in ['Snowy', 'Sunny']:
    input_dict[f'weather_{val}'] = int(weather == val)

# Encode time
for val in ['7AM', '10AM', '2PM', '6PM', '10PM']:
    input_dict[f'time_{val}'] = int(time == val)

# Encode age
for val in ['26', '31', '36', '41', '46', '50plus', 'below21']:
    input_dict[f'age_{val}'] = int(age == val)

# Fill missing categorical fields with 0
extra_keys = [
    'coupon_Carry out & Take away', 'coupon_Coffee House', 'coupon_Restaurant(20-50)', 'coupon_Restaurant(<20)',
    'expiration_2h', 'gender_Male',
    'maritalStatus_Married partner', 'maritalStatus_Single', 'maritalStatus_Unmarried partner', 'maritalStatus_Widowed',
    'education_Bachelors degree', 'education_Graduate degree (Masters or Doctorate)', 'education_High School Graduate',
    'education_Some High School', 'education_Some college - no degree'
] + [
    f'occupation_{job}' for job in [
        'Arts Design Entertainment Sports & Media', 'Building & Grounds Cleaning & Maintenance', 'Business & Financial',
        'Community & Social Services', 'Computer & Mathematical', 'Construction & Extraction',
        'Education&Training&Library', 'Farming Fishing & Forestry', 'Food Preparation & Serving Related',
        'Healthcare Practitioners & Technical', 'Healthcare Support', 'Installation Maintenance & Repair',
        'Legal', 'Life Physical Social Science', 'Management', 'Office & Administrative Support', 'Personal Care & Service',
        'Production Occupations', 'Protective Service', 'Retired', 'Sales & Related', 'Student',
        'Transportation & Material Moving', 'Unemployed'
    ]
] + [
    f'income_{inc}' for inc in [
        '$12500 - $24999', '$25000 - $37499', '$37500 - $49999', '$50000 - $62499', '$62500 - $74999',
        '$75000 - $87499', '$87500 - $99999', 'Less than $12500'
    ]
] + [
    f'{place}_{freq}' for place in ['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']
    for freq in ['4~8', 'Less than 1', 'More than 8', 'never']
]

for key in extra_keys:
    input_dict[key] = 0

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# --- Prediction ---
if st.button("Predict Coupon Acceptance"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.markdown(f"**Coupon will be {'ACCEPTED' if prediction == 1 else 'DECLINED'}**")
    st.markdown(f"Probability of acceptance: **{proba:.2%}**")
