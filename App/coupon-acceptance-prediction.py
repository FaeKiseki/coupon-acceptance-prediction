import pickle
import os

# Define the relative path from App folder to the model file
model_path = os.path.join('..', 'Models', 'random_forest_model.pkl')

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)


st.title("Coupon Acceptance Prediction")

# --- User Inputs ---
# Adapt these options based on your dataset categories
weather_options = ['Sunny', 'Rainy', 'Snowy', 'Cloudy', 'Other']
passenger_options = ['Alone', 'Friends', 'Kids', 'Other']
age_options = ['21-30', '31-40', '41-50', '51+']

# Dropdown for weather selection
weather = st.selectbox('Weather', weather_options)

# Slider to select the hour of the day (0-23)
time = st.slider('Hour of the day', 0, 23, 12)

# Dropdown for passenger type
passenger = st.selectbox('Passenger', passenger_options)

# Dropdown for age group
age = st.selectbox('Age group', age_options)

# Slider for temperature input
temperature = st.slider('Temperature (°C)', -10, 40, 20)

# --- Encoding functions ---
def encode_weather(w):
    # Map weather categories to numerical values
    mapping = {'Sunny': 0, 'Rainy': 1, 'Snowy': 2, 'Cloudy': 3, 'Other': 4}
    return mapping.get(w, 4)

def encode_passenger(p):
    # Map passenger categories to numerical values
    mapping = {'Alone': 0, 'Friends': 1, 'Kids': 2, 'Other': 3}
    return mapping.get(p, 3)

def encode_age(a):
    # Map age groups to numerical values
    mapping = {'21-30': 0, '31-40': 1, '41-50': 2, '51+': 3}
    return mapping.get(a, 3)

# Encode categorical inputs
weather_encoded = encode_weather(weather)
passenger_encoded = encode_passenger(passenger)
age_encoded = encode_age(age)

# Prepare feature vector in the expected order
X = np.array([[weather_encoded, time, passenger_encoded, age_encoded, temperature]])

# --- Prediction ---
if st.button("Predict Coupon Acceptance"):
    # Predict class label (0 = reject, 1 = accept)
    prediction = model.predict(X)[0]

    # Predict probability of acceptance
    proba = model.predict_proba(X)[0][1]

    # Display the result with probability
    if prediction == 1:
        st.success(f"Coupon accepted with probability of {proba:.2%}")
    else:
        st.error(f"Coupon rejected with probability of {1 - proba:.2%}")