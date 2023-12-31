import streamlit as st
import pickle

st.title('Form using Streamlit for Iris Database')


# Define los diccionarios de mapeo
sex_parse_dict = {"Male": 0, "Female": 1, "None": 2}
smoker_parse_dict = {"Yes": 1, "No": 0}
region_parse_dict = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}

from joblib import load

# Carga los modelos
scaler_model = load("../models/scaler.pkl")
model = load("../models/lineal_regresion_model.pkl")

# Streamlit widgets to accept user input
age_val = st.slider(
    "Please enter your age",
    min_value=15,
    max_value=70,
    step=1
)

bmi_val = st.slider(
    "Enter the BMI",
    min_value=15.0,
    max_value=53.0,
    step=0.01
)

child_val = st.slider(
    "Child number",
    min_value=0,
    max_value=5,
    step=1
)

sex_val = st.selectbox(
    'Enter the sex',
    ("Male", "Female", "None")
)

smoker_val = st.selectbox(
    "Are you smoker?",
    ("Yes", "No")
)

region_val = st.selectbox(
    "Enter the region",
    ("Southwest", "Southeast", "Northwest", "Northeast")
)

# Prediction button callback
if st.button("Predecir"):
    # Collect input features into a list, applying the mapping from the dictionaries
    features = [
        age_val,
        bmi_val,
        child_val,
        sex_parse_dict[sex_val],
        smoker_parse_dict[smoker_val],
        region_parse_dict[region_val]
    ]
    
    # Scale the features using the previously loaded scaler model
    scaled_features = scaler_model.transform([features])
    
    # Predict the outcome using the previously loaded regression model
    prediction = model.predict(scaled_features)[0]
    
    # Display the prediction
    st.write("Predicted value:", prediction)