import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model2.pkl")  # Make sure this file is in the same repo

st.title("Heart Disease Prediction")

st.markdown("Provide patient data to predict heart disease likelihood.")

# Input fields
age = st.number_input("Age", 1, 120, 50)
gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
chestpain = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
restingBP = st.number_input("Resting BP", 50, 200, 120)
serumcholestrol = st.number_input("Serum Cholestrol", 100, 600, 240)
fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restingrelectro = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
maxheartrate = st.number_input("Max Heart Rate", 60, 220, 150)
exerciseangia = st.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
noofmajorvessels = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[
        age, gender, chestpain, restingBP, serumcholestrol, fastingbloodsugar,
        restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, noofmajorvessels
    ]], columns=[
        "age", "gender", "chestpain", "restingBP", "serumcholestrol", "fastingbloodsugar",
        "restingrelectro", "maxheartrate", "exerciseangia", "oldpeak", "slope", "noofmajorvessels"
    ])

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("High risk of heart disease.")
    else:
        st.success("Low risk of heart disease.")
