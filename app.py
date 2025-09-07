import streamlit as st
import pandas as pd
import joblib

# Load model and selected features
model = joblib.load("models/heart_disease_model.pkl")
selected_features = ['sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# App title
st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("This app predicts whether a patient is at risk of heart disease based on medical data.")

# Sidebar for user input
st.sidebar.header("Enter Patient Data")

def user_input():
    sex = st.sidebar.selectbox("Sex (1 = male, 0 = female)", [0, 1])
    cp = st.sidebar.selectbox("Chest Pain Type (1–4)", [1, 2, 3, 4])
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", [0, 1])
    oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3])
    ca = st.sidebar.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thal (3 = normal, 6 = fixed defect, 7 = reversible defect)", [3, 6, 7])

    data = {
        'sex': sex,
        'cp': cp,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame([data])

input_df = user_input()

# Prediction
prediction = model.predict(input_df[selected_features])[0]

# Display result
st.subheader("Prediction Result")
if prediction == 1:
    st.error("⚠️ The model predicts **HIGH RISK of heart disease.** Please consult a doctor.")
else:
    st.success("✅ The model predicts **LOW RISK of heart disease.**")

# Model info
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Model trained on UCI Heart Disease dataset.\nAccuracy ~87%.")
