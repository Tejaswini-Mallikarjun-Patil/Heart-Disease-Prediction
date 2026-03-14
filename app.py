import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5", compile=False)
    return model

@st.cache_resource
def load_scaler():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return scaler

model = load_model()
scaler = load_scaler()

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Prediction using ANN")
st.markdown("Enter patient medical details to predict heart disease risk.")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol Level", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    restecg = st.selectbox("Rest ECG Result", [0,1,2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST", [0,1,2])
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy", [0,1,2,3,4])
    thal = st.selectbox("Thalassemia", [0,1,2,3])

# Convert categorical values
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# -----------------------------
# Prediction Button
# -----------------------------
st.divider()

if st.button("🔍 Predict Heart Disease Risk"):

    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]])

    # Scale input
    features = scaler.transform(features)

    # Prediction
    prediction = model.predict(features)[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.5:
        st.error("⚠️ High Risk of Heart Disease")
        st.write(f"Risk Score: **{prediction:.2f}**")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.write(f"Risk Score: **{prediction:.2f}**")

