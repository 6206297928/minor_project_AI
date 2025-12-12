import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai

# ---------------------------------------------------------
# LOAD MODEL + SCALER + ENCODERS
# ---------------------------------------------------------
MODEL_PATH = "model/random_forest.pkl"
SCALER_PATH = "model/scaler.pkl"
ENCODER_PATH = "model/encoders.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODER_PATH)

# ---------------------------------------------------------
# CLEAN STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Student Performance AI", layout="centered")
st.title("🎓 Student Performance Prediction (AI Powered)")
st.write("Fill the inputs below and enter your Gemini API key to get AI-generated analysis.")


# ---------------------------------------------------------
# FUNCTION: GEMINI AI SUMMARY
# ---------------------------------------------------------
def summarize_with_gemini(api_key: str, prediction: float, details: dict) -> str:
    genai.configure(api_key=api_key)

    prompt = f"""
You are an educational data analyst AI. Analyze the student's academic profile:

🎯 Predicted Score: {prediction}

📌 Student Inputs:
- Gender: {details['Gender']}
- Age: {details['Age']}
- Study Hours per Week: {details['Study_Hours']}
- Attendance: {details['Attendance']}%
- Parent Education Level: {details['Parent_Education_Level']}
- Family Income: {details['Family_Income_Level']}
- Extracurricular Activities: {details['Extracurricular_Activities']}
- Internet Access at Home: {details['Internet_Access']}
- Stress Level: {details['Stress_Level']}
- Sleep Hours: {details['Sleep_Hours']}

Write a short summary covering:
1. Expected academic performance  
2. Strengths  
3. Risk factors  
4. Personalized improvement tips  
"""

    response = genai.generate_text(
        model="gemini-2.0-flash-lite-preview-02-05",
        prompt=prompt
    )

    return response.text


# ---------------------------------------------------------
# FUNCTION: BUILD MODEL INPUT VECTOR
# ---------------------------------------------------------
def build_feature_vector():
    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    age = st.slider("Age", 15, 30, 20)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study_hours = st.slider("Study Hours per Week", 0, 40, 10)
    extracurricular = st.selectbox("Extracurricular Activities", encoders["Extracurricular_Activities"].classes_)
    internet = st.selectbox("Internet Access at Home", encoders["Internet_Access_at_Home"].classes_)
    parent_edu = st.selectbox("Parent Education Level", encoders["Parent_Education_Level"].classes_)
    income = st.selectbox("Family Income Level", encoders["Family_Income_Level"].classes_)
    stress = st.slider("Stress Level (1–10)", 1, 10, 5)
    sleep = st.slider("Sleep Hours per Night", 3, 12, 7)
    dept = st.selectbox("Department", ["CS", "Engineering", "Mathematics"])

    user_details = {
        "Gender": gender,
        "Age": age,
        "Attendance": attendance,
        "Study_Hours": study_hours,
        "Extracurricular_Activities": extracurricular,
        "Internet_Access": internet,
        "Parent_Education_Level": parent_edu,
        "Family_Income_Level": income,
        "Stress_Level": stress,
        "Sleep_Hours": sleep,
        "Department": dept,
    }

    # -----------------------------
    # BUILD MODEL FEATURE VECTOR
    # -----------------------------
    X = {}

    # Label-encoded categorical features
    X["Gender"] = encoders["Gender"].transform([gender])[0]
    X["Extracurricular_Activities"] = encoders["Extracurricular_Activities"].transform([extracurricular])[0]
    X["Internet_Access_at_Home"] = encoders["Internet_Access_at_Home"].transform([internet])[0]
    X["Parent_Education_Level"] = encoders["Parent_Education_Level"].transform([parent_edu])[0]
    X["Family_Income_Level"] = encoders["Family_Income_Level"].transform([income])[0]

    # Numeric features
    X["Age"] = age
    X["Attendance (%)"] = attendance
    X["Study_Hours_per_Week"] = study_hours
    X["Stress_Level (1-10)"] = stress
    X["Sleep_Hours_per_Night"] = sleep

    # One-hot encoded department
    X["Department_CS"] = 1 if dept == "CS" else 0
    X["Department_Engineering"] = 1 if dept == "Engineering" else 0
    X["Department_Mathematics"] = 1 if dept == "Mathematics" else 0

    df = pd.DataFrame([X])
    df_scaled = scaler.transform(df)

    return df_scaled, user_details


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------
st.subheader("🔧 Enter Student Details")
input_data, details = build_feature_vector()

st.subheader("🔑 Gemini API Key")
api_key = st.text_input("Enter your Gemini API key", type="password")

if st.button("Predict & Generate AI Summary"):
    if not api_key:
        st.error("Please enter your Gemini API key.")
    else:
        # ML Prediction
        prediction = model.predict(input_data)[0]

        st.success(f"🎯 **Predicted Grade Score: {prediction}**")

        # AI Summary
        with st.spinner("Generating AI summary..."):
            ai_summary = summarize_with_gemini(api_key, prediction, details)

        st.subheader("🧠 AI Summary")
        st.write(ai_summary)
