import streamlit as st
import pandas as pd
import pickle
from google import genai
from google.genai.types import Content, Part
import os

# -----------------------------------------------------------
# Load model + scaler + encoders
# -----------------------------------------------------------
MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"
ENCODER_PATH = "model/encoders.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

# -----------------------------------------------------------
# Allowed categorical values (to prevent unseen label errors)
# -----------------------------------------------------------
allowed_categories = {
    "Gender": ["Male", "Female"],
    "Parent_Education_Level": ["High School", "Diploma", "Graduate", "Post-Graduate"],
    "Internet_Access": ["Yes", "No"],
    "Extra_Classes": ["Yes", "No"],
    "Sports_Activity": ["Yes", "No"]
}

# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.title("📊 Student Performance Prediction AI")
st.write("Enter the student details below:")

# User enters Gemini API key
api_key = st.text_input("🔐 Enter Gemini API Key", type="password")

# Prevent further UI unless key provided
if not api_key:
    st.warning("Please enter your Gemini API key to continue.")
    st.stop()

# -----------------------------------------------------------
# Input fields (clean)
# -----------------------------------------------------------
gender = st.selectbox("Gender", allowed_categories["Gender"])
parent_edu = st.selectbox("Parent Education Level", allowed_categories["Parent_Education_Level"])
study_hours = st.slider("Study Hours Per Day", 1, 12, 5)
attendance = st.slider("Attendance (in %)", 30, 100, 85)
internet = st.selectbox("Internet Access", allowed_categories["Internet_Access"])
extra = st.selectbox("Extra Classes", allowed_categories["Extra_Classes"])
sports = st.selectbox("Sports Activity", allowed_categories["Sports_Activity"])

# -----------------------------------------------------------
# Build feature vector safely
# -----------------------------------------------------------
def build_feature_vector():
    inp = {}

    # Encode using fitted encoders
    inp["Gender"] = encoders["Gender"].transform([gender])[0]
    inp["Parent_Education_Level"] = encoders["Parent_Education_Level"].transform([parent_edu])[0]
    inp["Internet_Access"] = encoders["Internet_Access"].transform([internet])[0]
    inp["Extra_Classes"] = encoders["Extra_Classes"].transform([extra])[0]
    inp["Sports_Activity"] = encoders["Sports_Activity"].transform([sports])[0]

    # Numeric fields
    inp["Study_Hours"] = study_hours
    inp["Attendance"] = attendance

    return pd.DataFrame([inp])

# -----------------------------------------------------------
# Summarize with Gemini
# -----------------------------------------------------------
def summarize_with_gemini(api_key: str, prediction: float, details: dict) -> str:
    client = genai.Client(api_key=api_key)

    prompt_text = f"""
    Summarize this student’s expected academic performance:

    Prediction Score: {prediction}

    Input Factors:
    - Gender: {details['Gender']}
    - Parent Education: {details['Parent_Education_Level']}
    - Study Hours: {details['Study_Hours']}
    - Attendance: {details['Attendance']}%
    - Internet Access: {details['Internet_Access']}
    - Extra Classes: {details['Extra_Classes']}
    - Sports Activity: {details['Sports_Activity']}

    Generate a short, student-friendly explanation.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite-preview-02-05",
        contents=[Content(parts=[Part.from_text(prompt_text)])],
    )

    return response.text


# -----------------------------------------------------------
# Predict Button
# -----------------------------------------------------------
if st.button("Predict Performance"):
    try:
        df = build_feature_vector()

        numeric_cols = ["Study_Hours", "Attendance"]
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        prediction = model.predict(df)[0]

        st.success(f"🎯 Predicted Performance Score: **{prediction:.2f}**")

        # Prepare summary details
        detail_map = {
            "Gender": gender,
            "Parent_Education_Level": parent_edu,
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Internet_Access": internet,
            "Extra_Classes": extra,
            "Sports_Activity": sports
        }

        st.info("⏳ Generating AI summary...")
        summary = summarize_with_gemini(api_key, prediction, detail_map)

        st.subheader("📘 AI Summary")
        st.write(summary)

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
