import streamlit as st
import pandas as pd
import joblib
import os
from gemini_agent import GeminiAgent

st.set_page_config(page_title="Student Performance Analyzer", layout="wide")
st.title("📊 Student Performance Analyzer — Single AI Agent")

# User API Key
st.subheader("🔑 Enter Gemini API Key")
api_key = st.text_input("Gemini API Key", type="password")
agent = GeminiAgent(api_key) if api_key else None

# Upload Dataset
st.subheader("📂 Upload Dataset (CSV)")
file = st.file_uploader("Choose CSV File", type=["csv"])

df = None
if file:
    df = pd.read_csv(file)
    st.write("### 🔍 Data Preview")
    st.dataframe(df.head())

# Load Model
model_path = "model/model.joblib"
scaler_path = "model/scaler.joblib"

model = joblib.load(model_path) if os.path.exists(model_path) else None
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

if st.button("🚀 Run Analysis"):
    if not api_key:
        st.error("❌ Enter Gemini API key.")
    elif df is None:
        st.error("❌ Upload a dataset.")
    elif model is None or scaler is None:
        st.error("❌ Model or scaler missing in /model folder.")
    else:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        scaled = scaler.transform(df[numeric_cols])
        preds = model.predict(scaled)
        df["Predicted_Grade"] = preds

        st.success("✅ Prediction completed!")
        st.dataframe(df.head())

        with st.spinner("🤖 AI is generating insights..."):
            summary = agent.summarize(df, model)

        st.subheader("🧠 AI Summary Report")
        st.write(summary)
