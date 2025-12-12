import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gemini_agent import GeminiAgent

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Student Performance AI Agent", layout="wide")

st.title("🎓 Student Performance Analysis (AI Agent Enabled)")
st.write("Upload student features → Model predicts grade → Gemini summarizes insights.")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("masked_data.csv")

    # encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    le_map = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = le

    return df, le_map


df, le_map = load_data()


# ---------------------------------------------------------
# TRAIN MODEL IN-MEMORY
# ---------------------------------------------------------
@st.cache_resource
def train_model(df):
    X = df.drop("Grade", axis=1)
    y = df["Grade"]

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)

    return model, X.columns


model, feature_names = train_model(df)

st.success("✔ Model trained successfully!")


# ---------------------------------------------------------
# USER INPUT FORM
# ---------------------------------------------------------
st.header("📥 Enter Student Details")

input_data = {}

for col in feature_names:
    if df[col].dtype in ["float64", "int64"]:
        val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()))
    else:
        val = st.selectbox(f"{col}", sorted(df[col].unique()))
    input_data[col] = val

input_df = pd.DataFrame([input_data])

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("Predict Grade"):
    prediction = model.predict(input_df)[0]

    st.subheader("🎯 Predicted Grade")
    st.metric("Grade", prediction)

    # -----------------------------------------------------
    # GEMINI AGENT SUMMARY
    # -----------------------------------------------------
    with st.expander("✨ AI Agent Summary (Gemini)"):
        st.write("Enter Gemini API key to generate explanation.")

        api_key = st.text_input("Gemini API Key", type="password")

        if api_key:
            agent = GeminiAgent(api_key)

            summary = agent.summarize_prediction(input_data, str(prediction))

            st.write("### 📌 Summary")
            st.write(summary)
        else:
            st.info("Provide your Gemini API key to get AI-generated insights.")


st.caption("Built with ❤️ using Streamlit + Gemini + RandomForest")
