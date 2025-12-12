import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from gemini_agent import GeminiAgent


# -----------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------
st.set_page_config(page_title="AI Student Performance Predictor", layout="wide")
st.title("🎓 AI-Powered Student Performance Predictor")
st.markdown("Predict student grade + get AI-generated summary using Gemini.")



# -----------------------------------------
# LOAD DATA + TRAIN MODEL (NO PKL FILES)
# -----------------------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("masked_data.csv")

    # ---- Fill Missing ----
    df["Parent_Education_Level"] = df["Parent_Education_Level"].fillna("Bachelor's")

    # ---- Encode Binary ----
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({"No": 0, "Yes": 1})
    df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map({"No": 0, "Yes": 1})

    # ---- Ordinal Encoding ----
    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    df["Parent_Education_Level"] = df["Parent_Education_Level"].map(edu_map)

    income_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Family_Income_Level"] = df["Family_Income_Level"].map(income_map)

    # ---- One-hot Encoding Department ----
    df = pd.get_dummies(df, columns=["Department"], drop_first=True)

    # Fill missing numeric values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # ---- Label Encode Target ----
    encoder = LabelEncoder()
    df["Grade"] = encoder.fit_transform(df["Grade"])

    # ---- Feature Scaling ----
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Grade"]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ---- Train Model ----
    X = df.drop("Grade", axis=1)
    y = df["Grade"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, scaler, encoder, X.columns, df


model, scaler, encoder, model_columns, df_ref = load_and_train()



# -----------------------------------------
# GEMINI API KEY INPUT
# -----------------------------------------
st.sidebar.subheader("🔑 Enter Gemini API Key")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
agent = None

if api_key:
    agent = GeminiAgent(api_key)
else:
    st.warning("Please enter your Gemini API key to enable AI summary.")



# -----------------------------------------
# USER INPUT SECTION
# -----------------------------------------
st.sidebar.header("Student Parameters")

def user_input():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    dept = st.sidebar.selectbox("Department", ("CS", "Engineering", "Mathematics"))
    parent_edu = st.sidebar.selectbox("Parent Education", ("High School", "Bachelor's", "Master's", "PhD"))
    income = st.sidebar.selectbox("Family Income", ("Low", "Medium", "High"))
    extra = st.sidebar.selectbox("Extracurricular Activities", ("Yes", "No"))
    internet = st.sidebar.selectbox("Internet Access at Home", ("Yes", "No"))

    attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
    study_hours = st.sidebar.number_input("Study Hours per Week", 0, 80, 15)
    sleep = st.sidebar.number_input("Sleep Hours per Night", 1, 12, 7)
    stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)

    midterm = st.sidebar.number_input("Midterm Score", 0, 100, 70)
    final = st.sidebar.number_input("Final Score", 0, 100, 75)
    assignments = st.sidebar.number_input("Assignments Avg", 0, 100, 80)
    projects = st.sidebar.number_input("Projects Score", 0, 100, 85)

    # ---- Auto-computed fields ----
    quizzes = df_ref["Quizzes_Avg"].mean()
    participation = df_ref["Participation_Score"].mean()
    age = df_ref["Age"].mean()

    total = midterm + final + assignments + projects + quizzes + participation

    d = {
        "Gender": gender,
        "Age": age,
        "Attendance (%)": attendance,
        "Midterm_Score": midterm,
        "Final_Score": final,
        "Assignments_Avg": assignments,
        "Quizzes_Avg": quizzes,
        "Participation_Score": participation,
        "Projects_Score": projects,
        "Total_Score": total,
        "Study_Hours_per_Week": study_hours,
        "Extracurricular_Activities": extra,
        "Internet_Access_at_Home": internet,
        "Parent_Education_Level": parent_edu,
        "Family_Income_Level": income,
        "Stress_Level (1-10)": stress,
        "Sleep_Hours_per_Night": sleep,
        "Department": dept
    }

    return pd.DataFrame([d])


input_df = user_input()
st.subheader("📌 Input Preview")
st.dataframe(input_df, use_container_width=True)



# -----------------------------------------
# PREPROCESS INPUT BEFORE PREDICTION
# -----------------------------------------
def preprocess_input(df):
    df = df.copy()

    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({"No": 0, "Yes": 1})
    df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map({"No": 0, "Yes": 1})

    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    df["Parent_Education_Level"] = df["Parent_Education_Level"].map(edu_map)

    income_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Family_Income_Level"] = df["Family_Income_Level"].map(income_map)

    df = pd.get_dummies(df, columns=["Department"], drop_first=True)

    # Align columns with model
    df = df.reindex(columns=model_columns, fill_value=0)

    # Scale numeric features
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = scaler.transform(df[num_cols])

    return df



# -----------------------------------------
# PREDICTION BUTTON
# -----------------------------------------
if st.button("🔮 Predict Grade"):
    processed = preprocess_input(input_df)
    pred_encoded = model.predict(processed)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]

    st.success(f"### 🎯 Predicted Grade: **{pred_label}**")

    if agent:
        st.subheader("🧠 AI-Generated Performance Summary")
        summary = agent.get_summary(input_df, pred_label)
        st.write(summary)
