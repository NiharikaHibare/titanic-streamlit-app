# app.py - Streamlit Titanic survival predictor
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")

MODEL_PATH = "titanic_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Please place titanic_model.pkl in the same folder as app.py.")
    st.stop()

# Load trained pipeline (preprocessing + classifier)
model = joblib.load(MODEL_PATH)

st.markdown("Enter passenger details on the left and click **Predict**.")

# Sidebar inputs
st.sidebar.header("Passenger details")
pclass = st.sidebar.selectbox("Pclass", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
sibsp = st.sidebar.number_input("SibSp (siblings/spouses)", min_value=0, max_value=10, value=0, step=1)
parch = st.sidebar.number_input("Parch (parents/children)", min_value=0, max_value=10, value=0, step=1)
fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=10000.0, value=32.0, step=0.1)
embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])

# Prepare input dataframe
input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

st.subheader("Input preview")
st.write(input_df)

# Predict button
if st.button("Predict Survival"):
    # Model pipeline expects the same columns
    prob = model.predict_proba(input_df)[:, 1][0]
    pred = int(model.predict(input_df)[0])
    if pred == 1:
        st.success(f"💚 Predicted to survive — probability = {prob:.3f}")
    else:
        st.error(f"💔 Predicted not to survive — probability = {prob:.3f}")

st.markdown("---")
st.subheader("Batch prediction (CSV upload)")
st.write("Upload a CSV that contains these columns exactly: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked")

uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Uploaded file missing required columns: {missing}")
    else:
        df["survival_probability"] = model.predict_proba(df[required])[:, 1]
        st.write(df.head(10))
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", csv, "predictions.csv")
