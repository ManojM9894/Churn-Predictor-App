import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import requests
import io
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---------- Load model & encoders ----------

@st.cache_resource
def load_artifacts():
    try:
        with open("customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model_data["model"], model_data["features_names"], encoders
    except Exception:
        model_url = "https://drive.google.com/uc?id=<your-model-file-id>"
        encoder_url = "https://drive.google.com/uc?id=<your-encoder-file-id>"
        model_data = pickle.load(io.BytesIO(requests.get(model_url).content))
        encoders = pickle.load(io.BytesIO(requests.get(encoder_url).content))
        return model_data["model"], model_data["features_names"], encoders

model, feature_names, encoders = load_artifacts()

# ---------- Load CSV of risky customers ----------

@st.cache_data
def load_customer_list():
    try:
        df = pd.read_csv("top_50_risky_customers.csv")
    except:
        url = "https://drive.google.com/uc?id=<your-top50-file-id>"
        df = pd.read_csv(url)
    return df

top_customers = load_customer_list()

# ---------- UI Layout ----------

st.title("Telco Churn Predictor + SHAP")
selected_id = st.selectbox("Select a Customer ID", top_customers["customerID"].values)

selected_row = top_customers[top_customers["customerID"] == selected_id].iloc[0]

# ---------- Auto-prefill ----------

def get_customer_input():
    default_input = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": int(selected_row["tenure"]),
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": float(selected_row["MonthlyCharges"]),
        "TotalCharges": float(selected_row["MonthlyCharges"] * selected_row["tenure"])
    }

    return default_input

input_data = get_customer_input()
input_df = pd.DataFrame([input_data])

# Encode input
for col, encoder in encoders.items():
    input_df[col] = encoder.transform(input_df[col])

input_df = input_df[feature_names]

# ---------- Predict ----------
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")
st.write(f"**Churn Prediction:** {'Yes' if prediction else 'No'}")
st.write(f"**Churn Probability:** {probability * 100:.2f}%")

# ---------- SHAP Explanation ----------
explainer = shap.Explainer(model, input_df)
shap_values = explainer(input_df)

st.subheader("SHAP Churn Drivers (Bar)")
fig, ax = plt.subplots()
shap.plots.bar(shap_values[0], show=False)
st.pyplot(fig)
