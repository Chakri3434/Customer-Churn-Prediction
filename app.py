# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 13:52:45 2025

@author: chakr
"""
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import json

model = pickle.load(open("C:/Users/chakr/Desktop/chakri/ML/Projects-Mine/Customer Churn Prediction/churn_model.pkl",'rb'))
scaler = pickle.load(open("C:/Users/chakr/Desktop/chakri/ML/Projects-Mine/Customer Churn Prediction/scaler.pkl",'rb'))
model_columns = json.load(open("C:/Users/chakr/Desktop/chakri/ML/Projects-Mine/Customer Churn Prediction/model_columns.json", 'r'))

st.title("Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on their profile and usage details.")

col1,col2,col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", 0, 72, 12)

with col2:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col3:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)


input_dict = {
    'gender': gender,
    'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
input_scaled = scaler.transform(input_encoded)


if st.button("Predict Churn"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer is **likely to churn** (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer will **likely stay** (Probability: {prob:.2f})")