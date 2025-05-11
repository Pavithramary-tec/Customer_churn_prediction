import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('churn_logreg_model (1).pkl')

st.title("Bank Customer Churn Prediction")
st.write("Enter customer details below to check the likelihood of churn.")

# User Inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Account Balance", min_value=0.0, step=100.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=1000.0, value=50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])

# Encode categorical variables manually to match model input
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0
has_credit_card = 1 if has_credit_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# Create input vector
input_data = np.array([[credit_score, age, tenure, balance, num_products,
                        has_credit_card, is_active_member, estimated_salary,
                        geo_germany, geo_spain, gender_male]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"This customer is likely to churn. 🔴 (Probability: {probability:.2f})")
    else:
        st.success(f"This customer is not likely to churn. 🟢 (Probability: {probability:.2f})")

