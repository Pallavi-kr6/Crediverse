import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="CrediVerse Risk Intelligence", layout="wide")

st.title("CrediVerse — AI Credit Risk Intelligence")

# Load model
model = joblib.load("crediverse_model.pkl")

# --- User Inputs ---
st.sidebar.header("Applicant Information")

age = st.sidebar.slider("Age", 18, 80, 30)
income = st.sidebar.number_input("Monthly Income", 1000, 100000, 5000)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
utilization = st.sidebar.slider("Credit Utilization", 0.0, 1.0, 0.3)

# Alternative features
utility_score = st.sidebar.slider("Utility Payment Score", 0.0, 1.0, 0.8)
wallet_score = st.sidebar.slider("Wallet Stability", 0.0, 1.0, 0.7)
cashflow_vol = st.sidebar.slider("Cashflow Volatility", 0.0, 1.0, 0.4)

if st.button("Assess Risk"):

    input_data = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": utilization,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0,
    "MonthlyIncome": income,
    "NumberOfOpenCreditLinesAndLoans": 0,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0
}])

    prob = model.predict_proba(input_data)[0][1]

    def decision_engine(prob):
        if prob < 0.25:
            return "LOW RISK — APPROVE (8.5%)"
        elif prob < 0.50:
            return "MODERATE RISK — REVIEW (12%)"
        else:
            return "HIGH RISK — DECLINE"

    decision = decision_engine(prob)

    st.subheader("Risk Assessment Result")
    st.metric("Default Probability", f"{round(prob*100, 2)}%")
    st.success(decision)
