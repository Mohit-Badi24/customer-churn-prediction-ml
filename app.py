import streamlit as st
import pandas as pd
import joblib
st.sidebar.title("ℹ️ About This App")
st.sidebar.info(
    """
    This application predicts customer churn in the telecom domain
    using a machine learning model optimized for recall.

    **Model:** Logistic Regression  
    **Metric Focus:** Recall (Churn)  
    """
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("📉 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn based on service details.")
@st.cache_resource
def load_model():
    model = joblib.load("models/churn_model.pkl")
    features = joblib.load("models/model_features.pkl")
    return model, features

model, feature_columns = load_model()
st.subheader("Enter Customer Details")

st.subheader("🧾 Customer Information")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    with col2:
        tenure = st.slider(
            "Tenure (months)", 
            0, 72, 24,
            help="Number of months the customer has stayed with the company"
        )

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            help="Longer contracts usually indicate lower churn risk"
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=75.0,
            step=5.0
        )

    submit = st.form_submit_button("Predict Churn")

    senior = 1 if senior == "Yes" else 0

if submit:
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "Contract": contract,
        "MonthlyCharges": monthly_charges
    }])
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )
    churn_prob = model.predict_proba(input_encoded)[0][1]
    st.divider()
    st.subheader("📊 Churn Risk Assessment")

    st.subheader("Prediction Result")
    st.progress(int(churn_prob * 100))

    #st.write(f"### 🔍 Churn Probability: **{churn_prob:.2%}**")
    st.metric(
    label="Churn Risk Score",
    value=f"{int(churn_prob * 100)}%"
    )


    if churn_prob >= 0.7:
        st.error("🚨 High Risk: Immediate retention action recommended.")
    elif churn_prob >= 0.4:
        st.warning("⚠️ Medium Risk: Monitor and consider engagement offers.")
    else:
        st.success("✅ Low Risk: Customer likely to stay.")
    st.info(
    "Note: This prediction is based on historical customer behavior patterns "
    "and should be used as a decision-support tool, not a final judgment."
)


st.markdown("---")
st.caption(
    "Built by Mohit Badi | "
    "[GitHub](https://github.com/Mohit-Badi24/customer-churn-prediction-ml)"
)
