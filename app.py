import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval Prediction System", page_icon="🏦", layout="wide")
st.title("🏦 Loan Approval Prediction System")
st.caption("AIML Minor Project 1 | Machine-learning demonstration")
st.info("Academic demo only. Do not use this prototype for real lending or credit decisions.")

MODEL_PATH = "loan_approval_model.joblib"
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Download loan_approval_model.joblib from Google Colab and put it beside app.py.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)

model = get_model()
page = st.sidebar.radio("Navigate", ["Make a Prediction", "Model Performance", "Feature Importance", "About"])

if page == "Make a Prediction":
    st.subheader("Loan Application Details")
    with st.form("loan_application"):
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with b:
            self_employed = st.selectbox("Self-employed", ["No", "Yes"])
            applicant_income = st.number_input("Applicant income", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Co-applicant income", min_value=0, value=0, step=500)
            loan_amount = st.number_input("Loan amount", min_value=0.0, value=120.0, step=10.0)
        with c:
            loan_term = st.selectbox("Loan term (months)", [360, 300, 240, 180, 120, 84, 60, 36, 12])
            credit_history = st.selectbox("Credit history", [1.0, 0.0], format_func=lambda x: "Good / 1" if x == 1.0 else "Poor / 0")
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
        submitted = st.form_submit_button("Predict Loan Outcome", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history, "Property_Area": property_area
        }])
        try:
            prediction = int(model.predict(row)[0])
            st.divider()
            if prediction == 1:
                st.success("Model prediction: Likely approved")
            else:
                st.warning("Model prediction: Likely not approved")
            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)
                probs = model.predict_proba(row)[0]
                if 1 in classes:
                    p = float(probs[classes.index(1)])
                    st.metric("Model-estimated approval probability", f"{p:.1%}")
                    st.progress(max(0.0, min(1.0, p)))
            st.caption("This is a statistical estimate, not a guarantee or an actual bank decision.")
            st.dataframe(row, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}. Check that the model was trained using the expected Loan Prediction dataset.")

elif page == "Model Performance":
    st.subheader("Model Performance")
    path = "model_evaluation_results.csv"
    if os.path.exists(path):
        results = pd.read_csv(path)
        st.dataframe(results.round(3), use_container_width=True, hide_index=True)
        if {"Model", "Accuracy", "ROC-AUC"}.issubset(results.columns):
            st.bar_chart(results.set_index("Model")[["Accuracy", "ROC-AUC"]])
    else:
        st.warning("Upload model_evaluation_results.csv from your Colab run to show actual metrics.")
    st.markdown("Accuracy measures overall correct predictions; precision and recall measure different types of errors; F1 balances precision and recall; ROC-AUC measures ranking performance across thresholds.")

elif page == "Feature Importance":
    st.subheader("Feature Importance")
    path = "feature_importance.csv"
    if os.path.exists(path):
        imp = pd.read_csv(path)
        if {"Feature", "Importance"}.issubset(imp.columns):
            imp = imp.sort_values("Importance", ascending=True).tail(15)
            st.bar_chart(imp.set_index("Feature")["Importance"], horizontal=True)
            st.dataframe(imp.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.warning("CSV is missing Feature or Importance columns.")
    else:
        st.warning("Upload feature_importance.csv from your Colab run to show the actual chart.")
    st.caption("Feature importance is model-specific and does not prove causation or fairness.")

else:
    st.subheader("About")
    st.write("This project demonstrates data preprocessing, categorical encoding, classification, confusion matrix, ROC-AUC, and feature importance using Python and scikit-learn.")
    st.warning("Historical data can contain bias. This website is for educational demonstration only.")

st.divider()
st.caption("Loan Approval Prediction System | AIML Minor Project 1")
