import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="LoanWise | Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Styling --------------------
st.markdown(
    """
<style>
    .stApp {
        background: #ffffff;
        color: #000000;
    }

    .stApp p,
    .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #000000;
    }

    [data-testid="stSidebar"] {
        background: #102447;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        color: #ffffff;
        background: #102447;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: #ffffff;
    }

    .hero p {
        color: #ffffff;
    }

    .section-note {
        padding: 0.9rem 1rem;
        background: #f3f4f6;
        border-left: 4px solid #102447;
        border-radius: 8px;
        color: #000000;
        margin: .4rem 0 1rem 0;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        color: #000000;
        border: 1px solid #d1d5db;
        padding: 1rem;
        border-radius: 14px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.8rem;
    }

    .small-muted {
        color: #444444;
        font-size: .9rem;
    }
</style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "loan_approval_model.joblib"
EVALUATION_PATH = "model_evaluation_results.csv"
IMPORTANCE_PATH = "feature_importance.csv"

# -------------------- Header --------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏦 LoanWise</h1>
        <p>Loan Approval Prediction System · AIML Minor Project 1</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Educational machine-learning prototype. Predictions are not bank decisions and must not be used for real lending.")
st.info("Privacy note: avoid entering real personal or financial information. This demo is intended for sample data only.")

# -------------------- Load model --------------------
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: `{MODEL_PATH}`. Make sure it is in the same GitHub repository folder as `app.py`."
    )
    st.stop()


@st.cache_resource
def get_model(model_path: str):
    return joblib.load(model_path)


try:
    model = get_model(MODEL_PATH)
except Exception as exc:
    st.error("The model could not be loaded. Check that the model's scikit-learn version matches `requirements.txt`.")
    st.exception(exc)
    st.stop()

# -------------------- Navigation --------------------
pages = [
    "🏠 Dashboard",
    "📝 Make a Prediction",
    "🧮 EMI Calculator",
    "📊 Model Performance",
    "🔎 Feature Importance",
    "🕘 Prediction History",
    "ℹ️ About Project",
]
page = st.sidebar.radio("NAVIGATION", pages)
st.sidebar.divider()
st.sidebar.caption("Built with Python, Streamlit and scikit-learn")
st.sidebar.caption("AIML Minor Project 1")

# -------------------- Session history --------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


def make_prediction(row: pd.DataFrame):
    """Return predicted class and optional model-estimated approval probability."""
    prediction = int(model.predict(row)[0])
    approval_probability = None

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probabilities = model.predict_proba(row)[0]
        if 1 in classes:
            approval_probability = float(probabilities[classes.index(1)])

    return prediction, approval_probability


# -------------------- Dashboard --------------------
if page == "🏠 Dashboard":
    st.subheader("Welcome to your loan analysis workspace")
    st.markdown(
        '<div class="section-note">Explore a sample application, estimate a monthly EMI, and review model metrics. '
        "Use synthetic/sample values only.</div>",
        unsafe_allow_html=True,
    )

    history = st.session_state.prediction_history
    total = len(history)
    approved = sum(item["Prediction"] == "Likely approved" for item in history)
    not_approved = total - approved

    c1, c2, c3 = st.columns(3)
    c1.metric("Predictions this session", total)
    c2.metric("Likely approved", approved)
    c3.metric("Likely not approved", not_approved)

    st.markdown("### What you can do")
    left, right = st.columns(2)
    with left:
        st.markdown("#### 📝 Predict an outcome")
        st.write("Enter sample applicant and loan details and view the model's prediction.")
        st.caption("Choose **📝 Make a Prediction** from the navigation menu in the sidebar.")
        st.markdown("#### 📊 Explore model metrics")
        st.write("View evaluation results if the project CSV is available.")
    with right:
        st.markdown("#### 🧮 Calculate an EMI")
        st.write("Estimate monthly payments using a loan amount, annual interest rate, and tenure.")
        st.caption("Choose **🧮 EMI Calculator** from the navigation menu in the sidebar.")
        st.markdown("#### 🔎 Understand features")
        st.write("Review feature importance if `feature_importance.csv` is included.")

    st.markdown("### Project overview")
    x, y, z = st.columns(3)
    x.metric("Application inputs", "11")
    y.metric("Prediction type", "Classification")
    z.metric("Mode", "Educational demo")

# -------------------- Prediction form --------------------
elif page == "📝 Make a Prediction":
    st.subheader("Loan application")
    st.write("Fill in the sample details below. Fields are grouped to make the form easier to complete.")

    with st.form("loan_application", clear_on_submit=False):
        st.markdown("#### Applicant profile")
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        with b:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self-employed", ["No", "Yes"])
            applicant_income = st.number_input(
                "Applicant income", min_value=0, value=5000, step=500,
                help="Use the same income units as the training dataset.",
            )
        with c:
            coapplicant_income = st.number_input(
                "Co-applicant income", min_value=0, value=0, step=500,
                help="Use 0 if there is no co-applicant.",
            )
            credit_history = st.selectbox(
                "Credit history", [1.0, 0.0],
                format_func=lambda value: "Good / 1" if value == 1.0 else "Poor / 0",
                help="This must match the encoding used in your training dataset.",
            )
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

        st.markdown("#### Loan details")
        d, e = st.columns(2)
        with d:
            loan_amount = st.number_input(
                "Loan amount", min_value=0.0, value=120.0, step=10.0,
                help="Enter the amount in the same units used by the model's training data.",
            )
        with e:
            loan_term = st.selectbox("Loan term (months)", [360, 300, 240, 180, 120, 84, 60, 36, 12])

        submitted = st.form_submit_button(
            "✨ Predict loan outcome", type="primary", use_container_width=True
        )

    if submitted:
        row = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
        }])

        try:
            prediction, approval_probability = make_prediction(row)
            label = "Likely approved" if prediction == 1 else "Likely not approved"

            st.divider()
            st.markdown("### Prediction result")
            result_col, probability_col = st.columns([1, 1])
            with result_col:
                if prediction == 1:
                    st.success("✅ Model prediction: Likely approved")
                else:
                    st.warning("⚠️ Model prediction: Likely not approved")
            with probability_col:
                if approval_probability is not None:
                    st.metric("Model-estimated approval probability", f"{approval_probability:.1%}")
                    st.progress(max(0.0, min(1.0, approval_probability)))
                else:
                    st.info("This model does not provide probability estimates.")

            st.caption("This is a model estimate, not a guarantee, financial recommendation, or actual bank decision.")
            with st.expander("Review the values sent to the model"):
                st.dataframe(row, use_container_width=True, hide_index=True)

            history_item = {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Prediction": label,
                "Approval probability": (
                    f"{approval_probability:.1%}" if approval_probability is not None else "Not available"
                ),
                "Applicant income": applicant_income,
                "Co-applicant income": coapplicant_income,
                "Loan amount": loan_amount,
                "Loan term (months)": loan_term,
                "Credit history": "Good / 1" if credit_history == 1.0 else "Poor / 0",
                "Property area": property_area,
            }
            st.session_state.prediction_history.insert(0, history_item)
        except Exception as exc:
            st.error(
                f"Prediction failed: {exc}. Confirm that the model was trained using the expected Loan Prediction dataset and column names."
            )

# -------------------- EMI calculator --------------------
elif page == "🧮 EMI Calculator":
    st.subheader("Monthly EMI calculator")
    st.write("Estimate the monthly instalment for a hypothetical loan. This calculation is separate from the ML prediction.")
    with st.form("emi_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            principal = st.number_input("Loan principal (₹)", min_value=0.0, value=500000.0, step=25000.0)
        with c2:
            annual_rate = st.number_input("Annual interest rate (%)", min_value=0.0, value=9.0, step=0.25)
        with c3:
            years = st.number_input("Tenure (years)", min_value=1, max_value=40, value=5, step=1)
        calculate_emi = st.form_submit_button("Calculate EMI", type="primary", use_container_width=True)

    if calculate_emi:
        months = int(years * 12)
        monthly_rate = annual_rate / 12 / 100
        if principal <= 0:
            st.error("Enter a loan principal greater than ₹0.")
        elif monthly_rate == 0:
            emi = principal / months
            total_payment = principal
            total_interest = 0.0
            st.caption("A 0% interest rate was used.")
            st.metric("Estimated monthly EMI", f"₹{emi:,.2f}")
            e1, e2 = st.columns(2)
            e1.metric("Total payment", f"₹{total_payment:,.2f}")
            e2.metric("Total interest", f"₹{total_interest:,.2f}")
        else:
            emi = principal * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)
            total_payment = emi * months
            total_interest = total_payment - principal
            st.metric("Estimated monthly EMI", f"₹{emi:,.2f}")
            e1, e2 = st.columns(2)
            e1.metric("Total payment", f"₹{total_payment:,.2f}")
            e2.metric("Total interest", f"₹{total_interest:,.2f}")
            chart = pd.DataFrame(
                {"Amount (₹)": [principal, total_interest]},
                index=["Principal", "Estimated interest"],
            )
            st.markdown("#### Payment breakdown")
            st.bar_chart(chart)
        st.caption("Illustrative estimate only. Actual lender charges, fees, rate changes, and repayment rules may differ.")

# -------------------- Model performance --------------------
elif page == "📊 Model Performance":
    st.subheader("Model performance")
    if os.path.exists(EVALUATION_PATH):
        results = pd.read_csv(EVALUATION_PATH)
        st.dataframe(results.round(3), use_container_width=True, hide_index=True)
        if {"Model", "Accuracy", "ROC-AUC"}.issubset(results.columns):
            st.markdown("#### Compare evaluation metrics")
            st.bar_chart(results.set_index("Model")[["Accuracy", "ROC-AUC"]])
        elif "Accuracy" in results.columns:
            st.bar_chart(results.set_index("Model")["Accuracy"] if "Model" in results.columns else results["Accuracy"])
    else:
        st.warning("No evaluation file found. Add `model_evaluation_results.csv` from your Colab evaluation run to display real metrics.")
    with st.expander("What do the metrics mean?"):
        st.markdown(
            """
            - **Accuracy:** fraction of predictions that were correct.
            - **Precision:** among predicted positives, how many were actually positive.
            - **Recall:** among actual positives, how many were identified.
            - **F1-score:** balances precision and recall.
            - **ROC-AUC:** measures how well the model ranks positive examples above negative examples across thresholds.
            """
        )
    st.caption("Only report metrics generated from your actual held-out evaluation data. Do not invent or manually estimate model scores.")

# -------------------- Feature importance --------------------
elif page == "🔎 Feature Importance":
    st.subheader("Feature importance")
    if os.path.exists(IMPORTANCE_PATH):
        importance = pd.read_csv(IMPORTANCE_PATH)
        if {"Feature", "Importance"}.issubset(importance.columns):
            importance = importance.dropna(subset=["Feature", "Importance"])
            importance["Importance"] = pd.to_numeric(importance["Importance"], errors="coerce")
            importance = importance.dropna(subset=["Importance"])
            top = importance.sort_values("Importance", ascending=False).head(15)
            if not top.empty:
                st.markdown("#### Top features")
                st.bar_chart(top.set_index("Feature")["Importance"].sort_values(ascending=True), horizontal=True)
                with st.expander("View feature importance table"):
                    st.dataframe(top, use_container_width=True, hide_index=True)
            else:
                st.warning("No usable importance values were found in the CSV.")
        else:
            st.warning("The CSV needs columns named `Feature` and `Importance`.")
    else:
        st.warning("No feature importance file found. Add `feature_importance.csv` from your Colab run to show the actual chart.")
    st.caption("Importance is model-specific and does not prove causation, fairness, or that a feature should determine a real lending decision.")

# -------------------- Prediction history --------------------
elif page == "🕘 Prediction History":
    st.subheader("Prediction history")
    st.write("History is stored only for this active browser session and is not saved to a database.")
    history = st.session_state.prediction_history
    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download session history as CSV",
            data=csv_data,
            file_name="loan_prediction_history.csv",
            mime="text/csv",
        )
        if st.button("Clear history", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions yet. Make a prediction to see it listed here.")

# -------------------- About --------------------
elif page == "ℹ️ About Project":
    st.subheader("About this project")
    st.markdown(
        """
        **LoanWise — Loan Approval Prediction System** is an educational AIML project demonstrating how a trained
        classification model can process applicant information and produce a predicted class.

        **Technology stack**
        - Python and pandas for data handling
        - scikit-learn for machine-learning preprocessing and classification
        - joblib for loading the trained model
        - Streamlit for the interactive website

        **Current website features**
        - Loan outcome prediction
        - Optional model-estimated approval probability
        - Model performance report (when the evaluation CSV is supplied)
        - Feature importance chart (when the feature-importance CSV is supplied)
        - EMI calculator and temporary prediction history
        """
    )
    st.warning(
        "Important limitation: historical loan datasets may contain bias and may not represent current lending rules. "
        "This prototype is for classroom demonstration only, not for actual credit decisions."
    )

st.divider()
st.caption("LoanWise · Loan Approval Prediction System · AIML Minor Project 1")
