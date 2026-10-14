# LoanWise — Loan Approval Prediction System | Minor Project 1

**Student:** RAHUL KUMAR  
**Roll number:** 301302224267  
**College:** RUNGTA COLLEGE OF ENGINEERING AND TECHNOLOGY  
**Department/branch:** Computer Science and Engineering (CSE Core)  
**Academic session:** 2024–2028  
**Model:** Logistic Regression (as specified for the project)

## Project links
- GitHub: https://github.com/rahulkt2411/loan-approval-prediction-system
- Colab notebook: https://colab.research.google.com/drive/1HWvlstT-uvcoJwy5BrhV5cRekh0WQ8lC#scrollTo=P9UzhP-ZUfxz
- Deployed Streamlit app: https://loan-approval-prediction-system-8dn6o5usdvqfd2y8tvmcjw.streamlit.app/

## Suggested repository structure
```text
loan-approval-prediction-system/
├── app.py
├── loan_approval_model.joblib
├── requirements.txt
├── model_evaluation_results.csv
├── feature_importance.csv
├── report/
│   └── Minor_Project_1_Loan_Approval_Report_Rahul_Kumar.pdf
├── presentation/
│   └── Minor_Project_1_Loan_Approval_Presentation_Rahul_Kumar.pptx
└── screenshots/
    ├── 01_prediction_result.png
    ├── 02_emi_calculator.png
    └── 03_prediction_input_form.png
```

## Verified from supplied screenshots
- Applicant profile and loan details form is visible.
- One example output shows “Likely approved” and a model-estimated approval probability of 63.3%.
- The EMI calculator screenshot shows principal ₹500,000, annual interest 9%, tenure 5 years, monthly EMI ₹10,379.18, total payment ₹622,750.66, and total interest ₹122,750.66.
- These screenshots do **not** establish overall model accuracy.

## Final checklist
- [ ] Fill in dataset name/source/licence, data dimensions, target labels, input features, preprocessing, and split strategy from Colab.
- [ ] Rerun model evaluation and record genuine metrics only.
- [ ] Check that `model_evaluation_results.csv` contains real results, not only headers.
- [ ] Confirm that Logistic Regression is the model loaded by the deployed app.
- [ ] Add any faculty guide/certificate/declaration required by college format.
- [ ] Review the report and presentation placeholders and remove all unresolved placeholders.
- [ ] Check screenshots and ensure no private data is visible.
- [ ] Upload report, PPT, dataset or an accessible dataset source, screenshots, source code, and other required files to GitHub or Drive.
- [ ] Ensure GitHub/Drive sharing works for evaluators.
- [ ] Open the deployed app and all links in a signed-out/incognito window.
- [ ] Submit the repository or Drive folder through the college form.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important
The displayed 63.3% is one example's model-estimated approval probability, **not accuracy**. The report intentionally avoids claiming accuracy, precision, recall, F1-score, or ROC-AUC until actual evaluation results are verified. This is an academic prototype, not a real lending decision system.
