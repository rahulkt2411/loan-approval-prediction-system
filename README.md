# Loan Approval Prediction System Website

## Required files
Place these files beside `app.py`:
- `loan_approval_model.joblib` (required)
- `model_evaluation_results.csv` (optional)
- `feature_importance.csv` (optional)

## Run locally
Install Python 3.10+ and run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Export files from Colab
In the notebook, the final cell saves the model and CSV results. In Colab, click the folder icon on the left, refresh the file list, then right-click each file and select Download. Put downloaded files beside `app.py`.

## Publish online
Upload `app.py`, `requirements.txt`, and the exported model/CSV files to a GitHub repository, then deploy `app.py` through Streamlit Community Cloud.

Educational prototype only; not for real lending decisions.
