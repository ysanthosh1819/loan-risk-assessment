"""
predict.py
----------
Helper module that loads the trained pipeline and dataset metadata,
and provides a function to predict the risk level for a new applicant.

Used by app.py (Flask web application).

Dataset: Loan_default.csv
Target : Default  (0 = No Default / Low Risk, 1 = Default / High Risk)
"""

import os
import math
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "loan_risk_model.pkl")
DATA_PATH = os.path.join("data", "Loan_default.csv")

NUMERICAL_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]

CATEGORICAL_FEATURES = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Reasonable validation ranges, derived from the training dataset
# (see train_model.py Step 2 printout for the exact min/max observed).
# min/max are inclusive; "min_exclusive" fields must be strictly > 0.
VALID_RANGES = {
    "Age": (18, 100),
    "Income": (0, 10_000_000),
    "LoanAmount": (0, 10_000_000),
    "CreditScore": (300, 850),
    "MonthsEmployed": (0, 600),
    "NumCreditLines": (0, 50),
    "InterestRate": (0, 100),
    "LoanTerm": (1, 480),
    "DTIRatio": (0, 1),
}

# Fields that must be strictly greater than zero (zero is not a realistic value)
STRICTLY_POSITIVE = ["Age", "Income", "LoanAmount", "CreditScore", "NumCreditLines", "InterestRate", "LoanTerm"]


def load_model():
    """Load the trained pipeline from disk. Raises FileNotFoundError with a
    friendly message if the model hasn't been trained yet."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found at 'models/loan_risk_model.pkl'. "
            "Please run 'python train_model.py' first to train the model."
        )
    return joblib.load(MODEL_PATH)


def get_dropdown_options():
    """Read the dataset to get the valid category values for each
    categorical feature, so the web form always matches the real data."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found at 'data/Loan_default.csv'. Cannot build form options."
        )
    df = pd.read_csv(DATA_PATH)
    options = {}
    for col in CATEGORICAL_FEATURES:
        options[col] = sorted(df[col].dropna().unique().tolist())
    return options


def validate_input(form_data):
    """Validate raw form input. Returns (cleaned_dict, list_of_errors)."""
    errors = []
    cleaned = {}

    # Numerical fields
    for field in NUMERICAL_FEATURES:
        raw_value = form_data.get(field, "").strip()
        if raw_value == "":
            errors.append(f"'{field}' is required.")
            continue
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"'{field}' must be a valid number.")
            continue

        if math.isnan(value) or math.isinf(value):
            errors.append(f"'{field}' must be a finite number.")
            continue

        if field in STRICTLY_POSITIVE and value <= 0:
            errors.append(f"'{field}' must be greater than zero.")
            continue

        min_val, max_val = VALID_RANGES.get(field, (None, None))
        if min_val is not None and value < min_val:
            errors.append(f"'{field}' cannot be below {min_val}.")
            continue
        if max_val is not None and value > max_val:
            errors.append(f"'{field}' seems unrealistically high (max {max_val}).")
            continue
        cleaned[field] = value

    # Categorical fields
    try:
        valid_options = get_dropdown_options()
    except FileNotFoundError as e:
        errors.append(str(e))
        return cleaned, errors

    for field in CATEGORICAL_FEATURES:
        raw_value = form_data.get(field, "").strip()
        if raw_value == "":
            errors.append(f"'{field}' is required.")
            continue
        if raw_value not in valid_options[field]:
            errors.append(f"'{field}' has an invalid value.")
            continue
        cleaned[field] = raw_value

    # LoanTerm should be one of the discrete terms seen in training (12/24/36/48/60 months)
    if "LoanTerm" in cleaned and cleaned["LoanTerm"] not in (12, 24, 36, 48, 60):
        errors.append("'LoanTerm' must be one of: 12, 24, 36, 48, 60 months.")

    return cleaned, errors


def predict_risk(cleaned_data, pipeline):
    """Run the trained pipeline on a single new applicant and return
    the predicted label + probability."""
    input_df = pd.DataFrame([cleaned_data], columns=FEATURE_COLUMNS)
    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]

    # classes_ tells us which probability column corresponds to which class
    classes = list(pipeline.named_steps["classifier"].classes_)
    high_risk_idx = classes.index(1)
    low_risk_idx = classes.index(0)

    result = {
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
        "high_risk_probability": round(probabilities[high_risk_idx] * 100, 1),
        "low_risk_probability": round(probabilities[low_risk_idx] * 100, 1),
    }
    return result
