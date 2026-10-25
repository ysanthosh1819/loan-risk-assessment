"""
app.py
------
Loan Risk Assessment System - Flask Web Application

Pages:
    /            Home page
    /predict     Prediction form (GET) + handle submission (POST)
    /result      Shows the prediction result
    /performance Shows model evaluation metrics & graphs
    /about       About the project

Run with:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash

from predict import (
    load_model,
    get_dropdown_options,
    validate_input,
    predict_risk,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

app = Flask(__name__)
app.secret_key = "loan-risk-assessment-college-project"  # fine for a local college project

MODEL_PATH = os.path.join("models", "loan_risk_model.pkl")
RESULTS_PATH = os.path.join("outputs", "evaluation_results.json")

# Human-friendly labels for form fields
FIELD_LABELS = {
    "Age": "Age (years)",
    "Income": "Annual Income (Rs.)",
    "LoanAmount": "Loan Amount (Rs.)",
    "CreditScore": "Credit Score",
    "MonthsEmployed": "Months Employed",
    "NumCreditLines": "Number of Credit Lines",
    "InterestRate": "Interest Rate (%)",
    "LoanTerm": "Loan Term (months)",
    "DTIRatio": "Debt-to-Income Ratio",
    "Education": "Education",
    "EmploymentType": "Employment Type",
    "MaritalStatus": "Marital Status",
    "HasMortgage": "Has Mortgage",
    "HasDependents": "Has Dependents",
    "LoanPurpose": "Loan Purpose",
    "HasCoSigner": "Has Co-Signer",
}

# Cache the trained pipeline once at startup (reloaded lazily if missing)
_pipeline_cache = {"pipeline": None}


def get_pipeline():
    if _pipeline_cache["pipeline"] is None:
        _pipeline_cache["pipeline"] = load_model()
    return _pipeline_cache["pipeline"]


@app.route("/")
def home():
    model_ready = os.path.exists(MODEL_PATH)
    return render_template("index.html", model_ready=model_ready)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model_ready = os.path.exists(MODEL_PATH)
    if not model_ready:
        flash("Model file not found. Please run 'python train_model.py' first.", "error")
        return redirect(url_for("home"))

    try:
        dropdown_options = get_dropdown_options()
    except FileNotFoundError as e:
        flash(str(e), "error")
        return redirect(url_for("home"))

    if request.method == "POST":
        cleaned_data, errors = validate_input(request.form)

        if errors:
            for err in errors:
                flash(err, "error")
            return render_template(
                "predict.html",
                numerical_features=NUMERICAL_FEATURES,
                categorical_features=CATEGORICAL_FEATURES,
                dropdown_options=dropdown_options,
                field_labels=FIELD_LABELS,
                form_values=request.form,
            )

        try:
            pipeline = get_pipeline()
            result = predict_risk(cleaned_data, pipeline)
        except Exception as e:
            flash(f"Something went wrong while making the prediction: {e}", "error")
            return render_template(
                "predict.html",
                numerical_features=NUMERICAL_FEATURES,
                categorical_features=CATEGORICAL_FEATURES,
                dropdown_options=dropdown_options,
                field_labels=FIELD_LABELS,
                form_values=request.form,
            )

        return render_template(
            "result.html",
            result=result,
            applicant=cleaned_data,
            field_labels=FIELD_LABELS,
        )

    # GET request -> show empty form
    return render_template(
        "predict.html",
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        dropdown_options=dropdown_options,
        field_labels=FIELD_LABELS,
        form_values={},
    )


@app.route("/performance")
def performance():
    if not os.path.exists(RESULTS_PATH):
        flash("Evaluation results not found. Please run 'python train_model.py' first.", "error")
        return redirect(url_for("home"))

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    return render_template("performance.html", results=results)


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(404)
def not_found(e):
    return render_template("about.html"), 404


@app.errorhandler(500)
def server_error(e):
    flash("An unexpected server error occurred. Please try again.", "error")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
