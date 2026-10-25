# Loan Risk Assessment System

A Machine Learning based B.Tech college project that predicts whether a loan
applicant is **Low Risk** (no predicted default) or **High Risk** (predicted
default), using the best of three trained classifiers and a simple Flask
web application.

## Project Overview

Banks and financial institutions receive a large number of loan applications
and need to assess the risk of each applicant before approving a loan. This
project builds a Machine Learning pipeline that learns from historical
applicant data (income, credit score, loan amount, employment type, existing
debt, etc.) and predicts whether a new applicant is likely to **Default**
(High Risk) or **Not Default** (Low Risk) on their loan.

## Features

- Data preprocessing pipeline (missing values, duplicates, encoding)
- Exploratory Data Analysis (EDA) with saved graphs
- Random Forest Classifier trained with an 80/20 train-test split
- Model evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Feature importance analysis
- Flask web application with:
  - Home page
  - Prediction form (based on real dataset columns)
  - Result page (Low Risk / High Risk + probability)
  - Model performance page (metrics + graphs)
  - About page

## Technology Used

**Machine Learning:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
**Backend:** Flask
**Frontend:** HTML, CSS (no external frameworks)

## Dataset

`data/Loan_default.csv` — 255,347 loan applicant records with the following columns:

| Column | Description | Used as feature? |
|---|---|---|
| `LoanID` | Row identifier | No — identifier, not predictive |
| `Age` | Applicant age | Yes |
| `Income` | Annual income | Yes |
| `LoanAmount` | Requested loan amount | Yes |
| `CreditScore` | Credit score (300–850) | Yes |
| `MonthsEmployed` | Months in current employment | Yes |
| `NumCreditLines` | Number of open credit lines | Yes |
| `InterestRate` | Loan interest rate (%) | Yes |
| `LoanTerm` | Loan term in months (12/24/36/48/60) | Yes |
| `DTIRatio` | Debt-to-income ratio | Yes |
| `Education` | Highest education level | Yes |
| `EmploymentType` | Full-time / Part-time / Self-employed / Unemployed | Yes |
| `MaritalStatus` | Single / Married / Divorced | Yes |
| `HasMortgage` | Yes / No | Yes |
| `HasDependents` | Yes / No | Yes |
| `LoanPurpose` | Auto / Business / Education / Home / Other | Yes |
| `HasCoSigner` | Yes / No | Yes |
| `Default` | Target: 0 = No Default / Low Risk, 1 = Default / High Risk | Target column |

The dataset is imbalanced: about 88.4% No Default vs 11.6% Default. This is
handled using stratified train/test splitting plus `class_weight="balanced"`
(Logistic Regression, Random Forest) / `scale_pos_weight` (XGBoost) — no rows
are deleted to fix the imbalance.

## Machine Learning Method

Three candidate algorithms are trained and evaluated on the **same**
preprocessed train/test split, each wrapped in a Scikit-learn `Pipeline`
together with a `ColumnTransformer` that:
- Imputes missing numerical values (median) and scales them (`StandardScaler`)
- Imputes missing categorical values (most frequent) and one-hot encodes them
  (`OneHotEncoder(handle_unknown="ignore")`)

Candidate algorithms:
- **Logistic Regression** — linear baseline model
- **Random Forest Classifier** — ensemble of decision trees
- **XGBoost Classifier** — gradient-boosted trees

The algorithm with the **highest F1-score** on the test set (the metric
that best balances precision and recall on the imbalanced Default class)
is automatically selected as the deployed model. See
`outputs/evaluation_results.txt` for the full comparison table and current
numbers — the winner can change between dataset/runs, which is why the
comparison is run fresh every time `train_model.py` executes.

Saving the whole pipeline (not just the model) with `joblib` ensures that
new applicant data is preprocessed **exactly the same way** during
prediction as it was during training — no train/predict mismatch.

## Project Structure

```text
loan-risk-assessment/
│
├── app.py                  # Flask web application
├── train_model.py          # Training script (run this first)
├── predict.py               # Shared prediction/validation helper used by app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Loan_default.csv     # Dataset
│
├── models/
│   └── loan_risk_model.pkl  # Saved trained pipeline (created by train_model.py)
│
├── notebooks/
│   └── analysis.ipynb       # Exploratory analysis notebook
│
├── static/
│   ├── css/style.css
│   └── images/              # Graphs copied here for the web app to display
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   ├── performance.html
│   └── about.html
│
└── outputs/
    ├── target_distribution.png
    ├── numerical_distributions.png
    ├── risk_vs_numerical.png
    ├── risk_vs_categorical.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── evaluation_results.json
    └── evaluation_results.txt
```

## Installation

Open the project folder in VS Code, then in a terminal:

```powershell
cd loan-risk-assessment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(On Mac/Linux, use `source venv/bin/activate` instead.)

## Model Training

Train the model (this also generates the EDA graphs and evaluation results):

```powershell
python train_model.py
```

This will:
1. Load and inspect `data/Loan_default.csv`
2. Preprocess the data
3. Generate EDA graphs into `outputs/`
4. Split the data 80/20 (train/test)
5. Train 3 candidate models and select the best by F1-score
6. Evaluate it and print/save the metrics
7. Compute feature importance
8. Save the full pipeline to `models/loan_risk_model.pkl`

## Starting Flask

Once the model is trained:

```powershell
python app.py
```

Open your browser at:

```
http://127.0.0.1:5000
```

## How to Use the Website

1. Go to the Home page.
2. Click **Predict Loan Risk**.
3. Fill in the applicant's income, age, experience, marital status, house
   ownership, car ownership, profession, current job years, current house
   years, and state.
4. Click **Predict Risk**.
5. View the result: **LOW RISK** or **HIGH RISK**, with the predicted
   probability for each class.
6. Click **Model Performance** to see accuracy, precision, recall, F1-score,
   the confusion matrix, and feature importance — all computed from the
   actual trained model.

## Model Evaluation

Run `python train_model.py` to see the latest results printed to the console
and saved to `outputs/evaluation_results.txt`. On the last training run:

- **Accuracy**: overall percentage of correct predictions.
- **Precision**: of all applicants predicted High Risk, how many actually
  were High Risk.
- **Recall**: of all applicants who actually were High Risk, how many the
  model correctly caught.
- **F1-Score**: the balance between precision and recall.

Exact numbers depend on the training run (see `outputs/evaluation_results.txt`
for the current values) since they come directly from evaluating the trained
model on the test set — nothing is hard-coded.

## Screenshots

_(Add screenshots of the Home, Predict, Result, and Performance pages here
after running the project for your project report.)_

## Future Improvements

- Try additional models (Gradient Boosting, XGBoost) and compare performance.
- Perform hyperparameter tuning (GridSearchCV) for better accuracy.
- Add more applicant features if a richer dataset becomes available.
- Add user authentication for loan officers.
- Deploy the Flask app to a cloud platform for wider access.
- Add explainability tools (e.g., SHAP) for individual predictions.

## Important Notes

- This is an academic / college project. Predictions are for demonstration
  purposes only and are **not** real financial or loan approval decisions.
- The dataset's `CITY` column was excluded from the model because it has 317
  unique values (too high cardinality for a simple model); `STATE` (29
  values) is used instead to capture geographic information.
