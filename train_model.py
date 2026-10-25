"""
train_model.py
----------------
Loan Risk Assessment System - Model Training & Comparison Script
(Dataset: Loan_default.csv)

This script:
1. Loads Loan_default.csv, validates required columns
2. Drops LoanID (identifier only, not a predictive feature)
3. Reports dataset shape, dtypes, missing values, duplicates,
   target distribution, and numerical ranges
4. Splits data into train/test sets (80/20, stratified)
5. Builds a preprocessing pipeline:
     - numerical: median imputation + StandardScaler
     - categorical: most-frequent imputation + OneHotEncoder(handle_unknown="ignore")
6. Trains THREE candidate models on the same data:
     - Logistic Regression  (linear baseline model)
     - Random Forest Classifier
     - XGBoost Classifier
   all using class-imbalance handling (class_weight="balanced" /
   scale_pos_weight), since Default=1 is the minority class.
7. Evaluates all three: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
8. Selects the BEST model (highest F1-score on the Default=1 class)
9. Computes feature importance for the best model
10. Saves the best trained pipeline (preprocessing + model) to
    models/loan_risk_model.pkl
11. Writes evaluation_results.txt / .json with metrics for ALL three models

Run with:
    python train_model.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNGs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "Loan_default.csv")
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_risk_model.pkl")
RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "Default"
ID_COLUMN = "LoanID"

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
REQUIRED_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. DATA COLLECTION & VALIDATION
# ---------------------------------------------------------------------------
section("STEP 1: DATA COLLECTION & VALIDATION")
if not os.path.exists(DATA_PATH):
    sys.exit(f"ERROR: dataset not found at '{DATA_PATH}'. Place Loan_default.csv in data/ and rerun.")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded from {DATA_PATH}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    sys.exit(f"ERROR: required column(s) missing from dataset: {missing_cols}")
print("All required columns are present.")

print("\nData types:")
print(df.dtypes)

# ---------------------------------------------------------------------------
# 2. DATA PREPROCESSING / INSPECTION
# ---------------------------------------------------------------------------
section("STEP 2: DATA PREPROCESSING / INSPECTION")

missing = df.isnull().sum()
total_missing = int(missing.sum())
print(f"Total missing values: {total_missing}")
if total_missing > 0:
    print(missing[missing > 0])
else:
    print("No missing values found in the raw data. "
          "(The training pipeline still includes an imputer as a safety net "
          "for any missing values in future/unseen data.)")

duplicates = int(df.duplicated().sum())
print(f"Duplicate rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} exact duplicate rows. Rows remaining: {len(df)}")
else:
    print("No duplicate rows found. Nothing removed.")

print(f"\nTarget column '{TARGET_COLUMN}' unique values: {sorted(df[TARGET_COLUMN].unique())}")
print("Interpretation: 0 = No Default / Low Risk, 1 = Default / High Risk")

class_counts = df[TARGET_COLUMN].value_counts()
class_pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100
print("\nTarget class distribution:")
for cls in class_counts.index:
    label = "No Default / Low Risk" if cls == 0 else "Default / High Risk"
    print(f"  {cls} ({label}): {class_counts[cls]} rows ({class_pct[cls]:.2f}%)")

print(f"\nDropped column: {ID_COLUMN} (unique row identifier, not predictive; "
      f"kept OUT of the feature set)")
print(f"Numerical features ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
print(f"Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
print(f"Target column excluded from features: {TARGET_COLUMN}")

print("\nNumerical feature ranges:")
print(df[NUMERICAL_FEATURES].describe().T[["min", "max", "mean"]])

print("\nUnique values per categorical feature:")
for c in CATEGORICAL_FEATURES:
    print(f"  {c}: {sorted(df[c].dropna().unique().tolist())}")

# ---------------------------------------------------------------------------
# 3. FEATURE SELECTION (drop LoanID; no target leakage)
# ---------------------------------------------------------------------------
section("STEP 3: FEATURE SELECTION")
X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()
print(f"Final feature set ({len(FEATURE_COLUMNS)} columns): {FEATURE_COLUMNS}")
print(f"Target: {TARGET_COLUMN}")
print("Confirmed: LoanID and Default are NOT present in X (no identifier leakage, no target leakage).")

# ---------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
section("STEP 4: EXPLORATORY DATA ANALYSIS")
sns.set_style("whitegrid")

plt.figure(figsize=(6, 4.5))
ax = sns.countplot(x=TARGET_COLUMN, data=df, hue=TARGET_COLUMN,
                    palette=["#2E7D32", "#C62828"], legend=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["No Default (0)", "Default (1)"])
plt.title("Target Class Distribution (Default)")
plt.ylabel("Number of Applicants")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"), dpi=120)
plt.close()
print("Saved: outputs/target_distribution.png")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flatten(), ["Income", "Age", "LoanAmount", "CreditScore"]):
    sns.histplot(df[col], bins=30, kde=True, ax=ax, color="#1565C0")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "numerical_distributions.png"), dpi=120)
plt.close()
print("Saved: outputs/numerical_distributions.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(x=TARGET_COLUMN, y="CreditScore", data=df, hue=TARGET_COLUMN,
            palette=["#2E7D32", "#C62828"], legend=False, ax=axes[0])
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["No Default", "Default"])
axes[0].set_title("Credit Score vs Default")
sns.boxplot(x=TARGET_COLUMN, y="DTIRatio", data=df, hue=TARGET_COLUMN,
            palette=["#2E7D32", "#C62828"], legend=False, ax=axes[1])
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["No Default", "Default"])
axes[1].set_title("DTI Ratio vs Default")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "risk_vs_numerical.png"), dpi=120)
plt.close()
print("Saved: outputs/risk_vs_numerical.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
emp_risk = pd.crosstab(df["EmploymentType"], df[TARGET_COLUMN], normalize="index") * 100
emp_risk.plot(kind="bar", stacked=True, ax=axes[0], color=["#2E7D32", "#C62828"])
axes[0].set_title("Default % by Employment Type"); axes[0].set_ylabel("Percentage")
axes[0].legend(["No Default", "Default"]); axes[0].tick_params(axis="x", rotation=20)
cos_risk = pd.crosstab(df["HasCoSigner"], df[TARGET_COLUMN], normalize="index") * 100
cos_risk.plot(kind="bar", stacked=True, ax=axes[1], color=["#2E7D32", "#C62828"])
axes[1].set_title("Default % by Has Co-Signer"); axes[1].set_ylabel("Percentage")
axes[1].legend(["No Default", "Default"]); axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "risk_vs_categorical.png"), dpi=120)
plt.close()
print("Saved: outputs/risk_vs_categorical.png")

plt.figure(figsize=(7, 5.5))
corr_cols = NUMERICAL_FEATURES + [TARGET_COLUMN]
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=120)
plt.close()
print("Saved: outputs/correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ---------------------------------------------------------------------------
section("STEP 5: TRAIN-TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
)
print(f"Training set: {X_train.shape[0]} rows")
print(f"Testing set: {X_test.shape[0]} rows")
print("Stratified split used because Default is imbalanced "
      f"(~{class_pct[0]:.1f}% No Default vs ~{class_pct[1]:.1f}% Default) - "
      "stratification keeps the same ratio in both train and test sets.")
print("Preprocessing (imputers, scaler, encoder) is fit ONLY on X_train, "
      "inside the sklearn Pipeline below, then applied unchanged to X_test.")

# ---------------------------------------------------------------------------
# 6. MODEL TRAINING - THREE CANDIDATE ALGORITHMS
# ---------------------------------------------------------------------------
section("STEP 6: MODEL TRAINING (Logistic Regression, Random Forest, XGBoost)")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ]
)

neg, pos = class_counts.get(0, 0), class_counts.get(1, 0)
scale_pos_weight = neg / pos
print(f"Class imbalance ratio (neg/pos) = {neg}/{pos} = {scale_pos_weight:.2f}")
print("Handling: class_weight='balanced' for Logistic Regression & Random Forest, "
      f"scale_pos_weight={scale_pos_weight:.2f} for XGBoost. This makes errors on the "
      "minority Default=1 class count more during training, instead of letting the "
      "model take the 'always predict No Default' shortcut. No rows were deleted.")

CANDIDATES = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
}

trained_pipelines = {}
comparison = {}

for name, clf in CANDIDATES.items():
    section(f"Training: {name}")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, list(pipe.named_steps["classifier"].classes_).index(1)]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Default", "Default"])

    print(f"Accuracy : {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  "
          f"F1: {f1:.4f}  ROC-AUC: {auc:.4f}")

    trained_pipelines[name] = pipe
    comparison[name] = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "roc_auc": auc, "confusion_matrix": cm.tolist(), "classification_report": report,
    }

# ---------------------------------------------------------------------------
# 7. SELECT THE BEST MODEL
# ---------------------------------------------------------------------------
section("STEP 7: SELECTING THE BEST MODEL")
# Best model = highest F1-score on the Default=1 (minority, harder) class.
# Accuracy alone is misleading here: ~88.4% of applicants do NOT default, so a
# model that always predicts "No Default" would already score ~88% accuracy
# while catching zero real defaulters. F1 balances precision (how many flagged
# defaulters really default) and recall (how many real defaulters are caught).
BEST_MODEL_NAME = max(comparison, key=lambda k: comparison[k]["f1_score"])
best_pipeline = trained_pipelines[BEST_MODEL_NAME]
best_metrics = comparison[BEST_MODEL_NAME]

print("Model comparison (sorted by F1-score):")
for name, m in sorted(comparison.items(), key=lambda kv: -kv[1]["f1_score"]):
    print(f"  {name:22s}  Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
          f"Rec={m['recall']:.4f}  F1={m['f1_score']:.4f}  AUC={m['roc_auc']:.4f}")
print(f"\n>>> BEST MODEL SELECTED: {BEST_MODEL_NAME} (F1-Score = {best_metrics['f1_score']:.4f})")

accuracy, precision, recall, f1, roc_auc = (
    best_metrics["accuracy"], best_metrics["precision"], best_metrics["recall"],
    best_metrics["f1_score"], best_metrics["roc_auc"],
)
cm = np.array(best_metrics["confusion_matrix"])
report = best_metrics["classification_report"]

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
plt.title(f"Confusion Matrix - {BEST_MODEL_NAME} (Best Model)")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=120)
plt.close()
print("Saved: outputs/confusion_matrix.png")

metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
model_names = list(comparison.keys())
x = np.arange(len(metric_names))
width = 0.25
plt.figure(figsize=(9.5, 5.2))
colors = ["#8E8E8E", "#1565C0", "#2E7D32"]
for i, name in enumerate(model_names):
    vals = [comparison[name][m] for m in metric_names]
    plt.bar(x + (i - 1) * width, vals, width,
            label=name + (" *BEST*" if name == BEST_MODEL_NAME else ""),
            color=colors[i % len(colors)])
plt.xticks(x, metric_labels)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Model Comparison: Logistic Regression vs Random Forest vs XGBoost")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=120)
plt.close()
print("Saved: outputs/model_comparison.png")

# ---------------------------------------------------------------------------
# 8. FEATURE IMPORTANCE (for the best model)
# ---------------------------------------------------------------------------
section("STEP 8: FEATURE IMPORTANCE")

ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
all_feature_names = NUMERICAL_FEATURES + cat_feature_names

classifier = best_pipeline.named_steps["classifier"]
if hasattr(classifier, "feature_importances_"):
    importances = classifier.feature_importances_
elif hasattr(classifier, "coef_"):
    importances = np.abs(classifier.coef_[0])
else:
    importances = np.zeros(len(all_feature_names))

importance_df = pd.DataFrame({"feature": all_feature_names, "importance": importances}) \
    .sort_values("importance", ascending=False)


def base_feature(name):
    for c in CATEGORICAL_FEATURES:
        if name.startswith(c + "_"):
            return c
    return name


importance_df["base_feature"] = importance_df["feature"].apply(base_feature)
grouped_importance = importance_df.groupby("base_feature")["importance"].sum().sort_values(ascending=False)
grouped_importance = grouped_importance / grouped_importance.sum()

print(f"Feature importance (grouped) - {BEST_MODEL_NAME}:")
print(grouped_importance)

plt.figure(figsize=(7.5, 5.5))
grouped_importance.sort_values().plot(kind="barh", color="#1565C0")
plt.title(f"Feature Importance - {BEST_MODEL_NAME} (Best Model)")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=120)
plt.close()
print("Saved: outputs/feature_importance.png")

# ---------------------------------------------------------------------------
# 9. SAVE BEST MODEL + VERIFY IT LOADS BACK
# ---------------------------------------------------------------------------
section("STEP 9: SAVING BEST MODEL")
joblib.dump(best_pipeline, MODEL_PATH)
print(f"Best pipeline ({BEST_MODEL_NAME}) saved to: {MODEL_PATH}")

reloaded = joblib.load(MODEL_PATH)
sanity_row = X_test.iloc[[0]]
_ = reloaded.predict(sanity_row)
print("Verified: saved model reloads from disk and produces a prediction successfully.")

# ---------------------------------------------------------------------------
# SAVE EVALUATION RESULTS (used by the Flask "Model Performance" page)
# ---------------------------------------------------------------------------
results = {
    "dataset": "Loan_default.csv",
    "best_model": BEST_MODEL_NAME,
    "selection_metric": "f1_score",
    "accuracy": accuracy, "precision": precision, "recall": recall,
    "f1_score": f1, "roc_auc": roc_auc,
    "confusion_matrix": cm.tolist(),
    "class_labels": ["No Default", "Default"],
    "train_rows": int(X_train.shape[0]), "test_rows": int(X_test.shape[0]),
    "total_rows": int(df.shape[0]),
    "duplicates_removed": duplicates,
    "feature_importance": grouped_importance.round(4).to_dict(),
    "class_distribution": {
        "No Default": int(class_counts.get(0, 0)), "Default": int(class_counts.get(1, 0)),
    },
    "model_comparison": {
        name: {k: v for k, v in m.items() if k != "classification_report"}
        for name, m in comparison.items()
    },
}

with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "evaluation_results.txt"), "w") as f:
    f.write("LOAN RISK ASSESSMENT SYSTEM - MODEL EVALUATION RESULTS\n")
    f.write("Dataset: Loan_default.csv\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Total rows in dataset : {df.shape[0]}\n")
    f.write(f"Duplicate rows removed: {duplicates}\n")
    f.write(f"Training rows         : {X_train.shape[0]}\n")
    f.write(f"Testing rows          : {X_test.shape[0]}\n\n")
    f.write("MODEL COMPARISON (Logistic Regression vs Random Forest vs XGBoost)\n")
    f.write("-" * 78 + "\n")
    f.write(f"{'Model':22s}{'Accuracy':>10s}{'Precision':>11s}{'Recall':>9s}{'F1-Score':>10s}{'ROC-AUC':>10s}\n")
    for name, m in sorted(comparison.items(), key=lambda kv: -kv[1]["f1_score"]):
        marker = "  <-- BEST" if name == BEST_MODEL_NAME else ""
        f.write(f"{name:22s}{m['accuracy']:>10.4f}{m['precision']:>11.4f}"
                f"{m['recall']:>9.4f}{m['f1_score']:>10.4f}{m['roc_auc']:>10.4f}{marker}\n")
    f.write(f"\nBest model selected: {BEST_MODEL_NAME} (highest F1-score)\n\n")
    f.write(f"BEST MODEL METRICS ({BEST_MODEL_NAME})\n")
    f.write("-" * 40 + "\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-Score  : {f1:.4f}\n")
    f.write(f"ROC-AUC   : {roc_auc:.4f}\n\n")
    f.write("Confusion Matrix (rows = actual, cols = predicted):\n")
    f.write(f"                 Predicted No-Def  Predicted Default\n")
    f.write(f"Actual No-Def    {cm[0][0]:<18} {cm[0][1]}\n")
    f.write(f"Actual Default   {cm[1][0]:<18} {cm[1][1]}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write(f"Feature Importance (grouped, {BEST_MODEL_NAME}):\n")
    for feat, imp in grouped_importance.items():
        f.write(f"  {feat}: {imp:.4f}\n")

print("Saved: outputs/evaluation_results.json")
print("Saved: outputs/evaluation_results.txt")

import shutil
STATIC_IMAGES_DIR = os.path.join("static", "images")
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
for fname in os.listdir(OUTPUT_DIR):
    if fname.endswith(".png"):
        shutil.copy(os.path.join(OUTPUT_DIR, fname), os.path.join(STATIC_IMAGES_DIR, fname))
print(f"Copied graph images to {STATIC_IMAGES_DIR}/ for the web app")

section("TRAINING COMPLETE")
print(f"Best model ({BEST_MODEL_NAME}) is now used by the app.")
