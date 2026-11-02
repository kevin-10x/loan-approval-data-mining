"""
LOAN APPROVAL PREDICTION - END-TO-END PIPELINE
==============================================
Dataset : Kaggle "Loan Approval Prediction Dataset" (loan_approval_dataset.csv)
Target  : loan_status  (Approved / Rejected)

Every step is numbered and commented. Run with:
    python loan_approval_pipeline.py [path_to_csv]

Outputs (written next to this script, in ./outputs_loan/):
    figures/*.png, metrics.json, loan_model.joblib, model_comparison.csv
"""

import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42  # fixed seed => identical results on every run

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "data", "loan_approval_dataset.csv")
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{CSV_PATH}'. Download the Kaggle file and place it in the data folder, "
        "or pass it explicitly: python loan_approval_pipeline.py /path/to/loan_approval_dataset.csv"
    )
OUT_DIR = os.path.join(BASE_DIR, "outputs_loan")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
metrics = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ----------------------------------------------------------------------
# STEP 1. LOAD DATA
# ----------------------------------------------------------------------
section("STEP 1: LOAD DATA")
df = pd.read_csv(CSV_PATH)
print("Raw shape:", df.shape)
print("Raw column names:", df.columns.tolist())
metrics["raw_shape"] = list(df.shape)

# ----------------------------------------------------------------------
# STEP 2. DATA CLEANING
# ----------------------------------------------------------------------
section("STEP 2: DATA CLEANING")

# 2a. Column names contain a leading space (e.g. ' education'). Strip them.
df.columns = df.columns.str.strip()

# 2b. Text values also contain a leading space (e.g. ' Graduate'). Strip them.
for col in df.select_dtypes(exclude="number").columns:
    df[col] = df[col].str.strip()
print("Whitespace removed from column names and text values.")

# 2c. Missing values
missing = df.isna().sum()
print("Missing values per column:\n", missing[missing > 0] if missing.any() else "None")
metrics["missing_values_total"] = int(missing.sum())

# 2d. Duplicate rows (ignoring the ID column, which is unique by design)
dups = int(df.drop(columns="loan_id").duplicated().sum())
print("Duplicate rows:", dups)
if dups:
    df = df.drop_duplicates(subset=df.columns.difference(["loan_id"]))
metrics["duplicates_removed"] = dups

# 2e. Invalid values: asset values cannot be negative.
asset_cols = ["residential_assets_value", "commercial_assets_value",
              "luxury_assets_value", "bank_asset_value"]
neg_counts = {c: int((df[c] < 0).sum()) for c in asset_cols}
print("Negative asset values:", neg_counts)
metrics["negative_asset_values_fixed"] = neg_counts
# Only 28 rows (0.66%), all residential = -100000 => data-entry errors.
# We clip to 0 (the lowest valid value) instead of deleting rows.
df[asset_cols] = df[asset_cols].clip(lower=0)

# 2f. Range sanity checks against known business limits
assert df["cibil_score"].between(300, 900).all(), "CIBIL must be in 300-900"
assert df["loan_term"].between(1, 30).all()
assert df["no_of_dependents"].between(0, 10).all()
print("Range checks passed (CIBIL 300-900, term, dependents).")

# 2g. Outlier check (IQR). Values are valid business values (large loans
#     and incomes exist), so we REPORT but do not remove them. Tree models
#     are insensitive to outliers; Logistic Regression is scaled.
num_cols_raw = ["income_annum", "loan_amount", "cibil_score"] + asset_cols
outliers = {}
for c in num_cols_raw:
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    outliers[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
print("IQR outlier counts (kept):", outliers)
metrics["iqr_outliers_kept"] = outliers

# ----------------------------------------------------------------------
# STEP 3. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------
section("STEP 3: EXPLORATORY DATA ANALYSIS")

# 3a. Encode target: Approved = 1, Rejected = 0
df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
balance = df["loan_status"].value_counts(normalize=True).round(4).to_dict()
print("Class balance (1=Approved):", balance)
metrics["class_balance"] = {str(k): v for k, v in balance.items()}

# 3b. Plots
sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df["loan_status"].map({1: "Approved", 0: "Rejected"}).value_counts().plot.bar(
    ax=ax[0], color=["#2a9d8f", "#e76f51"])
ax[0].set_title("Class balance")
ax[0].tick_params(axis="x", rotation=0)
sns.histplot(data=df, x="cibil_score", hue=df["loan_status"].map({1: "Approved", 0: "Rejected"}),
             bins=40, ax=ax[1], palette=["#2a9d8f", "#e76f51"])
ax[1].set_title("CIBIL score by outcome")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/01_target_and_cibil.png", dpi=130); plt.close()

bins = pd.cut(df["cibil_score"], [299, 500, 549, 600, 700, 900])
rate = df.groupby(bins, observed=True)["loan_status"].agg(["mean", "count"])
print("Approval rate by CIBIL band:\n", rate.round(3))
metrics["approval_rate_by_cibil_band"] = {str(k): round(float(v), 4) for k, v in rate["mean"].items()}
rate["mean"].plot.bar(figsize=(7, 4), color="#264653")
plt.title("Approval rate by CIBIL band"); plt.ylabel("Approval rate"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/02_approval_by_cibil_band.png", dpi=130); plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df.drop(columns="loan_id").select_dtypes("number").corr(), annot=True,
            fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_correlation.png", dpi=130); plt.close()

corr_target = (df.drop(columns="loan_id").select_dtypes("number").corr()["loan_status"]
               .drop("loan_status").sort_values(ascending=False).round(3))
print("Correlation with target:\n", corr_target)
metrics["corr_with_target"] = corr_target.to_dict()

# ----------------------------------------------------------------------
# STEP 4. FEATURE ENGINEERING & ENCODING
# ----------------------------------------------------------------------
section("STEP 4: FEATURE ENGINEERING & ENCODING")

# 4a. Drop loan_id: an identifier, carries no predictive information.
df = df.drop(columns="loan_id")

# 4b. Encode categorical variables (all binary => 0/1 mapping)
df["education"] = df["education"].map({"Not Graduate": 0, "Graduate": 1})
df["self_employed"] = df["self_employed"].map({"No": 0, "Yes": 1})

# 4c. Domain-driven features that lenders actually use
df["total_assets"] = df[asset_cols].sum(axis=1)
df["loan_to_income"] = df["loan_amount"] / df["income_annum"]
df["loan_to_assets"] = df["loan_amount"] / (df["total_assets"] + 1)
df["monthly_installment"] = df["loan_amount"] / (df["loan_term"] * 12)
df["installment_to_income"] = df["monthly_installment"] / (df["income_annum"] / 12)
df["good_credit"] = (df["cibil_score"] >= 550).astype(int)  # business-rule flag from EDA
print("Engineered: total_assets, loan_to_income, loan_to_assets, "
      "monthly_installment, installment_to_income, good_credit")

X = df.drop(columns="loan_status")
y = df["loan_status"]
print("Feature matrix:", X.shape, "| features:", X.columns.tolist())
metrics["features"] = X.columns.tolist()

# ----------------------------------------------------------------------
# STEP 5. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
section("STEP 5: TRAIN / TEST SPLIT")
# Stratified 80/20 keeps the 62/38 class ratio in both sets.
# The test set is untouched until the final evaluation (no data leakage).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
metrics["train_size"], metrics["test_size"] = len(X_train), len(X_test)

# ----------------------------------------------------------------------
# STEP 6. MODEL COMPARISON (5-fold stratified CV on TRAIN only)
# ----------------------------------------------------------------------
section("STEP 6: MODEL COMPARISON (5-fold CV)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
candidates = {
    # Scaling lives INSIDE the pipeline so it is fit on training folds only.
    "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
rows = []
for name, model in candidates.items():
    acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    rows.append({"Model": name, "CV Accuracy": acc.mean(), "CV Acc Std": acc.std(), "CV F1": f1.mean()})
    print(f"{name:22s} acc={acc.mean():.4f} (+/-{acc.std():.4f})  f1={f1.mean():.4f}")
comp = pd.DataFrame(rows).sort_values("CV Accuracy", ascending=False)
comp.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
metrics["model_comparison"] = comp.round(4).to_dict(orient="records")

# ----------------------------------------------------------------------
# STEP 7. HYPERPARAMETER TUNING (best families)
# ----------------------------------------------------------------------
section("STEP 7: HYPERPARAMETER TUNING")
searches = {
    "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1), {
        "n_estimators": [200, 400], "max_depth": [None, 8, 14],
        "min_samples_leaf": [1, 2, 4]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED), {
        "n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 4]}),
}
best_models, tuned_rows = {}, []
for name, (est, grid) in searches.items():
    gs = GridSearchCV(est, grid, cv=cv, scoring="accuracy", n_jobs=-1).fit(X_train, y_train)
    best_models[name] = gs.best_estimator_
    tuned_rows.append({"Model": name, "Best CV Accuracy": round(gs.best_score_, 4),
                       "Best Params": gs.best_params_})
    print(f"{name}: CV acc={gs.best_score_:.4f}  params={gs.best_params_}")
metrics["tuning"] = tuned_rows

winner_name = max(tuned_rows, key=lambda r: r["Best CV Accuracy"])["Model"]
final_model = best_models[winner_name]
print("\nSelected final model:", winner_name)
metrics["final_model"] = winner_name

# ----------------------------------------------------------------------
# STEP 8. FINAL EVALUATION ON THE HELD-OUT TEST SET
# ----------------------------------------------------------------------
section("STEP 8: FINAL TEST EVALUATION")
final_model.fit(X_train, y_train)
pred = final_model.predict(X_test)
proba = final_model.predict_proba(X_test)[:, 1]

metrics["test"] = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
}
cm = confusion_matrix(y_test, pred)
metrics["confusion_matrix"] = cm.tolist()
print(json.dumps(metrics["test"], indent=2))
print(classification_report(y_test, pred, target_names=["Rejected", "Approved"], digits=4))
print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)

plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (test)")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/04_confusion_matrix.png", dpi=130); plt.close()

fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(5, 4.5))
plt.plot(fpr, tpr, label=f"AUC = {metrics['test']['roc_auc']:.3f}", color="#264653")
plt.plot([0, 1], [0, 1], "--", color="grey")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curve (test)"); plt.legend()
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/05_roc_curve.png", dpi=130); plt.close()

imp = pd.Series(final_model.feature_importances_, index=X.columns).sort_values()
imp.tail(10).plot.barh(figsize=(7, 5), color="#2a9d8f")
plt.title("Top 10 feature importances"); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_feature_importance.png", dpi=130); plt.close()
metrics["top_features"] = imp.sort_values(ascending=False).head(8).round(4).to_dict()
print("Top features:\n", imp.sort_values(ascending=False).head(8).round(4))

# ----------------------------------------------------------------------
# STEP 9. ROBUSTNESS CHECK: repeated CV on the FULL dataset
# ----------------------------------------------------------------------
section("STEP 9: ROBUSTNESS (10-fold CV on all data)")
full_cv = cross_val_score(final_model, X, y,
                          cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                          scoring="accuracy")
print(f"10-fold accuracy: {full_cv.mean():.4f} (+/-{full_cv.std():.4f})")
metrics["cv10_accuracy_mean"] = round(full_cv.mean(), 4)
metrics["cv10_accuracy_std"] = round(full_cv.std(), 4)

# ----------------------------------------------------------------------
# STEP 10. SAVE MODEL + INFERENCE HELPER
# ----------------------------------------------------------------------
section("STEP 10: SAVE MODEL")
final_model.fit(X, y)  # refit on 100% of the data for deployment
joblib.dump({"model": final_model, "features": X.columns.tolist()}, f"{OUT_DIR}/loan_model.joblib")
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print("Saved loan_model.joblib and metrics.json")


def predict_loan(applicant: dict, model_path=f"{OUT_DIR}/loan_model.joblib"):
    """Predict approval for one applicant.

    Required keys: no_of_dependents, education ('Graduate'/'Not Graduate'),
    self_employed ('Yes'/'No'), income_annum, loan_amount, loan_term,
    cibil_score, residential_assets_value, commercial_assets_value,
    luxury_assets_value, bank_asset_value
    """
    bundle = joblib.load(model_path)
    a = dict(applicant)
    a["education"] = 1 if str(a["education"]).strip() == "Graduate" else 0
    a["self_employed"] = 1 if str(a["self_employed"]).strip() == "Yes" else 0
    row = pd.DataFrame([a])
    row["total_assets"] = row[asset_cols].sum(axis=1)
    row["loan_to_income"] = row["loan_amount"] / row["income_annum"]
    row["loan_to_assets"] = row["loan_amount"] / (row["total_assets"] + 1)
    row["monthly_installment"] = row["loan_amount"] / (row["loan_term"] * 12)
    row["installment_to_income"] = row["monthly_installment"] / (row["income_annum"] / 12)
    row["good_credit"] = (row["cibil_score"] >= 550).astype(int)
    row = row[bundle["features"]]
    p = bundle["model"].predict_proba(row)[0, 1]
    return {"decision": "Approved" if p >= 0.5 else "Rejected", "approval_probability": round(float(p), 4)}


if __name__ == "__main__":
    demo = {"no_of_dependents": 2, "education": "Graduate", "self_employed": "No",
            "income_annum": 9600000, "loan_amount": 29900000, "loan_term": 12,
            "cibil_score": 778, "residential_assets_value": 2400000,
            "commercial_assets_value": 17600000, "luxury_assets_value": 22700000,
            "bank_asset_value": 8000000}
    print("\nDemo prediction:", predict_loan(demo))
