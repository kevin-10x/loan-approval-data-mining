# Loan Approval Prediction

A supervised machine learning (classification) project that predicts whether a loan application will be approved or rejected.

## Project Developers
- Kevin Kimutai
- Brian Rotich
- Marries Achieng

## Dataset
Kaggle "Loan Approval Prediction Dataset" (`loan_approval_dataset.csv`, 4,269 rows, 13 columns).
Download it from Kaggle and place it at `data/loan_approval_dataset.csv`
(not redistributed here; check the dataset's terms).

## Algorithms
Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (final model, tuned with GridSearchCV).

## How to Run
```bash
pip install -r requirements.txt
python loan_approval_pipeline.py data/loan_approval_dataset.csv
python fit_diagnostics.py data/loan_approval_dataset.csv
```

## Results (held-out test set, 854 applications)
Accuracy 99.88% | Precision 99.81% | Recall 100% | F1 99.91% | ROC-AUC 1.00

## Contents
- `loan_approval_pipeline.py` - cleaning, EDA, preprocessing, training, evaluation, model saving
- `fit_diagnostics.py` - overfitting/underfitting checks
- `Loan_Approval_Documentation.md` - step-by-step process documentation
- `Loan_Approval_Prediction_Data_Mining_Project.pdf` - project report
- `outputs_loan/` - figures, metrics, comparison table, trained model

## Limitations
Educational prototype. The dataset appears clean and strongly rule-driven (CIBIL score dominates), so real-world accuracy would likely be lower. Not for real lending decisions without validation and a fairness review.
