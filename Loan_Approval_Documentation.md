# Loan Approval Prediction: Process Documentation

Dataset: Kaggle *Loan Approval Prediction Dataset* (4,269 applicants, 13 columns)
Target: `loan_status` (Approved / Rejected)
Code: `loan_approval_pipeline.py` (every step below maps to a numbered step in the script)

## Results at a glance

| Metric (held-out test set, 854 rows) | Value |
|---|---|
| Accuracy | 99.88% |
| Precision | 99.81% |
| Recall | 100% |
| F1 | 99.91% |
| ROC-AUC | 1.000 |
| 10-fold CV accuracy (all data) | 99.95% (±0.09%) |

Confusion matrix: 322 true rejections, 531 true approvals, 1 false approval, 0 false rejections.

## Results and Discussion
The model achieved a strong overall performance on the held-out dataset, with an accuracy of 99.88%, precision of 99.81%, recall of 100%, F1-score of 99.91%, and ROC-AUC of 1.000. These results show that the classifier was highly effective in separating approved and rejected applicants when tested on unseen data. The low false-positive and false-negative counts further show that the model is reliable for this particular dataset and meets the project objective of producing a meaningful and accurate loan approval prediction system.

The model therefore meets the required objectives for this academic project because it demonstrates a high level of predictive power, strong discrimination between the two classes, and stable validation performance. However, the dataset is highly structured and strongly dominated by CIBIL score, which suggests the task is almost rule-driven rather than challenging from a machine learning perspective. In real-world lending, additional factors such as income streams, repayment history, and policy changes may create more variation, so the model should be treated as a prototype that requires further validation before being used in live decision-making.

---

## Step 1: Load the data
Loaded with `pd.read_csv`. Raw shape: 4,269 rows × 13 columns.

## Step 2: Data cleaning

| Issue found | Action | Why |
|---|---|---|
| Column names had a leading space (`' education'`) | `df.columns.str.strip()` | Otherwise `df['education']` raises a KeyError |
| Text values had a leading space (`' Graduate'`, `' Approved'`) | `.str.strip()` on all text columns | Otherwise `'Graduate'` ≠ `' Graduate'` and encoding silently fails |
| Missing values | None found (0) | No imputation needed |
| Duplicate rows | None found (0) | Nothing removed |
| 28 negative values in `residential_assets_value` (all −100,000) | Clipped to 0 | Assets cannot be negative; these are data-entry errors. Clipping keeps the rows (0.66% of data) |
| Range validation | CIBIL 300–900, term, dependents all valid (asserted in code) | Guards against corrupt data |
| Outliers (IQR rule) | Found in residential (52), commercial (37), bank assets (5). **Kept.** | These are plausible wealthy-applicant values, not errors. Tree models are not sensitive to them |

## Step 3: Exploratory data analysis (figures in `outputs_loan/figures/`)

- **Class balance:** 62.2% Approved, 37.8% Rejected. Mild imbalance, so stratified splitting is used and no resampling is needed.
- **Strongest signal:** `cibil_score` has a 0.77 correlation with approval. Every other numeric feature is below 0.12 (largest: `loan_term` at −0.11).
- **The key finding:** approval rate by CIBIL band is about 10% for scores ≤549, and about 99.4% for scores ≥550. This is a near-deterministic cutoff around 550.

| CIBIL band | Approval rate |
|---|---|
| 300–500 | 10.7% |
| 501–549 | 9.2% |
| 550–600 | 99.4% |
| 601–700 | 99.4% |
| 701–900 | 99.5% |

## Step 4: Feature engineering and encoding

1. Dropped `loan_id` (an identifier with no predictive value).
2. Encoded `education` (Graduate=1) and `self_employed` (Yes=1) as 0/1. Both are binary, so one-hot encoding is unnecessary.
3. Target encoded as Approved=1, Rejected=0.
4. Added lender-style features: `total_assets`, `loan_to_income`, `loan_to_assets`, `monthly_installment`, `installment_to_income`, and `good_credit` (CIBIL ≥ 550).

**Honest caveat on `good_credit`:** the 550 threshold was spotted during EDA on the full dataset, so it is lightly informed by the labels. The model reaches ~99.97% CV accuracy even from raw CIBIL alone (the plain Decision Tree below), so the flag adds little. If you want a strictly leakage-free setup, delete that one line and re-run.

## Step 5: Train/test split
Stratified 80/20 split with `random_state=42`: 3,415 training rows, 854 test rows. The test set is not touched until Step 8.

## Step 6: Model comparison (5-fold stratified CV, training data only)

| Model | CV accuracy | CV F1 |
|---|---|---|
| Decision Tree | 99.97% | 99.98% |
| Gradient Boosting | 99.91% | 99.93% |
| Random Forest | 99.80% | 99.84% |
| Logistic Regression | 97.16% | 97.70% |

Logistic Regression is lower because the CIBIL relationship is a sharp step, not a smooth curve. Scaling is placed inside its pipeline so it is fit on training folds only.

## Step 7: Hyperparameter tuning (GridSearchCV, 5-fold)

| Model | Best CV accuracy | Best parameters |
|---|---|---|
| Gradient Boosting | 99.97% | learning_rate=0.05, max_depth=2, n_estimators=200 |
| Random Forest | 99.80% | n_estimators=400, max_depth=None, min_samples_leaf=1 |

Selected: **Gradient Boosting**. It is a more stable choice than a single decision tree, which can overfit on noisier data.

## Step 8: Final evaluation on the untouched test set
See the results table above. Top features by importance: `good_credit` (0.43), `cibil_score` (0.41), `installment_to_income` (0.13). Everything else combined is under 0.05.

## Step 9: Robustness check
10-fold CV on all data gives 99.95% ± 0.09%, consistent with the test result, so the score is not a lucky split.

## Step 10: Save and use the model
The final model is refit on 100% of the data and saved to `outputs_loan/loan_model.joblib`. Use it like this:

```python
from loan_approval_pipeline import predict_loan

predict_loan({
    "no_of_dependents": 2, "education": "Graduate", "self_employed": "No",
    "income_annum": 9600000, "loan_amount": 29900000, "loan_term": 12,
    "cibil_score": 778, "residential_assets_value": 2400000,
    "commercial_assets_value": 17600000, "luxury_assets_value": 22700000,
    "bank_asset_value": 8000000,
})
# {'decision': 'Approved', 'approval_probability': 0.9978}
```

(Importing the module re-runs training; for production, copy `predict_loan` into its own file.)

## Limitations
- The dataset appears synthetic: approval is driven almost entirely by CIBIL ≥ 550, whereas real lending also weighs income, debt, and history in more nuanced ways.
- Because of that, scores near 100% will likely not carry over to real bank data.
- Fairness: features like education and self-employment should be reviewed for bias before any real-world use.

## Summary and Conclusion
This project successfully developed and evaluated a machine learning model for predicting loan approval decisions using a structured lending dataset. The workflow included data cleaning, exploratory analysis, feature engineering, model comparison, tuning, validation, and deployment of the final trained classifier. The Gradient Boosting model was selected as the final model because it achieved the best balance of predictive performance, stability, and robustness, producing extremely high scores on both the held-out test set and cross-validation evaluation.

Overall, the project met its main objective of building an accurate and dependable loan approval prediction model. The results confirm that the model can correctly identify approved and rejected applicants with minimal error. To improve functionality in future work, the model should be retrained on broader and more diverse real-world lending data, assessed for fairness across protected groups, and monitored for drift over time. Additional safeguards such as human review for borderline cases, monthly performance tracking, and retraining when model behavior changes would further strengthen the system and make it more suitable for actual lending applications.

## GitHub Repository Link
GitHub Repository Link: [Add your GitHub repository URL here after uploading the project to GitHub]

## Reproduce
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python loan_approval_pipeline.py loan_approval_dataset.csv
```
The fixed seed (42) makes results identical on every run.
