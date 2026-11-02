"""Under/overfitting diagnostics for the loan approval model."""
import os, sys, json, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score,
                                     learning_curve, validation_curve)
warnings.filterwarnings("ignore")
SEED = 42
BASE_DIR = "/home/zral/Downloads/loan-approval-prediction"
DEFAULT_CSV_PATH = f"{BASE_DIR}/data/loan_approval_dataset.csv"
df_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
if not os.path.exists(df_path):
    raise FileNotFoundError(
        f"Dataset not found at '{df_path}'. Download the Kaggle file and place it in the data folder, "
        "or pass it explicitly: python fit_diagnostics.py /path/to/loan_approval_dataset.csv"
    )
df = pd.read_csv(df_path)
df.columns = df.columns.str.strip()
for c in df.select_dtypes(exclude="number").columns: df[c] = df[c].str.strip()
A = ["residential_assets_value","commercial_assets_value","luxury_assets_value","bank_asset_value"]
df[A] = df[A].clip(lower=0)
df["y"] = (df.loan_status == "Approved").astype(int)
df["education"] = (df.education == "Graduate").astype(int)
df["self_employed"] = (df.self_employed == "Yes").astype(int)
df["total_assets"] = df[A].sum(axis=1)
df["loan_to_income"] = df.loan_amount / df.income_annum
df["loan_to_assets"] = df.loan_amount / (df.total_assets + 1)
df["monthly_installment"] = df.loan_amount / (df.loan_term * 12)
df["installment_to_income"] = df.monthly_installment / (df.income_annum / 12)
df["good_credit"] = (df.cibil_score >= 550).astype(int)
X_all = df.drop(columns=["loan_id","loan_status","y"]); y = df.y
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
P = dict(learning_rate=0.05, max_depth=2, n_estimators=200, random_state=SEED)
out = {}

# 1. Train vs CV vs test gap (both feature sets)
for name, X in {"with_good_credit": X_all, "leakage_free(no good_credit)": X_all.drop(columns="good_credit")}.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    m = GradientBoostingClassifier(**P).fit(Xtr, ytr)
    cvs = cross_val_score(GradientBoostingClassifier(**P), Xtr, ytr, cv=cv)
    out[name] = dict(train=round(m.score(Xtr,ytr),4), cv=round(cvs.mean(),4),
                     cv_std=round(cvs.std(),4), test=round(m.score(Xte,yte),4))
    out[name]["train_minus_test_gap"] = round(out[name]["train"]-out[name]["test"],4)
print(json.dumps(out, indent=2))

# 2. Learning curve (leakage-free)
X = X_all.drop(columns="good_credit")
sz, tr, va = learning_curve(GradientBoostingClassifier(**P), X, y, cv=cv,
                            train_sizes=np.linspace(.1,1,8), scoring="accuracy", n_jobs=-1)
fig, ax = plt.subplots(1, 2, figsize=(11,4))
ax[0].plot(sz, tr.mean(1), "o-", label="Train"); ax[0].plot(sz, va.mean(1), "o-", label="Validation")
ax[0].set_title("Learning curve"); ax[0].set_xlabel("Training rows"); ax[0].set_ylabel("Accuracy"); ax[0].legend()
print("Learning curve final train/val:", tr.mean(1)[-1].round(4), va.mean(1)[-1].round(4))

# 3. Validation curve: model complexity (tree depth)
d = [1,2,3,4,6,8]
t, v = validation_curve(GradientBoostingClassifier(learning_rate=.05, n_estimators=200, random_state=SEED),
                        X, y, param_name="max_depth", param_range=d, cv=cv, n_jobs=-1)
ax[1].plot(d, t.mean(1), "o-", label="Train"); ax[1].plot(d, v.mean(1), "o-", label="Validation")
ax[1].set_title("Complexity sweep (max_depth)"); ax[1].set_xlabel("max_depth"); ax[1].legend()
plt.tight_layout(); plt.savefig("outputs_loan/figures/07_fit_diagnostics.png", dpi=130)
print("Depth sweep val acc:", dict(zip(d, v.mean(1).round(4))))

# 4. Noise robustness: flip 5% of TRAIN labels; test on clean labels
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
rng = np.random.RandomState(SEED); yn = ytr.to_numpy().copy()
idx = rng.choice(len(yn), int(.05*len(yn)), replace=False); yn[idx] = 1 - yn[idx]
mn = GradientBoostingClassifier(**P).fit(Xtr, yn)
print("5% label noise -> train acc (noisy):", round(mn.score(Xtr, yn),4), "| clean test acc:", round(mn.score(Xte,yte),4))

# 5. Repeated random splits: stability
accs = []
for s in range(20):
    a,b,c,e = train_test_split(X, y, test_size=.2, stratify=y, random_state=s)
    accs.append(GradientBoostingClassifier(**P).fit(a,c).score(b,e))
print(f"20 random splits: mean={np.mean(accs):.4f} min={np.min(accs):.4f} max={np.max(accs):.4f}")
