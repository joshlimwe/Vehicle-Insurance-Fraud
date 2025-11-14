import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import re, warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# this cleans column names as LightGBM cannot handle symbols like ':' or '-'
def clean_columns(df):
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in df.columns]
    return df

train = clean_columns(train)
test = clean_columns(test)

# target column
target_col = "FraudFound"

X = train.drop(columns=[target_col])
y = train[target_col]

# encodes Categorical Columns
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

for col in test.columns:
    if test[col].dtype == "object":
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col].astype(str))

if target_col in test.columns:
    test = test.drop(columns=[target_col])
test = test[X.columns]

# prepare Stratified K-Fold 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), 3))  # for XGB, LGB, CatBoost
test_preds = np.zeros((len(test), 3))

# this trains the base models
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n🔹 Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # handles imbalance ratio
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        early_stopping_rounds=30
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds[val_idx, 0] = xgb_model.predict_proba(X_val)[:, 1]
    test_preds[:, 0] += xgb_model.predict_proba(test)[:, 1] / skf.n_splits

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc"
    )
    oof_preds[val_idx, 1] = lgb_model.predict_proba(X_val)[:, 1]
    test_preds[:, 1] += lgb_model.predict_proba(test)[:, 1] / skf.n_splits

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=8,
        eval_metric="AUC",
        class_weights=[1, 10],
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    oof_preds[val_idx, 2] = cat_model.predict_proba(X_val)[:, 1]
    test_preds[:, 2] += cat_model.predict_proba(test)[:, 1] / skf.n_splits

# trains meta model next
meta_model = LogisticRegression(random_state=42)
meta_model.fit(oof_preds, y)

# evaluates stacking model
meta_val_preds = meta_model.predict_proba(oof_preds)[:, 1]
roc_auc = roc_auc_score(y, meta_val_preds)

# maximises recall with min precision = 0.5
precisions, recalls, thresholds = precision_recall_curve(y, meta_val_preds)
target_precision = 0.5
best_thr = 0.5
best_recall = 0
for p, r, t in zip(precisions, recalls, thresholds):
    if p >= target_precision and r > best_recall:
        best_recall, best_thr = r, t

print(f"\n⚙️ Optimal threshold for ≥{target_precision*100:.0f}% precision: {best_thr:.3f}")

ensemble_class = (meta_val_preds >= best_thr).astype(int)
precision = precision_score(y, ensemble_class)
recall = recall_score(y, ensemble_class)
f1 = f1_score(y, ensemble_class)

print(" Threshold-Tuned Stacking Ensemble Results (Recall-Focused)")
print(f"ROC-AUC Score : {roc_auc:.5f}")
print(f"Precision Score: {precision:.5f}")
print(f"Recall Score   : {recall:.5f}")
print(f"F1 Score       : {f1:.5f}")

final_preds = meta_model.predict_proba(test_preds)[:, 1]
pd.DataFrame({"Fraud_Probability": final_preds}).to_csv(
    "stacking_predictions_highrecall.csv", index=False
)

print("\n Saved as 'stacking_predictions_highrecall.csv'")
