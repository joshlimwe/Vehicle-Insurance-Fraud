import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
N_FOLDS = 5

# ================================================================
# 0) Load
# ================================================================
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET = "FraudFound"
assert TARGET in train.columns, f"Target column '{TARGET}' missing in train.csv"

# ================================================================
# 1) Pre-clean helpers for original (un-cleaned) column names
#    Convert ordinal-like strings to numeric BEFORE renaming columns
# ================================================================
# month/day mappings for later calendar features (works pre/post clean)
MONTH_MAP = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
}
DOW_MAP = {
    "Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7
}

def _to_float_or_nan(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        xs = str(x).strip()
        if xs == "" or xs.lower() in {"nan", "none"}:
            return np.nan
        return float(xs)
    except Exception:
        return np.nan

def parse_price_mid(x):
    """VehiclePrice like '20,000 to 29,000' -> midpoint (e.g., 24500)."""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if not isinstance(x, str):
        return np.nan
    s = x.replace(",", "").strip()
    if "to" in s:
        try:
            a, b = [float(t.strip()) for t in s.split("to")]
            return (a + b) / 2.0
        except Exception:
            return np.nan
    # single number?
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_range_mid_or_years(x):
    """
    Handles:
      '2 to 4'   -> 3.0
      '26 to 30' -> 28.0
      '6 years'  -> 6.0
      '1 year'   -> 1.0
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if not isinstance(x, str):
        return np.nan
    s = x.strip().lower()
    if "to" in s:
        try:
            a, b = s.replace("years","").replace("year","").split("to")
            return (float(a.strip()) + float(b.strip())) / 2.0
        except Exception:
            return np.nan
    if "year" in s:
        try:
            return float(s.replace("years","").replace("year","").strip())
        except Exception:
            return np.nan
    # maybe plain number
    return _to_float_or_nan(s)

def parse_more_than_or_num(x):
    """
    'more than 30' -> 30.0
    'more than 5'  -> 5.0
    plain '7'      -> 7.0
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if not isinstance(x, str):
        return np.nan
    s = x.strip().lower()
    if s.startswith("more than"):
        try:
            val = s.replace("more than", "").strip()
            # take that number as the floor (conservative)
            return float(val)
        except Exception:
            return np.nan
    # maybe '3 to 4' style slipped here
    if "to" in s:
        try:
            a, b = s.split("to")
            return (float(a.strip()) + float(b.strip())) / 2.0
        except Exception:
            return np.nan
    return _to_float_or_nan(s)

def parse_number_of_cars(x):
    """
    '1 vehicle' -> 1
    '2 vehicles' -> 2
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x)
    if not isinstance(x, str):
        return np.nan
    s = x.strip().lower()
    parts = s.split()
    for p in parts:
        if p.isdigit():
            return int(p)
    return _to_float_or_nan(s)

def map_address_change(x):
    """
    AddressChange-Claim categories → ordinal scale
      'no change'        -> 0
      'under 6 months'   -> 1
      '1 year'           -> 2
      '2 to 3 years'     -> 3
      'over 3 years'     -> 4
    """
    if not isinstance(x, str):
        return np.nan
    s = x.strip().lower()
    mapping = {
        "no change": 0,
        "under 6 months": 1,
        "1 year": 2,
        "2 to 3 years": 3,
        "over 3 years": 4
    }
    return mapping.get(s, np.nan)

def convert_ordinal_strings(df):
    """Apply to ORIGINAL column names (with colons / hyphens) BEFORE cleaning."""
    df = df.copy()

    # VehiclePrice: price range -> midpoint
    if "VehiclePrice" in df.columns:
        df["VehiclePrice"] = df["VehiclePrice"].apply(parse_price_mid)

    # Ranges / years to numeric
    for c in ["AgeOfVehicle", "AgeOfPolicyHolder", "PastNumberOfClaims"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_range_mid_or_years)

    # 'more than N' or 'A to B' or plain numbers for day-gap style columns
    if "Days:Policy-Accident" in df.columns:
        df["Days:Policy-Accident"] = df["Days:Policy-Accident"].apply(parse_more_than_or_num)
    if "Days:Policy-Claim" in df.columns:
        df["Days:Policy-Claim"] = df["Days:Policy-Claim"].apply(parse_more_than_or_num)

    # NumberOfSuppliments sometimes has 'more than 5'
    if "NumberOfSuppliments" in df.columns:
        df["NumberOfSuppliments"] = df["NumberOfSuppliments"].apply(parse_more_than_or_num)

    # NumberOfCars: '1 vehicle', '2 vehicles'
    if "NumberOfCars" in df.columns:
        df["NumberOfCars"] = df["NumberOfCars"].apply(parse_number_of_cars)

    # AddressChange-Claim categorical ordering
    if "AddressChange-Claim" in df.columns:
        df["AddressChange-Claim"] = df["AddressChange-Claim"].apply(map_address_change)

    # Map Month / DayOfWeek / MonthClaimed / DayOfWeekClaimed to numeric for diffs
    for month_col in ["Month", "MonthClaimed"]:
        if month_col in df.columns:
            df[month_col + "_num"] = df[month_col].map(MONTH_MAP)

    for dow_col in ["DayOfWeek", "DayOfWeekClaimed"]:
        if dow_col in df.columns:
            df[dow_col + "_num"] = df[dow_col].map(DOW_MAP)

    return df

train = convert_ordinal_strings(train)
test  = convert_ordinal_strings(test)

# ================================================================
# 2) Clean column names AFTER conversions (so math worked)
# ================================================================
def clean_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in df.columns]
    return df

train = clean_columns(train)
test  = clean_columns(test)

# Split y out, drop from train
y = train[TARGET]
train = train.drop(columns=[TARGET], errors="ignore")

# ================================================================
# 3) Feature Engineering (post-clean names)
# ================================================================
def basic_feature_engineering(df):
    df = df.copy()

    # Count encodings for object columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col + "_count"] = df[col].map(df[col].value_counts())

    # Calendar diffs / (in)consistencies (safe if available)
    def add_diff(a, b, name):
        if a in df.columns and b in df.columns:
            df[name] = df[a] - df[b]

    add_diff("Month_num", "MonthClaimed_num", "Month_diff")
    add_diff("DayOfWeek_num", "DayOfWeekClaimed_num", "DOW_diff")
    add_diff("WeekOfMonth", "WeekOfMonthClaimed", "WOM_diff")

    # Ratios / interactions — only if both exist and numeric
    def add_ratio(n, d, name):
        if n in df.columns and d in df.columns:
            dn = pd.to_numeric(df[d], errors="coerce").fillna(0) + 1e-6
            df[name] = pd.to_numeric(df[n], errors="coerce") / dn

    def add_product(a, b, name):
        if a in df.columns and b in df.columns:
            df[name] = pd.to_numeric(df[a], errors="coerce") * pd.to_numeric(df[b], errors="coerce")

    # Classic pairs from your schema
    add_product("Age", "DriverRating", "Age_x_Rating")
    add_product("VehiclePrice", "AgeOfVehicle", "Price_x_VehicleAge")
    add_ratio("Days_Policy_Accident", "Days_Policy_Claim", "PolicyAcc_to_Claim_ratio")

    # Buckets
    def add_bucket(col, bins, name):
        if col in df.columns:
            df[name] = pd.cut(pd.to_numeric(df[col], errors="coerce"),
                              bins=bins, labels=False, include_lowest=True)

    add_bucket("Age", [0, 20, 30, 45, 60, 120], "Age_bucket")
    add_bucket("AgeOfVehicle", [0, 1, 4, 10, 25], "VehicleAge_bucket")
    add_bucket("AgeOfPolicyHolder", [0, 20, 30, 45, 60, 120], "PolicyHolderAge_bucket")

    return df

train_fe = basic_feature_engineering(train)
test_fe  = basic_feature_engineering(test)

# ================================================================
# 4) Impute simple missing values BEFORE label encoding
# ================================================================
for col in train_fe.columns:
    if train_fe[col].dtype == "object":
        train_fe[col] = train_fe[col].fillna("missing")
        if col in test_fe:
            test_fe[col] = test_fe[col].fillna("missing")
    else:
        med = pd.to_numeric(train_fe[col], errors="coerce").median()
        train_fe[col] = pd.to_numeric(train_fe[col], errors="coerce").fillna(med)
        if col in test_fe:
            test_fe[col] = pd.to_numeric(test_fe[col], errors="coerce").fillna(med)

# Keep list of categorical columns BEFORE label encoding (for target encoding)
cat_cols_for_te = [c for c in train_fe.columns if train_fe[c].dtype == "object"]

# ================================================================
# 5) Label Encoding across train+test to keep category alignment
# ================================================================
full = pd.concat([train_fe, test_fe], axis=0, ignore_index=True)
for col in full.columns:
    if full[col].dtype == "object":
        le = LabelEncoder()
        full[col] = le.fit_transform(full[col].astype(str))

train_fe = full.iloc[:len(train_fe)].reset_index(drop=True)
test_fe  = full.iloc[len(train_fe):].reset_index(drop=True)

# ================================================================
# 6) Fold-wise Target Encoding (no leakage)
# ================================================================
def add_target_encoding(X, y, X_test, cols, n_splits=5, random_state=RANDOM_STATE):
    X = X.copy()
    X_test = X_test.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for col in cols:
        # If column was label-encoded already, it's numeric — still ok to TE.
        oof_te = np.zeros(len(X))
        test_fold_vals = []

        for tr_idx, va_idx in skf.split(X, y):
            tr_X, va_X = X.iloc[tr_idx], X.iloc[va_idx]
            tr_y = y.iloc[tr_idx]

            means = tr_X.groupby(col)[tr_y.name].apply(
                lambda _: tr_y[tr_X[col] == _.name].mean()
            )
            # Map to validation
            oof_te[va_idx] = va_X[col].map(means).fillna(tr_y.mean()).values
            # Map to test
            test_fold_vals.append(X_test[col].map(means).fillna(tr_y.mean()).values)

        X[col + "_te"] = oof_te
        X_test[col + "_te"] = np.mean(test_fold_vals, axis=0)

    return X, X_test

# guard: use only columns that actually exist (after encoding)
cat_cols_for_te = [c for c in cat_cols_for_te if c in train_fe.columns]
train_fe, test_fe = add_target_encoding(train_fe, y, test_fe, cat_cols_for_te, n_splits=N_FOLDS, random_state=RANDOM_STATE)

# ================================================================
# 7) Prepare matrices
# ================================================================
X = train_fe
T = test_fe

print(f"[INFO] X shape: {X.shape} | Test shape: {T.shape}")

# ================================================================
# 8) Train base models with CV and stack
# ================================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros((len(X), 3))
test_preds = np.zeros((len(T), 3))

print("\n==================== TRAINING MODEL STACK ====================\n")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"🔹 Fold {fold}/{N_FOLDS}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # imbalance handling for XGB
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    scale_pos_weight = (neg / max(pos, 1))

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=900,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        early_stopping_rounds=40
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds[va_idx, 0] = xgb.predict_proba(X_va)[:, 1]
    test_preds[:, 0] += xgb.predict_proba(T)[:, 1] / N_FOLDS

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc")
    oof_preds[va_idx, 1] = lgb.predict_proba(X_va)[:, 1]
    test_preds[:, 1] += lgb.predict_proba(T)[:, 1] / N_FOLDS

    # CatBoost
    cat = CatBoostClassifier(
        iterations=900,
        learning_rate=0.03,
        depth=8,
        class_weights=[1.0, float(max(1.0, scale_pos_weight))],
        random_seed=RANDOM_STATE,
        verbose=False,
        loss_function="Logloss",
        eval_metric="AUC"
    )
    cat.fit(X_tr, y_tr)
    oof_preds[va_idx, 2] = cat.predict_proba(X_va)[:, 1]
    test_preds[:, 2] += cat.predict_proba(T)[:, 1] / N_FOLDS

# ================================================================
# 9) Meta model + threshold tuning
# ================================================================
meta = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
meta.fit(oof_preds, y)

meta_val = meta.predict_proba(oof_preds)[:, 1]
roc_auc = roc_auc_score(y, meta_val)

prec, rec, thr = precision_recall_curve(y, meta_val)
target_precision = 0.50
best_thr = 0.5
best_rec = 0.0
for p, r, t in zip(prec, rec, thr):
    if p >= target_precision and r > best_rec:
        best_rec, best_thr = r, t

print("\n==================== METRICS ====================")
print(f"Stack ROC-AUC            : {roc_auc:.6f}")
print(f"Optimal thr (≥{int(target_precision*100)}% precision): {best_thr:.4f}")
pred_class = (meta_val >= best_thr).astype(int)
print(f"Precision                : {precision_score(y, pred_class):.6f}")
print(f"Recall                   : {recall_score(y, pred_class):.6f}")
print(f"F1                       : {f1_score(y, pred_class):.6f}")

# ================================================================
# 10) Final predictions
# ================================================================
final_prob = meta.predict_proba(test_preds)[:, 1]
pd.DataFrame({"Fraud_Probability": final_prob}).to_csv("stacking_predictions_highrecall_FE.csv", index=False)
print("\nSaved: stacking_predictions_highrecall_FE.csv")
