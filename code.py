# BT4012 Vehicle Insurance Fraud Detection 
# Group 27
# This includes : 
# Full ordinal cleaning functions
# Feature engineering upgrades
# Rare category smoothing
# Label encoding on combined train+test
# 10-fold stratified CV stacking
# XGBoost, LightGBM, CatBoost base models
# Gradient Boosting meta-model
# Isotonic-calibration of meta output
# Multi-seed blending (42, 99, 123)

import pandas as pd
import numpy as np
import re, warnings
warnings.filterwarnings("ignore") 

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

TARGET = "FraudFound"
RANDOM_SEEDS = [42, 99, 123]
N_FOLDS = 10

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

assert TARGET in train.columns, f"Target column {TARGET} missing."
y = train[TARGET]
train = train.drop(columns=[TARGET])

# Converts ordinal string to numeric values

MONTH_MAP = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
    "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
}
DOW_MAP = {
    "Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,
    "Friday":5,"Saturday":6,"Sunday":7
}

def _to_float_or_nan(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return np.nan
        return float(s)
    except:
        return np.nan

def parse_price_mid(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if not isinstance(x, str): return np.nan
    s = x.replace(",", "").strip()
    if "to" in s:
        try:
            a, b = [float(t.strip()) for t in s.split("to")]
            return (a+b) / 2
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def parse_range_mid_or_years(x):
    if isinstance(x, (int,float,np.integer,np.floating)):
        return float(x)
    if not isinstance(x,str):
        return np.nan
    s = x.strip().lower()
    if "to" in s:
        try:
            a,b = s.replace("years","").replace("year","").split("to")
            return (float(a.strip()) + float(b.strip()))/2
        except:
            return np.nan
    if "year" in s:
        try:
            return float(s.replace("years","").replace("year","").strip())
        except:
            return np.nan
    return _to_float_or_nan(s)

def parse_more_than_or_num(x):
    if isinstance(x,(int,float,np.integer,np.floating)):
        return float(x)
    if not isinstance(x,str): return np.nan
    s = x.strip().lower()
    if s.startswith("more than"):
        try:
            v = s.replace("more than","").strip()
            return float(v)
        except:
            return np.nan
    if "to" in s:
        try:
            a,b = s.split("to")
            return (float(a.strip()) + float(b.strip())) / 2
        except:
            return np.nan
    return _to_float_or_nan(s)

def parse_number_of_cars(x):
    if isinstance(x,(int,float,np.integer,np.floating)): return int(x)
    if not isinstance(x,str): return np.nan
    for p in x.lower().split():
        if p.isdigit():
            return int(p)
    return _to_float_or_nan(x)

def map_address_change(x):
    if not isinstance(x,str): return np.nan
    s = x.strip().lower()
    mapping = {
        "no change": 0,
        "under 6 months": 1,
        "1 year": 2,
        "2 to 3 years": 3,
        "over 3 years": 4
    }
    return mapping.get(s,np.nan)

def convert_ordinal_strings(df):
    df = df.copy()

    if "VehiclePrice" in df.columns:
        df["VehiclePrice"] = df["VehiclePrice"].apply(parse_price_mid)

    for c in ["AgeOfVehicle","AgeOfPolicyHolder","PastNumberOfClaims"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_range_mid_or_years)

    if "Days:Policy-Accident" in df.columns:
        df["Days:Policy-Accident"] = df["Days:Policy-Accident"].apply(parse_more_than_or_num)
    if "Days:Policy-Claim" in df.columns:
        df["Days:Policy-Claim"] = df["Days:Policy-Claim"].apply(parse_more_than_or_num)

    if "NumberOfSuppliments" in df.columns:
        df["NumberOfSuppliments"] = df["NumberOfSuppliments"].apply(parse_more_than_or_num)

    if "NumberOfCars" in df.columns:
        df["NumberOfCars"] = df["NumberOfCars"].apply(parse_number_of_cars)

    if "AddressChange-Claim" in df.columns:
        df["AddressChange-Claim"] = df["AddressChange-Claim"].apply(map_address_change)

    for m in ["Month","MonthClaimed"]:
        if m in df.columns:
            df[m+"_num"] = df[m].map(MONTH_MAP)

    for d in ["DayOfWeek","DayOfWeekClaimed"]:
        if d in df.columns:
            df[d+"_num"] = df[d].map(DOW_MAP)

    return df


train = convert_ordinal_strings(train)
test  = convert_ordinal_strings(test)


# Clean up column names
def clean_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+","_",c) for c in df.columns]
    return df

train = clean_columns(train)
test  = clean_columns(test)


# Improved feature engineering
def enhanced_feature_engineering(df):
    df = df.copy()

    # Count encoding for categorical
    for col in df.columns:
        if df[col].dtype == "object":
            df[col + "_count"] = df[col].map(df[col].value_counts())
    
    #timing features
    if "DayOfWeek_num" in df.columns:
        df["IsWeekend"] = df["DayOfWeek_num"].isin([6, 7]).astype(int)

    if "Month_num" in df.columns:
        df["IsMonthEnd"] = df["Month_num"].isin([3, 6, 9, 12]).astype(int)

    if "Days_Policy_Claim" in df.columns and "Days_Policy_Accident" in df.columns:
        df["SameDayReport"] = (df["Days_Policy_Claim"] == df["Days_Policy_Accident"]).astype(int)
        df["ClaimDelay"] = df["Days_Policy_Claim"] - df["Days_Policy_Accident"]
    
    # incosistency features
    df["InconsistencyScore"] = 0

    if "Month_num" in df.columns and "MonthClaimed_num" in df.columns:
        df["InconsistencyScore"] += (df["Month_num"] != df["MonthClaimed_num"]).astype(int)

    if "DayOfWeek_num" in df.columns and "DayOfWeekClaimed_num" in df.columns:
        df["InconsistencyScore"] += (df["DayOfWeek_num"] != df["DayOfWeekClaimed_num"]).astype(int)

    if "WeekOfMonth" in df.columns and "WeekOfMonthClaimed" in df.columns:
        df["InconsistencyScore"] += (df["WeekOfMonth"] != df["WeekOfMonthClaimed"]).astype(int)

    if "AddressChange_Claim" in df.columns:
        df["InconsistencyScore"] += (df["AddressChange_Claim"] > 1).astype(int)
    
    # Basic diffs
    def add_diff(a,b,name):
        if a in df.columns and b in df.columns:
            df[name] = df[a] - df[b]

    add_diff("Month_num","MonthClaimed_num","Month_diff")
    add_diff("DayOfWeek_num","DayOfWeekClaimed_num","DOW_diff")
    add_diff("WeekOfMonth","WeekOfMonthClaimed","WOM_diff")

    # Interaction features

    def add_interaction(a, b, name):
        if a in df.columns and b in df.columns:
            df[name] = pd.to_numeric(df[a], errors="coerce") * pd.to_numeric(df[b], errors="coerce")

    add_interaction("VehiclePrice", "AgeOfVehicle", "Price_x_AgeVehicle")
    add_interaction("Deductible", "DriverRating", "Deductible_x_Rating")
    add_interaction("VehiclePrice", "PastNumberOfClaims", "Price_x_PastClaims")
    add_interaction("Age", "NumberOfCars", "Age_x_NumCars")

    def add_ratio(a, b, name):
        if a in df.columns and b in df.columns:
            df[name] = (
                pd.to_numeric(df[a], errors="coerce") /
                (pd.to_numeric(df[b], errors="coerce") + 1e-6)
            )

    add_ratio("Deductible", "VehiclePrice", "Deductible_per_Price")
    add_ratio("PastNumberOfClaims", "NumberOfCars", "Claims_per_Car")
    add_ratio("Days_Policy_Accident", "Days_Policy_Claim", "Acc_to_Claim_Ratio")

    def add_product(a,b,name):
        if a in df.columns and b in df.columns:
            df[name] = pd.to_numeric(df[a],errors="coerce") * pd.to_numeric(df[b],errors="coerce")

    add_ratio("Deductible","VehiclePrice","Deductible_per_VehiclePrice")
    add_product("Age","DriverRating","Age_x_Rating")
    add_product("VehiclePrice","AgeOfVehicle","Price_x_VehicleAge")
    add_ratio("Days_Policy_Accident","Days_Policy_Claim","PolicyAcc_to_Claim_ratio")

    # Claim delay
    if "Days_Policy_Claim" in df.columns and "Days_Policy_Accident" in df.columns:
        df["ClaimDelay"] = df["Days_Policy_Claim"] - df["Days_Policy_Accident"]
        df["ClaimDelay_bucket"] = pd.cut(
            df["ClaimDelay"],
            bins=[-99,0,5,15,30,999],
            labels=False
        )

    # Buckets
    def add_bucket(col,bins,name):
        if col in df.columns:
            df[name] = pd.cut(pd.to_numeric(df[col],errors="coerce"),
                              bins=bins,labels=False,include_lowest=True)

    add_bucket("Age",[0,20,30,45,60,120],"Age_bucket")
    add_bucket("AgeOfVehicle",[0,1,4,10,25],"VehicleAge_bucket")

    add_bucket("Age",[0,20,30,45,60,120],"Age_bucket")
    add_bucket("AgeOfVehicle",[0,1,4,10,25],"VehicleAge_bucket")

    #target encoding
    target_enc_cols = [
        "AccidentArea", "VehicleMaker", "VehicleModel", "PolicyType",
        "BasePolicy", "Make", "DayOfWeek", "DayOfWeekClaimed"
    ]

    global_mean = y.mean()

    for col in target_enc_cols:
        if col in df.columns:
            means = df.groupby(col)["FraudFound"].mean() if "FraudFound" in df.columns else {}
            df[col + "_te"] = df[col].map(means).fillna(global_mean)

    return df

train_fe = enhanced_feature_engineering(train)
test_fe  = enhanced_feature_engineering(test)


# Rare category smoothing
def smooth_rare_categories(df, min_count=10):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            vc = df[col].value_counts()
            rare = vc[vc < min_count].index
            df[col] = df[col].apply(lambda x: "RARE" if x in rare else x)
    return df

train_fe = smooth_rare_categories(train_fe)
test_fe  = smooth_rare_categories(test_fe)


# Label encoding (for both train+test)
full = pd.concat([train_fe,test_fe], ignore_index=True)
for col in full.columns:
    if full[col].dtype == "object":
        le = LabelEncoder()
        full[col] = le.fit_transform(full[col].astype(str))

train_fe = full.iloc[:len(train_fe)]
test_fe  = full.iloc[len(train_fe):]


# Multi-seed training function
def train_one_seed(seed):
    print(f"\n========== SEED {seed} ==========")
    skf = StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed)

    oof = np.zeros((len(train_fe), 3))
    test_stack = np.zeros((len(test_fe), 3))

    for fold,(tr,va) in enumerate(skf.split(train_fe,y),1):
        print(f"  Fold {fold}/{N_FOLDS}")

        Xtr,Xva = train_fe.iloc[tr], train_fe.iloc[va]
        ytr,yva = y.iloc[tr], y.iloc[va]

        pos = ytr.sum()
        neg = len(ytr)-pos
        spw = neg / max(pos,1)

        # XGBoost 
        xgb = XGBClassifier(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=1,
            tree_method="hist",
            eval_metric="auc",
            scale_pos_weight=spw,
            early_stopping_rounds=40,
            random_state=seed
        )
        xgb.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
        oof[va,0] = xgb.predict_proba(Xva)[:,1]
        test_stack[:,0] += xgb.predict_proba(test_fe)[:,1] / N_FOLDS

        # LightGBM 
        lgb = LGBMClassifier(
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=31,
            min_data_in_leaf=50,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed
        )
        lgb.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc")
        oof[va,1] = lgb.predict_proba(Xva)[:,1]
        test_stack[:,1] += lgb.predict_proba(test_fe)[:,1] / N_FOLDS

        # CatBoost 
        cat = CatBoostClassifier(
            iterations=900,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5,
            bagging_temperature=0.8,
            class_weights=[1, float(spw)],
            eval_metric="AUC",
            loss_function="Logloss",
            random_seed=seed,
            verbose=False
        )
        cat.fit(Xtr,ytr)
        oof[va,2] = cat.predict_proba(Xva)[:,1]
        test_stack[:,2] += cat.predict_proba(test_fe)[:,1] / N_FOLDS

    # Meta Model (GradientBoostingClassifier) + Calibration (Isotonic Regression)
    meta = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=seed
    )
    meta.fit(oof, y)
    meta_val = meta.predict_proba(oof)[:,1]
    auc_raw = roc_auc_score(y, meta_val)

    # Calibrate
    cal = CalibratedClassifierCV(meta, cv="prefit", method="isotonic")
    cal.fit(oof,y)
    meta_val_cal = cal.predict_proba(oof)[:,1]
    auc_cal = roc_auc_score(y, meta_val_cal)

    print(f"[Seed {seed}] Raw AUC = {auc_raw:.6f} | Calibrated AUC = {auc_cal:.6f}")

    return meta_val_cal, cal.predict_proba(test_stack)[:,1]


# Running multi-seed training and blending 
all_oof = []
all_test = []

for seed in RANDOM_SEEDS:
    oof_s, test_s = train_one_seed(seed)
    all_oof.append(oof_s)
    all_test.append(test_s)

oof_blend = np.mean(all_oof, axis=0)
test_blend = np.mean(all_test, axis=0)

final_auc = roc_auc_score(y, oof_blend)
print("\n==================== FINAL STACK AUC ====================")
print(f"Final ROC-AUC (multi-seed, calibrated) = {final_auc:.6f}")


prec, rec, thr = precision_recall_curve(y, oof_blend)

best_thr = 0.5
best_rec = 0

for p, r, t in zip(prec, rec, thr):
    if p >= 0.50 and r > best_rec:
        best_rec = r
        best_thr = t

print("\n===== THRESHOLD TUNING =====")
print(f"Optimal threshold (>=50% precision): {best_thr:.4f}")

pred_class = (oof_blend >= best_thr).astype(int)
print(f"Precision: {precision_score(y,pred_class):.6f}")
print(f"Recall   : {recall_score(y,pred_class):.6f}")
print(f"F1       : {f1_score(y,pred_class):.6f}")


# Save predictions
pd.DataFrame({"Fraud_Probability": test_blend}).to_csv(
    "stacking_predictions_highauc_v2.csv", index=False
)

print("\nSaved: stacking_predictions_highauc_v2.csv")
print("Done.")
