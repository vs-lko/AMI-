# Final XGBoost pipeline — Colab-ready
# - Robust xgboost import (auto-install)
# - XGBoost 3.x-compatible training (eval_metric in constructor, callbacks for early stopping)
# - Time-aware feature engineering from /content/relevant_labs.csv
# - Saves model + scaler/features to /content

import warnings, os, pickle, sys, subprocess, traceback
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression

# ---------- Paths (Colab) ----------
LABS_CSV = "/content/relevant_labs.csv"
DIAG_CSV = "/content/diagnoses_icd.csv"
MODEL_OUT = "/content/xgb_ami_model_fromscratch.pkl"
SCALER_OUT = "/content/xgb_scaler_and_features_fromscratch.pkl"

# ---------- Preferred biomarkers (will fallback to top-by-presence if not found) ----------
PREFERRED_BIOMARKERS = [
    "hs-cTn I","CK-MB","CRP","IL-6","TNF-α",
    "MPO","PTX-3","suPAR","NGAL","NLR","MPV","PLR"
]

# ---------- Helper functions ----------
def detect_ami_codes(s):
    if pd.isna(s):
        return False
    s = str(s).upper()
    if s.startswith("410") or s.startswith("I21"):
        return True
    if "AMI" in s or "ACUTE MYOCARDIAL" in s or "MYOCARDIAL INFARCTION" in s:
        return True
    return False

def slope_of_series(times, values):
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan
    X = times[mask].reshape(-1,1)
    y = values[mask]
    lr = LinearRegression()
    lr.fit(X,y)
    return float(lr.coef_[0])

def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    for c in cols:
        if c.lower() in [x.lower() for x in candidates]:
            return c
    return None

# ---------- Robust xgboost import/install ----------
def ensure_xgboost_installed(install_if_missing=True):
    try:
        import xgboost as xgb
        print("xgboost import OK — version:", getattr(xgb, "__version__", "unknown"))
        return xgb
    except Exception as first_err:
        print("Initial import of xgboost failed:", str(first_err))
        if not install_if_missing:
            raise
        print("Attempting to install/upgrade xgboost into the current interpreter...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "xgboost"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import importlib
            xgb = importlib.import_module("xgboost")
            print("xgboost imported after install — version:", getattr(xgb, "__version__", "unknown"))
            return xgb
        except Exception as second_err:
            print("xgboost import failed even after attempting installation.")
            print("=== First import exception ===")
            traceback.print_exception(type(first_err), first_err, first_err.__traceback__)
            print("\n=== Second import/install exception ===")
            traceback.print_exception(type(second_err), second_err, second_err.__traceback__)
            raise RuntimeError("xgboost import/install failed. Try restarting the runtime and re-running the notebook. See tracebacks above.") from second_err

xgb = ensure_xgboost_installed(install_if_missing=True)

# ---------- Load and normalize input CSVs ----------
print("Reading labs CSV (may take a moment)...")
sample = pd.read_csv(LABS_CSV, nrows=200)
cols = sample.columns.tolist()

patient_col = find_col(cols, ["subject_id","patient_id","hadm_id"])
time_col = find_col(cols, ["charttime","time","datetime","date"])
itemid_col = find_col(cols, ["itemid","label","lab_name","test_name","component"])
value_col = find_col(cols, ["valuenum","value_num","value","result","lab_value"])
ref_low_col = find_col(cols, ["ref_range_lower","ref_low","low"])
ref_high_col = find_col(cols, ["ref_range_upper","ref_high","high"])
flag_col = find_col(cols, ["flag","abnormal"])

usecols = [c for c in [patient_col, time_col, itemid_col, value_col, ref_low_col, ref_high_col, flag_col] if c is not None]
labs = pd.read_csv(LABS_CSV, usecols=usecols, parse_dates=[time_col] if time_col else None)

# Rename consistent columns
labs = labs.rename(columns={patient_col:"subject_id", time_col:"charttime", itemid_col:"label", value_col:"valuenum"})
if ref_low_col: labs = labs.rename(columns={ref_low_col:"ref_low"})
if ref_high_col: labs = labs.rename(columns={ref_high_col:"ref_high"})
if flag_col: labs = labs.rename(columns={flag_col:"flag"})

labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")

# Load diagnoses and detect AMI patients
diags = pd.read_csv(DIAG_CSV)
diag_patient = find_col(diags.columns.tolist(), ["subject_id","patient_id","hadm_id"])
diag_code = find_col(diags.columns.tolist(), ["icd_code","code","diagnosis"])
if diag_patient is None or diag_code is None:
    diag_patient = diag_patient or diags.columns[0]
    diag_code = diag_code or diags.columns[1] if len(diags.columns)>1 else diags.columns[0]
diags = diags.rename(columns={diag_patient:"subject_id", diag_code:"icd_code"})
diags["is_ami"] = diags["icd_code"].apply(detect_ami_codes)
ami_patients = set(diags.loc[diags["is_ami"], "subject_id"].unique())

# ---------- Choose biomarkers (preferred or top-by-presence) ----------
available = labs["label"].dropna().unique().tolist()
use_biomarkers = []
for b in PREFERRED_BIOMARKERS:
    for a in available:
        if str(a).strip().lower() == str(b).strip().lower():
            use_biomarkers.append(a)
            break

if len(use_biomarkers) == 0:
    print("Preferred biomarkers not found. Selecting top 20 by presence.")
    last_vals = labs.sort_values(["subject_id","charttime"]).groupby(["subject_id","label"])["valuenum"].last().reset_index()
    pivot = last_vals.pivot(index="subject_id", columns="label", values="valuenum")
    top = pivot.notna().sum().sort_values(ascending=False).index[:20].tolist()
    use_biomarkers = top
print("Using biomarkers:", use_biomarkers)

# ---------- Feature engineering (time-aware) ----------
print("Starting feature engineering (this may take several minutes on large files)...")
global_max_time = labs["charttime"].max()
rows = []
subjects = []
grouped = labs.groupby("subject_id")

for i, (pid, g) in enumerate(grouped):
    if i % 500 == 0 and i > 0:
        print(f"Processed {i} patients...")
    subjects.append(pid)
    feat = {}
    g = g.sort_values("charttime")
    feat["num_tests_total"] = len(g)
    feat["num_distinct_tests"] = g["label"].nunique()
    if g["charttime"].notna().sum() >= 2:
        feat["overall_time_span_hours"] = (g["charttime"].max() - g["charttime"].min()).total_seconds()/3600.0
    else:
        feat["overall_time_span_hours"] = 0.0

    for biom in use_biomarkers:
        sub = g[g["label"]==biom].dropna(subset=["charttime"])
        vals = sub["valuenum"].values.astype(float) if len(sub)>0 else np.array([])
        times = sub["charttime"].astype('datetime64[ns]').astype('int64') // 10**9 if len(sub)>0 else np.array([])
        times_h = (times - times.min())/3600.0 if len(times)>0 else np.array([])
        pfx = str(biom).replace(" ","_").replace("/","_").replace("-","_")
        feat[f"{pfx}__count"] = len(vals)
        feat[f"{pfx}__last_value"] = float(vals[-1]) if len(vals)>0 else np.nan
        feat[f"{pfx}__mean"] = float(np.nanmean(vals)) if len(vals)>0 else np.nan
        feat[f"{pfx}__std"] = float(np.nanstd(vals)) if len(vals)>0 else np.nan
        feat[f"{pfx}__min"] = float(np.nanmin(vals)) if len(vals)>0 else np.nan
        feat[f"{pfx}__max"] = float(np.nanmax(vals)) if len(vals)>0 else np.nan
        feat[f"{pfx}__median"] = float(np.nanmedian(vals)) if len(vals)>0 else np.nan
        if len(sub)>0:
            last_time = sub["charttime"].max()
            feat[f"{pfx}__last_time_recency_hours"] = (global_max_time - last_time).total_seconds()/3600.0
            feat[f"{pfx}__time_span_hours"] = (sub["charttime"].max() - sub["charttime"].min()).total_seconds()/3600.0 if sub["charttime"].notna().sum()>1 else 0.0
        else:
            feat[f"{pfx}__last_time_recency_hours"] = np.nan
            feat[f"{pfx}__time_span_hours"] = np.nan
        feat[f"{pfx}__slope_per_hour"] = slope_of_series(np.array(times_h), np.array(vals)) if len(vals)>1 else np.nan
        try:
            feat[f"{pfx}__ewma"] = float(pd.Series(vals).ewm(span=3, adjust=False).mean().iloc[-1]) if len(vals)>0 else np.nan
        except:
            feat[f"{pfx}__ewma"] = np.nan
        if "ref_low" in sub.columns and "ref_high" in sub.columns:
            valid = (~sub["ref_low"].isna()) & (~sub["ref_high"].isna()) & (~sub["valuenum"].isna())
            if valid.sum()>0:
                is_ab = ((sub.loc[valid,"valuenum"] < sub.loc[valid,"ref_low"]) | (sub.loc[valid,"valuenum"] > sub.loc[valid,"ref_high"])).sum()/float(valid.sum())
                feat[f"{pfx}__fraction_abnormal"] = float(is_ab)
            else:
                feat[f"{pfx}__fraction_abnormal"] = np.nan
        else:
            feat[f"{pfx}__fraction_abnormal"] = np.nan

    rows.append(feat)

# ---------- Build DataFrame ----------
X = pd.DataFrame(rows, index=subjects).reset_index().rename(columns={"index":"subject_id"})
for c in X.columns:
    if X[c].dtype == object:
        X[c] = pd.to_numeric(X[c], errors='coerce')
X = X.fillna(X.median())
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
y = X["subject_id"].apply(lambda s: 1 if s in ami_patients else 0).values

print("Final feature matrix:", X[num_cols].shape, "Positive rate:", y.mean())

# ---------- Train/test split + scale ----------
X_train, X_test, y_train, y_test = train_test_split(X[num_cols].values, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_pos, n_neg = (y_train==1).sum(), (y_train==0).sum()
scale_pos_weight = n_neg / max(1, n_pos)
print("Class balance:", n_pos, "AMI vs", n_neg, "non-AMI, scale_pos_weight =", scale_pos_weight)

# ---------- Train XGBoost (XGBoost 3.x-compatible) ----------
print("Training XGBoost model now...")

# Use callback-based early stopping
# ***FIX: Define the callback *before* initializing the classifier***
early_stop = xgb.callback.EarlyStopping(rounds=20, save_best=True)

clf = xgb.XGBClassifier(
    n_estimators=300,
    random_state=42,
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    n_jobs=4,
    eval_metric="auc",         # set in constructor
    callbacks=[early_stop]   # ***FIX: Pass callbacks to the constructor***
)

# ***FIX: Remove 'callbacks' from the .fit() method***
clf.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=True
)

model = clf
model_name = "xgboost"

# ---------- Evaluate ----------
probs = model.predict_proba(X_test_scaled)[:,1]
auc = roc_auc_score(y_test, probs)
print(f"\nModel: {model_name}  |  AUC: {auc:.4f}")
print("Accuracy (thr=0.5):", accuracy_score(y_test, (probs>0.5).astype(int)))

for thr in [0.5, 0.4, 0.3, 0.2]:
    print(f"\n--- Threshold {thr} ---")
    print(classification_report(y_test, (probs>thr).astype(int), digits=4))

# ---------- Save artifacts ----------
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)
with open(SCALER_OUT, "wb") as f:
    pickle.dump({"scaler":scaler, "feature_names":num_cols, "used_biomarkers":use_biomarkers}, f)

print("\nSaved model to:", MODEL_OUT)
print("Saved scaler/features to:", SCALER_OUT)