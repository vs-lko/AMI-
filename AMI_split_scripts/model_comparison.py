#!/usr/bin/env python3
"""
Model Comparison Harness for AMI Prediction
-------------------------------------------
- Runs the same data pipeline as before (Steps 1-6)
- Trains and evaluates multiple classifiers:
  - LogisticRegression
  - GaussianNB
  - RandomForest
  - AdaBoost
  - LightGBM
  - XGBoost (as baseline)
- Saves a separate model file (.pkl) and PR curve (.png) for each model.
"""

import os
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

# --- SKLearn Models ---
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)

# --- Boosting Models ---
from xgboost import XGBClassifier
import lightgbm as lgb

# ---------------- PATHS ----------------
LABS_FILE = "/content/relevant_labs.csv"
DIAG_FILE = "/content/diagnoses_icd.csv"
OUT_DIR = "/content/model_comparison" # Made a new dir for outputs
CHUNKSIZE = 200000

# ---------------- CONFIG ----------------
PREFERRED_BIOMARKERS = [
    "hs-cTn I", "CK-MB", "CRP", "IL-6", "TNF-α",
    "MPO", "PTX-3", "suPAR", "NGAL", "NLR", "MPV", "PLR"
]
THRESHOLDS = [0.5, 0.4, 0.3, 0.2]

# ---------------- HELPERS ----------------
# (Helper functions are identical to your script)
def detect_ami_codes(s):
    if pd.isna(s):
        return False
    s = str(s).upper()
    if s.startswith("410") or s.startswith("I21"):
        return True
    if "AMI" in s or "MYOCARDIAL INFARCTION" in s:
        return True
    return False

def pick_col(cols, candidates):
    cols_lower = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols[cols_lower.index(cand.lower())]
    return None

# ---------------- PIPELINE ----------------
def run_model_comparison():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Steps 1-6: Data Preparation (Identical to your script) ---

    # Step 1: detect key columns
    sample = pd.read_csv(LABS_FILE, nrows=5000, low_memory=False)
    pcol = pick_col(sample.columns, ["subject_id", "patient_id", "hadm_id"])
    labcol = pick_col(sample.columns, ["label", "lab_name", "test_name", "component", "itemid"])
    valcol = pick_col(sample.columns, ["valuenum", "value", "result", "value_num", "test_value"])
    if not (pcol and labcol and valcol):
        raise ValueError(f"Column detection failed. Found columns: {sample.columns.tolist()}")
    print(f"[Detected columns] patient: {pcol} | lab: {labcol} | value: {valcol}")

    # Step 2: chunked aggregation
    agg = defaultdict(lambda: [0.0, 0])
    reader = pd.read_csv(LABS_FILE, chunksize=CHUNKSIZE, low_memory=True)
    for i, chunk in enumerate(reader, 1):
        df = chunk[[pcol, labcol, valcol]].rename(columns={pcol: "patient_id", labcol: "lab_name", valcol: "value"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["patient_id", "lab_name"])
        grp = df.groupby(["patient_id", "lab_name"])["value"].agg(["sum", "count"]).reset_index()
        for _, r in grp.iterrows():
            key = (r["patient_id"], r["lab_name"])
            agg[key][0] += float(r["sum"])
            agg[key][1] += int(r["count"])
        if i % 5 == 0:
            print(f"  processed {i} chunks...")

    print(f"[Aggregation done] Unique patient-lab pairs: {len(agg):,}")

    # Step 3: per-patient means
    rows = [(pid, lab, s / c if c > 0 else np.nan) for (pid, lab), (s, c) in agg.items()]
    feat_df = pd.DataFrame(rows, columns=["patient_id", "lab_name", "mean_value"])
    pivot = feat_df.pivot(index="patient_id", columns="lab_name", values="mean_value")
    pivot_filled = pivot.fillna(pivot.median())
    features_file = os.path.join(OUT_DIR, "patient_static_features.parquet")
    pivot_filled.reset_index().to_parquet(features_file)
    print(f"[Saved static features] {features_file}")

    # Step 4: load diagnoses for AMI labels
    if not os.path.exists(DIAG_FILE):
        print("No diagnoses file found, skipping model training.")
        return

    diags = pd.read_csv(DIAG_FILE, low_memory=False)
    pcol_diag = pick_col(diags.columns, ["subject_id", "patient_id", "hadm_id"])
    codecol = pick_col(diags.columns, ["icd_code", "code", "diagnosis", "icd9_code"])
    if pcol_diag is None or codecol is None:
        print("diagnoses_icd.csv column detection failed, skipping training.")
        return

    diags = diags.rename(columns={pcol_diag: "patient_id", codecol: "icd_code"})
    diags["is_ami"] = diags["icd_code"].apply(detect_ami_codes)
    ami_patients = set(diags.loc[diags["is_ami"], "patient_id"].unique())
    pivot_filled["is_ami"] = pivot_filled.index.to_series().apply(lambda pid: 1 if pid in ami_patients else 0).values
    print(f"[Labels created] Positive cases: {pivot_filled['is_ami'].sum()} / {pivot_filled.shape[0]}")

    # Step 5: select biomarkers
    available = list(pivot_filled.columns)
    used_biomarkers = [b for b in PREFERRED_BIOMARKERS if b in available]
    if len(used_biomarkers) == 0:
        print("Preferred biomarkers not found, using top 20 by RandomForest importance.")
        rf_feat = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        X_tmp = pivot_filled.drop(columns=["is_ami"]).fillna(0).values
        y_tmp = pivot_filled["is_ami"].astype(int).values
        rf_feat.fit(X_tmp, y_tmp)
        idx = np.argsort(rf_feat.feature_importances_)[::-1][:20]
        used_biomarkers = [pivot_filled.columns[i] for i in idx]
        print(f"Top 20 biomarkers: {used_biomarkers}")
    else:
        print(f"Using preferred biomarkers: {used_biomarkers}")

    # Step 6: prepare data
    X = pivot_filled[used_biomarkers].values
    y = pivot_filled["is_ami"].astype(int).values

    if len(np.unique(y)) < 2:
        print("Only one class present in labels — cannot train. Exiting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nData preparation complete. Starting model comparison...")

    # --- NEW: Step 7: Define All Models ---

    # Calculate imbalance weight once
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"Calculated scale_pos_weight for balancing: {scale_pos_weight:.2f}")

    models_to_test = {
        "LogisticRegression": LogisticRegression(
            random_state=42,
            class_weight='balanced', # Use built-in balancing
            max_iter=1000,
            n_jobs=-1
        ),

        "GaussianNB": GaussianNB(),
        # Note: GaussianNB assumes features are normally distributed
        # and doesn't have a simple class_weight parameter.
        # Its performance might be poor on imbalanced, non-Gaussian data.

        "RandomForest": RandomForestClassifier(
            random_state=42,
            class_weight='balanced', # Use built-in balancing
            n_jobs=-1,
            n_estimators=200 # Match n_estimators from your feature selector
        ),

        "AdaBoost": AdaBoostClassifier(
            random_state=42,
            n_estimators=300 # Match n_estimators from your XGB
        ),

        "LightGBM": lgb.LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            n_estimators=300,
            scale_pos_weight=scale_pos_weight # Use same balancing as XGB
        ),

        "XGBoost (Baseline)": XGBClassifier( # Your original model
            n_estimators=300,
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )
    }

    # --- NEW: Steps 8 & 9: Loop, Train, Evaluate, and Save ---

    for model_name, model in models_to_test.items():
        print(f"\n{'='*20} TRAINING: {model_name} {'='*20}")

        # --- Train ---
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Failed to train {model_name}. Error: {e}")
            continue

        # --- Evaluate ---
        # Get prediction probabilities for PR/ROC curves
        if not hasattr(model, "predict_proba"):
            print(f"Model {model_name} does not support 'predict_proba'. Skipping.")
            continue

        preds = model.predict_proba(X_test)[:, 1]

        # --- Print Reports ---
        print(f"\nAUC: {roc_auc_score(y_test, preds):.4f}")
        print(f"Accuracy (thr=0.5): {accuracy_score(y_test, (preds > 0.5).astype(int)):.4f}")
        for thr in THRESHOLDS:
            print(f"\n--- Threshold {thr} (for {model_name}) ---")
            print(classification_report(y_test, (preds > thr).astype(int), digits=4))

        # --- PR Curve ---
        precision, recall, _ = precision_recall_curve(y_test, preds)
        ap = average_precision_score(y_test, preds)
        print(f"\nAverage Precision (AP) for {model_name}: {ap:.4f}")

        plt.figure(figsize=(6, 6))
        plt.step(recall, precision, where='post', label=f'AP = {ap:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR curve - {model_name}')
        plt.legend()
        plt.grid(True)
        pr_path = os.path.join(OUT_DIR, f"pr_curve_{model_name.replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(pr_path, dpi=150)
        plt.close()
        print(f"Saved PR curve to: {pr_path}")

        # --- Save Model ---
        out_path = os.path.join(OUT_DIR, f"model_{model_name.replace(' ', '_')}.pkl")
        with open(out_path, "wb") as f:
            # Save the trained model, the scaler, and the feature list
            pickle.dump({
                "model": model,
                "scaler": scaler,
                "features": used_biomarkers
            }, f)
        print(f"[Model saved] {out_path}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    run_model_comparison()