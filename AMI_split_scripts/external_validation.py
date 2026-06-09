# Full training + external validation script with all-NaN column handling
import os, pandas as pd, numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report

# paths
patient_csv = "/content/patient_level_from_labs.csv"
dataset_xlsx = "/content/dataset.xlsx"

# check files
for p in (patient_csv, dataset_xlsx):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing file: {p}")

# load
patient_df = pd.read_csv(patient_csv)
ds = pd.read_excel(dataset_xlsx)

# normalize column names (trim spaces)
patient_df.columns = [c.strip() for c in patient_df.columns]
ds.columns = [c.strip() for c in ds.columns]

# find dataset target column (best-effort)
target_ds = None
for c in ds.columns:
    if 'group' in c.lower() and ('ami' in c.lower() or 'control' in c.lower()):
        target_ds = c
        break
if target_ds is None:
    target_ds = 'group  (control:0, AMI:1)' if 'group  (control:0, AMI:1)' in ds.columns else None
if target_ds is None:
    raise RuntimeError("Couldn't find target column in dataset.xlsx. Expected 'group  (control:0, AMI:1)' or similar.")

# determine features (exclude ids and targets)
exclude = {'subject_id','is_ami', target_ds}
patient_feats = [c for c in patient_df.columns if c not in exclude]
ds_feats = [c for c in ds.columns if c not in exclude]
common = [c for c in patient_feats if c in ds_feats]
if not common:
    raise RuntimeError("No overlapping feature columns between the two datasets. Check column names.")

# prepare X and X_ext
X = patient_df[common].copy()
y = patient_df['is_ami'].copy()
X_ext = ds[common].copy()
y_ext = ds[target_ds].copy()

# === NEW BLOCK: convert to numeric, drop all-NaN cols in training, then impute ===
for c in common:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X_ext[c] = pd.to_numeric(X_ext[c], errors="coerce")

cols_allnan_train = [c for c in common if X[c].isna().all()]
cols_allnan_ext = [c for c in common if X_ext[c].isna().all()]

to_drop = set(cols_allnan_train)  # at minimum drop columns with no values in training
if to_drop:
    print("Dropping all-NaN columns (train):", sorted(to_drop))

common_reduced = [c for c in common if c not in to_drop]
if not common_reduced:
    raise RuntimeError("No usable features remain after dropping all-NaN columns. Inspect data.")

imp = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imp.fit_transform(X[common_reduced]), columns=common_reduced, index=X.index)
X_ext_imp = pd.DataFrame(imp.transform(X_ext[common_reduced]), columns=common_reduced, index=X_ext.index)
# === END NEW BLOCK ===

# train/val split
X_tr, X_val, y_tr, y_val = train_test_split(X_imp, y, stratify=y, test_size=0.2, random_state=42)

# model (try xgboost, fallback to RandomForest)
try:
    import xgboost as xgb
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=4)
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=4)

model.fit(X_tr, y_tr)

# eval helper
def eval_model(m, Xs, ys):
    probs = m.predict_proba(Xs)[:,1] if hasattr(m, "predict_proba") else m.predict(Xs)
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(ys, probs) if len(np.unique(ys))>1 else float('nan')
    return {
        "auc": auc,
        "accuracy": accuracy_score(ys, preds),
        "f1": f1_score(ys, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(ys, preds),
        "report": classification_report(ys, preds, zero_division=0)
    }

res_tr = eval_model(model, X_tr, y_tr)
res_val = eval_model(model, X_val, y_val)

# external evaluation — only rows with non-missing external target
valid_ext_idx = y_ext.dropna().index
if len(valid_ext_idx) == 0:
    raise RuntimeError("No labeled rows in external dataset to evaluate on.")
res_ext = eval_model(model, X_ext_imp.loc[valid_ext_idx], y_ext.loc[valid_ext_idx].astype(int))

# summary
print("\nCommon features originally:", len(common))
print("Features used after dropping all-NaN in train:", len(common_reduced))
print("Features dropped (train all-NaN):", sorted(cols_allnan_train))
print("\n=== Metrics ===")
print("Train  AUC / Acc / F1:", res_tr["auc"], res_tr["accuracy"], res_tr["f1"])
print("Val    AUC / Acc / F1:", res_val["auc"], res_val["accuracy"], res_val["f1"])
print("External AUC / Acc / F1:", res_ext["auc"], res_ext["accuracy"], res_ext["f1"])
print("\nExternal classification report:\n", res_ext["report"])
print("External confusion matrix:\n", res_ext["confusion_matrix"])

# save model + used feature list
import joblib
joblib.dump(model, "/content/xgb_model_patient_level.pkl")
pd.Series(common_reduced).to_csv("/content/used_features.csv", index=False)
print("\nSaved model -> /content/xgb_model_patient_level.pkl")
print("Saved used features -> /content/used_features.csv")
