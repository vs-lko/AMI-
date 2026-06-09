import pickle
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


MODEL_PATH = "/content/xgb_static_ami_model.pkl"
FEATURES_PATH = "/content/patient_static_features.parquet"

with open(MODEL_PATH, "rb") as f:
    obj = pickle.load(f)

model = obj["model"]
scaler = obj["scaler"]
feature_names = obj["features"]

print("Loaded model with features:", feature_names)


df = pd.read_parquet("/content/patient_static_features.parquet")

print(df.columns)  # sanity check


# Load diagnoses
diags = pd.read_csv("/content/diagnoses_icd.csv", low_memory=False)

def detect_ami_codes(s):
    if pd.isna(s):
        return False
    s = str(s).upper()
    if s.startswith("410") or s.startswith("I21"):
        return True
    if "AMI" in s or "MYOCARDIAL INFARCTION" in s:
        return True
    return False

# Detect patient id column
pcol = None
for c in ["subject_id", "patient_id", "hadm_id"]:
    if c in diags.columns:
        pcol = c
        break

# Detect ICD column
codecol = None
for c in ["icd_code", "code", "diagnosis", "icd9_code"]:
    if c in diags.columns:
        codecol = c
        break

diags = diags.rename(columns={pcol: "patient_id", codecol: "icd_code"})
diags["is_ami"] = diags["icd_code"].apply(detect_ami_codes)

ami_patients = set(diags.loc[diags["is_ami"], "patient_id"].unique())

# Add label to feature table
df["is_ami"] = df["patient_id"].apply(lambda x: 1 if x in ami_patients else 0)

print("AMI positives:", df["is_ami"].sum())


X = df[feature_names].values
y = df["is_ami"].astype(int).values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_test = scaler.transform(X_test)
X_test_df = pd.DataFrame(X_test, columns=feature_names)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df)

shap.summary_plot(
    shap_values,
    X_test_df,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("/content/shap_global_importance.png", dpi=150)
plt.close()


shap.summary_plot(
    shap_values,
    X_test_df,
    show=False
)
plt.tight_layout()
plt.savefig("/content/shap_summary.png", dpi=150)
plt.close()


ami_idx = np.where(y_test == 1)[0][0]

shap.force_plot(
    explainer.expected_value,
    shap_values[ami_idx],
    X_test_df.iloc[ami_idx],
    matplotlib=True,
    show=False
)
plt.savefig("/content/shap_local_ami.png", dpi=150)
plt.close()
