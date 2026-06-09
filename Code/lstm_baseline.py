
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import pickle

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception as e:
    tf = None
    print("TensorFlow not available. Install with: pip install tensorflow")

# --- Parameters ---
LABS_CSV = "relevant_labs.csv"
DIAG_CSV = "diagnoses_icd.csv"
MODEL_OUT = "lstm_ami_model.h5"
SCALER_OUT = "scaler_and_weights.pkl"
N_TIMESTEPS = 10

# Preferred biomarkers list
PREFERRED_BIOMARKERS = [
    "hs-cTn I","CK-MB","CRP","IL-6","TNF-α",
    "MPO","PTX-3","suPAR","NGAL","NLR","MPV","PLR"
]

# --- Helpers ---
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        if col.lower() in [x.lower() for x in candidates]:
            return col
    return None

def detect_ami_codes(s):
    if pd.isna(s):
        return False
    s = str(s).upper()
    if s.startswith("410") or s.startswith("I21"):
        return True
    if "AMI" in s or "ACUTE MYOCARDIAL" in s or "MYOCARDIAL INFARCTION" in s:
        return True
    return False

# --- Load data ---
labs = pd.read_csv(LABS_CSV)
diags = pd.read_csv(DIAG_CSV)

# Infer columns
patient_col = find_col(labs,["patient_id","subject_id","hadm_id"])
time_col_lab = find_col(labs,["charttime","time","datetime","date"])
lab_name_col = find_col(labs,["lab_name","itemid","test_name","component","test"])
lab_value_col = find_col(labs,["value","lab_value","value_num","test_value","result"])

labs = labs.rename(columns={patient_col:"patient_id", lab_name_col:"lab_name", lab_value_col:"value"})
if time_col_lab: labs = labs.rename(columns={time_col_lab:"charttime"})
labs["value"] = pd.to_numeric(labs["value"], errors="coerce")

patient_col_diag = find_col(diags,["patient_id","subject_id","hadm_id"])
diag_code_col = find_col(diags,["icd_code","code","diagnosis"])
diags = diags.rename(columns={patient_col_diag:"patient_id", diag_code_col:"icd_code"})

# Labels
diags["is_ami"] = diags["icd_code"].apply(detect_ami_codes)
ami_patients = set(diags.loc[diags["is_ami"],"patient_id"].unique())

# --- Biomarker set ---
all_biomarkers = labs["lab_name"].unique().tolist()
use_biomarkers = [b for b in PREFERRED_BIOMARKERS if b in all_biomarkers]

if len(use_biomarkers) == 0:
    print("Preferred biomarkers not found. Selecting top 20 features from RandomForest...")
    labs_sorted = labs.sort_values(["patient_id","charttime"])
    last_vals = labs_sorted.groupby(["patient_id","lab_name"]) ["value"].last().reset_index()
    pivot = last_vals.pivot(index="patient_id", columns="lab_name", values="value")
    pivot = pivot.fillna(pivot.median())
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(pivot.values, [1 if pid in ami_patients else 0 for pid in pivot.index])
    top_features = rf.feature_importances_.argsort()[::-1]
    use_biomarkers = pivot.columns[top_features[:20]].tolist()
    print("Using top RF features:", use_biomarkers)
else:
    print("Using preferred biomarkers:", use_biomarkers)

# --- Build patient sequences with consistent feature set ---
X_seqs, y_labels = [], []

for pid, group in labs.groupby("patient_id"):
    group = group.sort_values("charttime")
    pivoted = group.pivot_table(index="charttime", columns="lab_name", values="value")
    for b in use_biomarkers:
        if b not in pivoted.columns:
            pivoted[b] = np.nan
    pivoted = pivoted[use_biomarkers]

    seq = pivoted.tail(N_TIMESTEPS).values
    if seq.shape[0] < N_TIMESTEPS:
        pad = np.full((N_TIMESTEPS - seq.shape[0], len(use_biomarkers)), np.nan)
        seq = np.vstack([pad, seq])

    X_seqs.append(seq)
    y_labels.append(1 if pid in ami_patients else 0)

X = np.array(X_seqs)
y = np.array(y_labels)

# Impute missing values with median per feature
for f in range(X.shape[2]):
    col = X[:,:,f].flatten()
    median = np.nanmedian(col)
    X[:,:,f] = np.where(np.isnan(X[:,:,f]), median, X[:,:,f])

print("Final data shape:", X.shape)

# --- Train/test split and scaling ---
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
ns,nt,nf = X_train.shape

scaler = StandardScaler()
X_train_2d = X_train.reshape((ns*nt,nf))
X_test_2d = X_test.reshape((X_test.shape[0]*X_test.shape[1],nf))
scaler.fit(X_train_2d)
X_train_scaled = scaler.transform(X_train_2d).reshape((ns,nt,nf))
X_test_scaled = scaler.transform(X_test_2d).reshape((X_test.shape[0],nt,nf))

# --- LSTM model ---
if tf is None:
    raise ImportError("TensorFlow required.")

from tensorflow.keras import backend as K
K.clear_session()
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(nt,nf)))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

es = EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)
model.fit(X_train_scaled,y_train,validation_split=0.2,epochs=30,batch_size=32,callbacks=[es],verbose=1)

preds = model.predict(X_test_scaled).ravel()
print('AUC:', roc_auc_score(y_test, preds))
print('Accuracy:', accuracy_score(y_test, (preds>0.5).astype(int)))
print(classification_report(y_test,(preds>0.5).astype(int)))

model.save(MODEL_OUT)
with open(SCALER_OUT,'wb') as f:
    pickle.dump({'scaler':scaler,'biomarkers':use_biomarkers},f)

print('Saved model to', MODEL_OUT)
print('Saved scaler/weights to', SCALER_OUT)