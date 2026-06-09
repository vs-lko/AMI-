import pandas as pd
from pathlib import Path

# ---------- FILE PATHS ----------
relevant_labs_path = "/content/relevant_labs.csv"
diagnoses_icd_path = "/content/diagnoses_icd.csv"
dataset_xlsx_path = "/content/dataset.xlsx"
output_path = "/content/patient_level_from_labs.csv"

# ---------- STEP 1: Identify AMI subjects ----------
diag = pd.read_csv(diagnoses_icd_path, usecols=["subject_id", "icd_code"], dtype={"icd_code": str})
ami_mask = diag["icd_code"].str.startswith(("I21", "I22", "410"), na=False)
ami_subjects = set(diag.loc[ami_mask, "subject_id"].unique())

# ---------- STEP 2: Chunked aggregation ----------
chunksize = 200000
usecols = ["subject_id", "label", "valuenum"]
partial_aggs = []

for chunk in pd.read_csv(relevant_labs_path, usecols=usecols, chunksize=chunksize):
    chunk = chunk.dropna(subset=["valuenum"])
    chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
    chunk = chunk.dropna(subset=["valuenum"])
    if chunk.empty:
        continue
    g = chunk.groupby(["subject_id", "label"])["valuenum"].agg(["sum", "count"]).reset_index()
    partial_aggs.append(g)

if not partial_aggs:
    raise RuntimeError("No numeric lab values found after dropping NaNs.")

combined = pd.concat(partial_aggs, ignore_index=True)
combined = combined.groupby(["subject_id", "label"], as_index=False).sum()
combined["mean"] = combined["sum"] / combined["count"]

# ---------- STEP 3: Pivot to patient-level ----------
patient_labs = combined.pivot(index="subject_id", columns="label", values="mean").reset_index()
patient_labs["is_ami"] = patient_labs["subject_id"].apply(lambda x: 1 if x in ami_subjects else 0)

# ---------- STEP 4: Align with dataset.xlsx columns ----------
ds_cols = pd.read_excel(dataset_xlsx_path, nrows=0).columns.tolist()

def normalize(s):
    return str(s).lower().replace(" ", "").replace("(", "").replace(")", "").replace(":", "").replace("-", "").replace("/", "")

norm_target = {normalize(c): c for c in ds_cols}
col_mapping = {}

# direct name match
for col in patient_labs.columns:
    nc = normalize(col)
    if nc in norm_target:
        col_mapping[col] = norm_target[nc]

# heuristic mapping for common labs
heuristic_map = {
    "wbc": ["wbc","whiteblood"],
    "rbc": ["rbc","redblood"],
    "hgb": ["hgb","hemoglobin","haemoglobin"],
    "mpv": ["mpv","meanplatelet"],
    "plt": ["plt","platelet"],
    "ne": ["neut","neu","neutrophil"],
    "ly": ["lymph","ly"],
    "hct": ["hct","hematocrit","haematocrit"],
    "mcv": ["mcv"],
    "rdw-sd": ["rdw-sd","rdwsd"],
    "rdw-cv": ["rdw-cv","rdwcv"],
    "mch": ["mch"],
    "mchc": ["mchc"],
    "pdw": ["pdw"],
    "pct": ["pct"]
}
for col in patient_labs.columns:
    if col in col_mapping:
        continue
    nc = normalize(col)
    for target_norm, variants in heuristic_map.items():
        if any(v in nc for v in variants) and target_norm in norm_target:
            col_mapping[col] = norm_target[target_norm]
            break

# ---------- STEP 5: Build final aligned DataFrame ----------
final_cols = ["subject_id"] + ds_cols + ["is_ami"]
final_df = pd.DataFrame(patient_labs["subject_id"])

for src, tgt in col_mapping.items():
    if tgt in final_df.columns:
        continue
    final_df[tgt] = patient_labs[src]

for c in ds_cols:
    if c not in final_df.columns:
        final_df[c] = pd.NA

final_df["is_ami"] = patient_labs["is_ami"].values

# ---------- STEP 6: Save result ----------
final_df.to_csv(output_path, index=False)

print(f"✅ Saved: {output_path}")
print(f"Subjects processed: {final_df.shape[0]}")
print(f"Columns from dataset.xlsx: {len(ds_cols)}")
print(f"Mapped columns: {len(col_mapping)}")
print("Sample column mappings:")
for k, v in list(col_mapping.items())[:15]:
    print(f"  {k} → {v}")
