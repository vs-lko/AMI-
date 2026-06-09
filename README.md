Stratification of Acute Myocardial Infarction (AMI) from non-AMI in admitted patients using static and time-dependent Cardiac and Hematological biomarkers applying Explainable Artificial Intelligence (AI)

Overview

This repository contains the code, processed datasets, and experimental workflow developed for the study:

Prediction of Acute Myocardial Infarction (AMI) using Static and Time-Dependent Cardiac and Hematological Biomarkers Applying Explainable AI

The objective of this work is to investigate whether temporal and non-temporal machine learning models can predict Acute Myocardial Infarction (AMI) using routinely collected laboratory biomarkers from electronic health records.

Two modelling strategies were evaluated:

Temporal model: Long Short-Term Memory (LSTM)
Non-temporal models: XGBoost, Random Forest, LightGBM, AdaBoost, Logistic Regression, and Gaussian Naïve Bayes

Model interpretability was performed using SHAP (SHapley Additive Explanations).

Dataset

The study was conducted using the MIMIC-IV (v3.1) database developed by Beth Israel Deaconess Medical Center and hosted on PhysioNet.

Source

Johnson et al., MIMIC-IV: A Freely Accessible Electronic Health Record Dataset.

The dataset contains:

Adult ICU admissions
Laboratory measurements
Diagnosis records
Demographic information
Clinical outcomes
AMI Identification

AMI patients were identified using:

ICD-9: 410.x
ICD-10: I21.x
ICD-10: I22.x

Patients without AMI-related diagnosis codes were treated as controls.

Biomarker Categories

The study focuses on three clinically relevant biomarker groups.

Cardiac Biomarkers
Troponin T
CK-MB
Creatine Kinase (CK)
NT-proBNP
Hematological Biomarkers
Red Blood Cells (RBC)
White Blood Cells (WBC)
Hematocrit
Hemoglobin
Neutrophil-related indices
Metabolic and Biochemical Biomarkers
Creatinine
Potassium
Sodium
Lactate Dehydrogenase (LD)
Data Processing Pipeline

The workflow consisted of:

Extraction of laboratory records from MIMIC-IV
Identification of AMI and non-AMI patients using ICD codes
Cleaning and merging laboratory and diagnosis tables
Missing value analysis and imputation
Feature engineering
Construction of temporal sequences
Generation of static patient-level summaries
Train-test split using stratified sampling
Model training and evaluation
SHAP-based explainability analysis
Modelling Approaches
Temporal Modelling (LSTM)

The LSTM model was trained on time-dependent laboratory measurements to capture biomarker evolution over time.

Challenges:

Irregular sampling intervals
Sparse laboratory observations
Class imbalance

Performance:

Metric	Value
Accuracy	0.81–0.88
Recall (AMI)	Up to 0.75
Precision (AMI)	0.15–0.18

The temporal model showed good sensitivity but limited precision due to irregular laboratory sampling.

Non-Temporal Modelling

Static patient-level biomarker summaries were used to train:

XGBoost
Random Forest
LightGBM
AdaBoost
Logistic Regression
Gaussian Naïve Bayes
Best Performing Model: XGBoost
Metric	Value
Accuracy	0.968
AUC	0.945
Precision (AMI)	0.60
Recall (AMI)	0.56

Tree-based ensemble methods consistently outperformed temporal and linear models.

Explainable AI (SHAP)

SHAP was used to identify the biomarkers contributing most strongly to AMI prediction.

Top Predictive Biomarkers
Troponin T
Red Blood Cells
CK-MB Isoenzyme
Creatinine
Hematocrit
Potassium
Lactate Dehydrogenase (LD)

SHAP analysis confirmed that model predictions were driven by clinically meaningful biomarkers consistent with established AMI pathophysiology.

Repository Structure
AMI/
│
├── mimic-iv-3.1/
│   ├── diagnoses_icd.csv
│   ├── d_labitems.csv
│   ├── labevents.csv
│   └── relevant_labs.csv
│
├── AMI_Prediction_FINAL.ipynb
├── README.md
└── .gitattributes

Reproducing the Study
Requirements

pip install pandas numpy scikit-learn
pip install xgboost lightgbm
pip install tensorflow
pip install shap
pip install matplotlib
pip install imbalanced-learn

Running the Notebook

Open:

AMI_Prediction_FINAL.ipynb
Execute all cells sequentially to reproduce:

Data preprocessing
Feature engineering
Temporal LSTM model
Non-temporal machine learning models
Performance evaluation
SHAP explainability analysis

Key Findings
Non-temporal tree-based models outperformed temporal deep-learning approaches.
XGBoost achieved the highest predictive performance.
Irregular laboratory sampling reduced the effectiveness of temporal modelling.
SHAP identified clinically established cardiac biomarkers as the strongest predictors.
Ensemble learning demonstrated robustness to missing values and class imbalance.
Limitations
Irregular and sparse temporal laboratory measurements.
Severe class imbalance.
Missing availability of several specialized cardiac biomarkers.
External validation dataset contained feature mismatches.
Single-center retrospective study design.
Citation

If you use this repository, please cite:
Srivastava V., Chaudhary K., Bandyopadhyay D.

Stratification of Acute Myocardial Infarction (AMI) from non-AMI in admitted patients using static and time-dependent Cardiac and Hematological biomarkers applying Explainable Artificial Intelligence (AI)

Authors
Vaibhav Srivastava
Kartikay Chaudhary
Dr. Debashree Bandyopadhyay

Department of Biological Sciences
BITS Pilani Hyderabad Campus
Birla Institute of Technology and Science,
Pilani Hyderabad Campus, 2026.



Repository Structure
