# AI-Driven Account Prioritization Scoring Model  
**Author:** Muhammad Yahya  
**Context:** RainFocus Business Systems Internship  
**Status:** Internal / Private Repository Only

---

## Overview
This repository contains the full technical implementation and documentation for an AI-driven account prioritization model developed during the RainFocus Business Systems Internship. The goal of the project was to automate the ranking of enterprise accounts based on their likelihood of converting, replacing manual and subjective prioritization methods with a data-driven scoring framework.

The project includes:
- A clean, leakage-free modeling pipeline
- Curated pre-conversion features
- A baseline XGBoost classifier
- End-to-end preprocessing infrastructure
- Detailed technical report in LaTeX/PDF
- Synthetic placeholders for diagrams and datasets (no real corporate data)

This repository remains **private** due to confidentiality constraints.

---

## Objectives
The primary objectives of the model were:

1. **Automate account prioritization** using statistically sound machine learning.
2. **Integrate multi-source Salesforce-derived data** into a unified feature set.
3. **Prevent temporal leakage** using a strict cutoff-date methodology.
4. **Develop a deployable pipeline** using scikit-learn and XGBoost.
5. **Deliver explainable results** to support sales and marketing strategy decisions.

---

## Repository Structure
├── src/
│ ├── preprocessing.py # Imputers, encoders, scalers
│ ├── stage_booster.py # Custom transformer for stage weighting
│ ├── model_training.py # XGBoost pipeline and RandomizedSearchCV
│ ├── evaluate.py # Metrics, AUC, threshold analysis
│ └── utils.py # Helper functions
│
├── data/
│ └── synthetic_sample.csv # Synthetic example dataset (safe)
│
├── paper/
│ ├── report.tex # Full technical report (LaTeX)
│ ├── report.pdf # Compiled PDF
│ ├── matrix.png # Confusion matrix (safe)
│ ├── process.png # Workflow diagram (safe)
│ └── logo.jpg # Placeholder logo
│
├── notebooks/
│ └── exploratory.ipynb # Safe notebook demonstrating pipeline steps
│
└── README.md # This file


No proprietary datasets, dashboards, or internal documents are included.

---

## Methodology Summary

### **1. Data Consolidation**
Multiple Salesforce-derived sources were merged into a single unified account-level dataset. Only pre-conversion signals were retained.

### **2. Leakage Prevention**
A dynamic cutoff timestamp ensured that:
- no post-conversion fields,
- no updated sales stages,
- no retroactive engagement data  
were included in training.

### **3. Feature Engineering**
The baseline model used six non-leaking features:
- `stage_idx_cleaned`
- `days_since_created_date`
- `days_since_first_engagement_date`
- `Billing Country`
- `Vertical`
- `Industry`

Temporal recency and cleaned sales-stage progression were especially important.

### **4. Modeling Pipeline**
The ML pipeline included:
- Median/mode imputation
- Standardization
- OneHotEncoding
- XGBoost classifier
- Custom StageBooster transformer
- RandomizedSearchCV for hyperparameter tuning

### **5. Evaluation**
The model achieved:
- **ROC AUC:** 0.9586  
- **PR AUC:** 0.7059  
- **Precision (positive class):** 0.77  
- **Recall (positive class):** 0.63  

Despite class imbalance, the model demonstrated strong discriminative power and high precision.

---

## Limitations
- Small number of positive-class examples
- Limited depth of pre-conversion signals
- Absence of granular engagement intent timestamps
- Narrow feature set by design (to ensure zero leakage)

---

## Future Improvements
Planned enhancements include:
- Engagement velocity features  
- Intent-topic aggregation signals  
- Multi-model ensembling  
- Probability calibration  
- Weekly scoring automation  
- Salesforce writeback integration  

These improvements will make the system production-ready.

---

## Confidentiality Notice
This repository is kept **private** because the underlying business context and modeling approach were developed during an internal internship. **No real RainFocus data, dashboards, internal slides, or confidential documents** are included.

All datasets contained in this repository are:
- synthetic  
- anonymized  
- structurally representative only  

This ensures compliance with confidentiality and data-handling standards.

---

## Contact
For verification or questions regarding the project:  
**Muhammad Yahya**  
Knox College — Integrative Business & Data Science  
Email upon request
