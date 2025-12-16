# AI-Driven Account Prioritization Scoring Model  
**Role:** Data Science / Business Systems Internship  

---

## **Confidentiality Notice**

This repository is a methodological case study created for portfolio and educational purposes.  
It contains **no raw company data**, **no client-identifiable information**, and **no proprietary schemas**.  
All dataset descriptions, feature definitions, and results are presented at a high level and are intentionally abstracted.

---

## Overview

This project demonstrates an end-to-end scoring workflow for estimating the likelihood that an account converts into a customer. The goal is to support sales and operations teams with data-driven prioritization and clearer pipeline decisioning.

**Included in this repository**
- A reproducible preprocessing + modeling workflow (scikit-learn pipeline + XGBoost)
- Example feature engineering patterns for CRM-style signals (sanitized / synthetic)
- A technical methodology report (PDF)

**Not included**
- Any raw exports, internal tables, or proprietary schemas
- Any client-identifiable data
- Production credentials, connections, or deployment scripts tied to internal systems

---

## Objectives

1. Automate account prioritization via a binary classification score  
2. Combine multi-source inputs into a unified account-level feature table (synthetic/sanitized)  
3. Prevent temporal leakage through pre-outcome cutoffs and careful feature selection  
4. Package modeling into a deployable pipeline (imputation → encoding → model)  
5. Provide interpretable outputs (feature importance and evaluation summaries)

---

## Methodology Summary

### 1) Data Consolidation  
Synthetic/sanitized datasets representing firmographics, engagement activity, and pipeline signals are merged to an account-level table.

### 2) Leakage Prevention  
A cutoff timestamp is enforced so that only information available **before** the outcome window is used for training and scoring.

### 3) Feature Engineering (Example Set)  
The public version uses a deliberately small feature set to demonstrate the approach without exposing sensitive detail:

- `stage_idx_cleaned`  
- `days_since_created_date`  
- `days_since_first_engagement_date`  
- `billing_country`  
- `vertical`  
- `industry`  

> Note: Feature names may appear in snake_case in the public pipeline for consistency.

### 4) Modeling Pipeline  
- Missing value handling (median/mode)  
- Standardization for numeric fields  
- One-hot encoding for categorical fields  
- XGBoost classifier  
- Randomized hyperparameter search with stratified CV

### 5) Evaluation  
The report and code include evaluation outputs (ROC AUC / PR AUC / precision / recall). Any numeric results shown in this repository are **demonstration-only** and derived from **synthetic/sanitized** inputs.

---

## BI Dashboard  
A BI dashboard was used to visualize scoring outputs (distributions, segments, and trends).  
The dashboard artifact is **not included** in this repository.

---

## Limitations
- Synthetic inputs cannot reproduce full production complexity  
- Feature space is intentionally constrained to reduce leakage and exposure risk  
- This public version focuses on methodology and structure rather than production deployment

---

## Future Improvements
- Engagement velocity and trend features  
- Better intent/topic aggregation (where available)  
- Probability calibration  
- Cost-sensitive optimization for higher recall under class imbalance  
- CRM scoring integration patterns (generic)

---

## Contact
**Muhammad Yahya**  
Email: yahyaehtisham2004@gmail.com
