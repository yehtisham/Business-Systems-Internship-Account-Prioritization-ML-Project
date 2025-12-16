# AI-Driven Account Prioritization Scoring Model  
**Role:** Data Science / Business Systems Internship
---

##Confidentiality Notice
This repository is a methodological case study created for portfolio and educational purposes.
It contains no raw company data, no client-identifiable information, and no proprietary schemas.
All dataset descriptions and metrics are presented at a high level and are intentionally abstracted. 


## Overview
This project presents an end-to-end machine learning framework for predicting the likelihood that an account converts into a customer. It simulates a real-world enterprise scenario where sales teams require data-driven prioritization to improve pipeline efficiency.

The repository includes:
- A clean, leakage-free modeling pipeline
- Curated synthetic pre-conversion features
- A baseline XGBoost classifier
- End-to-end preprocessing code
- A fully detailed technical report written in LaTeX/PDF
- Synthetic placeholders replicating realistic ML workflows

This version is fully sanitized and safe for public distribution.

---

## Objectives

1. **Automate account prioritization** using a binary classification model.  
2. **Integrate multi-source synthetic datasets** into a unified feature table.  
3. **Prevent temporal leakage** through strict cutoff enforcement.  
4. **Design a deployable scikit-learn / XGBoost pipeline.**  
5. **Produce explainable outputs** including feature importance and evaluation metrics.  

---

## Methodology Summary

### **1. Data Consolidation (Synthetic)**
Synthetic datasets representing account demographics, engagement activity, and sales stages were merged into a unified account-level table.

### **2. Leakage Prevention**
A dynamic cutoff timestamp ensured that:
- only pre-conversion features were used,  
- no future information leaked into the model,  
- temporal validity was maintained.

### **3. Feature Engineering**
The model uses six non-leaking features:

- `stage_idx_cleaned`  
- `days_since_created_date`  
- `days_since_first_engagement_date`  
- `Billing Country`  
- `Vertical`  
- `Industry`  

These replicate realistic CRM-style signals without exposing any real data.

### **4. Modeling Pipeline**
The pipeline includes:
- Median/mode imputation  
- Standard scaling  
- OneHotEncoding  
- Custom StageBooster transformer  
- XGBoost classifier  
- Randomized hyperparameter search  

### **5. Evaluation**
On synthetic test data, the model achieved:

- **ROC AUC:** ≈ 0.95  
- **PR AUC:** ≈ 0.70  
- **Precision:** ≈ 0.75  
- **Recall:** ≈ 0.60  

Numbers are representative but fully synthetic.

---

## Looker Dashboard (Not Included)
An interactive dashboard was developed to visualize account-scoring outputs, including prediction distributions and segmentation insights.  
Due to confidentiality and platform restrictions, the dashboard is **not included** in this public repository.

---

## Limitations
- Synthetic dataset cannot fully replicate real engagement complexity  
- Feature space intentionally limited to avoid leakage  
- No deep behavioral sequences or intent embeddings  

---

## Future Improvements
Planned enhancements include:
- Engagement velocity features  
- Intent-topic aggregation  
- Probability calibration  
- Cost-sensitive training to improve recall  
- Integration with CRM scoring workflows  

---

## Contact
For questions or verification:  
**Muhammad Yahya**  
Email: *yahyaehtisham2004@gmail.com*
