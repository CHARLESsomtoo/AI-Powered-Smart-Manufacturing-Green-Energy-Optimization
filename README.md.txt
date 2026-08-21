# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project develops a machine learning model to detect fraudulent credit card transactions. The objective is to help financial institutions identify fraudulent activities accurately while minimizing false alarms. Multiple machine learning algorithms were developed, evaluated, and compared to determine the most effective model for fraud detection.

## Business Problem

Credit card fraud is a major challenge for financial institutions, resulting in significant financial losses and reduced customer trust. Since fraudulent transactions represent only a very small percentage of all transactions, detecting them accurately is a difficult task. The goal of this project is to build a machine learning model that can identify fraudulent transactions while minimizing false positives and false negatives.



## Dataset

The dataset used in this project is the **Credit Card Fraud Detection Dataset**, which contains anonymized credit card transactions made by European cardholders.

Key characteristics of the dataset include:

- **Total Transactions:** 284,807
- **Features:** 30 predictor variables (`Time`, `Amount`, `V1`–`V28`)
- **Target Variable:** `Class`
  - `0` = Legitimate Transaction
  - `1` = Fraudulent Transaction
- **Fraudulent Transactions:** 492 (approximately **0.17%** of the dataset)

The severe class imbalance makes this dataset a challenging and realistic machine learning problem for fraud detection.

## Project Workflow (PACE Framework)

### P – Plan

The project began by understanding the business problem of credit card fraud detection. The objective was to develop a machine learning model capable of accurately identifying fraudulent transactions while minimizing false positives and false negatives. Success would be measured using evaluation metrics suitable for imbalanced datasets, such as Precision, Recall, F1-Score, ROC-AUC, and Average Precision.

### A – Analyze

The dataset was explored to understand its structure and quality before model development. This involved checking the dataset dimensions, identifying missing values, removing duplicate records, analyzing the distribution of the target variable, examining the transaction amount, and studying the relationships between features using visualizations and correlation analysis. The analysis revealed that the dataset was highly imbalanced, with fraudulent transactions accounting for only about 0.17% of all transactions.

### C – Construct

The data was prepared for machine learning by separating the predictor variables from the target variable and splitting the dataset into training and testing sets using stratified sampling to preserve the class distribution. Four machine learning models were developed and compared: Logistic Regression, Decision Tree, Random Forest, and an Optimized Random Forest using RandomizedSearchCV. The models were evaluated using metrics appropriate for imbalanced classification problems.

### E – Execute

The trained models were evaluated using Precision, Recall, F1-Score, ROC-AUC, and Average Precision. The Optimized Random Forest achieved the best overall performance, with a Precision of **97.22%**, Recall of **73.68%**, and an F1-Score of **83.83%**. Based on these results, the Optimized Random Forest was selected as the final model because it provided the best balance between detecting fraudulent transactions and minimizing false alarms.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|---------:|----------:|--------:|---------:|
| Logistic Regression | 99.91% | 84.62% | 57.89% | 68.75% |
| Decision Tree | 100.00% | 72.00% | 71.00% | 71.00% |
| Random Forest | 99.95% | 97.18% | 72.63% | 83.13% |
| Optimized Random Forest | **99.95%** | **97.22%** | **73.68%** | **83.83%** |

## Project Results

The Optimized Random Forest model achieved the best overall performance for detecting fraudulent credit card transactions. It demonstrated a strong ability to identify fraudulent activities while maintaining a low false positive rate.

**Key achievements:**

- Precision: **97.22%**
- Recall: **73.68%**
- F1-Score: **83.83%**
- ROC-AUC Score: **95.40%**
- Average Precision Score: **79.05%**

These results indicate that the model is capable of effectively distinguishing between legitimate and fraudulent transactions, making it a strong candidate for real-world fraud detection applications.

## Project Visualizations

### Feature Importance

![Feature Importance](images/feature_importance.png)

### ROC Curve

![ROC Curve](images/roc_curve.png)

### Precision–Recall Curve

![Precision–Recall Curve](images/precision_recall_curve.png)

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

