# AI-Powered Smart Manufacturing & Green Energy Optimization

## 📌 Project Overview

This project develops an AI-powered decision-support system for smart
manufacturing operations by combining **predictive maintenance** with
**industrial energy optimization**.

The system uses machine-learning techniques to identify maintenance risk
from machine operating conditions and combines industrial energy analytics
with optimization logic to identify opportunities for improving energy
efficiency.

The final solution is presented through an interactive **Streamlit dashboard**.

---

## 🎯 Business Problem

Manufacturing organizations face two major operational challenges:

1. Unexpected machine failures can cause production downtime and maintenance
   costs.
2. High or inefficient energy consumption increases operating costs and
   environmental impact.

Traditional monitoring systems often treat these problems separately.

This project combines both perspectives into a single decision-support
application.

---

## 🎯 Project Objectives

The project aims to:

- Predict machine maintenance requirements using machine-learning models.
- Identify the most important machine operating conditions associated with
  maintenance requirements.
- Measure machine-level energy efficiency.
- Compare actual energy consumption with forecast consumption.
- Identify machines requiring energy-efficiency intervention.
- Analyze energy performance across production modes.
- Provide actionable management recommendations.
- Develop an interactive Streamlit application.
- Create a reproducible data-science portfolio project.

---

# 🏭 Predictive Maintenance

The predictive-maintenance component uses the following machine-condition
variables:

- Temperature
- Vibration
- Humidity
- Pressure
- Energy consumption

The target variable is:

`maintenance_required`

Two classification models were evaluated:

- Logistic Regression
- Random Forest

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 63.5% | 29.7% | 62.5% | 40.3% | 69.3% |
| Random Forest | **89.1%** | **99.9%** | 44.9% | **62.0%** | **72.4%** |

Random Forest was selected as the provisional predictive-maintenance model
because it achieved the strongest overall performance.

### Important limitation

Although Random Forest achieved high precision and accuracy, its recall was
44.9%.

This means that some machines requiring maintenance may not be identified.

Therefore, the model is intended as a **decision-support tool** rather than
an autonomous maintenance system.

---

## 🔍 Feature Importance

The Random Forest model identified the following feature importance:

| Feature | Importance |
|---|---:|
| Temperature | 36.5% |
| Vibration | 22.4% |
| Humidity | 14.7% |
| Energy consumption | 13.3% |
| Pressure | 13.1% |

Temperature and vibration were the strongest model features.

---

# ⚡ Green Energy Optimization

The energy component evaluates:

- Energy consumption
- Production output
- Energy efficiency
- Forecast energy consumption
- Energy variance
- Machine utilization
- Production mode

## Energy Efficiency

Energy efficiency is calculated as:

`Production Output / Energy Consumption`

Higher values indicate greater production output per unit of energy consumed.

### Machine efficiency results

| Machine | Average Efficiency |
|---|---:|
| MCH_3 | **11.388** |
| MCH_1 | 9.909 |
| MCH_5 | 9.222 |
| MCH_4 | 7.347 |
| MCH_2 | 6.887 |

MCH_3 achieved the strongest energy-efficiency performance, while MCH_2
recorded the lowest efficiency.

---

## 🔎 Energy Variance

Energy variance is calculated as:

`Actual Energy - Forecast Energy`

Positive values indicate energy consumption above forecast.

| Machine | Average Variance |
|---|---:|
| MCH_5 | **+0.125** |
| MCH_4 | +0.053 |
| MCH_2 | +0.037 |
| MCH_3 | -0.024 |
| MCH_1 | -0.041 |

MCH_5 recorded the largest positive average energy variance.

---

## 🚨 Optimization Priority

The optimization framework combines:

- Energy efficiency
- Energy variance

to rank machines according to potential energy-efficiency intervention.

The analysis identified:

### MCH_4

as the primary optimization-priority machine because it combines relatively
low energy efficiency with positive energy variance.

MCH_5 represents a different type of opportunity because it has relatively
good efficiency but consistently consumes more energy than forecast.

---

# 📊 Production Mode Analysis

Energy performance was also examined across production modes.

| Production Mode | Avg. Energy | Avg. Efficiency | Avg. Variance |
|---|---:|---:|---:|
| Setup | 11.672 | **9.132** | -0.012 |
| Maintenance | 11.805 | 9.030 | +0.087 |
| Idle | 11.904 | 8.860 | -0.083 |
| Production | **11.992** | 8.794 | **+0.126** |

Production mode showed the least favorable energy-efficiency profile in the
dataset.

This finding represents an association rather than proof of causation.

---

# 🧠 Decision-Support Framework

The final system combines three analytical components:

```text
Machine Operating Conditions
            │
            ▼
   Predictive Maintenance
            │
            ▼
     Maintenance Risk
            │
            │
            ├──────────────────┐
            │                  │
            ▼                  ▼
     Energy Analytics    Optimization Logic
            │                  │
            ▼                  │
    Energy Efficiency         │
    Energy Variance ──────────┘
            │
            ▼
     Recommendations
            │
            ▼
       Streamlit App