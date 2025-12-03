# ⚙️ Technical Report: Predictive Maintenance & Machine Failure Analysis

**Predictive Maintenance AI System**

Unplanned machine failures in industrial processes lead to costly downtime, production delays, and resource wastage. The **Predictive Maintenance AI System** leverages machine learning to anticipate failures, identify failure types, and provide real-time actionable insights to minimize downtime, optimize maintenance, and extend machine lifespan.

### Key Features

* **Failure Prediction:** Detects potential machine failures before they occur.
* **Failure Classification:** Identifies specific failure types (HDF, PWF, OSF, TWF, RNF).
* **Data-Driven Monitoring:** Analyzes historical data (temperature, torque, speed, tool wear).
* **Condition-Based Maintenance:** Replaces reactive and scheduled maintenance with intelligent scheduling.
* **Explainable AI:** Uses **SHAP** to provide transparent, trustworthy predictions.
* **Resource Optimization:** Helps plan maintenance schedules and allocate resources efficiently.
* **User-Friendly Dashboard:** Streamlit-based interface for real-time monitoring and insights.

---

**Date:** November 18, 2025

This repository contains the machine learning models and analysis developed for a **Predictive Maintenance (PdM)** system, aiming to anticipate equipment failures using sensor telemetry data.

---

## 🚀 1. Executive Summary

The primary objective was to develop machine learning models to shift equipment maintenance from **reactive** to **predictive**. By analyzing sensor readings (temperature, speed, torque, tool wear), we successfully built models for:

1.  **Binary Classification:** Predicting "Failure" or "No Failure."
2.  **Multi-class Classification:** Diagnosing the specific **Failure Mode**.

The project identified that **Random Forest** and **XGBoost** models, optimized using **SMOTE** (Synthetic Minority Over-sampling Technique), achieved superior predictive performance, with **AUC scores exceeding 0.99** for failure detection. Physics-based **feature engineering** was key to improving model accuracy and interpretability.

---

## 🎯 2. Problem Statement & Data Overview

Unplanned equipment downtime leads to significant financial loss and operational disruption. The goal is to mitigate this by predicting impending failure using historical sensor data.

### 2.1 Dataset Summary

The dataset comprises **10,000 records** of telemetry data, including:

* **Categorical:** `Type` (L, M, H quality variants).
* **Sensor Metrics (Input Features):** `Air Temperature [K]`, `Process Temperature [K]`, `Rotational Speed [rpm]`, `Torque [Nm]`, `Tool Wear [min]`.
* **Target Variables:**
    * `Machine failure` (Binary: 0 or 1).
    * `Failure Modes` (TWF, HDF, PWF, OSF, RNF).

### 2.2 Data Quality & Imbalance

* **Missing Values:** None detected.
* **Class Imbalance:** The dataset is highly imbalanced, with only **3.4%** (339 out of 10,000) of records representing a machine failure event. This necessitated robust imbalance handling techniques.

---

## 🔬 3. Methodology

### 3.1 Feature Engineering

Domain knowledge was utilized to create synthetic, **physics-based features** that better represent the machine's physical state:

* `Temp_Delta`: Difference between Process and Air temperature.
* `Power [W]`: Calculated as $Torque \times Rotational\ Speed \times constant$.
* `Wear_per_Torque`: Ratio of tool wear to torque (indicating strain).
* `Speed_Torque_Ratio`: Captures drivetrain stress relationships.

### 3.2 Key EDA Insights

* **Correlation:** A strong **inverse correlation (-0.88)** was observed between `Rotational Speed` and `Torque`.
* **Failure Profile:** Failures are strongly associated with **higher Tool Wear, Torque, and Power**, and conversely, **lower Rotational Speed**.
* **Failure Type:** **Heat Dissipation Failure (HDF)** was identified as the most common specific failure type.
* **Feature Significance:** `Torque`, `Rotational Speed`, and `Tool Wear` were identified as the most statistically significant predictors (via ANOVA F-tests and Mutual Information).

### 3.3 Preprocessing

1.  **Scaling:** Applied **StandardScaler** and **RobustScaler** to normalize numerical features.
2.  **Imbalance Handling:** **SMOTE** was crucially applied to the training data to generate synthetic samples of the minority (failure) class, preventing model bias.

---

## 📈 4. Modeling & Results

### 4.1 Phase 1: Binary Classification (Failure Detection)

Three models were tested for predicting an imminent failure (Binary 0/1). Evaluation focused on **AUC Score** due to the class imbalance.

| Model | AUC Score | Performance Notes |
| :--- | :--- | :--- |
| **Random Forest** | **1.00** | Excellent precision and recall; misclassified only 77 samples. |
| **XGBoost** | **1.00** | **Best performer**; misclassified only 55 samples. |
| **Logistic Regression** | 0.94 | Served as a good baseline but struggled with non-linear relationships. |

**Key Finding:** Ensemble tree-based models (Random Forest and XGBoost) significantly outperformed the linear model.

> **Explainability (SHAP Analysis):** **Rotational Speed**, **Power**, and **Tool Wear** were confirmed to be the primary drivers of the models' failure predictions.

### 4.2 Phase 2: Multi-class Classification (Failure Diagnosis)

The goal was to predict the specific failure type (TWF, HDF, PWF, OSF, RNF, or No Failure).

**Technique:** **SMOTE** was applied to address extreme imbalance across the specific failure sub-types.

**Result: Random Forest Superiority**
The **Random Forest** model again demonstrated superior robustness, achieving **nearly perfect scores (1.00 Precision/Recall)** for most specific failure classes on the test set. It successfully identified the vast majority of "No Failure" cases (**1,928 instances**) and accurately classified rare events like Overstrain Failure (OSF) with minimal error.

---

## 💡 5. Conclusion & Future Steps

The developed **Random Forest** and **XGBoost** models are highly effective for this predictive maintenance task, providing high accuracy in both detecting and diagnosing potential equipment failures.

### Key Takeaways

1.  **Model Robustness:** **Random Forest** proved the most reliable model for handling the complex multi-class classification and non-linear interactions within this tabular dataset.
2.  **Feature Importance:** **Physics-based features** like `Power` and `Temp_Delta` were crucial, providing interpretability that raw sensor data alone lacked.
3.  **Metrics:** Given the **96.6% "No Failure" rate**, metrics like **F1-Score** and **ROC-AUC** were essential for valid model evaluation, while simple Accuracy was misleading.

### Recommendations for Deployment

* **Model Serialization:** The final **Random Forest** model should be saved using `joblib` or `pickle` for quick loading in a production environment.
* **Real-time Interface:** Develop a **Streamlit/Web Application** where operators can input live sensor readings and immediately receive a real-time failure probability and the predicted specific failure type.
* **Threshold Tuning:** The production probability threshold for flagging a failure must be carefully tuned to minimize the overall cost of **False Positives** (false alarms) versus the cost of **Missed Failures**.

---

Would you like me to generate a sample SHAP summary plot or a confusion matrix image tag to include in the "Modeling & Results" section for better visualization of the models' performance?
