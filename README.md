# ⚙️ Technical Report: Predictive Maintenance & Machine Failure Analysis

## **Predictive Maintenance AI System for Milling Machines**

Unplanned machine failures in industrial operations lead to costly downtime, production delays, and resource wastage. The **Predictive Maintenance AI System** leverages advanced machine learning to detect early warning signs of malfunction, predict failures, classify failure modes, and provide real-time actionable insights. This empowers maintenance teams to reduce downtime, optimize scheduling, enhance operational efficiency, and extend machinery lifespan.

---

## 🔑 Key Features

- **Failure Prediction:** Detects potential machine failures before they occur.  
- **Failure Classification:** Identifies specific modes (HDF, PWF, OSF, TWF, RNF).  
- **Data-Driven Monitoring:** Analyzes operational metrics such as temperature, torque, speed, and tool wear.  
- **Condition-Based Maintenance:** Replaces reactive and scheduled maintenance with intelligent, predictive scheduling.  
- **Explainable AI:** Uses **SHAP** for transparent and interpretable model decisions.  
- **Resource Optimization:** Supports maintenance planning and resource allocation.  
- **User-Friendly Dashboard:** Streamlit-based interface for real-time analytics and visualizations.

---

## 📦 Dataset: AI4I 2020 Predictive Maintenance Dataset

This project utilizes the **original AI4I 2020 dataset**, a synthetic dataset modeled after a real milling machine, containing **10,000 rows** and **14 features** describing machine operating conditions.

### **Dataset Features**
- **UID** — Unique identifier  
- **Product ID**, **Type (L/M/H)** — Product categories  
- **Air Temperature [K]**  
- **Process Temperature [K]**  
- **Rotational Speed [rpm]**  
- **Torque [Nm]**  
- **Tool Wear [min]**  
- **Machine Failure** (Binary)

### **Failure Modes**
- **TWF** — Tool Wear Failure  
- **HDF** — Heat Dissipation Failure  
- **PWF** — Power Failure  
- **OSF** — Overstrain Failure  
- **RNF** — Random Failure

### **Dataset Citation**
> S. Matzka, “Explainable Artificial Intelligence for Predictive Maintenance Applications,”  
> *2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, 2020,  
> pp. 69–74, doi: 10.1109/AI4I49448.2020.00023.

---

**Date:** November 18, 2025

This repository contains machine learning models and analytical workflows developed for a **Predictive Maintenance (PdM)** system aimed at forecasting equipment failures using sensor telemetry.

---

# 🚀 1. Executive Summary

The objective of this project is to transition from **reactive** to **predictive** maintenance using machine learning. By analyzing key machine telemetry—temperature, torque, speed, and tool wear—two predictive tasks were successfully implemented:

1. **Binary Classification:** Predict whether a failure will occur.  
2. **Multi-Class Classification:** Diagnose the specific failure type.

**Random Forest** and **XGBoost** delivered the best performance, achieving **AUC > 0.99** for failure detection. Physics-informed **feature engineering** significantly improved model interpretability and performance.

---

# 🎯 2. Problem Statement & Data Overview

Unplanned equipment downtime causes financial losses and process inefficiencies. This system aims to predict failures using historical machine telemetry.

## **2.1 Dataset Summary**
The dataset contains **10,000 records**, including:

- **Categorical:** `Type` (L, M, H)  
- **Sensor Inputs:**  
  - Air Temperature  
  - Process Temperature  
  - Rotational Speed  
  - Torque  
  - Tool Wear  
- **Targets:**  
  - `Machine Failure` (Binary)  
  - `Failure Modes`: TWF, HDF, PWF, OSF, RNF  

## **2.2 Data Quality & Class Imbalance**
- **No missing values**  
- Highly imbalanced: **3.4% failure events (339/10,000)**  
- Required class balancing using **SMOTE**

---

# 🔬 3. Methodology

## **3.1 Feature Engineering**
Physics-based synthetic features were created to capture machine behavior:

- `Temp_Delta` = Process Temp − Air Temp  
- `Power [W]` = Torque × Rotational Speed × constant  
- `Wear_per_Torque` = Tool Wear ÷ Torque  
- `Speed_Torque_Ratio` = Rotational Speed ÷ Torque  

## **3.2 EDA Insights**
- **Strong negative correlation** between Torque and Rotational Speed (−0.88).  
- Failures correlated with **higher torque, tool wear, power**, and **lower speed**.  
- **HDF** was the most common failure mode.  
- Top predictors: **Torque**, **Rotational Speed**, **Tool Wear** (via ANOVA & Mutual Information).

## **3.3 Preprocessing**
- Numeric scaling using **StandardScaler** and **RobustScaler**  
- **SMOTE** used to correct class imbalance during training  

---

# 📈 4. Modeling & Results

## **4.1 Binary Classification (Failure Detection)**

| Model | Accuracy | Notes |
|-------|----------|--------|
| **Random Forest** | **1.00** | Excellent precision/recall; 77 misclassifications |
| **XGBoost** | **1.00** | Best performer; 55 misclassifications |
| **Logistic Regression** | 0.94 | Good baseline; limited by non-linearity |

**SHAP Explainability:**  
Top contributors: **Rotational Speed**, **Power**, **Tool Wear**.

---

## **4.2 Multi-Class Classification (Failure Diagnosis)**

Goal: Predict specific failure types.  
Approach: Applied **SMOTE** to each minority class.

**Best Model: Random Forest**

- Near-perfect performance across failure types  
- Accurately predicted **1,928 "No Failure"** cases  
- Successfully classified rare failure events like **OSF**  

---

# 💡 5. Conclusion & Future Steps

The system’s **Random Forest** and **XGBoost** models provide highly reliable predictive and diagnostic capabilities for maintenance optimization.

## **Key Takeaways**
1. **Random Forest** is the most robust model for multi-class tabular prediction.  
2. Physics-informed features significantly improved interpretability and accuracy.  
3. Accuracy alone is misleading due to imbalance; **F1-Score** and **ROC-AUC** were essential.

## **Deployment Recommendations**
- Serialize model using **joblib** or **pickle**  
- Deploy a **Streamlit dashboard** for real-time predictions  
- Tune probability thresholds to balance false alarms vs. missed failures  

---

Would you like me to format this as a **GitHub README**, add **visual placeholders** (e.g., confusion matrix, SHAP plot), or include a **system architecture diagram**?
