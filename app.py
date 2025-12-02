import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit.components.v1 import html

# ---------------------------
# 1. PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="MachineGuard Predictive Maintenance App", layout="wide")
st.title("🛠️ MachineGuard Predictive Maintenance Dashboard")
st.write("Enter machine sensor data to predict status and fault type.")

# ---------------------------
# 2. LOAD SCALER & MODELS
# ---------------------------
scaler = joblib.load(r"preprocess/scaler_singleClass.joblib")
single_model = joblib.load(r"model/single_class_best_xgboost_model.joblib")
multi_model = joblib.load(r"model/multi_class_best_xgboost_model.joblib")

# ---------------------------
# 3. FEATURES
# ---------------------------
numeric_features = [
    "Type", "Air_temperature_[K]", "Process_temperature_[K]",
    "Rotational_speed_[rpm]", "Torque_[Nm]", "Tool_wear_[min]",
    "Temp_Delta", "Power_[W]", "Wear_per_Torque", "Speed_Torque_Ratio"
]

categorical_features = ["No_failure"]
all_features = numeric_features + categorical_features
target_classes = ["No_failure", "TWF", "RNF", "PWF", "HDF", "OSF"]

# ---------------------------
# 4. SIDEBAR INPUT
# ---------------------------
st.sidebar.header("Input Machine Sensor Data")
user_input = {}
for feature in numeric_features[:6]:  # First 6 manually
    user_input[feature] = st.sidebar.number_input(feature, value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# ---------------------------
# 5. ENGINEERED FEATURES
# ---------------------------
input_df["Temp_Delta"] = input_df["Process_temperature_[K]"] - input_df["Air_temperature_[K]"]
input_df["Power_[W]"] = input_df["Torque_[Nm]"] * (input_df["Rotational_speed_[rpm]"] * 2 * np.pi / 60)
input_df["Wear_per_Torque"] = input_df["Tool_wear_[min]"] / (input_df["Torque_[Nm]"] + 1e-5)
input_df["Speed_Torque_Ratio"] = input_df["Rotational_speed_[rpm]"] / (input_df["Torque_[Nm]"] + 1e-5)

# ---------------------------
# 6. SCALE NUMERIC FEATURES
# ---------------------------
input_df[numeric_features] = scaler.transform(input_df[numeric_features])

# ---------------------------
# 7. SINGLE-CLASS MODEL → Predict No_failure
# ---------------------------
single_pred = single_model.predict(input_df[numeric_features])[0]
single_status = "No Failure" if single_pred == 1 else "Failure Detected"

# ---------------------------
# 8. MULTI-CLASS MODEL INPUT
# ---------------------------
input_df["No_failure"] = 0 if single_pred == 1 else 1

# ✅ FIX APPLIED HERE — correct multi-class prediction
multi_pred_vector = multi_model.predict_proba(input_df[all_features])[0]

pred_idx = np.argmax(multi_pred_vector)
multi_class = target_classes[pred_idx]
multi_status = "Good condition" if multi_class == "No_failure" else "Fault identified"

# ---------------------------
# 9. DASHBOARD DISPLAY
# ---------------------------
st.subheader("Model Predictions")
st.markdown(f"**Single-Class Model:** {single_status}")
st.markdown(f"**Multi-Class Model Status:** {multi_status}")
if multi_status == "Fault identified":
    st.markdown(f"**Predicted Fault Type:** {multi_class}")

# Display predicted probabilities for multi-class model
probs = multi_model.predict_proba(input_df[all_features])[0]
prob_df = pd.DataFrame({"Fault Type": target_classes, "Probability": probs})
st.subheader("Multi-Class Predicted Probabilities")
st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

# ---------------------------
# 10. SHAP EXPLAINER
# ---------------------------
explainer = shap.Explainer(multi_model, feature_names=all_features)
shap_values = explainer(input_df)

def st_shap(plot, height=350):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    html(shap_html, height=height)

st.subheader("Feature Impact (SHAP)")
force_plot = shap.plots.force(
    explainer.expected_value[pred_idx],
    shap_values.values[0, :, pred_idx],
    feature_names=all_features,
    matplotlib=False
)
st_shap(force_plot, height=300)

waterfall_expl = shap.Explanation(
    values=shap_values.values[0, :, pred_idx],
    base_values=explainer.expected_value[pred_idx],
    data=input_df.iloc[0],
    feature_names=all_features
)
waterfall_ax = shap.plots.waterfall(waterfall_expl, show=False)
st.pyplot(waterfall_ax.figure)

st.subheader("Global Feature Importance (SHAP Summary)")
if input_df.shape[0] > 1:
    shap_values_arr = np.abs(shap_values.values)
    shap.summary_plot(shap_values_arr, input_df, feature_names=all_features, show=False)
    st.pyplot(plt.gcf())
else:
    st.info("SHAP summary requires multiple rows. Force and waterfall plots shown for single prediction.")
