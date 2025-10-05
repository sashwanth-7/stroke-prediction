# ===== Stroke Risk Prediction App =====
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import shap

# ===== Load Model & Scaler =====
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ===== Page Configuration =====
st.set_page_config(page_title="Stroke Risk Prediction", page_icon="🧠", layout="centered")

# ===== Header =====
st.title("🧠 Stroke Risk Prediction")
st.markdown("Enter the patient details below to predict stroke risk. The app will also show feature importance and explainability.")

# ===== User Inputs =====
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])

with col2:
    work_type = st.selectbox("Work Type", ["Government", "Private", "Self-employed", "Children", "Never worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=400.0, value=110.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"])

# ===== Manual Encoding =====
gender_map = {"Male": 1, "Female": 0}
hypertension_map = {"Yes": 1, "No": 0}
heart_disease_map = {"Yes": 1, "No": 0}
ever_married_map = {"Yes": 1, "No": 0}
work_type_map = {"Government":0, "Private":1, "Self-employed":2, "Children":3, "Never worked":4}
residence_map = {"Urban":1, "Rural":0}
smoke_map = {"Never smoked":0, "Formerly smoked":1, "Smokes":2, "Unknown":3}

input_dict = {
    "gender": gender_map[gender],
    "age": age,
    "hypertension": hypertension_map[hypertension],
    "heart_disease": heart_disease_map[heart_disease],
    "ever_married": ever_married_map[ever_married],
    "work_type": work_type_map[work_type],
    "Residence_type": residence_map[residence_type],
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoke_map[smoking_status]
}

input_df = pd.DataFrame([input_dict])

# Scale numeric columns
num_cols = ['age','avg_glucose_level','bmi']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# ===== Predict & Show Results =====
if st.button("🔎 Predict Stroke Risk"):
    prob = model.predict_proba(input_df)[0][1]
    pred_class = model.predict(input_df)[0]

    if pred_class == 1:
        st.error(f"⚠️ High Risk of Stroke! Probability: {prob*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Stroke. Probability: {prob*100:.2f}%")

    st.progress(min(int(prob*100),100))

    # ===== Feature Importance Plot =====
    st.subheader("📊 Feature Importance (Overall Model)")
    feat_imp = pd.DataFrame({
        'Feature': input_df.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, ax=ax, palette="viridis")
    st.pyplot(fig)

    # ===== SHAP Explainability for this prediction =====
    st.subheader("🔍 Why This Prediction? (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    shap.initjs()
    st.markdown("Feature impact for this specific patient:")
    st_shap = shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=True, show=False)
    plt.tight_layout()
    st.pyplot(bbox_inches='tight')

# ===== Footer =====
st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This tool is for educational purposes only. Not a substitute for medical advice.")
st.markdown("Made with ❤️ by Sashwanth | ML Stroke Prediction Project")
