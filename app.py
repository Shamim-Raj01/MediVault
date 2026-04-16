import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from inference_improved_model import ImprovedModelInference

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_models():
    """Load trained models and preprocessors with file existence checks."""
    improved_paths = {
        'model_improved.pkl': BASE_DIR / 'model_improved.pkl',
        'le_classification_improved.pkl': BASE_DIR / 'le_classification_improved.pkl',
        'feature_names_improved.pkl': BASE_DIR / 'feature_names_improved.pkl'
    }
    legacy_paths = {
        'model.pkl': BASE_DIR / 'model.pkl',
        'reg_model.pkl': BASE_DIR / 'reg_model.pkl',
        'le_classification.pkl': BASE_DIR / 'le_classification.pkl',
        'scaler.pkl': BASE_DIR / 'scaler.pkl'
    }

    model_inference = None
    if all(path.exists() for path in improved_paths.values()):
        try:
            model_inference = ImprovedModelInference(model_dir=BASE_DIR)
        except Exception as exc:
            st.warning(
                "Improved model files were found but could not be loaded. "
                "The app will fall back to the legacy model."
            )
            st.write(f"Debug: {exc}")

    if model_inference is not None:
        missing_files = [name for name, path in legacy_paths.items() if not path.exists()]
        if missing_files:
            st.error("Legacy model files are required for regression and fallback support.")
            st.write("Missing files:")
            for missing in missing_files:
                st.write(f"- {missing}")
            st.stop()

        with open(legacy_paths['reg_model.pkl'], 'rb') as f:
            reg_model = pickle.load(f)
        with open(legacy_paths['le_classification.pkl'], 'rb') as f:
            le_clf = pickle.load(f)
        with open(legacy_paths['scaler.pkl'], 'rb') as f:
            scaler_reg = pickle.load(f)
        return model_inference, reg_model, le_clf, scaler_reg

    missing_files = [name for name, path in legacy_paths.items() if not path.exists()]
    if missing_files:
        st.error("Model files not found. Please run the training notebook first.")
        st.write("Missing files:")
        for missing in missing_files:
            st.write(f"- {missing}")
        st.info("Run `fixed_project.ipynb` to generate the required model files.")
        st.stop()

    with open(legacy_paths['model.pkl'], 'rb') as f:
        clf_model = pickle.load(f)
    with open(legacy_paths['reg_model.pkl'], 'rb') as f:
        reg_model = pickle.load(f)
    with open(legacy_paths['le_classification.pkl'], 'rb') as f:
        le_clf = pickle.load(f)
    with open(legacy_paths['scaler.pkl'], 'rb') as f:
        scaler_reg = pickle.load(f)
    return clf_model, reg_model, le_clf, scaler_reg

def load_data():
    """Load dataset and precautions with robust column handling."""
    df = pd.read_csv(BASE_DIR / 'healthcare-chatbot' / 'Data' / 'Training.csv')
    symptoms = [col for col in df.columns if col != 'prognosis']
    precautions_df = pd.read_csv(
        BASE_DIR / 'healthcare-chatbot' / 'MasterData' / 'symptom_precaution.csv'
    )
    normalized_cols = {str(col).strip().lower(): col for col in precautions_df.columns}
    if 'disease' in normalized_cols:
        disease_col = normalized_cols['disease']
    elif 'prognosis' in normalized_cols:
        disease_col = normalized_cols['prognosis']
    else:
        disease_col = precautions_df.columns[0]  # fallback
    precaution_cols = [
        col for col in precautions_df.columns
        if col != disease_col
    ][:4]  # take first 4
    precautions = {}
    for _, row in precautions_df.iterrows():
        disease = str(row[disease_col]).strip()
        if not disease or disease.lower() == 'nan':
            continue
        values = []
        for col in precaution_cols:
            val = row[col]
            if pd.notna(val):
                val = str(val).strip()
                if val:
                    values.append(val)
        precautions[disease] = values
    return symptoms, precautions

def preprocess_input(symptoms_list, all_symptoms):
    """Create feature vector from selected symptoms."""
    input_vector = np.zeros(len(all_symptoms))
    for symptom in symptoms_list:
        if symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            input_vector[idx] = 1
    return input_vector

def predict_disease(model, input_vector, le_clf):
    """Predict disease and confidence."""
    pred_encoded = model.predict([input_vector])[0]
    disease = le_clf.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba([input_vector])[0]
    confidence = np.max(proba) * 100
    return disease, confidence

def predict_disease_safe(model, selected_symptoms, all_symptoms, le_clf):
    """Predict disease using the improved model when available, otherwise fallback."""
    if isinstance(model, ImprovedModelInference):
        disease, confidence = model.predict_disease(selected_symptoms, all_symptoms)
        return disease, confidence * 100

    input_vector = preprocess_input(selected_symptoms, all_symptoms)
    return predict_disease(model, input_vector, le_clf)


def predict_progression(reg_model, disease, le_clf, scaler_reg):
    """Predict progression stage using default patient data."""
    default_data = {
        'Age': 50,
        'Gender': 1,
        'Disease': disease,
        'MedicalHistory': 1,
        'Lifestyle': 1,
        'BiomarkerScore': 0.5,
        'MedicationDose': 50.0,
        'HeartRate': 75,
        'BloodPressure_Systolic': 130,
        'BloodPressure_Diastolic': 85,
        'Cholesterol': 200,
        'BMI': 25.0,
        'SleepHours': 7.0,
        'StepsPerDay': 8000,
        'MedicationAdherence': 1,
        'StressLevel': 5,
        'CognitiveScore': 75,
        'MoodScore': 5,
        'Smoker': 0,
        'AlcoholUse': 1,
        'SupportSystem': 1,
        'HasCaregiver': 0,
        'EmploymentStatus': 1
    }

    input_df = pd.DataFrame([default_data])

    # Encode Disease safely and keep prediction stable
    try:
        if disease in le_clf.classes_:
            input_df['Disease'] = le_clf.transform([disease])[0]
        else:
            input_df['Disease'] = 0
    except Exception:
        input_df['Disease'] = 0

    # Ensure regression input ordering matches scaler training
    feature_order = getattr(scaler_reg, 'feature_names_in_', None)
    if feature_order is not None:
        input_df = input_df.reindex(columns=feature_order)

    try:
        input_scaled = scaler_reg.transform(input_df)
        stage = reg_model.predict(input_scaled)[0]
    except Exception:
        stage = np.nan
    return stage

# Load data and models
clf_model, reg_model, le_clf, scaler_reg = load_models()
symptoms, precautions = load_data()

# Styled header
st.markdown(
    "<div style='text-align: center; padding: 20px; color: #0a3161; font-size:32px; font-weight:700;'>🩺 AI Health Diagnosis System</div>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    "This AI system predicts diseases based on symptoms using Random Forest classification and estimates disease progression stages using regression models."
)

st.sidebar.header("🔍 How it Works")
st.sidebar.write(
    "1. Select symptoms you're experiencing.\n2. Click 'Predict Disease' to get results.\n3. View predicted disease, confidence, progression stage, and precautions."
)

st.sidebar.header("📊 Model Info")
st.sidebar.write(f"- **Classification:** Random Forest ({len(symptoms)} symptoms)")
st.sidebar.write("- **Regression:** Random Forest Regressor")
st.sidebar.write(f"- **Total Symptoms:** {len(symptoms)}")

st.sidebar.success("✅ System Ready")
st.sidebar.warning("⚠️ This system is for educational purposes only. Consult a doctor.")

# Tabs
tab1, tab2 = st.tabs(["🔮 Prediction", "📈 Insights"])

with tab1:
    st.subheader("Select Your Symptoms")
    selected_symptoms = st.multiselect("Choose symptoms you are experiencing:", symptoms, key="symptoms")

    if selected_symptoms:
        st.markdown("**Selected Symptoms:**")
        st.write(selected_symptoms)
    else:
        st.info("Select symptoms from the list to begin prediction.")

    st.write("")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        predict_btn = st.button("🔍 Predict Disease", type="primary")
    with col3:
        st.write("")
        reset_btn = st.button("🔄 Reset", on_click=lambda: st.experimental_rerun())

    if predict_btn:
        if selected_symptoms:
            with st.spinner("Analyzing your symptoms..."):
                try:
                    predicted_disease, confidence = predict_disease_safe(
                        clf_model,
                        selected_symptoms,
                        symptoms,
                        le_clf
                    )
                    progression_stage = predict_progression(reg_model, predicted_disease, le_clf, scaler_reg)
                except Exception as exc:
                    st.error("Disease prediction failed. Please try again later.")
                    st.exception(exc)
                    predicted_disease = None
                    confidence = 0.0
                    progression_stage = np.nan

                if predicted_disease is not None:
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.success(f"**Predicted Disease:** {predicted_disease}")
                        st.info(f"**Confidence:** {confidence:.1f}%")
                    with col_right:
                        if np.isnan(progression_stage):
                            st.warning("**Progression Stage:** Unable to estimate reliably")
                        else:
                            st.warning(f"**Progression Stage:** {progression_stage:.2f} / 5.0")
                else:
                    st.warning("Prediction is not available at this time.")

                if predicted_disease in precautions:
                    st.subheader("🛡️ Precautions")
                    for i, prec in enumerate(precautions[predicted_disease], 1):
                        if pd.notna(prec) and prec:
                            st.write(f"{i}. {prec}")
                else:
                    st.warning("Please consult a healthcare professional for personalized advice.")
        else:
            st.error("Please select at least one symptom.")

with tab2:
    st.subheader("Top Feature Importance")
    if hasattr(clf_model, 'feature_importances_'):
        importances = clf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_symptoms = [symptoms[i] for i in indices][::-1]
        top_importances = importances[indices][::-1]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_symptoms, top_importances, color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        ax.grid(axis='x', alpha=0.2)
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")