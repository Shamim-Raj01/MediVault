import os
from pathlib import Path
import time

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from inference_improved_model import ImprovedModelInference
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="MediVault AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

def get_top_predictions(model, selected_symptoms, all_symptoms, le_clf, top_k=3):
    """Get top-K predictions with probabilities."""
    if isinstance(model, ImprovedModelInference):
        return model.get_top_predictions(selected_symptoms, all_symptoms, top_k)
    else:
        # For legacy model, approximate top predictions
        input_vector = preprocess_input(selected_symptoms, all_symptoms)
        probas = model.predict_proba([input_vector])[0]
        top_indices = np.argsort(probas)[::-1][:top_k]
        results = []
        for idx in top_indices:
            disease = le_clf.inverse_transform([idx])[0]
            prob = float(probas[idx])
            results.append((disease, prob))
        return results

def calculate_health_risk_score(confidence):
    """Calculate health risk score based on confidence."""
    # Higher confidence = lower risk (more certain diagnosis)
    risk_score = max(0, 100 - confidence)
    if risk_score < 30:
        level = "Low Risk"
    elif risk_score < 70:
        level = "Medium Risk"
    else:
        level = "High Risk"
    return risk_score, level

def generate_ai_reasoning(selected_symptoms, predicted_disease):
    """Generate human-readable AI reasoning."""
    symptom_text = ", ".join(selected_symptoms[:3])  # Show first 3
    if len(selected_symptoms) > 3:
        symptom_text += f" and {len(selected_symptoms) - 3} more"
    return f"Based on your symptoms ({symptom_text}), the AI model identified patterns consistent with {predicted_disease}."

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

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 100%);
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1f77b4 0%, #0056b3 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(31, 119, 180, 0.3);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .diagnosis-card {
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.8) 0%, rgba(0, 86, 179, 0.8) 100%);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 12px 40px rgba(31, 119, 180, 0.4);
        text-align: center;
    }
    .metric-card {
        background: rgba(46, 46, 46, 0.8);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 25px;
        color: #888;
        font-size: 14px;
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(5px);
        border-radius: 10px;
        margin-top: 30px;
    }
    .empty-state {
        text-align: center;
        padding: 50px;
        color: #cccccc;
        font-size: 18px;
    }
    .progress-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.2);
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #1f77b4, #00d4aa);
        border-radius: 4px;
        transition: width 1s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color: #1f77b4; font-size: 48px; margin: 0;'>🩺 MediVault AI</h1>
    <p style='color: #cccccc; font-size: 18px; margin: 5px 0;'>Smart Disease Prediction System</p>
</div>
""", unsafe_allow_html=True)

# Navigation Menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Prediction", "Insights", "About"],
    icons=["house", "search", "bar-chart", "info-circle"],
    menu_icon="cast",
    default_index=1,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0e1117"},
        "icon": {"color": "#1f77b4", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#2e2e2e",
        },
        "nav-link-selected": {"background-color": "#1f77b4"},
    }
)

# Content based on navigation
if selected == "Home":
    st.markdown("""
    <div class='glass-card'>
        <h2 style='text-align: center; color: #1f77b4;'>Welcome to MediVault AI</h2>
        <p style='text-align: center; font-size: 18px; line-height: 1.6;'>
            Our advanced AI system helps predict diseases based on your symptoms and provides valuable insights for better health management.
        </p>
        <div style='text-align: center; margin-top: 30px;'>
            <p style='font-size: 16px; color: #cccccc;'>Navigate using the menu above to start predicting or explore insights.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif selected == "Prediction":
    st.markdown("## 🔮 Disease Prediction")
    
    with st.expander("🧾 Select Your Symptoms", expanded=True):
        selected_symptoms = st.multiselect("Choose symptoms you are experiencing:", symptoms, key="symptoms")
        
        if selected_symptoms:
            st.markdown("**Selected Symptoms:**")
            st.write(", ".join(selected_symptoms))
        else:
            st.markdown("""
            <div class='empty-state'>
                <h3>🩺 Start Your Diagnosis</h3>
                <p>Select symptoms from the list above to begin your personalized health analysis.</p>
            </div>
            """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        predict_btn = st.button("🔍 Predict Disease", type="primary")
    with col3:
        reset_btn = st.button("🔄 Reset", on_click=lambda: st.experimental_rerun())
    
    if predict_btn and selected_symptoms:
        with st.spinner("🧠 AI is analyzing your symptoms..."):
            time.sleep(0.8)  # Add slight delay for realism
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
                # Personalized greeting
                st.markdown(f"### 👋 Here's your personalized health analysis, based on {len(selected_symptoms)} symptom{'s' if len(selected_symptoms) > 1 else ''}")
                
                # Diagnosis Summary Card
                st.markdown(f"""
                <div class='diagnosis-card'>
                    <h2 style='color: white; margin: 0; font-size: 32px;'>🩺 {predicted_disease}</h2>
                    <div style='display: flex; justify-content: space-around; margin-top: 20px;'>
                        <div>
                            <h3 style='color: #00d4aa; margin: 0;'>📊 Confidence</h3>
                            <p style='font-size: 24px; margin: 5px 0; color: white;'>{confidence:.1f}%</p>
                        </div>
                        <div>
                            <h3 style='color: #ff6b6b; margin: 0;'>📈 Stage</h3>
                            <p style='font-size: 24px; margin: 5px 0; color: white;'>{progression_stage:.2f}/5.0</p>
                        </div>
                        <div>
                            <h3 style='color: #ffd93d; margin: 0;'>⚕️ Risk</h3>
                            <p style='font-size: 18px; margin: 5px 0; color: white;'>{calculate_health_risk_score(confidence)[1]}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Animated Confidence Bar
                st.markdown("### 📊 Confidence Level")
                progress_html = f"""
                <div class='progress-bar'>
                    <div class='progress-fill' style='width: {confidence}%;'></div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
                
                # AI Reasoning Section
                st.markdown("### 🤖 AI Reasoning")
                st.markdown(f"""
                <div class='glass-card'>
                    <p style='font-size: 16px; line-height: 1.6;'>{generate_ai_reasoning(selected_symptoms, predicted_disease)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Next Steps Section
                st.markdown("### 📌 Recommended Next Steps")
                st.markdown("""
                <div class='glass-card'>
                    <ul style='font-size: 16px; line-height: 1.8;'>
                        <li><strong>Consult a healthcare professional</strong> for accurate diagnosis and treatment</li>
                        <li><strong>Monitor your symptoms</strong> and track any changes</li>
                        <li><strong>Follow the precautions below</strong> to manage your condition</li>
                        <li><strong>Stay hydrated and rest</strong> if experiencing fatigue or weakness</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Precautions
                if predicted_disease in precautions:
                    st.markdown("### 🛡️ Precautions")
                    for i, prec in enumerate(precautions[predicted_disease], 1):
                        if pd.notna(prec) and prec:
                            st.write(f"{i}. {prec}")
                else:
                    st.warning("Please consult a healthcare professional for personalized advice.")
            else:
                st.warning("Prediction is not available at this time.")
    elif predict_btn:
        st.error("Please select at least one symptom.")

elif selected == "Insights":
    st.markdown("## 📈 Model Insights")
    
    # Top 3 Predictions (if symptoms selected)
    if st.session_state.get("symptoms"):
        selected_symptoms = st.session_state["symptoms"]
        if selected_symptoms:
            st.markdown("### 🏆 Top 3 Disease Predictions")
            try:
                top_predictions = get_top_predictions(clf_model, selected_symptoms, symptoms, le_clf, 3)
                diseases = [pred[0] for pred in top_predictions]
                probs = [pred[1] * 100 for pred in top_predictions]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(diseases[::-1], probs[::-1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax.set_xlabel('Probability (%)')
                ax.set_title('Top 3 Disease Predictions')
                ax.grid(axis='x', alpha=0.2)
                for bar, prob in zip(bars, probs[::-1]):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{prob:.1f}%', 
                           ha='left', va='center', fontweight='bold')
                st.pyplot(fig)
            except Exception:
                st.info("Top predictions not available for this model configuration.")
    
    # Feature Importance
    if hasattr(clf_model, 'feature_importances_'):
        st.markdown("### 📊 Feature Importance")
        importances = clf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_symptoms = [symptoms[i] for i in indices][::-1]
        top_importances = importances[indices][::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_symptoms, top_importances, color='#1f77b4')
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        ax.grid(axis='x', alpha=0.2)
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")
    
    # Model Statistics
    st.markdown("### 📊 Model Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Symptoms", len(symptoms))
    with col2:
        st.metric("Model Type", "Random Forest" if not isinstance(clf_model, ImprovedModelInference) else "XGBoost")

elif selected == "About":
    st.markdown("## ℹ️ About MediVault AI")
    st.markdown("""
    <div class='glass-card'>
        <h3>How It Works</h3>
        <p>This AI system predicts diseases based on symptoms using advanced machine learning models and estimates disease progression stages.</p>
        <ol>
            <li>Select symptoms you're experiencing</li>
            <li>Click 'Predict Disease' to get results</li>
            <li>View predicted disease, confidence, progression stage, and precautions</li>
        </ol>
    </div>
    <div class='glass-card'>
        <h3>Model Information</h3>
        <ul>
            <li><strong>Classification:</strong> Random Forest or XGBoost ({len(symptoms)} symptoms)</li>
            <li><strong>Regression:</strong> Random Forest Regressor</li>
            <li><strong>Features:</strong> Symptom-based prediction with confidence scoring</li>
        </ul>
    </div>
    <div class='glass-card'>
        <h3>Important Notice</h3>
        <p style='color: #ff6b6b;'>⚠️ This system is for educational purposes only. Always consult a healthcare professional for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    Built with ❤️ using Machine Learning & Streamlit | 👨‍⚕️ Consult a doctor for professional medical advice
</div>
""", unsafe_allow_html=True)