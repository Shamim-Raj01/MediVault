from pathlib import Path
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
CLASSIFICATION_ACCURACY = 0.94
REGRESSION_R2 = 0.88

SPECIALIST_MAP = {
    "acne": "Dermatologist",
    "allergy": "Allergist / Immunologist",
    "arthritis": "Rheumatologist",
    "bronchial asthma": "Pulmonologist",
    "cervical spondylosis": "Orthopedic Specialist",
    "chicken pox": "General Physician",
    "common cold": "General Physician",
    "dengue": "Infectious Disease Specialist",
    "diabetes": "Endocrinologist",
    "dimorphic hemmorhoids(piles)": "Gastroenterologist",
    "drug reaction": "General Physician",
    "fungal infection": "Dermatologist",
    "gastroenteritis": "Gastroenterologist",
    "gerd": "Gastroenterologist",
    "heart attack": "Cardiologist",
    "hypertension": "Cardiologist",
    "hyperthyroidism": "Endocrinologist",
    "hypoglycemia": "Endocrinologist",
    "hypothyroidism": "Endocrinologist",
    "impetigo": "Dermatologist",
    "jaundice": "Hepatologist",
    "malaria": "Infectious Disease Specialist",
    "migraine": "Neurologist",
    "osteoarthristis": "Orthopedic Specialist",
    "paralysis (brain hemorrhage)": "Neurologist",
    "peptic ulcer diseae": "Gastroenterologist",
    "pneumonia": "Pulmonologist",
    "psoriasis": "Dermatologist",
    "tuberculosis": "Pulmonologist",
    "urinary tract infection": "Urologist",
    "varicose veins": "Vascular Surgeon",
    "(vertigo) paroymsal positional vertigo": "ENT Specialist",
}

DISEASE_INFO = {
    "acne": "A skin condition caused by blocked hair follicles, often leading to pimples or inflammation.",
    "allergy": "An immune response to a trigger such as food, pollen, medication, or environmental exposure.",
    "arthritis": "Inflammation of one or more joints that can cause pain, swelling, and stiffness.",
    "bronchial asthma": "A chronic airway condition that can cause wheezing, chest tightness, and breathing difficulty.",
    "chicken pox": "A contagious viral infection that commonly causes an itchy rash and fever.",
    "common cold": "A mild viral infection affecting the nose and throat, usually with sneezing and congestion.",
    "dengue": "A mosquito-borne viral illness that may cause fever, headache, muscle pain, and fatigue.",
    "diabetes": "A metabolic condition where the body has difficulty controlling blood sugar levels.",
    "drug reaction": "An unwanted reaction to medication that can involve skin, breathing, or systemic symptoms.",
    "fungal infection": "An infection caused by fungi, often affecting the skin, nails, or other moist areas.",
    "gastroenteritis": "Inflammation of the stomach and intestines that often causes diarrhea, vomiting, and cramps.",
    "gerd": "A digestive disorder where stomach acid frequently flows back into the food pipe.",
    "heart attack": "A medical emergency caused by reduced blood flow to the heart muscle.",
    "hypertension": "Persistently elevated blood pressure that can increase cardiovascular risk over time.",
    "jaundice": "Yellowing of the skin or eyes, often linked to liver, bile duct, or blood-related conditions.",
    "malaria": "A parasitic infection spread by mosquitoes that commonly causes fever, chills, and weakness.",
    "migraine": "A neurological headache disorder often associated with throbbing pain, nausea, or light sensitivity.",
    "pneumonia": "An infection of the lungs that can cause cough, fever, chest pain, and shortness of breath.",
    "psoriasis": "A chronic skin condition that causes rapid skin-cell buildup and scaly patches.",
    "tuberculosis": "A serious bacterial infection that often affects the lungs and may cause cough and weight loss.",
    "urinary tract infection": "An infection in the urinary system that may cause burning, urgency, or pelvic discomfort.",
}

GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}


@st.cache_data
def load_models():
    """Load trained models and preprocessors with file existence checks."""
    model_files = {
        "model.pkl": BASE_DIR / "model.pkl",
        "reg_model.pkl": BASE_DIR / "reg_model.pkl",
        "le_classification.pkl": BASE_DIR / "le_classification.pkl",
        "scaler.pkl": BASE_DIR / "scaler.pkl",
    }

    missing_files = [name for name, path in model_files.items() if not path.exists()]
    if missing_files:
        st.error("Model files not found. Please run the training notebook first.")
        st.write("Missing files:")
        for missing in missing_files:
            st.write(f"- {missing}")
        st.info("Run `fixed_project.ipynb` to generate the required model files.")
        st.stop()

    with open(model_files["model.pkl"], "rb") as f:
        clf_model = pickle.load(f)
    with open(model_files["reg_model.pkl"], "rb") as f:
        reg_model = pickle.load(f)
    with open(model_files["le_classification.pkl"], "rb") as f:
        le_clf = pickle.load(f)
    with open(model_files["scaler.pkl"], "rb") as f:
        scaler_reg = pickle.load(f)
    return clf_model, reg_model, le_clf, scaler_reg


@st.cache_data
def load_data():
    """Load dataset and precautions with robust column handling."""
    df = pd.read_csv(BASE_DIR / "healthcare-chatbot" / "Data" / "Training.csv")
    symptoms = [col for col in df.columns if col != "prognosis"]

    precautions_df = pd.read_csv(
        BASE_DIR / "healthcare-chatbot" / "MasterData" / "symptom_precaution.csv"
    )
    normalized_cols = {str(col).strip().lower(): col for col in precautions_df.columns}

    if "disease" in normalized_cols:
        disease_col = normalized_cols["disease"]
    elif "prognosis" in normalized_cols:
        disease_col = normalized_cols["prognosis"]
    else:
        disease_col = precautions_df.columns[0]

    precaution_cols = [col for col in precautions_df.columns if col != disease_col][:4]

    precautions = {}
    for _, row in precautions_df.iterrows():
        disease = str(row[disease_col]).strip()
        if not disease or disease.lower() == "nan":
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
    symptom_index = {symptom: idx for idx, symptom in enumerate(all_symptoms)}
    for symptom in symptoms_list:
        idx = symptom_index.get(symptom)
        if idx is not None:
            input_vector[idx] = 1
    return input_vector


def predict_disease(model, input_vector, le_clf):
    """Predict disease and confidence."""
    pred_encoded = model.predict([input_vector])[0]
    disease = le_clf.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba([input_vector])[0]
    confidence = float(np.max(proba) * 100)
    return disease, confidence


def get_top_predictions(model, input_vector, le_clf, top_n=3):
    """Return the top-N disease predictions with probabilities."""
    probabilities = model.predict_proba([input_vector])[0]
    top_indices = np.argsort(probabilities)[::-1][: min(top_n, len(probabilities))]
    return [
        (le_clf.inverse_transform([idx])[0], float(probabilities[idx] * 100))
        for idx in top_indices
    ]


def predict_progression(reg_model, disease, le_clf, scaler_reg, age=50, gender="Male"):
    """Predict progression stage using user-specific demographic inputs."""
    default_data = {
        "Age": age,
        "Gender": GENDER_MAP.get(gender, 1),
        "Disease": disease,
        "MedicalHistory": 1,
        "Lifestyle": 1,
        "BiomarkerScore": 0.5,
        "MedicationDose": 50.0,
        "HeartRate": 75,
        "BloodPressure_Systolic": 130,
        "BloodPressure_Diastolic": 85,
        "Cholesterol": 200,
        "BMI": 25.0,
        "SleepHours": 7.0,
        "StepsPerDay": 8000,
        "MedicationAdherence": 1,
        "StressLevel": 5,
        "CognitiveScore": 75,
        "MoodScore": 5,
        "Smoker": 0,
        "AlcoholUse": 1,
        "SupportSystem": 1,
        "HasCaregiver": 0,
        "EmploymentStatus": 1,
    }

    input_df = pd.DataFrame([default_data])

    try:
        if disease in le_clf.classes_:
            input_df["Disease"] = le_clf.transform([disease])[0]
        else:
            input_df["Disease"] = 0
    except Exception:
        input_df["Disease"] = 0

    feature_order = getattr(scaler_reg, "feature_names_in_", None)
    if feature_order is not None:
        input_df = input_df.reindex(columns=feature_order, fill_value=0)

    try:
        input_scaled = scaler_reg.transform(input_df)
        stage = float(reg_model.predict(input_scaled)[0])
    except Exception:
        stage = np.nan
    return stage


def get_stage_info(stage_value):
    """Convert the numeric progression estimate into a label and explanation."""
    if np.isnan(stage_value):
        return {
            "label": "Unavailable",
            "display_value": None,
            "explanation": "The progression model could not produce a reliable estimate for this prediction.",
        }

    clipped_stage = float(np.clip(stage_value, 0, 5))
    if clipped_stage < 1:
        label = "Very Mild"
        explanation = "Symptoms appear early or relatively limited. Continue monitoring and follow precautions closely."
    elif clipped_stage < 2:
        label = "Mild"
        explanation = "The pattern suggests a mild condition that still deserves attention and symptom tracking."
    elif clipped_stage < 3:
        label = "Moderate"
        explanation = "The condition may need timely medical guidance, especially if symptoms are worsening."
    elif clipped_stage < 4:
        label = "Severe"
        explanation = "The estimated progression is advanced and a clinical evaluation is strongly recommended."
    else:
        label = "Critical"
        explanation = "This estimate indicates potentially high severity. Seek urgent medical care as soon as possible."

    return {
        "label": label,
        "display_value": clipped_stage,
        "explanation": explanation,
    }


def normalize_key(value):
    return str(value).strip().lower()


def get_precautions_for_disease(disease, precautions):
    """Match precautions case-insensitively."""
    disease_key = normalize_key(disease)
    for precaution_disease, items in precautions.items():
        if normalize_key(precaution_disease) == disease_key:
            return items
    return []


def get_doctor_recommendation(disease):
    """Map predicted diseases to the most relevant specialist."""
    return SPECIALIST_MAP.get(normalize_key(disease), "General Physician")


def get_disease_description(disease):
    """Return a short description for the predicted disease."""
    return DISEASE_INFO.get(
        normalize_key(disease),
        "This condition may require a clinician's review for confirmation and the right treatment plan.",
    )


def build_hospital_link(disease, specialist):
    """Create a Google Maps search URL for nearby hospitals."""
    query = f"{specialist} hospital near me for {disease}"
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"


def get_confidence_message(confidence):
    """Explain the confidence level in plain language."""
    if confidence >= 80:
        return "The model sees a strong match between your selected symptoms and this prediction."
    if confidence >= 60:
        return "The prediction is reasonably strong, but overlapping conditions are still possible."
    return "The model is less certain here. Consider adding more symptoms or consulting a clinician."


def add_history_entry(history, disease, confidence, stage_label, doctor, symptom_count):
    """Append a compact prediction record to session history."""
    history.append(
        {
            "disease": disease,
            "confidence": round(confidence, 1),
            "stage": stage_label,
            "doctor": doctor,
            "symptom_count": symptom_count,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        }
    )


def reset_inputs():
    """Clear interactive inputs and rerun the app."""
    st.session_state["symptoms"] = []
    st.session_state["age"] = 35
    st.session_state["gender"] = "Male"
    st.rerun()


st.set_page_config(page_title="AI Healthcare Assistant", page_icon="🩺", layout="wide")

clf_model, reg_model, le_clf, scaler_reg = load_models()
symptoms, precautions = load_data()

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []
if "symptoms" not in st.session_state:
    st.session_state["symptoms"] = []
if "age" not in st.session_state:
    st.session_state["age"] = 35
if "gender" not in st.session_state:
    st.session_state["gender"] = "Male"

st.title("AI Healthcare Assistant")
st.caption(
    "Disease prediction, progression estimation, and basic care guidance for educational use."
)

st.sidebar.header("About")
st.sidebar.write(
    "This assistant predicts likely diseases from symptoms, estimates progression, and surfaces practical next steps."
)

st.sidebar.header("Model Performance")
col_a, col_b = st.sidebar.columns(2)
col_a.metric("Accuracy", f"{CLASSIFICATION_ACCURACY:.0%}")
col_b.metric("R^2", f"{REGRESSION_R2:.2f}")
st.sidebar.write(f"Classification model: Random Forest across {len(symptoms)} symptoms")
st.sidebar.write("Progression model: Random Forest Regressor")

st.sidebar.header("Prediction History")
history = st.session_state["prediction_history"]
if history:
    for item in reversed(history[-5:]):
        st.sidebar.markdown(
            f"**{item['disease']}**\n"
            f"{item['confidence']}% confidence | {item['stage']}\n"
            f"{item['doctor']} | {item['timestamp']}"
        )
else:
    st.sidebar.info("No predictions yet.")

st.sidebar.warning("This tool is educational and should not replace professional medical care.")

tab1, tab2 = st.tabs(["Prediction", "Insights"])

with tab1:
    st.subheader("Patient Inputs")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        age = st.slider("Age", min_value=1, max_value=100, value=st.session_state["age"], key="age")
    with info_col2:
        gender = st.selectbox("Gender", options=list(GENDER_MAP.keys()), key="gender")

    st.subheader("Symptoms")
    selected_symptoms = st.multiselect(
        "Choose the symptoms you are experiencing:",
        symptoms,
        key="symptoms",
    )

    if selected_symptoms:
        st.write("Selected symptoms:", ", ".join(selected_symptoms))
    else:
        st.info("Select symptoms from the list to begin prediction.")

    if 0 < len(selected_symptoms) < 2:
        st.warning("Selecting at least 2 symptoms usually produces more reliable results.")

    button_col1, button_col2, _ = st.columns([1, 1, 3])
    with button_col1:
        predict_btn = st.button("Predict Disease", type="primary")
    with button_col2:
        reset_btn = st.button("Reset")

    if reset_btn:
        reset_inputs()

    if predict_btn:
        if len(selected_symptoms) < 2:
            st.error("Please select at least 2 symptoms before prediction.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                input_vector = preprocess_input(selected_symptoms, symptoms)
                predicted_disease, confidence = predict_disease(clf_model, input_vector, le_clf)
                top_predictions = get_top_predictions(clf_model, input_vector, le_clf, top_n=3)
                progression_stage = predict_progression(
                    reg_model,
                    predicted_disease,
                    le_clf,
                    scaler_reg,
                    age=age,
                    gender=gender,
                )
                stage_info = get_stage_info(progression_stage)
                doctor = get_doctor_recommendation(predicted_disease)
                disease_description = get_disease_description(predicted_disease)
                disease_precautions = get_precautions_for_disease(predicted_disease, precautions)
                hospital_link = build_hospital_link(predicted_disease, doctor)

            add_history_entry(
                history,
                predicted_disease,
                confidence,
                stage_info["label"],
                doctor,
                len(selected_symptoms),
            )

            summary_col1, summary_col2 = st.columns([1.2, 1])
            with summary_col1:
                st.success(f"Predicted disease: {predicted_disease}")
                st.write(disease_description)
                st.info(f"Recommended specialist: {doctor}")
                st.markdown(f"[Find nearby hospitals on Google Maps]({hospital_link})")
            with summary_col2:
                if stage_info["display_value"] is None:
                    st.warning("Progression stage: Unavailable")
                else:
                    st.warning(
                        f"Progression stage: {stage_info['label']} ({stage_info['display_value']:.2f} / 5.00)"
                    )
                st.write(stage_info["explanation"])

            st.subheader("Confidence")
            st.progress(min(100, max(0, int(round(confidence)))))
            st.write(f"Prediction confidence: {confidence:.1f}%")
            st.caption(get_confidence_message(confidence))

            # Highlight the most influential symptoms used by the classifier overall.
            if hasattr(clf_model, "feature_importances_"):
                top_feature_indices = np.argsort(clf_model.feature_importances_)[::-1][:5]
                top_feature_names = [symptoms[idx] for idx in top_feature_indices]
                st.subheader("🧠 Why this prediction?")
                st.write(", ".join(top_feature_names))

            st.subheader("Top 3 likely diseases")
            top_predictions_df = pd.DataFrame(
                top_predictions,
                columns=["Disease", "Probability (%)"],
            )
            top_predictions_df["Probability (%)"] = top_predictions_df["Probability (%)"].map(
                lambda value: f"{value:.1f}"
            )
            st.dataframe(top_predictions_df, use_container_width=True, hide_index=True)

            st.subheader("Precautions")
            if disease_precautions:
                for idx, precaution in enumerate(disease_precautions, start=1):
                    st.write(f"{idx}. {precaution}")
            else:
                st.warning("No precaution list was found for this disease. Please consult a healthcare professional.")

with tab2:
    st.subheader("Model Insights")
    st.write(
        "The chart below shows which symptoms contribute most to the classifier's decisions overall."
    )

    if hasattr(clf_model, "feature_importances_"):
        importances = clf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_symptoms = [symptoms[i] for i in indices][::-1]
        top_importances = importances[indices][::-1]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_symptoms, top_importances, color="#6baed6")
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Feature Importances")
        ax.grid(axis="x", alpha=0.2)
        st.pyplot(fig)
    else:
        st.info("Feature importance is not available for this model.")

    st.subheader("How to interpret predictions")
    st.write(
        "The disease prediction comes from the classification model, while the progression estimate is a separate regression output informed by the disease plus baseline patient factors."
    )
