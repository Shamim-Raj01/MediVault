import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

MIN_SYMPTOMS_REQUIRED = 3
CONFIDENCE_WARNING_THRESHOLD = 0.4
FOLLOW_UP_SUGGESTION_COUNT = 6
CRITICAL_SYMPTOM_WEIGHTS = {
    "fever": 1.2,
    "breathlessness": 1.5,
    "shortness_of_breath": 1.5,
    "difficulty_breathing": 1.5,
    "difficulty_in_breathing": 1.5,
    "chest_pain": 1.5,
    "sharp_chest_pain": 1.5,
    "chest_tightness": 1.4,
    "irregular_heartbeat": 1.4,
    "palpitations": 1.3,
    "vomiting_blood": 1.6,
    "blood_in_stool": 1.5,
    "blood_in_urine": 1.5,
    "seizures": 1.6,
    "fainting": 1.5,
    "weakness": 1.2,
    "focal_weakness": 1.4,
    "yellowish_skin": 1.3,
}
SUPPORT_DATA_DIR = Path("healthcare-chatbot") / "Data" / "Support"
MEDICAL_INFO_FALLBACK = "Information not available"


def format_symptom(symptom):
    return symptom.replace("_", " ").title()


def format_disease_name(disease):
    return disease.replace("_", " ").title()


def normalize_disease_key(disease):
    return " ".join(str(disease).replace("_", " ").strip().lower().split())


def parse_support_list(value):
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [item.strip() for item in text.split(",") if item.strip()]


@st.cache_data
def load_medical_data():
    support_data = {
        "descriptions": {},
        "precautions": {},
        "diets": {},
        "medications": {},
        "workouts": {},
    }

    description_path = SUPPORT_DATA_DIR / "description.csv"
    precautions_path = SUPPORT_DATA_DIR / "precautions.csv"
    diets_path = SUPPORT_DATA_DIR / "diets.csv"
    medications_path = SUPPORT_DATA_DIR / "medications.csv"
    workout_path = SUPPORT_DATA_DIR / "workout.csv"

    if description_path.exists():
        df = pd.read_csv(description_path)
        for _, row in df.iterrows():
            disease_key = normalize_disease_key(row.get("Disease", ""))
            if disease_key:
                support_data["descriptions"][disease_key] = str(row.get("Description", "")).strip()

    if precautions_path.exists():
        df = pd.read_csv(precautions_path)
        precaution_columns = [col for col in df.columns if col.lower().startswith("precaution")]
        for _, row in df.iterrows():
            disease_key = normalize_disease_key(row.get("Disease", ""))
            if disease_key:
                support_data["precautions"][disease_key] = [
                    str(row[col]).strip()
                    for col in precaution_columns
                    if pd.notna(row.get(col)) and str(row.get(col)).strip()
                ]

    if diets_path.exists():
        df = pd.read_csv(diets_path)
        for _, row in df.iterrows():
            disease_key = normalize_disease_key(row.get("Disease", ""))
            if disease_key:
                support_data["diets"][disease_key] = parse_support_list(row.get("Diet", ""))

    if medications_path.exists():
        df = pd.read_csv(medications_path)
        for _, row in df.iterrows():
            disease_key = normalize_disease_key(row.get("Disease", ""))
            if disease_key:
                support_data["medications"][disease_key] = parse_support_list(row.get("Medication", ""))

    if workout_path.exists():
        df = pd.read_csv(workout_path)
        for _, row in df.iterrows():
            disease_key = normalize_disease_key(row.get("Disease", ""))
            if disease_key:
                support_data["workouts"][disease_key] = parse_support_list(row.get("Workouts", ""))

    return support_data


def get_disease_support_info(disease_name, medical_data):
    disease_key = normalize_disease_key(disease_name)
    return {
        "description": medical_data["descriptions"].get(disease_key, MEDICAL_INFO_FALLBACK),
        "precautions": medical_data["precautions"].get(disease_key, []),
        "diets": medical_data["diets"].get(disease_key, []),
        "medications": medical_data["medications"].get(disease_key, []),
        "workouts": medical_data["workouts"].get(disease_key, []),
    }


def render_medical_list(items):
    if not items:
        st.write(MEDICAL_INFO_FALLBACK)
        return

    for item in items:
        cleaned_item = str(item).replace("_", " ").strip()
        st.markdown(f"- {cleaned_item}")


def build_symptom_mapping(feature_names):
    display_to_original = {}
    for feature in feature_names:
        display_name = format_symptom(feature)
        display_to_original[display_name] = feature
    return display_to_original


def preprocess_input(selected_display_names, model_feature_names, display_to_original):
    selected_original = [
        display_to_original[name]
        for name in selected_display_names
        if name in display_to_original
    ]

    input_vector = np.zeros(len(model_feature_names), dtype=np.float32)
    feature_index = {feature: idx for idx, feature in enumerate(model_feature_names)}
    selected_set = set(selected_original)

    for symptom in selected_original:
        if symptom in feature_index:
            input_vector[feature_index[symptom]] = CRITICAL_SYMPTOM_WEIGHTS.get(symptom, 1.0)

    for feature_name, idx in feature_index.items():
        if "_and_" in feature_name:
            left, right = feature_name.split("_and_", 1)
        elif "_x_" in feature_name:
            left, right = feature_name.split("_x_", 1)
        else:
            continue

        if left in selected_set and right in selected_set:
            left_weight = CRITICAL_SYMPTOM_WEIGHTS.get(left, 1.0)
            right_weight = CRITICAL_SYMPTOM_WEIGHTS.get(right, 1.0)
            input_vector[idx] = left_weight * right_weight

    input_vector = input_vector.reshape(1, -1)
    print(f"[DEBUG] Selected symptoms count: {len(selected_set)}")
    print(f"[DEBUG] Input shape: {input_vector.shape}")
    return input_vector


def extract_top_predictions(probabilities, label_encoder, top_k=3):
    probs = probabilities[0]
    top_indices = np.argsort(probs)[-top_k:][::-1]

    predictions = []
    for idx in top_indices:
        disease = format_disease_name(label_encoder.inverse_transform([idx])[0])
        predictions.append((disease, float(probs[idx])))
    return predictions


def render_top_predictions(top_predictions):
    st.subheader("Top 3 Predictions")
    for disease, prob in top_predictions:
        st.write(f"**{disease}** - {prob * 100:.2f}%")
        st.progress(prob)


def render_prediction_comparison(previous_predictions, current_predictions):
    st.subheader("Before vs After Predictions")
    previous_map = {disease: prob for disease, prob in previous_predictions}
    current_map = {disease: prob for disease, prob in current_predictions}
    ordered_diseases = list(dict.fromkeys(
        [disease for disease, _ in previous_predictions] +
        [disease for disease, _ in current_predictions]
    ))

    for disease in ordered_diseases:
        before_prob = previous_map.get(disease, 0.0)
        after_prob = current_map.get(disease, 0.0)
        delta = after_prob - before_prob
        delta_color = "#34d399" if delta > 0 else "#f87171" if delta < 0 else "#cbd5e1"
        delta_label = f"{delta * 100:+.2f}%"

        st.markdown(
            f"""
            <div style="
                display: grid;
                grid-template-columns: minmax(180px, 1.5fr) 1fr 1fr auto;
                gap: 0.75rem;
                align-items: center;
                background: rgba(8, 16, 26, 0.38);
                border: 1px solid rgba(143, 227, 255, 0.10);
                border-radius: 14px;
                padding: 0.75rem 0.9rem;
                margin-bottom: 0.45rem;
            ">
                <div style="color: #f4fbff; font-weight: 600;">{disease}</div>
                <div style="color: #b7c9d6;">Before: {before_prob * 100:.2f}%</div>
                <div style="color: #b7c9d6;">After: {after_prob * 100:.2f}%</div>
                <div style="color: {delta_color}; font-weight: 700;">{delta_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_medical_context(predicted_disease, medical_data):
    support_info = get_disease_support_info(predicted_disease, medical_data)

    st.write("")
    st.subheader("Medical Context")

    about_col, precautions_col = st.columns(2)
    with about_col:
        st.markdown("#### About the Disease")
        st.write(support_info["description"] or MEDICAL_INFO_FALLBACK)
    with precautions_col:
        st.markdown("#### Precautions")
        render_medical_list(support_info["precautions"])

    diet_col, meds_col = st.columns(2)
    with diet_col:
        st.markdown("#### Recommended Diet")
        render_medical_list(support_info["diets"])
    with meds_col:
        st.markdown("#### Medications (General Info)")
        render_medical_list(support_info["medications"])

    st.markdown("#### Lifestyle / Exercise")
    render_medical_list(support_info["workouts"])
    st.warning("This is not medical advice. Please consult a doctor.")


def render_confidence_chart(confidence):
    labels = ["Confidence", "Remaining"]
    sizes = [confidence, 1 - confidence]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#31c5ff", "#1f3340"],
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)


def build_confidence_warning(confidence):
    if confidence < CONFIDENCE_WARNING_THRESHOLD:
        return (
            "Low-confidence prediction. Add a few more distinguishing symptoms before "
            "treating this ranking as useful triage guidance."
        )
    return None


def suggest_follow_up_symptoms(selected_display_names, model, model_feature_names, display_to_original):
    selected_original = {
        display_to_original[name]
        for name in selected_display_names
        if name in display_to_original
    }
    base_features = [name for name in model_feature_names if "_and_" not in name and "_x_" not in name]
    if not selected_original:
        return [format_symptom(name) for name in base_features[:FOLLOW_UP_SUGGESTION_COUNT]]

    base_model = getattr(model, "estimator", model)
    importance = getattr(base_model, "feature_importances_", None)

    if importance is None:
        calibrated_models = getattr(model, "calibrated_classifiers_", [])
        if calibrated_models:
            inner_model = getattr(calibrated_models[0], "estimator", None)
            importance = getattr(inner_model, "feature_importances_", None)

    candidate_scores = {}
    for feature_name in model_feature_names:
        if "_and_" in feature_name:
            left, right = feature_name.split("_and_", 1)
        elif "_x_" in feature_name:
            left, right = feature_name.split("_x_", 1)
        else:
            continue

        if left in selected_original and right not in selected_original:
            candidate_scores[right] = candidate_scores.get(right, 0.0) + 2.0
        if right in selected_original and left not in selected_original:
            candidate_scores[left] = candidate_scores.get(left, 0.0) + 2.0

    if importance is not None:
        usable = min(len(model_feature_names), len(importance))
        for feature_name, score in zip(model_feature_names[:usable], importance[:usable]):
            if feature_name in selected_original:
                continue
            if "_and_" in feature_name or "_x_" in feature_name:
                continue
            candidate_scores[feature_name] = candidate_scores.get(feature_name, 0.0) + float(score)

    ranked_candidates = [
        feature for feature, _ in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        if feature in display_to_original.values()
    ]

    if len(ranked_candidates) < FOLLOW_UP_SUGGESTION_COUNT:
        for feature_name in base_features:
            if feature_name in selected_original or feature_name in ranked_candidates:
                continue
            ranked_candidates.append(feature_name)
            if len(ranked_candidates) >= FOLLOW_UP_SUGGESTION_COUNT:
                break

    return [format_symptom(name) for name in ranked_candidates[:FOLLOW_UP_SUGGESTION_COUNT]]


def get_feature_importance_frame(model, model_feature_names):
    base_model = getattr(model, "estimator", model)
    importance = getattr(base_model, "feature_importances_", None)

    if importance is None:
        calibrated_models = getattr(model, "calibrated_classifiers_", [])
        if calibrated_models:
            inner_model = getattr(calibrated_models[0], "estimator", None)
            importance = getattr(inner_model, "feature_importances_", None)

    if importance is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    usable = min(len(model_feature_names), len(importance))
    importance_df = pd.DataFrame(
        {
            "Feature": model_feature_names[:usable],
            "Importance": importance[:usable],
        }
    ).sort_values(by="Importance", ascending=False).head(5)

    importance_df["Feature"] = (
        importance_df["Feature"].str.replace("_", " ", regex=False).str.title()
    )
    return importance_df


def render(context):
    model = context["model"]
    label_encoder = context["label_encoder"]
    feature_names = context["feature_names"]
    model_feature_names = context["model_feature_names"]
    expected_feature_count = context["expected_feature_count"]
    medical_data = load_medical_data()

    display_to_original = build_symptom_mapping(feature_names)
    display_features = sorted(display_to_original.keys())
    selected_key = "prediction_selected_symptoms"
    follow_up_key = "prediction_follow_up_symptoms"
    previous_result_key = "prediction_previous_result"

    if selected_key not in st.session_state:
        st.session_state[selected_key] = []
    if follow_up_key not in st.session_state:
        st.session_state[follow_up_key] = []
    if previous_result_key not in st.session_state:
        st.session_state[previous_result_key] = None

    st.title("🩺 AI Disease Prediction")
    st.caption("Select symptoms to predict possible diseases using the trained machine learning model.")
    st.write("")

    summary_col, stats_col = st.columns([3, 2])
    with summary_col:
        st.markdown(
            "Choose one or more symptoms below. The app converts the clean labels back to "
            "the original model feature names automatically before prediction."
        )
    with stats_col:
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Symptoms", len(feature_names))
        metric_b.metric("Model Inputs", expected_feature_count)

    st.write("")

    with st.container(border=True):
        selected = st.multiselect(
            "Search and Select Symptoms",
            display_features,
            default=st.session_state[selected_key],
            placeholder="Type to search symptoms...",
            help="You can search by typing part of a symptom name.",
        )

    st.write("")
    button_col, reset_col = st.columns([1, 1])
    with button_col:
        predict_clicked = st.button("Predict Disease", type="primary", use_container_width=True)
    with reset_col:
        if st.button("Reset Selection", use_container_width=True):
            st.session_state[selected_key] = []
            st.session_state[follow_up_key] = []
            st.session_state[previous_result_key] = None
            st.rerun()

    st.session_state[selected_key] = selected

    if not selected and not predict_clicked:
        st.info("Select symptoms to generate a ranked prediction with confidence scores.")
        return

    if not selected and predict_clicked:
        st.warning("Please select at least one symptom before predicting.")
        return

    if len(selected) < MIN_SYMPTOMS_REQUIRED:
        st.warning(
            f"Please select at least {MIN_SYMPTOMS_REQUIRED} symptoms before predicting. "
            "This helps reduce ambiguous real-world results."
        )
        return

    input_vector = preprocess_input(selected, model_feature_names, display_to_original)
    print(f"[DEBUG] Expected feature count: {expected_feature_count}")
    print(f"[DEBUG] Actual input feature count: {input_vector.shape[1]}")

    if input_vector.shape[1] != expected_feature_count:
        st.error(
            f"Feature mismatch: model expects {expected_feature_count} features, "
            f"but app built {input_vector.shape[1]}."
        )
        return

    if predict_clicked:
        try:
            with st.spinner("Analyzing symptoms..."):
                prediction = model.predict(input_vector)
                probabilities = model.predict_proba(input_vector)

            predicted_disease = format_disease_name(label_encoder.inverse_transform(prediction)[0])
            confidence = float(np.max(probabilities))
            top_predictions = extract_top_predictions(probabilities, label_encoder)
            confidence_warning = build_confidence_warning(confidence)
            follow_up_suggestions = suggest_follow_up_symptoms(
                selected,
                model,
                model_feature_names,
                display_to_original
            )
            st.session_state[follow_up_key] = follow_up_suggestions
            previous_result = st.session_state[previous_result_key]

            left_col, right_col = st.columns([1.25, 1])

            with left_col:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(49,197,255,0.18), rgba(9,32,46,0.88));
                        border: 1px solid rgba(143,227,255,0.22);
                        border-radius: 18px;
                        padding: 1rem 1.1rem;
                        margin-bottom: 1rem;
                    ">
                        <div style="
                            font-size: 0.78rem;
                            letter-spacing: 0.12em;
                            text-transform: uppercase;
                            color: #8fe3ff;
                            margin-bottom: 0.35rem;
                        ">
                            Primary Prediction
                        </div>
                        <div style="
                            font-size: 1.55rem;
                            font-weight: 700;
                            color: #f4fbff;
                        ">
                            {predicted_disease}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.info(f"Confidence: {confidence * 100:.2f}%")
                if previous_result is not None:
                    previous_confidence = previous_result["confidence"]
                    confidence_delta = confidence - previous_confidence
                    if confidence_delta >= 0:
                        st.success(
                            f"Confidence improved from {previous_confidence * 100:.2f}% "
                            f"to {confidence * 100:.2f}%."
                        )
                    else:
                        st.error(
                            f"Confidence changed from {previous_confidence * 100:.2f}% "
                            f"to {confidence * 100:.2f}%."
                        )
                if confidence_warning:
                    st.warning(confidence_warning)
                else:
                    st.success("Confidence is above the review threshold for this symptom set.")
                st.caption("This is decision support, not a medical diagnosis.")
                render_top_predictions(top_predictions)

            with right_col:
                st.subheader("Confidence Breakdown")
                render_confidence_chart(confidence)

            if previous_result is not None:
                st.write("")
                render_prediction_comparison(previous_result["top_predictions"], top_predictions)

            importance_df = get_feature_importance_frame(model, model_feature_names)
            st.write("")
            st.subheader("Top Influencing Symptoms")
            if importance_df.empty:
                st.info("Feature importance is not available for the loaded model.")
            else:
                st.bar_chart(importance_df.set_index("Feature"), use_container_width=True)

            render_medical_context(predicted_disease, medical_data)

            st.write("")
            st.subheader("Refine With Additional Symptoms")
            st.write(
                "These follow-up symptoms are suggested to help separate similar conditions and improve practical accuracy."
            )
            suggested_follow_ups = st.multiselect(
                "Suggested follow-up symptoms",
                st.session_state[follow_up_key],
                default=[],
                key="follow_up_selector",
                help="Choose any symptoms that also apply, then run prediction again.",
            )
            if suggested_follow_ups:
                combined_selection = list(dict.fromkeys(selected + suggested_follow_ups))
                st.session_state[selected_key] = combined_selection
                st.info(
                    f"Added {len(suggested_follow_ups)} follow-up symptom(s). "
                    "Click Predict Disease again to refine the ranking."
                )

            st.caption(
                "Use this result as decision support only. Please consult a qualified medical professional for diagnosis."
            )
            st.session_state[previous_result_key] = {
                "confidence": confidence,
                "top_predictions": top_predictions,
                "selected_symptoms": selected.copy(),
            }
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
