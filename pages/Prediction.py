import ast
import re
import json
import numpy as np
import os
import pandas as pd
import streamlit as st
from difflib import get_close_matches
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    genai = None

MIN_SYMPTOMS_REQUIRED = 3
CONFIDENCE_WARNING_THRESHOLD = 0.4
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
ALIASES_PATH = Path("aliases_generated.json")
MEDICAL_INFO_FALLBACK = "Information not available"
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-pro")
FAST_MODE = os.getenv("FAST_MODE", "true").lower() == "true"
SHAP_MAX_FEATURES = 10
SHAP_SUMMARY_FEATURE_LIMIT = 250
MEDICAL_FIELDS = ["description", "precautions", "diet", "medications", "lifestyle"]
DISEASE_ALIASES = {
    "flu": "influenza",
    "cold": "common cold",
    "high blood pressure": "hypertension",
    "bp high": "hypertension",
    "sugar": "diabetes",
    "high sugar": "diabetes",
    "low sugar": "hypoglycemia",
    "heart attack": "myocardial infarction",
    "stroke": "cerebrovascular accident",
    "thyroid": "thyroid disorder",
    "acidity": "gastroesophageal reflux disease",
    "gas": "gastroesophageal reflux disease",
    "stomach infection": "gastroenteritis",
    "food poisoning": "gastroenteritis",
    "lung infection": "pneumonia",
    "chest infection": "pneumonia",
}
NOISE_WORDS = {"disease", "disorder", "condition", "syndrome"}


def format_symptom(symptom):
    return symptom.replace("_", " ").title()


def format_disease_name(disease):
    return disease.replace("_", " ").title()


def get_matplotlib():
    import matplotlib.pyplot as plt
    return plt


def normalize(text):
    if not text:
        return ""
    text = str(text).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for token in text.split():
        if token in NOISE_WORDS:
            continue
        if token.endswith("infections"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens).strip()


def normalize_disease_key(disease):
    return normalize(disease)


def singularize_token(token):
    if token.endswith("ies") and len(token) > 3:
        return f"{token[:-3]}y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def build_variant_candidates(canonical_disease):
    normalized = normalize(canonical_disease)
    if not normalized:
        return set()

    raw_tokens = [singularize_token(token) for token in normalized.split() if token]
    meaningful_tokens = [token for token in raw_tokens if token not in NOISE_WORDS]
    base_tokens = meaningful_tokens or raw_tokens
    variants = set()

    def add_variant(tokens):
        cleaned = normalize(" ".join(tokens))
        if cleaned:
            variants.add(cleaned)

    add_variant(base_tokens)

    for start in range(len(base_tokens)):
        for end in range(start + 1, len(base_tokens) + 1):
            add_variant(base_tokens[start:end])

    for idx, token in enumerate(base_tokens):
        remaining = base_tokens[:idx] + base_tokens[idx + 1:]
        add_variant([token] + remaining)

    if len(base_tokens) > 1:
        add_variant(base_tokens[1:])
        add_variant(base_tokens[:-1])

    return variants


def load_persisted_aliases():
    if not ALIASES_PATH.exists():
        return {}

    try:
        with ALIASES_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return {
                normalize(key): normalize(value)
                for key, value in data.items()
                if normalize(key) and normalize(value)
            }
    except Exception:
        return {}

    return {}


@st.cache_resource
def build_alias_map(medical_store):
    canonical = list(medical_store.keys())
    auto_aliases = load_persisted_aliases()

    for canonical_disease in canonical:
        normalized_canonical = normalize(canonical_disease)
        if not normalized_canonical:
            continue
        auto_aliases.setdefault(normalized_canonical, normalized_canonical)
        for variant in build_variant_candidates(normalized_canonical):
            auto_aliases.setdefault(variant, normalized_canonical)

    final_aliases = {**auto_aliases, **DISEASE_ALIASES}

    with ALIASES_PATH.open("w", encoding="utf-8") as file:
        json.dump(final_aliases, file, indent=2, sort_keys=True)

    return final_aliases


def apply_disease_alias(disease, alias_map):
    normalized = normalize(disease)
    alias_value = alias_map.get(normalized)
    if alias_value:
        return normalize(alias_value), normalized
    return normalized, None


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


def create_empty_medical_entry():
    return {
        "description": "",
        "precautions": [],
        "diet": [],
        "medications": [],
        "lifestyle": [],
    }


def merge_medical_value(existing_value, new_value):
    if isinstance(existing_value, list):
        if existing_value:
            return existing_value
        return new_value if isinstance(new_value, list) else []

    if existing_value:
        return existing_value
    return new_value if isinstance(new_value, str) else ""


def update_medical_store_entry(medical_store, disease_name, field, value):
    disease_key = normalize(disease_name)
    if not disease_key:
        return

    entry = medical_store.setdefault(disease_key, create_empty_medical_entry())
    entry[field] = merge_medical_value(entry[field], value)


@st.cache_resource
def load_medical_store():
    medical_store = {}
    description_path = SUPPORT_DATA_DIR / "description.csv"
    precautions_path = SUPPORT_DATA_DIR / "precautions.csv"
    diets_path = SUPPORT_DATA_DIR / "diets.csv"
    medications_path = SUPPORT_DATA_DIR / "medications.csv"
    workout_path = SUPPORT_DATA_DIR / "workout.csv"

    if description_path.exists():
        df = pd.read_csv(description_path)
        for _, row in df.iterrows():
            update_medical_store_entry(
                medical_store,
                row.get("Disease", ""),
                "description",
                str(row.get("Description", "")).strip(),
            )

    if precautions_path.exists():
        df = pd.read_csv(precautions_path)
        precaution_columns = [col for col in df.columns if col.lower().startswith("precaution")]
        for _, row in df.iterrows():
            update_medical_store_entry(
                medical_store,
                row.get("Disease", ""),
                "precautions",
                [
                    str(row[col]).strip()
                    for col in precaution_columns
                    if pd.notna(row.get(col)) and str(row.get(col)).strip()
                ],
            )

    if diets_path.exists():
        df = pd.read_csv(diets_path)
        for _, row in df.iterrows():
            update_medical_store_entry(
                medical_store,
                row.get("Disease", ""),
                "diet",
                parse_support_list(row.get("Diet", "")),
            )

    if medications_path.exists():
        df = pd.read_csv(medications_path)
        for _, row in df.iterrows():
            update_medical_store_entry(
                medical_store,
                row.get("Disease", ""),
                "medications",
                parse_support_list(row.get("Medication", "")),
            )

    if workout_path.exists():
        df = pd.read_csv(workout_path)
        for _, row in df.iterrows():
            update_medical_store_entry(
                medical_store,
                row.get("Disease", ""),
                "lifestyle",
                parse_support_list(row.get("Workouts", "")),
            )

    return medical_store


def resolve_disease_key(disease, medical_store, alias_map):
    resolved_norm, alias_used = apply_disease_alias(disease, alias_map)
    if not resolved_norm or not medical_store:
        return None, None, resolved_norm, alias_used

    if resolved_norm in medical_store:
        return resolved_norm, resolved_norm, resolved_norm, alias_used

    resolved_tokens = set(resolved_norm.split())
    best_token_match = None
    best_token_score = 0
    for key in medical_store:
        key_tokens = set(key.split())
        overlap = resolved_tokens & key_tokens
        if overlap and len(overlap) > best_token_score:
            best_token_score = len(overlap)
            best_token_match = key

    if best_token_match:
        return best_token_match, best_token_match, resolved_norm, alias_used

    matches = get_close_matches(resolved_norm, list(medical_store.keys()), n=1, cutoff=0.5)
    if matches:
        matched_key = matches[0]
        return matched_key, matched_key, resolved_norm, alias_used

    for key in medical_store:
        if key in resolved_norm or resolved_norm in key:
            return key, key, resolved_norm, alias_used

    return None, None, resolved_norm, alias_used


def get_disease_support_info(disease_name, medical_store, alias_map):
    matched_key, debug_key, normalized_disease, alias_used = resolve_disease_key(
        disease_name,
        medical_store,
        alias_map,
    )
    if matched_key is None:
        return create_empty_medical_entry(), debug_key, normalized_disease, alias_used

    entry = medical_store.get(matched_key, create_empty_medical_entry())
    return {
        "description": entry.get("description") or "",
        "precautions": entry.get("precautions") or [],
        "diet": entry.get("diet") or [],
        "medications": entry.get("medications") or [],
        "lifestyle": entry.get("lifestyle") or [],
    }, debug_key, normalized_disease, alias_used


def parse_generated_sections(text):
    sections = safe_parse_fallback(text)
    return {
        "description": sections.get("description") or None,
        "precautions": sections.get("precautions", []),
        "diet": sections.get("diet", []),
        "medications": sections.get("medications", []),
        "lifestyle": sections.get("lifestyle", []),
    }


def safe_parse_fallback(text):
    sections = {
        "description": "",
        "precautions": [],
        "diet": [],
        "medications": [],
        "lifestyle": [],
    }
    current_key = None

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if "description" in lowered:
            current_key = "description"
            continue
        elif "precaution" in lowered:
            current_key = "precautions"
            continue
        elif "diet" in lowered:
            current_key = "diet"
            continue
        elif "medication" in lowered:
            current_key = "medications"
            continue
        elif "lifestyle" in lowered or "exercise" in lowered:
            current_key = "lifestyle"
            continue
        else:
            if current_key == "description":
                sections["description"] += f"{line} "
            elif current_key in sections and line.strip():
                sections[current_key].append(line.strip("-•* "))

    return sections


def clean_json(text):
    text = str(text).strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[-2]
    return text.strip().removeprefix("json").strip()


def generate_medical_info(disease):
    if genai is None:
        raise RuntimeError("google-generativeai is not installed.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    prompt = f"""
Return STRICT JSON only for {disease} with keys:

description (string)
precautions (list)
diet (list)
medications (list)
lifestyle (list)

Example:
{{
  "description": "...",
  "precautions": ["...", "..."],
  "diet": ["...", "..."],
  "medications": ["...", "..."],
  "lifestyle": ["...", "..."]
}}
"""
    response = model.generate_content(prompt)
    response_text = clean_json(getattr(response, "text", "") or "")

    try:
        parsed_json = json.loads(response_text)
        parsed = {
            "description": str(parsed_json.get("description", "")).strip() or None,
            "precautions": [str(item).strip() for item in parsed_json.get("precautions", []) if str(item).strip()],
            "diet": [str(item).strip() for item in parsed_json.get("diet", []) if str(item).strip()],
            "medications": [str(item).strip() for item in parsed_json.get("medications", []) if str(item).strip()],
            "lifestyle": [str(item).strip() for item in parsed_json.get("lifestyle", []) if str(item).strip()],
        }
    except Exception:
        parsed = parse_generated_sections(response_text)

    if not any([
        parsed["description"],
        parsed["precautions"],
        parsed["diet"],
        parsed["medications"],
        parsed["lifestyle"],
    ]):
        raise RuntimeError("Gemini returned no usable medical information.")
    return parsed


@st.cache_data
def generate_medical_info_cached(disease):
    return generate_medical_info(disease)


def generate_medical_field_with_gemini(disease, field):
    generated = generate_medical_info_cached(disease)
    value = generated.get(field)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return "" if field == "description" else []


def reset_prediction_state():
    st.session_state.selected_symptoms = []
    st.session_state.symptom_selector = []


def render_medical_list(items):
    if not items:
        st.write(MEDICAL_INFO_FALLBACK)
        return

    for item in items:
        cleaned_item = str(item).replace("_", " ").strip()
        st.markdown(f"- {cleaned_item}")


def get_selected_original(selected_display_names, display_to_original):
    return [
        display_to_original[name]
        for name in selected_display_names
        if name in display_to_original
    ]


def build_symptom_mapping(feature_names):
    display_to_original = {}
    for feature in feature_names:
        display_name = format_symptom(feature)
        display_to_original[display_name] = feature
    return display_to_original


def preprocess_input(
    selected_display_names,
    model_feature_names,
    display_to_original,
    feature_index,
    interaction_index,
):
    selected_original = get_selected_original(selected_display_names, display_to_original)

    input_vector = np.zeros(len(model_feature_names), dtype=np.float32)
    selected_set = set(selected_original)

    for symptom in selected_original:
        if symptom in feature_index:
            input_vector[feature_index[symptom]] = CRITICAL_SYMPTOM_WEIGHTS.get(symptom, 1.0)

    for (left, right), idx in interaction_index.items():
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


@st.cache_resource
def get_shap_explainer(model):
    shap_module = get_shap()
    if shap_module is None:
        return None
    try:
        base_model = getattr(model, "estimator", model)
        return shap_module.TreeExplainer(base_model)
    except Exception:
        return None


def get_shap():
    try:
        import shap
        return shap
    except ImportError:
        return None


def get_feature_importance_values(model):
    base_model = getattr(model, "estimator", model)
    importance = getattr(base_model, "feature_importances_", None)

    if importance is None:
        calibrated_models = getattr(model, "calibrated_classifiers_", [])
        if calibrated_models:
            inner_model = getattr(calibrated_models[0], "estimator", None)
            importance = getattr(inner_model, "feature_importances_", None)

    return importance


def explain_prediction(selected_symptoms, model_feature_names, feature_importance, top_features=10):
    if feature_importance is None:
        return []

    explanation = []
    selected_set = set(selected_symptoms)
    important_features = sorted(
        zip(model_feature_names[:len(feature_importance)], feature_importance),
        key=lambda item: item[1],
        reverse=True
    )[:top_features]

    for feat, _ in important_features:
        if feat in selected_set:
            explanation.append(f"{feat.replace('_', ' ').title()} (high impact)")
        elif "_and_" in feat:
            a, b = feat.split("_and_", 1)
            if a in selected_set and b in selected_set:
                explanation.append(f"{a.replace('_', ' ').title()} + {b.replace('_', ' ').title()} combination")
        elif "_x_" in feat:
            a, b = feat.split("_x_", 1)
            if a in selected_set and b in selected_set:
                explanation.append(f"{a.replace('_', ' ').title()} + {b.replace('_', ' ').title()} combination")

    return explanation


def get_important_features(model_feature_names, feature_importance, top_features=10):
    if feature_importance is None:
        return []
    usable = min(len(model_feature_names), len(feature_importance))
    return sorted(
        zip(model_feature_names[:usable], feature_importance[:usable]),
        key=lambda item: item[1],
        reverse=True
    )[:top_features]


def render_confidence_explanation(confidence):
    st.subheader("Why this prediction?")
    if confidence > 0.7:
        st.success("High confidence due to strong symptom alignment")
    elif confidence > 0.4:
        st.info("Moderate confidence - symptoms partially match")
    else:
        st.warning("Low confidence - symptoms are ambiguous")


def render_explanation_details(selected_symptoms, important_features):
    selected_set = set(selected_symptoms)
    rendered = 0
    for feat, _ in important_features[:5]:
        if feat in selected_set:
            st.write(f"- {feat.replace('_', ' ').title()} contributed significantly")
            rendered += 1
        elif "_and_" in feat:
            left, right = feat.split("_and_", 1)
            if left in selected_set and right in selected_set:
                st.write(
                    f"- {left.replace('_', ' ').title()} + {right.replace('_', ' ').title()} combination contributed significantly"
                )
                rendered += 1
        elif "_x_" in feat:
            left, right = feat.split("_x_", 1)
            if left in selected_set and right in selected_set:
                st.write(
                    f"- {left.replace('_', ' ').title()} + {right.replace('_', ' ').title()} combination contributed significantly"
                )
                rendered += 1
    return rendered


def render_top3_explanations(top_predictions, selected_symptoms, important_features):
    for disease, _ in top_predictions:
        with st.expander(f"Why {disease}?", expanded=False):
            explanation = explain_prediction(selected_symptoms, [feat for feat, _ in important_features], np.array([score for _, score in important_features]), top_features=min(10, len(important_features)))
            if explanation:
                for item in explanation:
                    st.write(f"- {item}")
            else:
                st.write("Prediction based on overall symptom pattern.")


def render_feature_importance_fallback(important_features):
    st.subheader("Important Features")
    if important_features:
        for feature_name, _ in important_features[:SHAP_MAX_FEATURES]:
            st.write(f"- {feature_name.replace('_', ' ').title()}")
    else:
        st.write(MEDICAL_INFO_FALLBACK)


def render_shap_or_fallback(model, input_vector, model_feature_names, important_features, prediction, confidence):
    if FAST_MODE:
        return

    st.subheader("Feature Contribution (SHAP)")

    if confidence < 0.4:
        st.info("Explanation limited due to low confidence.")
        render_feature_importance_fallback(important_features)
        return

    explainer = get_shap_explainer(model)
    shap_module = get_shap()
    if explainer is None or shap_module is None:
        st.info("SHAP visualization not available.")
        render_feature_importance_fallback(important_features)
        return

    try:
        with st.spinner("Explaining prediction..."):
            shap_values = explainer.shap_values(input_vector)
            if isinstance(shap_values, list):
                shap_values = shap_values[prediction[0]]

            shap_array = np.asarray(shap_values)
            if shap_array.ndim == 1:
                shap_array = shap_array.reshape(1, -1)

            if len(model_feature_names) <= SHAP_SUMMARY_FEATURE_LIMIT:
                plt = get_matplotlib()
                plt.figure(figsize=(10, 3.5))
                shap_module.summary_plot(
                    shap_array,
                    input_vector,
                    feature_names=model_feature_names,
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.clf()
            else:
                st.info("Using lightweight explanation view for large feature space.")

            top_idx = np.argsort(np.abs(shap_array))[0][-SHAP_MAX_FEATURES:]
            st.subheader("Top Contributing Symptoms")
            for i in reversed(top_idx):
                st.write(f"- {model_feature_names[i].replace('_', ' ').title()}")
            return
    except Exception:
        st.warning("Feature temporarily unavailable")
        render_feature_importance_fallback(important_features)


def render_medical_context(predicted_disease, medical_store, alias_map, show_debug=False):
    support_info, matched_key, norm_disease, alias_used = get_disease_support_info(
        predicted_disease,
        medical_store,
        alias_map,
    )
    missing_fields = [field for field in MEDICAL_FIELDS if support_info[field] in (None, "", [])]
    should_try_gemini = bool(missing_fields) and not FAST_MODE and os.getenv("GEMINI_API_KEY")

    if should_try_gemini:
        try:
            for field in missing_fields:
                generated_value = generate_medical_field_with_gemini(predicted_disease, field)
                if generated_value not in (None, "", []):
                    support_info[field] = generated_value
        except Exception:
            st.warning("Feature temporarily unavailable")

    has_dataset_match = matched_key is not None

    if show_debug:
        st.write("DEBUG FAST_MODE:", FAST_MODE)
        st.write("Alias map size:", len(alias_map))
        st.write("Predicted disease:", predicted_disease)
        st.write("Normalized disease:", norm_disease)
        st.write("Alias applied:", alias_used if alias_used else "None")
        st.write("Matched key:", matched_key if matched_key else "None")
        st.write("Final matched disease:", matched_key if matched_key else "None")
        st.write("Match found:", has_dataset_match)

    if not has_dataset_match:
        st.warning("No dataset match found. Showing limited information.")

    st.write("")
    with st.expander("Medical Details", expanded=True):
        st.subheader("About the Disease")
        st.write(support_info["description"] or MEDICAL_INFO_FALLBACK)

        st.subheader("Precautions")
        render_medical_list(support_info["precautions"])

        st.subheader("Diet")
        render_medical_list(support_info["diet"])

        st.subheader("Medications")
        render_medical_list(support_info["medications"])

        st.subheader("Lifestyle")
        render_medical_list(support_info["lifestyle"])

    if not FAST_MODE and not os.getenv("GEMINI_API_KEY"):
        st.info("Add GEMINI_API_KEY to enable enhanced medical insights.")
    st.warning("This is not medical advice. Please consult a doctor.")


def render_confidence_chart(confidence):
    plt = get_matplotlib()
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


def get_feature_importance_frame(model, model_feature_names):
    importance = get_feature_importance_values(model)

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
    feature_index = context["feature_index"]
    interaction_index = context["interaction_index"]
    medical_store = load_medical_store()
    alias_map = build_alias_map(medical_store)

    display_to_original = build_symptom_mapping(feature_names)
    display_features = sorted(display_to_original.keys())
    selected_key = "selected_symptoms"

    if selected_key not in st.session_state:
        st.session_state[selected_key] = []
    if "symptom_selector" not in st.session_state:
        st.session_state.symptom_selector = st.session_state[selected_key].copy()
    if "loaded" not in st.session_state:
        st.info("Loading model for the first time...")
        st.session_state.loaded = True

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
        st.caption("Fast mode enabled" if FAST_MODE else "Full explanation mode")

    st.write("")

    with st.container(border=True):
        selected = st.multiselect(
            "Search and Select Symptoms",
            display_features,
            default=st.session_state[selected_key],
            key="symptom_selector",
            placeholder="Type to search symptoms...",
            help="You can search by typing part of a symptom name.",
        )

    st.write("")
    button_col, reset_col = st.columns([1, 1])
    with button_col:
        predict_clicked = st.button("Predict Disease", type="primary", use_container_width=True)
    with reset_col:
        st.button(
            "Reset Selection",
            use_container_width=True,
            on_click=reset_prediction_state,
        )

    st.session_state[selected_key] = list(selected)

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

    show_debug = st.checkbox("Show Debug Info")
    input_vector = None
    if predict_clicked or show_debug:
        input_vector = preprocess_input(
            selected,
            model_feature_names,
            display_to_original,
            feature_index,
            interaction_index,
        )
        print(f"[DEBUG] Expected feature count: {expected_feature_count}")
        print(f"[DEBUG] Actual input feature count: {input_vector.shape[1]}")

        if input_vector.shape[1] != expected_feature_count:
            st.error(
                f"Feature mismatch: model expects {expected_feature_count} features, "
                f"but app built {input_vector.shape[1]}."
            )
            return

    if show_debug and input_vector is not None:
        st.write("Selected:", selected)
        st.write("Input shape:", input_vector.shape)

    if predict_clicked:
        try:
            with st.spinner("Analyzing symptoms..."):
                prediction = model.predict(input_vector)
                probabilities = model.predict_proba(input_vector)

            predicted_disease = format_disease_name(label_encoder.inverse_transform(prediction)[0])
            confidence = float(np.max(probabilities))
            top_predictions = extract_top_predictions(probabilities, label_encoder)
            selected_original = get_selected_original(selected, display_to_original)
            feature_importance = get_feature_importance_values(model)
            important_features = get_important_features(
                model_feature_names,
                feature_importance,
                top_features=10,
            )
            explanation = explain_prediction(
                selected_original,
                model_feature_names,
                feature_importance,
            )
            confidence_warning = build_confidence_warning(confidence)

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
                if confidence_warning:
                    st.warning(confidence_warning)
                else:
                    st.success("Confidence is above the review threshold for this symptom set.")
                st.caption("This is decision support, not a medical diagnosis.")
                render_top_predictions(top_predictions)

            with right_col:
                st.subheader("Confidence Breakdown")
                render_confidence_chart(confidence)

            if not FAST_MODE:
                try:
                    importance_df = get_feature_importance_frame(model, model_feature_names)
                    st.subheader("Top Influencing Symptoms")
                    if importance_df.empty:
                        st.info("Feature importance is not available for the loaded model.")
                    else:
                        st.bar_chart(importance_df.set_index("Feature"), use_container_width=True)
                except Exception:
                    st.warning("Feature temporarily unavailable")

                render_confidence_explanation(confidence)
                rendered_explanations = render_explanation_details(selected_original, important_features)
                if explanation and rendered_explanations == 0:
                    for item in explanation:
                        st.markdown(f"- {item}")
                elif rendered_explanations == 0:
                    st.write("Prediction based on overall symptom pattern.")

                render_top3_explanations(top_predictions, selected_original, important_features)

                render_shap_or_fallback(
                    model,
                    input_vector,
                    model_feature_names,
                    important_features,
                    prediction,
                    confidence,
                )

            render_medical_context(
                predicted_disease,
                medical_store,
                alias_map,
                show_debug=show_debug,
            )

            st.caption(
                "Use this result as decision support only. Please consult a qualified medical professional for diagnosis."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
