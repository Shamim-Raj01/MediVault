from pathlib import Path

import streamlit as st


def render(context):
    model = context["model"]
    feature_names = context["feature_names"]
    model_feature_names = context["model_feature_names"]
    expected_feature_count = context["expected_feature_count"]

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Platform Overview</div>
            <h1 style="margin: 0 0 0.35rem 0;">About This Dashboard</h1>
            <p class="hero-copy">A clinical-style interface for exploring disease prediction outputs and model metadata.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    intro_col, stack_col = st.columns([1.2, 1])

    with intro_col:
        st.subheader("What It Does")
        st.write(
            "This dashboard helps users explore symptom-driven disease predictions with a "
            "clean workflow, confidence visualization, and supporting model insight panels."
        )
        st.write(
            "The prediction page is designed for quick interaction, while the insights page "
            "surfaces feature coverage and estimator behavior for review."
        )

        st.subheader("Clinical Disclaimer")
        st.warning(
            "This tool is for decision support and demonstration purposes only. "
            "It should not replace professional medical evaluation, diagnosis, or treatment."
        )

    with stack_col:
        st.subheader("System Snapshot")
        st.metric("Model Artifact", Path("model_improved.pkl").name)
        st.metric("UI Symptom Count", len(feature_names))
        st.metric("Model Feature Count", expected_feature_count)
        st.metric("Estimator", type(getattr(model, "estimator", model)).__name__)

    st.subheader("Architecture")
    architecture = [
        "Prediction page collects symptoms and builds a model-aligned feature vector.",
        "Inference runs with the trained XGBoost-based calibrated classifier.",
        "Insights page visualizes feature usage, importances, and deployment metrics.",
        "Shared styling in app.py applies the medical background, overlay, and dark sidebar theme.",
    ]
    for item in architecture:
        st.write(f"- {item}")

    st.subheader("Data Alignment")
    engineered_count = max(0, len(model_feature_names) - len(feature_names))
    st.info(
        f"The UI exposes {len(feature_names)} selectable symptoms while the model expects "
        f"{expected_feature_count} total inputs, including {engineered_count} engineered features."
    )
