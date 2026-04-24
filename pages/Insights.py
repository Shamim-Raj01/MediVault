import numpy as np
import pandas as pd
import streamlit as st


def _get_feature_importance_frame(model, feature_names):
    base_model = getattr(model, "estimator", model)
    importances = getattr(base_model, "feature_importances_", None)

    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    usable_length = min(len(feature_names), len(importances))
    frame = pd.DataFrame(
        {
            "feature": feature_names[:usable_length],
            "importance": importances[:usable_length],
        }
    ).sort_values("importance", ascending=False)
    return frame


def render(context):
    model = context["model"]
    label_encoder = context["label_encoder"]
    feature_names = context["feature_names"]
    model_feature_names = context["model_feature_names"]
    expected_feature_count = context["expected_feature_count"]

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Model Insights</div>
            <h1 style="margin: 0 0 0.35rem 0;">Diagnostic Model Insights</h1>
            <p class="hero-copy">Review feature coverage, importance patterns, and model deployment metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Diseases Supported", len(label_encoder.classes_))
    metric_cols[1].metric("UI Symptoms", len(feature_names))
    metric_cols[2].metric("Model Features", expected_feature_count)
    metric_cols[3].metric("Calibration Folds", len(getattr(model, "calibrated_classifiers_", [])))

    hist_col, bar_col = st.columns(2)

    symptom_lengths = pd.DataFrame(
        {"Symptom Name Length": [len(name.replace("_", " ")) for name in feature_names]}
    )
    with hist_col:
        st.subheader("Symptom Label Length Histogram")
        st.caption("Distribution of display symptom lengths used in the UI.")
        st.bar_chart(
            symptom_lengths["Symptom Name Length"].value_counts().sort_index(),
            use_container_width=True,
        )

    importance_frame = _get_feature_importance_frame(model, model_feature_names)
    with bar_col:
        st.subheader("Top Feature Importance")
        st.caption("Most influential features according to the underlying XGBoost estimator.")
        if importance_frame.empty:
            st.info("Feature importance is not available for the loaded model.")
        else:
            top_features = importance_frame.head(12).set_index("feature")
            st.bar_chart(top_features, use_container_width=True)

    detail_col, sample_col = st.columns([1.2, 1])
    with detail_col:
        st.subheader("Model Metrics")
        metrics_frame = pd.DataFrame(
            [
                {"Metric": "Model Type", "Value": type(model).__name__},
                {"Metric": "Base Estimator", "Value": type(getattr(model, "estimator", model)).__name__},
                {"Metric": "Expected Input Shape", "Value": f"(1, {expected_feature_count})"},
                {"Metric": "Prediction Classes", "Value": len(label_encoder.classes_)},
            ]
        )
        st.dataframe(metrics_frame, use_container_width=True, hide_index=True)

    with sample_col:
        st.subheader("Feature Composition")
        engineered_count = max(0, len(model_feature_names) - len(feature_names))
        composition = pd.DataFrame(
            {
                "Category": ["Base Symptoms", "Engineered Features"],
                "Count": [len(feature_names), engineered_count],
            }
        ).set_index("Category")
        st.bar_chart(composition, use_container_width=True)

        st.caption(
            "Engineered features are automatically added during preprocessing to match "
            "the model's trained input schema."
        )
