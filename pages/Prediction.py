import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def preprocess_input(selected, feature_names):
    input_vector = np.zeros(len(feature_names), dtype=np.float32)
    feature_index = {f: i for i, f in enumerate(feature_names)}
    selected_set = set(selected)

    for symptom in selected:
        if symptom in feature_index:
            input_vector[feature_index[symptom]] = 1

    for feature_name, idx in feature_index.items():
        if "_and_" not in feature_name:
            continue

        left, right = feature_name.split("_and_", 1)
        if left in selected_set and right in selected_set:
            input_vector[idx] = 1

    input_vector = input_vector.reshape(1, -1)
    print(f"[DEBUG] Selected symptoms count: {len(selected_set)}")
    print(f"[DEBUG] Input shape: {input_vector.shape}")
    return input_vector


def render_top_predictions(probabilities, label_encoder):
    probs = probabilities[0]
    top_indices = np.argsort(probs)[-3:][::-1]

    st.subheader("Top Predictions")
    for i in top_indices:
        disease = label_encoder.inverse_transform([i])[0]
        prob = float(probs[i])
        st.write(f"**{disease}** - {prob * 100:.2f}%")
        st.progress(prob)


def render_confidence_chart(confidence):
    labels = ["Confidence", "Remaining"]
    sizes = [confidence, 1 - confidence]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)


def render(context):
    model = context["model"]
    label_encoder = context["label_encoder"]
    feature_names = context["feature_names"]
    model_feature_names = context["model_feature_names"]
    expected_feature_count = context["expected_feature_count"]

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Prediction Workspace</div>
            <h1 style="margin: 0 0 0.35rem 0;">🩺 AI-Powered Disease Prediction</h1>
            <p class="hero-copy">Select symptoms to predict possible diseases using machine learning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_col, stats_col = st.columns([3, 2])
    with info_col:
        st.caption("Choose the observed symptoms below and run a model-assisted assessment.")
    with stats_col:
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Symptoms Available", len(feature_names))
        metric_b.metric("Model Inputs", expected_feature_count)

    with st.container(border=True):
        selected = st.multiselect("Select Symptoms", feature_names)
        action_col, reset_col = st.columns([4, 1])
        with action_col:
            predict_clicked = st.button("Predict Disease", use_container_width=True)
        with reset_col:
            if st.button("Reset", use_container_width=True):
                st.rerun()

    if not predict_clicked:
        st.info("Pick one or more symptoms to generate a ranked disease prediction.")
        return

    if not selected:
        st.warning("Please select at least one symptom before predicting.")
        return

    input_vector = preprocess_input(selected, model_feature_names)
    print(f"[DEBUG] Expected feature count: {expected_feature_count}")
    print(f"[DEBUG] Actual input feature count: {input_vector.shape[1]}")

    if input_vector.shape[1] != expected_feature_count:
        st.error(
            f"Feature mismatch: model expects {expected_feature_count} features, "
            f"but app built {input_vector.shape[1]}."
        )
        return

    try:
        with st.spinner("Analyzing symptoms..."):
            prediction = model.predict(input_vector)
            probabilities = model.predict_proba(input_vector)

        predicted_disease = label_encoder.inverse_transform(prediction)[0]
        confidence = float(np.max(probabilities))

        summary_col, chart_col = st.columns([1.2, 1])
        with summary_col:
            st.success(f"Predicted Disease: {predicted_disease}")
            st.info(f"Confidence: {confidence * 100:.2f}%")
            st.warning("⚠ This is an AI prediction, not a medical diagnosis.")

            if confidence < 0.6:
                st.warning("Low confidence prediction. Please consult a doctor.")

            render_top_predictions(probabilities, label_encoder)

        with chart_col:
            st.subheader("Confidence Breakdown")
            render_confidence_chart(confidence)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
