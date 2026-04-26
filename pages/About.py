import streamlit as st


def render(context):
    st.title("ℹ About This Project")
    st.caption("A professional overview of the AI-powered disease prediction platform.")
    st.write("")

    st.markdown(
        """
        ### About the System

        This project is an **AI-based disease prediction system** built with Python, Streamlit,
        and machine learning. It uses a list of symptoms as input and provides **real-time disease
        predictions** through an interactive multi-page dashboard.

        The goal is to make model predictions easier to explore, present, and explain in a way
        that feels useful for demos, portfolios, and decision-support scenarios.
        """
    )

    st.write("")

    st.markdown(
        """
        ### Model Details

        The prediction engine is based on the **XGBoost algorithm**, a strong gradient boosting
        approach that performs well on structured tabular data. The system is framed as a
        **multi-class classification** problem, where the model predicts one disease class from
        many possible outcomes based on symptom patterns.
        """
    )

    st.write("")

    st.markdown(
        """
        ### Accuracy Improvement

        The improved version of the system was designed to deliver stronger and more stable results:

        - Combined multiple datasets to improve coverage and diversity
        - Handled class imbalance using **SMOTE**
        - Added **feature engineering** to capture richer symptom interactions
        - Applied **hyperparameter tuning** to improve model quality
        - Used **cross-validation** to improve reliability and reduce overfitting risk

        These changes significantly improved the overall quality of predictions compared with the
        earlier baseline model.
        """
    )

    st.write("")

    st.markdown(
        """
        ### Performance

        The current model achieves approximately **98% accuracy** with a **high F1 score** and
        more stable predictions than the earlier version of the project. This makes the dashboard
        well-suited for demonstrations of end-to-end ML product thinking, including training,
        evaluation, inference, and presentation.
        """
    )

    st.write("")

    st.markdown(
        """
        ### Limitations

        - Predictions depend on the quality and coverage of the training datasets
        - Symptom-only input cannot capture full clinical context, history, or lab results
        - Similar diseases may still produce overlapping symptom patterns
        - Confidence scores do not guarantee medical correctness
        """
    )

    st.write("")

    st.markdown(
        """
        ### Disclaimer

        This application is a machine learning demonstration and decision-support tool.
        It is **not** a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a licensed healthcare professional for clinical decisions.
        """
    )

    st.write("")

    st.info(
        "This application is intended for educational and demonstration purposes. "
        "It should not be used as a substitute for professional medical diagnosis."
    )
