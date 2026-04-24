import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pages import About, Insights, Prediction


st.set_page_config(
    page_title="AI Medical Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def build_medical_background() -> str:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#12354a"/>
          <stop offset="100%" stop-color="#08151f"/>
        </linearGradient>
      </defs>
      <rect width="1600" height="900" fill="url(#bg)"/>
      <g opacity="0.18" stroke="#80d8ff" stroke-width="3" fill="none">
        <path d="M0 500 H280 L340 400 L420 620 L500 260 L580 500 H1600"/>
        <path d="M1050 0 V900"/>
        <path d="M1240 0 V900"/>
        <path d="M0 180 H1600"/>
        <path d="M0 720 H1600"/>
      </g>
      <g opacity="0.15" fill="#8fe3ff">
        <circle cx="240" cy="180" r="90"/>
        <circle cx="245" cy="180" r="55" fill="#12354a"/>
        <rect x="226" y="118" width="38" height="125" rx="12"/>
        <rect x="182" y="161" width="126" height="38" rx="12"/>
      </g>
      <g opacity="0.12" fill="#d8f6ff">
        <circle cx="1280" cy="250" r="120"/>
        <circle cx="1280" cy="250" r="72" fill="#08151f"/>
        <rect x="1253" y="160" width="54" height="180" rx="16"/>
        <rect x="1190" y="223" width="180" height="54" rx="16"/>
      </g>
    </svg>
    """
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def inject_styles():
    background_url = build_medical_background()
    st.markdown(
        f"""
        <style>
        :root {{
            --panel-bg: rgba(10, 18, 28, 0.72);
            --panel-border: rgba(143, 227, 255, 0.18);
            --accent: #8fe3ff;
            --accent-strong: #31c5ff;
            --text-soft: #b7c9d6;
        }}

        [data-testid="stAppViewContainer"] {{
            background:
                linear-gradient(rgba(6, 12, 18, 0.68), rgba(6, 12, 18, 0.82)),
                url("{background_url}") center/cover fixed no-repeat;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #071019 0%, #0c1b28 100%);
            border-right: 1px solid rgba(143, 227, 255, 0.12);
        }}

        [data-testid="stSidebar"] * {{
            color: #eef7fb;
        }}

        [data-testid="stSidebar"] .stRadio > div {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(143, 227, 255, 0.12);
            border-radius: 16px;
            padding: 0.5rem;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            padding: 0.35rem 0.4rem;
            border-radius: 10px;
        }}

        [data-testid="stMetric"] {{
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            padding: 0.8rem;
            backdrop-filter: blur(6px);
        }}

        [data-testid="stVerticalBlock"] > [data-testid="element-container"] .stAlert {{
            border-radius: 16px;
        }}

        .dashboard-shell {{
            background: rgba(8, 16, 26, 0.48);
            border: 1px solid rgba(143, 227, 255, 0.10);
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(8px);
        }}

        .hero-card {{
            background: linear-gradient(135deg, rgba(7, 25, 39, 0.88), rgba(18, 53, 74, 0.78));
            border: 1px solid rgba(143, 227, 255, 0.16);
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
            color: white;
        }}

        .hero-kicker {{
            color: var(--accent);
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }}

        .hero-copy {{
            color: var(--text-soft);
            margin: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    model_path = Path("model_improved.pkl")
    label_encoder_path = Path("le_classification_improved.pkl")
    feature_names_path = Path("feature_names_improved.pkl")

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    feature_names = joblib.load(feature_names_path)

    if isinstance(feature_names, pd.Index):
        feature_names = feature_names.tolist()
    elif isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    else:
        feature_names = list(feature_names)

    if hasattr(model, "feature_names_in_"):
        model_feature_names = [str(name) for name in model.feature_names_in_]
    elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_names_in_"):
        model_feature_names = [str(name) for name in model.estimator.feature_names_in_]
    else:
        model_feature_names = feature_names

    expected_feature_count = getattr(model, "n_features_in_", len(model_feature_names))

    print(f"[DEBUG] Loaded model file: {model_path.resolve()}")
    print(f"[DEBUG] Loaded feature file: {feature_names_path.resolve()}")
    print(f"[DEBUG] model.n_features_in_: {expected_feature_count}")
    print(f"[DEBUG] feature_names length: {len(feature_names)}")
    print(f"[DEBUG] model feature_names_in_ length: {len(model_feature_names)}")

    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "model_feature_names": model_feature_names,
        "expected_feature_count": expected_feature_count,
    }


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div style="padding: 0.25rem 0 1rem 0;">
            <div style="font-size: 0.8rem; letter-spacing: 0.18em; text-transform: uppercase; color: #8fe3ff;">
                Clinical AI Suite
            </div>
            <div style="font-size: 1.6rem; font-weight: 700; margin-top: 0.35rem;">
                MediVault Dashboard
            </div>
            <div style="color: #b7c9d6; margin-top: 0.45rem;">
                Explore predictions, model insights, and deployment context in one place.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigate",
        ["Prediction", "Insights", "About"],
        label_visibility="visible",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Built for fast triage support and model transparency.")
    return page


def main():
    inject_styles()

    try:
        context = load_artifacts()
    except Exception as exc:
        st.error(f"Failed to load model files: {exc}")
        st.stop()

    page = render_sidebar()

    pages = {
        "Prediction": Prediction.render,
        "Insights": Insights.render,
        "About": About.render,
    }

    st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)
    pages[page](context)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
