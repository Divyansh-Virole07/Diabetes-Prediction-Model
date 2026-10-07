# ============================================================
# DIABETES AI - STREAMLIT APPLICATION
# Logistic Regression Diabetes Prediction
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import textwrap

from pathlib import Path
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes AI | Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Always resolve paths relative to this app.py file.
# This prevents problems when Streamlit is launched from
# a different working directory.

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "diabetes_pipeline.joblib"


# ============================================================
# HTML HELPER
# ============================================================

def render_html(html):
    """
    Safely render custom HTML without indentation being
    interpreted as a Markdown code block.
    """
    st.html(textwrap.dedent(html))


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 5%,
                rgba(37, 99, 235, 0.13),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(124, 58, 237, 0.12),
                transparent 30%
            ),
            #070b14;

        color: #f8fafc;
    }


    .main {
        padding-top: 1rem;
    }


    #MainMenu {
        visibility: hidden;
    }


    footer {
        visibility: hidden;
    }


    header {
        background: transparent !important;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #0b1120 0%,
                #080c16 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.07);
    }


    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {

        position: relative;

        padding: 42px 45px;

        border-radius: 26px;

        background:
            linear-gradient(
                135deg,
                rgba(37,99,235,0.17),
                rgba(124,58,237,0.14)
            );

        border:
            1px solid rgba(96,165,250,0.20);

        box-shadow:
            0 25px 70px rgba(0,0,0,0.35);

        backdrop-filter: blur(18px);

        overflow: hidden;

        margin-bottom: 30px;
    }


    .hero::after {

        content: "";

        position: absolute;

        width: 220px;
        height: 220px;

        right: -80px;
        top: -90px;

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                rgba(96,165,250,0.22),
                transparent 70%
            );
    }


    .hero-icon {

        font-size: 52px;

        margin-bottom: 5px;
    }


    .hero-title {

        font-size: 42px;

        font-weight: 850;

        letter-spacing: -1.5px;

        line-height: 1.15;

        color: #f8fafc;

        margin-bottom: 10px;
    }


    .hero-subtitle {

        max-width: 850px;

        font-size: 16px;

        line-height: 1.7;

        color: #94a3b8;
    }


    .hero-badge {

        display: inline-block;

        margin-top: 18px;

        padding: 7px 13px;

        border-radius: 999px;

        background:
            rgba(37,99,235,0.13);

        border:
            1px solid rgba(96,165,250,0.22);

        color: #93c5fd;

        font-size: 13px;

        font-weight: 700;
    }


    /* =====================================================
       CARDS
       ===================================================== */

    .glass-card {

        background:
            rgba(15,23,42,0.70);

        border:
            1px solid rgba(148,163,184,0.11);

        border-radius: 20px;

        padding: 24px;

        box-shadow:
            0 18px 45px rgba(0,0,0,0.20);

        backdrop-filter: blur(14px);
    }


    .metric-card {

        background:
            linear-gradient(
                145deg,
                rgba(15,23,42,0.94),
                rgba(30,41,59,0.68)
            );

        border:
            1px solid rgba(148,163,184,0.11);

        border-radius: 18px;

        padding: 22px;

        min-height: 125px;

        text-align: center;

        box-shadow:
            0 12px 35px rgba(0,0,0,0.15);
    }


    .metric-icon {

        font-size: 22px;

        margin-bottom: 7px;
    }


    .metric-value {

        font-size: 29px;

        font-weight: 850;

        color: #f8fafc;
    }


    .metric-label {

        color: #94a3b8;

        font-size: 13px;

        margin-top: 4px;
    }


    /* =====================================================
       SECTION HEADINGS
       ===================================================== */

    .section-title {

        font-size: 25px;

        font-weight: 800;

        color: #f8fafc;

        margin-top: 28px;

        margin-bottom: 16px;
    }


    .section-subtitle {

        color: #64748b;

        font-size: 14px;

        margin-top: -10px;

        margin-bottom: 18px;
    }


    /* =====================================================
       INFORMATION BOX
       ===================================================== */

    .info-box {

        background:
            rgba(30,41,59,0.50);

        border-left:
            4px solid #60a5fa;

        border-radius: 12px;

        padding: 16px 18px;

        margin: 12px 0 20px 0;

        color: #cbd5e1;

        line-height: 1.6;
    }


    /* =====================================================
       RESULT
       ===================================================== */

    .result-card {

        border-radius: 25px;

        padding: 34px;

        text-align: center;

        margin-top: 25px;

        margin-bottom: 25px;

        box-shadow:
            0 25px 65px rgba(0,0,0,0.25);
    }


    .result-icon {

        font-size: 55px;

        margin-bottom: 5px;
    }


    .result-title {

        font-size: 29px;

        font-weight: 850;

        margin-bottom: 5px;
    }


    .result-probability {

        font-size: 58px;

        font-weight: 900;

        letter-spacing: -2px;

        margin: 12px 0;
    }


    .result-description {

        color: #cbd5e1;

        font-size: 15px;
    }


    /* =====================================================
       GAUGE
       ===================================================== */

    .gauge-wrapper {

        display: flex;

        justify-content: center;

        align-items: center;

        margin: 20px 0;
    }


    .gauge {

        width: 190px;

        height: 190px;

        border-radius: 50%;

        display: flex;

        align-items: center;

        justify-content: center;

        position: relative;
    }


    .gauge-inner {

        width: 145px;

        height: 145px;

        border-radius: 50%;

        background: #0b1120;

        display: flex;

        flex-direction: column;

        align-items: center;

        justify-content: center;

        box-shadow:
            inset 0 0 25px rgba(0,0,0,0.5);
    }


    .gauge-value {

        font-size: 32px;

        font-weight: 900;

        color: #f8fafc;
    }


    .gauge-label {

        color: #64748b;

        font-size: 12px;

        margin-top: 2px;
    }


    /* =====================================================
       INPUT GROUP
       ===================================================== */

    .input-description {

        color: #64748b;

        font-size: 12px;

        margin-top: -7px;

        margin-bottom: 12px;
    }


    /* =====================================================
       RISK BADGE
       ===================================================== */

    .risk-badge {

        display: inline-block;

        padding: 8px 17px;

        border-radius: 999px;

        font-weight: 800;

        font-size: 14px;
    }


    /* =====================================================
       FEATURE CARDS
       ===================================================== */

    .feature-card {

        background:
            rgba(15,23,42,0.68);

        border:
            1px solid rgba(148,163,184,0.10);

        border-radius: 16px;

        padding: 20px;

        min-height: 145px;

        margin-bottom: 15px;
    }


    .feature-card-icon {

        font-size: 25px;

        margin-bottom: 8px;
    }


    .feature-card-title {

        font-size: 16px;

        font-weight: 750;

        color: #f8fafc;

        margin-bottom: 6px;
    }


    .feature-card-description {

        font-size: 13px;

        color: #94a3b8;

        line-height: 1.55;
    }


    /* =====================================================
       PIPELINE
       ===================================================== */

    .pipeline-step {

        text-align: center;

        background:
            rgba(15,23,42,0.75);

        border:
            1px solid rgba(96,165,250,0.12);

        border-radius: 15px;

        padding: 18px 10px;

        min-height: 100px;
    }


    .pipeline-icon {

        font-size: 27px;

        margin-bottom: 7px;
    }


    .pipeline-name {

        font-size: 13px;

        font-weight: 700;

        color: #e2e8f0;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {

        border-radius: 12px;

        border:
            1px solid rgba(96,165,250,0.25);

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #7c3aed
            );

        color: white;

        font-weight: 750;

        padding: 12px 20px;

        min-height: 45px;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }


    .stButton > button:hover {

        transform:
            translateY(-2px);

        box-shadow:
            0 12px 30px rgba(37,99,235,0.30);
    }


    /* =====================================================
       DOWNLOAD BUTTON
       ===================================================== */

    .stDownloadButton > button {

        border-radius: 12px;

        font-weight: 700;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .custom-footer {

        text-align: center;

        color: #64748b;

        font-size: 13px;

        line-height: 1.8;

        padding:
            45px 0 20px 0;

        margin-top: 40px;

        border-top:
            1px solid rgba(148,163,184,0.08);
    }


    .footer-title {

        color: #94a3b8;

        font-size: 15px;

        font-weight: 700;
    }


    /* =====================================================
       SIDEBAR BRAND
       ===================================================== */

    .sidebar-brand {

        text-align: center;

        padding:
            15px 0 25px 0;
    }


    .sidebar-logo {

        font-size: 48px;

        margin-bottom: 4px;
    }


    .sidebar-title {

        font-size: 22px;

        font-weight: 800;

        color: #f8fafc;
    }


    .sidebar-subtitle {

        color: #64748b;

        font-size: 13px;

        margin-top: 4px;
    }


    /* =====================================================
       TABLE
       ===================================================== */

    div[data-testid="stDataFrame"] {

        border-radius: 14px;

        overflow: hidden;
    }


    /* =====================================================
       MOBILE
       ===================================================== */

    @media (max-width: 768px) {

        .hero {
            padding: 28px;
        }

        .hero-title {
            font-size: 31px;
        }

        .result-probability {
            font-size: 45px;
        }

    }

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)

    except Exception as error:
        st.error(
            f"Unable to load the model: {error}"
        )
        return None


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


if "form_version" not in st.session_state:

    st.session_state.form_version = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                🩺
            </div>

            <div class="sidebar-title">
                Diabetes AI
            </div>

            <div class="sidebar-subtitle">
                Logistic Regression
            </div>

        </div>
        """
    )

    st.markdown("### Navigation")

    page = st.radio(
        "Navigation",
        [
            "🏠 Prediction",
            "📊 Model Insights",
            "📜 Prediction History",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model Status")

    if model is not None:

        st.success("✓ Model Loaded")

    else:

        st.error("✕ Model Not Found")

    st.caption(
        "Logistic Regression classification pipeline"
    )

    st.divider()

    st.markdown("### Pipeline")

    st.caption("✓ Missing value handling")
    st.caption("✓ Feature scaling")
    st.caption("✓ One-hot encoding")
    st.caption("✓ Logistic Regression")

    st.divider()

    st.caption(
        "⚠️ Educational and research use only."
    )


# ============================================================
# MODEL NOT FOUND
# ============================================================

if model is None:

    render_html(
        """
        <div class="hero">

            <div class="hero-icon">
                ⚠️
            </div>

            <div class="hero-title">
                Model File Not Found
            </div>

            <div class="hero-subtitle">
                The Streamlit application is running correctly,
                but the trained machine learning pipeline could
                not be found.
            </div>

        </div>
        """
    )

    st.markdown("### Expected model location")

    st.code(
        str(MODEL_PATH),
        language="text"
    )

    st.markdown("### Train the model")

    st.code(
        "python train_model.py "
        "--data data/diabetes_prediction_dataset.csv "
        "--model models/diabetes_pipeline.joblib",
        language="bash"
    )

    st.stop()


# ============================================================
# HERO HEADER
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="hero-icon">
            🩺
        </div>

        <div class="hero-title">
            Diabetes Prediction
        </div>

        <div class="hero-subtitle">
            AI-powered diabetes risk prediction using
            <b style="color:#c4b5fd;">
                Logistic Regression
            </b>.
            Enter patient health information to generate
            a model-based probability estimate.
        </div>

        <div class="hero-badge">
            ● MODEL READY &nbsp; • &nbsp; ML CLASSIFICATION
        </div>

    </div>
    """
)


# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "🏠 Prediction":

    render_html(
        """
        <div class="section-title">
            Patient Information
        </div>

        <div class="info-box">
            Enter the patient's demographic, medical and
            laboratory information below. The trained pipeline
            will automatically perform preprocessing before
            generating the prediction.
        </div>
        """
    )

    # Current form version
    v = st.session_state.form_version

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45,
            step=1,
            key=f"age_{v}"
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male",
                "Other"
            ],
            key=f"gender_{v}"
        )

    with col3:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            key=f"bmi_{v}"
        )

    # ========================================================
    # MEDICAL PARAMETERS
    # ========================================================

    render_html(
        """
        <div class="section-title">
            Medical Parameters
        </div>

        <div class="section-subtitle">
            Existing medical conditions and lifestyle information
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes",
            key=f"hypertension_{v}"
        )

    with col2:

        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes",
            key=f"heart_disease_{v}"
        )

    with col3:

        smoking_history = st.selectbox(
            "Smoking History",
            [
                "never",
                "No Info",
                "current",
                "former",
                "ever",
                "not current"
            ],
            key=f"smoking_{v}"
        )

    # ========================================================
    # LABORATORY PARAMETERS
    # ========================================================

    render_html(
        """
        <div class="section-title">
            Laboratory Measurements
        </div>

        <div class="section-subtitle">
            Important blood-sugar related measurements
        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        hba1c = st.number_input(
            "HbA1c Level (%)",
            min_value=3.0,
            max_value=20.0,
            value=5.7,
            step=0.1,
            key=f"hba1c_{v}"
        )

    with col2:

        glucose = st.number_input(
            "Blood Glucose Level",
            min_value=40,
            max_value=500,
            value=100,
            step=1,
            key=f"glucose_{v}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================
    # ACTION BUTTONS
    # ========================================================

    col1, col2, col3 = st.columns([1.2, 1.2, 3])

    with col1:

        predict_button = st.button(
            "🔍 Predict Diabetes",
            width="stretch"
        )

    with col2:

        reset_button = st.button(
            "↻ Reset",
            width="stretch"
        )

    if reset_button:

        st.session_state.form_version += 1

        st.rerun()

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        patient_data = pd.DataFrame(
            {
                "age": [age],
                "hypertension": [hypertension],
                "heart_disease": [heart_disease],
                "bmi": [bmi],
                "HbA1c_level": [hba1c],
                "blood_glucose_level": [glucose],
                "gender": [gender],
                "smoking_history": [smoking_history]
            }
        )

        try:

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            prediction = int(
                model.predict(patient_data)[0]
            )

            probability = float(
                model.predict_proba(patient_data)[0][1]
            )

            probability_percent = probability * 100

            # ------------------------------------------------
            # RISK CATEGORY
            # ------------------------------------------------

            if probability < 0.30:

                risk_level = "Low Risk"

                risk_color = "#34d399"

                risk_background = (
                    "rgba(6,78,59,0.45)"
                )

                risk_border = (
                    "rgba(52,211,153,0.25)"
                )

                risk_icon = "🟢"

                risk_message = (
                    "The model estimates a relatively "
                    "low probability of diabetes."
                )

            elif probability < 0.60:

                risk_level = "Moderate Risk"

                risk_color = "#fbbf24"

                risk_background = (
                    "rgba(120,53,15,0.40)"
                )

                risk_border = (
                    "rgba(251,191,36,0.25)"
                )

                risk_icon = "🟡"

                risk_message = (
                    "The model estimates a moderate "
                    "probability of diabetes."
                )

            else:

                risk_level = "High Risk"

                risk_color = "#f87171"

                risk_background = (
                    "rgba(127,29,29,0.45)"
                )

                risk_border = (
                    "rgba(248,113,113,0.25)"
                )

                risk_icon = "🔴"

                risk_message = (
                    "The model estimates a higher "
                    "probability of diabetes."
                )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if prediction == 1:

                result_title = (
                    "Model Predicts Diabetes"
                )

                result_icon = "⚠️"

            else:

                result_title = (
                    "Model Predicts No Diabetes"
                )

                result_icon = "✅"

            # ------------------------------------------------
            # GAUGE
            # ------------------------------------------------

            gauge_degree = probability * 360

            render_html(
                f"""
                <div class="result-card"
                     style="
                        background:
                            linear-gradient(
                                135deg,
                                {risk_background},
                                rgba(15,23,42,0.70)
                            );

                        border:
                            1px solid {risk_border};
                     ">

                    <div class="result-icon">
                        {result_icon}
                    </div>

                    <div class="result-title">
                        {result_title}
                    </div>

                    <div class="result-description">
                        Estimated diabetes probability
                    </div>

                    <div class="gauge-wrapper">

                        <div class="gauge"
                             style="
                                background:
                                    conic-gradient(
                                        {risk_color}
                                        0deg
                                        {gauge_degree}deg,
                                        rgba(148,163,184,0.10)
                                        {gauge_degree}deg
                                        360deg
                                    );
                             ">

                            <div class="gauge-inner">

                                <div class="gauge-value">
                                    {probability_percent:.1f}%
                                </div>

                                <div class="gauge-label">
                                    probability
                                </div>

                            </div>

                        </div>

                    </div>

                    <div>
                        <span class="risk-badge"
                              style="
                                background:
                                    {risk_background};

                                color:
                                    {risk_color};

                                border:
                                    1px solid
                                    {risk_border};
                              ">

                            {risk_icon}
                            {risk_level}

                        </span>
                    </div>

                </div>
                """
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            render_html(
                """
                <div class="section-title">
                    Prediction Summary
                </div>
                """
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                render_html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-icon">
                            🎯
                        </div>

                        <div class="metric-value">
                            {probability_percent:.1f}%
                        </div>

                        <div class="metric-label">
                            Probability
                        </div>

                    </div>
                    """
                )

            with c2:

                render_html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-icon">
                            {risk_icon}
                        </div>

                        <div class="metric-value"
                             style="font-size:21px;">
                            {risk_level}
                        </div>

                        <div class="metric-label">
                            Risk Category
                        </div>

                    </div>
                    """
                )

            with c3:

                render_html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-icon">
                            🧪
                        </div>

                        <div class="metric-value">
                            {hba1c:.1f}
                        </div>

                        <div class="metric-label">
                            HbA1c Level
                        </div>

                    </div>
                    """
                )

            with c4:

                render_html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-icon">
                            🩸
                        </div>

                        <div class="metric-value">
                            {glucose}
                        </div>

                        <div class="metric-label">
                            Blood Glucose
                        </div>

                    </div>
                    """
                )

            # ------------------------------------------------
            # PROBABILITY BAR
            # ------------------------------------------------

            render_html(
                """
                <div class="section-title">
                    Probability Analysis
                </div>
                """
            )

            st.progress(
                probability,
                text=(
                    f"Estimated diabetes probability: "
                    f"{probability_percent:.1f}%"
                )
            )

            st.info(risk_message)

            # ------------------------------------------------
            # PATIENT SUMMARY
            # ------------------------------------------------

            render_html(
                """
                <div class="section-title">
                    Patient Summary
                </div>
                """
            )

            summary = pd.DataFrame(
                {
                    "Parameter": [
                        "Age",
                        "Gender",
                        "BMI",
                        "Hypertension",
                        "Heart Disease",
                        "Smoking History",
                        "HbA1c Level",
                        "Blood Glucose"
                    ],

                    "Value": [
                        f"{age:.0f} years",
                        gender,
                        f"{bmi:.1f}",
                        "Yes" if hypertension else "No",
                        "Yes" if heart_disease else "No",
                        smoking_history,
                        f"{hba1c:.1f}%",
                        str(glucose)
                    ]
                }
            )

            st.dataframe(
                summary,
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------------
            # SAVE HISTORY
            # ------------------------------------------------

            record = {
                "Timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Age":
                    age,

                "Gender":
                    gender,

                "BMI":
                    bmi,

                "Hypertension":
                    "Yes" if hypertension else "No",

                "Heart Disease":
                    "Yes" if heart_disease else "No",

                "Smoking History":
                    smoking_history,

                "HbA1c":
                    hba1c,

                "Blood Glucose":
                    glucose,

                "Prediction":
                    "Diabetes"
                    if prediction == 1
                    else "No Diabetes",

                "Probability (%)":
                    round(
                        probability_percent,
                        2
                    ),

                "Risk":
                    risk_level
            }

            st.session_state.prediction_history.append(
                record
            )

            # ------------------------------------------------
            # MEDICAL DISCLAIMER
            # ------------------------------------------------

            st.warning(
                "This prediction is generated by a machine "
                "learning model and is not a medical diagnosis. "
                "Clinical decisions should be made by qualified "
                "healthcare professionals."
            )

        except Exception as error:

            st.error(
                "Prediction failed. Please verify the input "
                "values and make sure the trained pipeline "
                "matches the expected dataset columns."
            )

            with st.expander("Technical error"):

                st.code(
                    str(error)
                )


# ============================================================
# MODEL INSIGHTS PAGE
# ============================================================

elif page == "📊 Model Insights":

    render_html(
        """
        <div class="section-title">
            Model Insights
        </div>

        <div class="section-subtitle">
            Understand how the machine learning pipeline processes
            patient information.
        </div>

        <div class="glass-card">

            <h3>
                🤖 Logistic Regression
            </h3>

            <p style="
                color:#94a3b8;
                line-height:1.8;
            ">

                Logistic Regression is a supervised machine
                learning algorithm commonly used for binary
                classification.

                In this project, it estimates the probability
                that a patient belongs to the diabetes class
                based on demographic, medical and laboratory
                features.

            </p>

        </div>
        """
    )

    # ========================================================
    # PIPELINE
    # ========================================================

    render_html(
        """
        <div class="section-title">
            Machine Learning Pipeline
        </div>

        <div class="section-subtitle">
            The preprocessing steps are stored together with
            the trained model inside the Joblib pipeline.
        </div>
        """
    )

    pipeline_cols = st.columns(5)

    pipeline_data = [
        ("📋", "Input", "Patient data"),
        ("🧹", "Imputation", "Missing values"),
        ("📏", "Scaling", "StandardScaler"),
        ("🔤", "Encoding", "OneHotEncoder"),
        ("🧠", "Model", "Logistic Regression"),
    ]

    for column, item in zip(
        pipeline_cols,
        pipeline_data
    ):

        with column:

            icon, title, description = item

            render_html(
                f"""
                <div class="pipeline-step">

                    <div class="pipeline-icon">
                        {icon}
                    </div>

                    <div class="pipeline-name">
                        {title}
                    </div>

                    <div style="
                        color:#64748b;
                        font-size:11px;
                        margin-top:5px;
                    ">
                        {description}
                    </div>

                </div>
                """
            )

    # ========================================================
    # FEATURES
    # ========================================================

    render_html(
        """
        <div class="section-title">
            Model Features
        </div>
        """
    )

    features = [
        (
            "🎂",
            "Age",
            "Age of the patient."
        ),

        (
            "❤️",
            "Hypertension",
            "Whether the patient has hypertension."
        ),

        (
            "🫀",
            "Heart Disease",
            "Whether the patient has heart disease."
        ),

        (
            "⚖️",
            "BMI",
            "Body Mass Index."
        ),

        (
            "🧪",
            "HbA1c",
            "Long-term blood glucose indicator."
        ),

        (
            "🩸",
            "Blood Glucose",
            "Measured blood glucose level."
        ),

        (
            "👤",
            "Gender",
            "Categorical gender information."
        ),

        (
            "🚬",
            "Smoking History",
            "Patient smoking history."
        )
    ]

    for row_start in range(0, len(features), 4):

        cols = st.columns(4)

        for col, feature in zip(
            cols,
            features[row_start:row_start + 4]
        ):

            with col:

                icon, title, description = feature

                render_html(
                    f"""
                    <div class="feature-card">

                        <div class="feature-card-icon">
                            {icon}
                        </div>

                        <div class="feature-card-title">
                            {title}
                        </div>

                        <div class="feature-card-description">
                            {description}
                        </div>

                    </div>
                    """
                )

    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    render_html(
        """
        <div class="section-title">
            Technical Details
        </div>
        """
    )

    technical_data = pd.DataFrame(
        {
            "Component": [
                "Algorithm",
                "Numeric preprocessing",
                "Categorical preprocessing",
                "Missing values",
                "Train/Test split",
                "Random State",
                "Model storage"
            ],

            "Implementation": [
                "Logistic Regression",
                "StandardScaler",
                "OneHotEncoder",
                "SimpleImputer",
                "80% / 20%",
                "42",
                "Joblib"
            ]
        }
    )

    st.dataframe(
        technical_data,
        width="stretch",
        hide_index=True
    )

    st.info(
        "The deployed model uses the preprocessing pipeline "
        "saved during training. This prevents inconsistencies "
        "between training-time and prediction-time preprocessing."
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📜 Prediction History":

    render_html(
        """
        <div class="section-title">
            Prediction History
        </div>

        <div class="section-subtitle">
            Predictions made during the current application session.
        </div>
        """
    )

    history = st.session_state.prediction_history

    if not history:

        render_html(
            """
            <div class="glass-card"
                 style="text-align:center;">

                <div style="
                    font-size:45px;
                    margin-bottom:10px;
                ">
                    📭
                </div>

                <h3>
                    No Predictions Yet
                </h3>

                <p style="color:#64748b;">
                    Make a prediction from the Prediction page
                    and it will appear here.
                </p>

            </div>
            """
        )

    else:

        history_df = pd.DataFrame(history)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        total_predictions = len(history_df)

        diabetes_predictions = (
            history_df["Prediction"]
            == "Diabetes"
        ).sum()

        average_probability = (
            history_df["Probability (%)"]
            .mean()
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Total Predictions",
                total_predictions
            )

        with c2:

            st.metric(
                "Predicted Diabetes",
                diabetes_predictions
            )

        with c3:

            st.metric(
                "Average Probability",
                f"{average_probability:.1f}%"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv_data = history_df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Prediction History",
            data=csv_data,
            file_name="diabetes_prediction_history.csv",
            mime="text/csv",
            width="content"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # CLEAR
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear Session History"
        ):

            st.session_state.prediction_history = []

            st.rerun()


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About Project":

    render_html(
        """
        <div class="section-title">
            About Diabetes AI
        </div>

        <div class="section-subtitle">
            A machine learning project for diabetes
            classification and probability estimation.
        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        render_html(
            """
            <div class="glass-card">

                <h3>
                    🎯 Project Objective
                </h3>

                <p style="
                    color:#94a3b8;
                    line-height:1.8;
                ">

                    The objective of this project is to develop
                    a machine learning system that estimates
                    the likelihood of diabetes using patient
                    health information.

                </p>

                <h3>
                    🧠 Machine Learning Algorithm
                </h3>

                <p style="color:#94a3b8;">
                    Logistic Regression
                </p>

                <h3>
                    ⚙️ Technologies
                </h3>

                <p style="
                    color:#94a3b8;
                    line-height:2;
                ">

                    Python<br>
                    Pandas<br>
                    NumPy<br>
                    Scikit-learn<br>
                    Joblib<br>
                    Streamlit

                </p>

            </div>
            """
        )

    with col2:

        render_html(
            """
            <div class="glass-card">

                <h3>
                    📋 Input Parameters
                </h3>

                <ul style="
                    color:#94a3b8;
                    line-height:2;
                ">

                    <li>Age</li>
                    <li>Gender</li>
                    <li>BMI</li>
                    <li>Hypertension</li>
                    <li>Heart Disease</li>
                    <li>Smoking History</li>
                    <li>HbA1c Level</li>
                    <li>Blood Glucose Level</li>

                </ul>

                <h3>
                    🔐 Data Processing
                </h3>

                <p style="
                    color:#94a3b8;
                    line-height:1.7;
                ">

                    The application sends patient input to the
                    trained preprocessing and classification
                    pipeline. Categorical values are encoded
                    automatically and numerical values are scaled
                    before classification.

                </p>

            </div>
            """
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    render_html(
        """
        <div class="section-title">
            ⚠️ Important Disclaimer
        </div>

        <div class="glass-card"
             style="
                border-color:
                    rgba(251,191,36,0.20);
             ">

            <p style="
                color:#fbbf24;
                line-height:1.8;
            ">

                This application is intended for educational,
                demonstration and research purposes only.

                The model output is a statistical prediction
                and should not be interpreted as a medical
                diagnosis.

                Do not use this application to make medical
                decisions without consultation with a qualified
                healthcare professional.

            </p>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="custom-footer">

        <div class="footer-title">
            🩺 Diabetes AI Prediction Dashboard
        </div>

        <div>
            Built with Python • Scikit-learn • Streamlit
        </div>

        <div>
            Logistic Regression • Machine Learning • Healthcare Analytics
        </div>

        <div style="
            margin-top:10px;
            font-size:11px;
            color:#475569;
        ">
            For educational and research purposes only.
        </div>

    </div>
    """
)