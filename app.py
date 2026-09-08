
import json
import subprocess
import sys
from pathlib import Path

from groq import Groq
import pandas as pd
import streamlit as st
from streamlit_mic_recorder import speech_to_text

from src.ml_model import (
    load_model,
    predict_risk,
    classify_risk,
)

from src.copilot import chat

from src.knowledge import (
    search_knowledge,
    format_context,
    format_references,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "sources.json"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Paramedic AI",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# RUNSHEET-INSPIRED THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       FONTS
       ======================================================== */

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap'
    );


    /* ========================================================
       COLOR SYSTEM
       ======================================================== */

    :root {
        --bg: #060b09;
        --bg2: #09110e;

        --panel: #0c1713;
        --panel2: #101d18;
        --panel3: #13251e;

        --border: #1d342b;
        --border-light: #29483a;

        --text: #e5eee9;
        --muted: #91a69d;
        --muted2: #61776d;

        --green: #39e47f;
        --green-bright: #55ef96;
        --green-dark: #1b9f59;
        --green-soft: rgba(57, 228, 127, 0.09);

        --yellow: #f2c94c;
        --red: #ff5b5b;
        --blue: #5bbcff;
    }


    /* ========================================================
       GLOBAL APP
       ======================================================== */

    html,
    body,
    [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(57, 228, 127, 0.055),
                transparent 30%
            ),
            var(--bg);

        color: var(--text);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 4rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #07100d;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.1rem;
    }

    section[data-testid="stSidebar"] label {
        color: var(--muted);

        font-size: 0.68rem;
        font-weight: 700;

        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    section[data-testid="stSidebar"] input {
        background: #09140f !important;
    }


    /* ========================================================
       TOP BAR
       ======================================================== */

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;

        min-height: 64px;

        padding: 0 1.15rem;

        background: rgba(12, 23, 19, 0.94);

        border: 1px solid var(--border);

        margin-bottom: 1rem;

        box-shadow:
            0 10px 35px rgba(0, 0, 0, 0.20);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .brand-mark {
        width: 35px;
        height: 35px;

        display: flex;
        align-items: center;
        justify-content: center;

        background: var(--green);

        color: #041009;

        font-weight: 800;
        font-size: 1.05rem;

        border-radius: 3px;
    }

    .brand-title {
        color: var(--text);

        font-size: 0.94rem;
        font-weight: 800;

        letter-spacing: 0.06em;
    }

    .brand-subtitle {
        margin-top: 2px;

        color: var(--muted2);

        font-family: 'DM Mono', monospace;

        font-size: 0.58rem;

        text-transform: uppercase;
        letter-spacing: 0.10em;
    }

    .system-status {
        display: flex;
        align-items: center;

        gap: 0.45rem;

        color: var(--green);

        font-family: 'DM Mono', monospace;

        font-size: 0.63rem;

        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .status-dot {
        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: var(--green);

        box-shadow:
            0 0 12px rgba(57, 228, 127, 0.8);
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        position: relative;

        overflow: hidden;

        padding: 1.4rem 1.5rem;

        margin-bottom: 1.25rem;

        background:
            linear-gradient(
                135deg,
                rgba(26, 65, 48, 0.40),
                rgba(8, 17, 13, 0.95)
            );

        border: 1px solid var(--border);
    }

    .hero::after {
        content: "EMS";

        position: absolute;

        right: 1rem;
        bottom: -2rem;

        color: rgba(57, 228, 127, 0.025);

        font-family: 'DM Mono', monospace;

        font-size: 7rem;
        font-weight: 500;

        pointer-events: none;
    }

    .hero-title {
        color: var(--text);

        font-size: 1.65rem;
        font-weight: 800;

        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        max-width: 760px;

        margin-top: 0.35rem;

        color: var(--muted);

        font-size: 0.78rem;
        line-height: 1.55;
    }


    /* ========================================================
       SECTION LABEL
       ======================================================== */

    .section-label {
        display: flex;
        align-items: center;

        gap: 0.55rem;

        margin-top: 1.6rem;
        margin-bottom: 0.65rem;

        color: var(--muted);

        font-family: 'DM Mono', monospace;

        font-size: 0.62rem;

        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .section-label::before {
        content: "";

        width: 18px;
        height: 1px;

        background: var(--green);
    }


    /* ========================================================
       PANELS
       ======================================================== */

    .console-panel {
        background: var(--panel);

        border: 1px solid var(--border);

        padding: 1rem;
    }

    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;

        padding-bottom: 0.7rem;
        margin-bottom: 0.9rem;

        border-bottom: 1px solid var(--border);
    }

    .panel-title {
        color: var(--text);

        font-size: 0.75rem;
        font-weight: 800;

        text-transform: uppercase;
        letter-spacing: 0.09em;
    }

    .panel-code {
        color: var(--muted2);

        font-family: 'DM Mono', monospace;

        font-size: 0.58rem;

        text-transform: uppercase;
    }


    /* ========================================================
       VITALS
       ======================================================== */

    .vitals-grid {
        display: grid;

        grid-template-columns:
            repeat(6, minmax(0, 1fr));

        gap: 0.55rem;

        margin-top: 0.7rem;
    }

    .vital {
        background: #09140f;

        border: 1px solid var(--border);

        padding: 0.75rem;
    }

    .vital-label {
        color: var(--muted2);

        font-family: 'DM Mono', monospace;

        font-size: 0.55rem;

        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .vital-value {
        color: var(--text);

        margin-top: 0.3rem;

        font-family: 'DM Mono', monospace;

        font-size: 1rem;
        font-weight: 500;
    }

    .vital-unit {
        color: var(--muted2);

        font-size: 0.55rem;
    }


    /* ========================================================
       STREAMLIT METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: var(--panel);

        border: 1px solid var(--border);

        border-radius: 0;

        padding: 1rem;
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;

        font-size: 0.61rem !important;
        font-weight: 700 !important;

        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--text);

        font-family: 'DM Mono', monospace;

        font-size: 1.5rem;
    }


    /* ========================================================
       INPUTS
       ======================================================== */

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: #08120e !important;

        color: var(--text) !important;

        border: 1px solid var(--border) !important;

        border-radius: 2px !important;

        font-family: 'DM Mono', monospace !important;

        font-size: 0.78rem !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--green) !important;

        box-shadow:
            0 0 0 1px
            rgba(57, 228, 127, 0.15) !important;
    }


    /* ========================================================
       SELECTBOX
       ======================================================== */

    div[data-baseweb="select"] > div {
        background: #08120e !important;

        border-color: var(--border) !important;

        border-radius: 2px !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        min-height: 40px;

        border-radius: 2px !important;

        background: #10231b !important;

        color: var(--text) !important;

        border: 1px solid var(--border-light) !important;

        font-size: 0.67rem !important;

        font-weight: 800 !important;

        text-transform: uppercase;

        letter-spacing: 0.07em;
    }

    .stButton > button:hover {
        background: #153025 !important;

        color: var(--green) !important;

        border-color: var(--green) !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--green) !important;

        color: #041009 !important;

        border-color: var(--green) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--green-bright) !important;

        color: #041009 !important;
    }


    /* ========================================================
       BADGES
       ======================================================== */

    .badge {
        display: inline-flex;

        align-items: center;

        padding: 0.28rem 0.48rem;

        border-radius: 2px;

        font-family: 'DM Mono', monospace;

        font-size: 0.58rem;

        text-transform: uppercase;

        letter-spacing: 0.06em;
    }

    .badge-high {
        color: #ff7c7c;

        background: rgba(255, 91, 91, 0.08);

        border: 1px solid rgba(255, 91, 91, 0.3);
    }

    .badge-mid {
        color: var(--yellow);

        background: rgba(242, 201, 76, 0.08);

        border: 1px solid rgba(242, 201, 76, 0.25);
    }

    .badge-low {
        color: var(--green);

        background: var(--green-soft);

        border: 1px solid rgba(57, 228, 127, 0.25);
    }

    .badge-neutral {
        color: var(--muted);

        background: rgba(141, 166, 157, 0.06);

        border: 1px solid var(--border);
    }


    /* ========================================================
       CHAT
       ======================================================== */

    div[data-testid="stChatMessage"] {
        background: var(--panel) !important;

        border: 1px solid var(--border);

        border-radius: 2px !important;

        margin-bottom: 0.55rem;
    }

    div[data-testid="stChatMessage"] p {
        color: var(--text);

        font-size: 0.83rem;

        line-height: 1.65;
    }

    div[data-testid="stChatInput"] {
        border-color: var(--border) !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: #08120e !important;

        color: var(--text) !important;

        border-radius: 2px !important;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    div[data-testid="stExpander"] {
        background: var(--panel);

        border: 1px solid var(--border);

        border-radius: 2px;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 2px !important;

        border: 1px solid var(--border) !important;

        background: #0b1712 !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    section[data-testid="stFileUploaderDropzone"] {
        background: #09140f !important;

        border: 1px dashed var(--border-light) !important;

        border-radius: 2px !important;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
    }


    /* ========================================================
       SOURCE CARDS
       ======================================================== */

    .source-card {
        background: var(--panel);

        border: 1px solid var(--border);

        padding: 0.85rem 0.95rem;

        margin-bottom: 0.45rem;
    }

    .source-title {
        color: var(--text);

        font-size: 0.76rem;

        font-weight: 700;
    }

    .source-meta {
        margin-top: 0.35rem;

        color: var(--muted2);

        font-family: 'DM Mono', monospace;

        font-size: 0.57rem;

        line-height: 1.6;

        text-transform: uppercase;
    }


    /* ========================================================
       TTS
       ======================================================== */

    .tts-label {
        display: inline-flex;

        align-items: center;

        gap: 0.4rem;

        color: var(--green);

        font-family: 'DM Mono', monospace;

        font-size: 0.61rem;

        text-transform: uppercase;

        letter-spacing: 0.08em;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        margin-top: 3rem;

        padding-top: 1rem;

        border-top: 1px solid var(--border);

        color: var(--muted2);

        font-family: 'DM Mono', monospace;

        font-size: 0.58rem;

        line-height: 1.7;

        text-transform: uppercase;

        letter-spacing: 0.04em;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 900px) {

        .vitals-grid {
            grid-template-columns:
                repeat(3, minmax(0, 1fr));
        }

        .system-status {
            display: none;
        }

    }

    @media (max-width: 600px) {

        .vitals-grid {
            grid-template-columns:
                repeat(2, minmax(0, 1fr));
        }

        .hero-title {
            font-size: 1.3rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
    <div class="topbar">

        <div class="brand">

            <div class="brand-mark">
                +
            </div>

            <div>
                <div class="brand-title">
                    PARAMEDIC AI
                </div>

                <div class="brand-subtitle">
                    EMS EDUCATION / DECISION SUPPORT
                </div>
            </div>

        </div>

        <div class="system-status">

            <span class="status-dot"></span>

            SYSTEM ONLINE

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            FIELD INTELLIGENCE CONSOLE
        </div>

        <div class="hero-subtitle">
            Machine-learning risk assessment, grounded EMS
            knowledge retrieval, voice-enabled copilot assistance,
            and authorized reference management.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — PATIENT ASSESSMENT
# ============================================================

st.sidebar.markdown(
    """
    <div class="section-label">
        PATIENT ASSESSMENT
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="
        color:#39e47f;
        font-family:'DM Mono',monospace;
        font-size:0.59rem;
        margin-bottom:0.9rem;
        text-transform:uppercase;
        letter-spacing:0.09em;
    ">
        VITAL SIGNS / PRIMARY DATA
    </div>
    """,
    unsafe_allow_html=True,
)


age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=50,
)

heart_rate = st.sidebar.number_input(
    "Heart Rate",
    min_value=0.0,
    max_value=300.0,
    value=90.0,
)

systolic_bp = st.sidebar.number_input(
    "Systolic BP",
    min_value=0.0,
    max_value=300.0,
    value=120.0,
)

diastolic_bp = st.sidebar.number_input(
    "Diastolic BP",
    min_value=0.0,
    max_value=200.0,
    value=80.0,
)

respiratory_rate = st.sidebar.number_input(
    "Respiratory Rate",
    min_value=0.0,
    max_value=100.0,
    value=18.0,
)

spo2 = st.sidebar.number_input(
    "SpO₂",
    min_value=0.0,
    max_value=100.0,
    value=98.0,
)

temperature = st.sidebar.number_input(
    "Temperature °F",
    min_value=80.0,
    max_value=115.0,
    value=98.6,
)


st.sidebar.markdown(
    """
    <div style="
        margin-top:0.9rem;
        padding:0.75rem;
        background:#09140f;
        border:1px solid #1d342b;
    ">

        <div style="
            color:#39e47f;
            font-family:'DM Mono',monospace;
            font-size:0.58rem;
            text-transform:uppercase;
            letter-spacing:0.08em;
        ">
            INPUT STATUS
        </div>

        <div style="
            color:#61776d;
            font-size:0.68rem;
            margin-top:0.3rem;
            line-height:1.45;
        ">
            Patient values ready for assessment.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATIENT SNAPSHOT
# ============================================================

st.markdown(
    """
    <div class="section-label">
        PATIENT SNAPSHOT
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="console-panel">

        <div class="panel-header">

            <div class="panel-title">
                CURRENT VITALS
            </div>

            <div class="panel-code">
                LIVE INPUT
            </div>

        </div>

        <div class="vitals-grid">

            <div class="vital">
                <div class="vital-label">AGE</div>
                <div class="vital-value">{age}</div>
            </div>

            <div class="vital">
                <div class="vital-label">HR</div>
                <div class="vital-value">{heart_rate:.0f}
                    <span class="vital-unit">BPM</span>
                </div>
            </div>

            <div class="vital">
                <div class="vital-label">BP</div>
                <div class="vital-value">
                    {systolic_bp:.0f}/{diastolic_bp:.0f}
                </div>
            </div>

            <div class="vital">
                <div class="vital-label">RR</div>
                <div class="vital-value">{respiratory_rate:.0f}
                    <span class="vital-unit">/MIN</span>
                </div>
            </div>

            <div class="vital">
                <div class="vital-label">SPO₂</div>
                <div class="vital-value">{spo2:.0f}
                    <span class="vital-unit">%</span>
                </div>
            </div>

            <div class="vital">
                <div class="vital-label">TEMP</div>
                <div class="vital-value">{temperature:.1f}
                    <span class="vital-unit">°F</span>
                </div>
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ML ASSESSMENT
# ============================================================

st.markdown(
    """
    <div class="section-label">
        RISK ASSESSMENT
    </div>
    """,
    unsafe_allow_html=True,
)

col_input, col_result = st.columns(
    [0.75, 1.25]
)


with col_input:

    st.markdown(
        """
        <div class="console-panel">

            <div class="panel-header">

                <div class="panel-title">
                    ML ASSESSMENT
                </div>

                <div class="panel-code">
                    RISK MODEL
                </div>

            </div>

            <div style="
                color:#91a69d;
                font-size:0.72rem;
                line-height:1.55;
                margin-bottom:0.9rem;
            ">
                Evaluate the current patient inputs using
                the demonstration machine-learning model.
            </div>

        """,
        unsafe_allow_html=True,
    )

    run_assessment = st.button(
        "RUN ML ASSESSMENT",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


if run_assessment:

    patient = pd.DataFrame(
        [
            {
                "age": age,
                "heart_rate": heart_rate,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "respiratory_rate": respiratory_rate,
                "spo2": spo2,
                "temperature": temperature,
            }
        ]
    )

    try:

        model = load_model()

        probability = predict_risk(
            model,
            age,
            heart_rate,
            systolic_bp,
            diastolic_bp,
            respiratory_rate,
            spo2,
            temperature,
        )

        category = classify_risk(
            probability
        )

        badge_class = {
            "HIGHER RISK": "badge-high",
            "INTERMEDIATE RISK": "badge-mid",
            "LOWER RISK": "badge-low",
        }.get(
            category,
            "badge-neutral",
        )


        with col_result:

            metric_col, status_col = st.columns(2)

            with metric_col:

                st.metric(
                    "Estimated Risk",
                    f"{probability * 100:.1f}%",
                )

            with status_col:

                st.markdown(
                    f"""
                    <div class="console-panel">

                        <div style="
                            color:#61776d;
                            font-family:'DM Mono',monospace;
                            font-size:0.58rem;
                            text-transform:uppercase;
                            letter-spacing:0.08em;
                            margin-bottom:0.75rem;
                        ">
                            CLASSIFICATION
                        </div>

                        <span class="badge {badge_class}">
                            {category}
                        </span>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        with st.expander(
            "VIEW PATIENT DATA"
        ):

            st.dataframe(
                patient,
                use_container_width=True,
                hide_index=True,
            )


        st.warning(
            "DEMONSTRATION MODEL — NOT A CLINICAL DIAGNOSIS."
        )


    except Exception as error:

        st.error(
            f"ML prediction error: {error}"
        )


# ============================================================
# COPILOT
# ============================================================

st.markdown(
    """
    <div class="section-label">
        COPILOT
    </div>

    <div class="panel-header">

        <div class="panel-title">
            PARAMEDIC COPILOT
        </div>

        <div class="panel-code">
            GROUNDED / EMS KNOWLEDGE
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "copilot_messages" not in st.session_state:

    st.session_state.copilot_messages = []


if "voice_text" not in st.session_state:

    st.session_state.voice_text = ""


if "last_copilot_answer" not in st.session_state:

    st.session_state.last_copilot_answer = ""


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.copilot_messages:

    avatar = (
        "🚑"
        if message["role"] == "assistant"
        else "●"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# VOICE INPUT
# ============================================================

st.markdown(
    """
    <div class="tts-label">
        ● VOICE INPUT / REVIEW BEFORE SEND
    </div>
    """,
    unsafe_allow_html=True,
)


voice_question = speech_to_text(
    language="en",
    start_prompt="START SPEAKING",
    stop_prompt="STOP RECORDING",
    use_container_width=True,
    key="voice_input",
)


if voice_question:

    st.session_state.voice_text = (
        voice_question
    )


# ============================================================
# TEXT INPUT
# ============================================================

typed_question = st.chat_input(
    "Enter an EMS question..."
)


if typed_question:

    st.session_state.voice_text = (
        typed_question
    )


# ============================================================
# REVIEW TRANSCRIPTION
# ============================================================

if st.session_state.voice_text:

    st.markdown(
        """
        <div class="section-label">
            INPUT REVIEW
        </div>
        """,
        unsafe_allow_html=True,
    )


    reviewed_question = st.text_area(
        "REVIEW QUESTION",
        value=st.session_state.voice_text,
        height=120,
        key="reviewed_question",
        label_visibility="collapsed",
    )


    col1, col2 = st.columns(2)


    with col1:

        send_question = st.button(
            "SEND TO COPILOT",
            type="primary",
            use_container_width=True,
        )


    with col2:

        clear_question = st.button(
            "CLEAR INPUT",
            use_container_width=True,
        )


    if clear_question:

        st.session_state.voice_text = ""

        st.rerun()


    if send_question:

        question = reviewed_question.strip()


        if not question:

            st.warning(
                "Please enter or speak a question first."
            )


        else:

            # ------------------------------------------------
            # USER MESSAGE
            # ------------------------------------------------

            st.session_state.copilot_messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )


            with st.chat_message(
                "user",
                avatar="●",
            ):

                st.markdown(
                    question
                )


            # ------------------------------------------------
            # COPILOT RESPONSE
            # ------------------------------------------------

            with st.chat_message(
                "assistant",
                avatar="🚑",
            ):

                with st.spinner(
                    "SEARCHING KNOWLEDGE / PROCESSING..."
                ):

                    try:

                        # ----------------------------------------
                        # 1. SEARCH KNOWLEDGE
                        # ----------------------------------------

                        results = search_knowledge(
                            question,
                            top_k=3,
                        )

                        context = format_context(
                            results
                        )


                        # ----------------------------------------
                        # 2. LIMIT CONTEXT
                        # ----------------------------------------

                        MAX_CONTEXT_CHARS = 6000

                        if len(context) > MAX_CONTEXT_CHARS:

                            context = context[
                                :MAX_CONTEXT_CHARS
                            ]


                        # ----------------------------------------
                        # 3. LIMIT HISTORY
                        # ----------------------------------------

                        MAX_MESSAGES = 6

                        recent_messages = (
                            st.session_state
                            .copilot_messages[
                                -MAX_MESSAGES:
                            ]
                        )


                        # ----------------------------------------
                        # 4. CALL COPILOT
                        # ----------------------------------------

                        answer = chat(
                            messages=recent_messages,
                            context=context,
                        )


                        # ----------------------------------------
                        # 5. REFERENCES
                        # ----------------------------------------

                        references = (
                            format_references(
                                results
                            )
                        )


                        if references:

                            answer += (
                                "\n\n"
                                "### References used\n"
                                + "\n".join(
                                    references
                                )
                            )


                        # ----------------------------------------
                        # 6. DISPLAY
                        # ----------------------------------------

                        st.markdown(
                            answer
                        )


                        # ----------------------------------------
                        # 7. SAVE
                        # ----------------------------------------

                        st.session_state.copilot_messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )


                        st.session_state.last_copilot_answer = (
                            answer
                        )


                        # ----------------------------------------
                        # 8. CLEAR
                        # ----------------------------------------

                        st.session_state.voice_text = ""


                    except Exception as error:

                        st.error(
                            f"Copilot error: {error}"
                        )


# ============================================================
# TEXT TO SPEECH
# ============================================================

st.markdown(
    """
    <div class="section-label">
        AUDIO OUTPUT
    </div>
    """,
    unsafe_allow_html=True,
)


if st.button(
    "READ LATEST RESPONSE ALOUD",
    use_container_width=True,
):

    answer = (
        st.session_state
        .last_copilot_answer
    )


    if not answer:

        st.warning(
            "There is no Copilot response to read yet."
        )


    else:

        try:

            client = Groq(
                api_key=st.secrets[
                    "GROQ_API_KEY"
                ]
            )


            # Groq Orpheus currently limits
            # each TTS request to 200 chars.

            chunks = [
                answer[i:i + 200]
                for i in range(
                    0,
                    len(answer),
                    200,
                )
            ]


            for chunk in chunks:

                response = client.audio.speech.create(
                    model="canopylabs/orpheus-v1-english",
                    voice="troy",
                    input=chunk,
                    response_format="wav",
                )


                audio_bytes = response.read()


                st.audio(
                    audio_bytes,
                    format="audio/wav",
                )


            st.success(
                "AUDIO READY — PRESS PLAY."
            )


        except Exception as error:

            st.error(
                f"Groq TTS error: {error}"
            )


# ============================================================
# KNOWLEDGE MANAGER
# ============================================================

st.markdown(
    """
    <div class="section-label">
        KNOWLEDGE
    </div>

    <div class="panel-header">

        <div class="panel-title">
            KNOWLEDGE MANAGER
        </div>

        <div class="panel-code">
            AUTHORIZED REFERENCES
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


st.caption(
    "Upload and manage authorized EMS reference documents."
)


uploaded_file = st.file_uploader(
    "UPLOAD PDF OR WORD REFERENCE",
    type=[
        "pdf",
        "docx",
    ],
)


if uploaded_file:

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    destination = (
        DOCUMENTS_DIR
        / uploaded_file.name
    )


    destination.write_bytes(
        uploaded_file.getbuffer()
    )


    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    registry = {}


    if REGISTRY_PATH.exists():

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8-sig",
        ) as file:

            registry = json.load(
                file
            )


    existing = registry.get(
        uploaded_file.name,
        {}
    )


    with st.expander(
        "SOURCE METADATA",
        expanded=True,
    ):

        title = st.text_input(
            "Title",
            value=existing.get(
                "title",
                Path(
                    uploaded_file.name
                ).stem,
            ),
        )


        jurisdiction = st.text_input(
            "Jurisdiction",
            value=existing.get(
                "jurisdiction",
                "UNSPECIFIED",
            ),
        )


        document_type = st.selectbox(
            "Document Type",
            [
                "educational_reference",
                "agency_protocol",
                "medical_director_order",
                "manufacturer_reference",
                "other",
            ],
        )


        effective_date = st.text_input(
            "Effective Date",
            value=existing.get(
                "effective_date",
                "",
            ) or "",
        )


        version = st.text_input(
            "Version",
            value=existing.get(
                "version",
                "",
            ) or "",
        )


        authority = st.text_input(
            "Authority / Issuing Organization",
            value=existing.get(
                "authority",
                "",
            ) or "",
        )


        status = st.selectbox(
            "Status",
            [
                "active",
                "draft",
                "inactive",
                "retired",
                "superseded",
            ],
        )


        review_required = st.checkbox(
            "Requires Review",
            value=existing.get(
                "review_required",
                True,
            ),
        )


        if st.button(
            "SAVE METADATA"
        ):

            registry[
                uploaded_file.name
            ] = {

                "title": title,

                "jurisdiction": jurisdiction,

                "document_type": document_type,

                "effective_date": (
                    effective_date
                    or None
                ),

                "version": (
                    version
                    or None
                ),

                "authority": authority,

                "status": status,

                "review_required": (
                    review_required
                ),
            }


            REGISTRY_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            with open(
                REGISTRY_PATH,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    registry,
                    file,
                    indent=2,
                )


            st.success(
                "Metadata saved."
            )


        if st.button(
            "INGEST & REBUILD KNOWLEDGE INDEX"
        ):

            ingestion = subprocess.run(
                [
                    sys.executable,
                    str(
                        PROJECT_ROOT
                        / "src"
                        / "ingest.py"
                    ),
                ],
                capture_output=True,
                text=True,
            )


            if ingestion.returncode != 0:

                st.error(
                    "Document ingestion failed."
                )

                st.code(
                    ingestion.stderr
                )


            else:

                index_result = subprocess.run(
                    [
                        sys.executable,
                        str(
                            PROJECT_ROOT
                            / "src"
                            / "build_index.py"
                        ),
                    ],
                    capture_output=True,
                    text=True,
                )


                if index_result.returncode != 0:

                    st.error(
                        "Knowledge index rebuild failed."
                    )

                    st.code(
                        index_result.stderr
                    )


                else:

                    st.success(
                        "Knowledge index rebuilt successfully."
                    )

                    st.code(
                        index_result.stdout
                    )


# ============================================================
# REGISTERED SOURCES
# ============================================================

with st.expander(
    "REGISTERED SOURCES"
):

    if REGISTRY_PATH.exists():

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8-sig",
        ) as file:

            registry = json.load(
                file
            )


        if registry:

            for filename, metadata in registry.items():

                status = str(
                    metadata.get(
                        "status",
                        "UNKNOWN",
                    )
                ).lower()


                badge_class = {
                    "active": "badge-low",
                    "draft": "badge-mid",
                }.get(
                    status,
                    "badge-neutral",
                )


                st.markdown(
                    f"""
                    <div class="source-card">

                        <div style="
                            display:flex;
                            align-items:center;
                            justify-content:space-between;
                            gap:1rem;
                        ">

                            <div class="source-title">
                                {metadata.get(
                                    'title',
                                    filename
                                )}
                            </div>

                            <span class="badge {badge_class}">
                                {status.upper()}
                            </span>

                        </div>

                        <div class="source-meta">

                            FILE / {filename}

                            <br>

                            JURISDICTION /
                            {metadata.get(
                                'jurisdiction',
                                'UNSPECIFIED'
                            )}

                            <br>

                            TYPE /
                            {metadata.get(
                                'document_type',
                                'UNKNOWN'
                            )}

                            <br>

                            VERSION /
                            {metadata.get(
                                'version'
                            ) or '—'}

                            <br>

                            AUTHORITY /
                            {metadata.get(
                                'authority'
                            ) or 'UNSPECIFIED'}

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                if metadata.get(
                    "review_required",
                    True,
                ):

                    st.warning(
                        "REVIEW REQUIRED"
                    )

                else:

                    st.success(
                        "REVIEW COMPLETE"
                    )


        else:

            st.info(
                "No registered sources."
            )


    else:

        st.info(
            "No knowledge registry found."
        )


# ============================================================
# SAFETY NOTICE
# ============================================================

st.markdown(
    """
    <div class="footer">

        DEMONSTRATION SYSTEM /
        NOT FOR CLINICAL DECISION MAKING

        <br><br>

        Always follow current local EMS protocols,
        medical direction, scope of practice,
        manufacturer instructions, and applicable regulations.

    </div>
    """,
    unsafe_allow_html=True,
)
