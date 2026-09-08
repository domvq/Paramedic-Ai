import json
import subprocess
import sys
from pathlib import Path
from groq import Groq

import pandas as pd
import streamlit as st
import pyttsx3
import tempfile
import os
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
    page_title="Paramedic AI // Copilot",
    page_icon="🚑",
    layout="wide",
)


# ============================================================
# THEME — MATCHES runsheet.website EXACTLY
# ============================================================
# Same CSS variables, fonts, and component styling pulled straight
# from the runsheet.website stylesheet (Oswald display font, IBM
# Plex Sans body, IBM Plex Mono for labels/data, dark green cardiac
# palette). Visual theme only — no functional logic below is
# changed from the original app.

BG = "#0a1210"
BG_RAISED = "#101c19"
BG_CARD = "#0e1a17"
HAIRLINE = "#1e3229"
TRACE_GREEN = "#3ce089"
TRACE_AMBER = "#ffb020"
TRACE_RED = "#ff5a5a"
TEXT = "#e6f2ec"
TEXT_DIM = "#7fa196"
TEXT_DIMMER = "#4d685f"

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', sans-serif !important;
        }}

        .stApp {{
            background:
                radial-gradient(ellipse at top, #0d1a16 0%, {BG} 55%);
            color: {TEXT};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {BG_RAISED};
            border-right: 1px solid {HAIRLINE};
        }}

        section[data-testid="stSidebar"] * {{
            color: {TEXT} !important;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Oswald', sans-serif !important;
            font-weight: 600;
            letter-spacing: 0.01em;
            color: {TEXT} !important;
        }}

        code, pre, .stCode, div[data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace !important;
        }}

        /* top masthead, styled like the runsheet.website header bar */
        .rs-masthead {{
            border-bottom: 1px solid {HAIRLINE};
            background: rgba(10,18,16,0.92);
            padding: 0.9rem 1.2rem;
            margin-bottom: 1.4rem;
        }}

        .rs-masthead h1 {{
            margin: 0;
            font-size: 1.4rem;
            color: {TEXT} !important;
            letter-spacing: 0.06em;
        }}

        .rs-masthead .rs-sub {{
            color: {TEXT_DIM};
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-top: 0.2rem;
        }}

        /* section dividers — mirrors the "eyebrow" label on runsheet.website */
        .rs-section {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin: 1.6rem 0 0.6rem 0;
        }}

        .rs-section .rs-tag {{
            color: {TRACE_GREEN};
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 500;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            font-size: 0.7rem;
            white-space: nowrap;
        }}

        .rs-section .rs-line {{
            flex: 1;
            height: 1px;
            background: {HAIRLINE};
        }}

        /* panel / card container, matches .ecg-card / .result-card */
        .rs-panel {{
            border: 1px solid {HAIRLINE};
            border-left: 3px solid {TRACE_GREEN};
            background-color: {BG_CARD};
            padding: 1rem 1.2rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }}

        /* disclaimer boxes — dashed hairline border, dim mono text,
           exactly like .disclaimer on runsheet.website */
        .rs-disclaimer {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: {TEXT_DIMMER};
            border: 1px dashed {HAIRLINE};
            border-radius: 8px;
            padding: 12px 14px;
            line-height: 1.6;
            margin: 0.6rem 0;
        }}

        .rs-disclaimer strong {{
            color: {TRACE_AMBER};
        }}

        .rs-disclaimer.rs-red strong {{
            color: {TRACE_RED};
        }}

        /* risk banners */
        .rs-risk {{
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            font-family: 'Oswald', sans-serif;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            text-align: center;
            border: 1px solid;
        }}

        .rs-risk-high {{
            color: {TRACE_RED};
            border-color: {TRACE_RED};
            background: rgba(255,90,90,0.08);
        }}

        .rs-risk-mid {{
            color: {TRACE_AMBER};
            border-color: {TRACE_AMBER};
            background: rgba(255,176,32,0.08);
        }}

        .rs-risk-low {{
            color: {TRACE_GREEN};
            border-color: {TRACE_GREEN};
            background: rgba(60,224,137,0.08);
        }}

        div[data-testid="stMetric"] {{
            background-color: {BG_CARD};
            border: 1px solid {HAIRLINE};
            border-radius: 10px;
            padding: 0.6rem 0.9rem;
        }}

        div[data-testid="stMetricValue"] {{
            color: {TRACE_GREEN} !important;
        }}

        /* buttons, matches .btn-primary / .btn-ghost */
        .stButton > button {{
            background-color: transparent;
            color: {TEXT_DIM};
            border: 1px solid {HAIRLINE};
            border-radius: 8px;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.75rem;
            letter-spacing: 0.06em;
            padding: 12px 18px;
        }}

        .stButton > button:hover {{
            color: {TEXT};
            border-color: {TEXT_DIM};
        }}

        .stButton > button[kind="primary"] {{
            background-color: {TRACE_GREEN};
            color: #04140d;
            border: none;
            font-family: 'Oswald', sans-serif !important;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 0 22px rgba(60,224,137,.35);
        }}

        div[data-testid="stChatMessage"] {{
            background-color: {BG_CARD};
            border: 1px solid {HAIRLINE};
            border-radius: 10px;
        }}

        .rs-footer {{
            border-top: 1px solid {HAIRLINE};
            margin-top: 2rem;
            padding-top: 1rem;
            color: {TEXT_DIMMER};
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            text-align: center;
            letter-spacing: 0.05em;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def rs_section(tag: str):
    """Render a runsheet-style '// SECTION' divider."""

    st.markdown(
        f"""
        <div class="rs-section">
            <span class="rs-tag">// {tag}</span>
            <span class="rs-line"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="rs-masthead">
        <h1>🚑 PARAMEDIC AI // COPILOT</h1>
        <div class="rs-sub">EMS EDUCATION &amp; DECISION-SUPPORT DEMONSTRATION</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — PATIENT ASSESSMENT
# ============================================================

st.sidebar.markdown(
    "<div class='rs-tag' style=\"color:#3ce089;font-family:'IBM Plex Mono',monospace;"
    "font-weight:500;letter-spacing:0.22em;text-transform:uppercase;"
    "font-size:0.7rem;\">// PATIENT ASSESSMENT</div>",
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


# ============================================================
# ML ASSESSMENT
# ============================================================

rs_section("ML Risk Assessment")

if st.button(
    "Run ML Assessment",
    type="primary",
):

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

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Estimated Risk Probability",
                f"{probability * 100:.1f}%",
            )

        with col2:

            if category == "HIGHER RISK":

                st.markdown(
                    "<div class='rs-risk rs-risk-high'>⚠ HIGHER RISK</div>",
                    unsafe_allow_html=True,
                )

            elif category == "INTERMEDIATE RISK":

                st.markdown(
                    "<div class='rs-risk rs-risk-mid'>◐ INTERMEDIATE RISK</div>",
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    "<div class='rs-risk rs-risk-low'>✓ LOWER RISK</div>",
                    unsafe_allow_html=True,
                )

        with st.expander(
            "View patient data"
        ):

            st.dataframe(
                patient,
                use_container_width=True,
            )

        st.markdown(
            "<div class='rs-disclaimer'>This prediction is generated by a "
            "demonstration machine-learning model and is <b>NOT</b> a "
            "clinical diagnosis.</div>",
            unsafe_allow_html=True,
        )

    except Exception as error:

        st.markdown(
            f"<div class='rs-disclaimer rs-red'>ML prediction error: {error}</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# COPILOT
# ============================================================

st.divider()

rs_section("Paramedic Copilot")


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

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# VOICE INPUT
# ============================================================

st.caption(
    "🎙️ SPEAK YOUR QUESTION, THEN REVIEW THE TRANSCRIPTION BEFORE SENDING."
)

voice_question = speech_to_text(
    language="en",
    start_prompt="🎙️ Start speaking",
    stop_prompt="⏹️ Stop recording",
    use_container_width=True,
    key="voice_input",
)


if voice_question:

    st.session_state.voice_text = voice_question


# ============================================================
# TEXT INPUT
# ============================================================

typed_question = st.chat_input(
    "Type your question..."
)


if typed_question:

    st.session_state.voice_text = typed_question


# ============================================================
# REVIEW TRANSCRIPTION
# ============================================================

if st.session_state.voice_text:

    st.markdown("##### 📝 REVIEW YOUR QUESTION")

    reviewed_question = st.text_area(
        "Edit the transcription if needed:",
        value=st.session_state.voice_text,
        height=120,
        key="reviewed_question",
    )

    col1, col2 = st.columns(2)

    with col1:

        send_question = st.button(
            "📤 Send to Copilot",
            type="primary",
            use_container_width=True,
        )

    with col2:

        clear_question = st.button(
            "🗑️ Clear",
            use_container_width=True,
        )

    if clear_question:

        st.session_state.voice_text = ""

        st.rerun()


    if send_question:

        question = reviewed_question.strip()

        if not question:

            st.markdown(
                "<div class='rs-disclaimer'>Please enter or speak a question first.</div>",
                unsafe_allow_html=True,
            )
        else:

            # ----------------------------------------------------
            # USER MESSAGE
            # ----------------------------------------------------

            st.session_state.copilot_messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):

                st.markdown(question)


            # ----------------------------------------------------
            # COPILOT RESPONSE
            # ----------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching knowledge and thinking..."
                ):
                    try:

                        # ------------------------------------------------
                        # 1. SEARCH KNOWLEDGE
                        # ------------------------------------------------

                        results = search_knowledge(
                            question,
                            top_k=3,
                        )

                        context = format_context(
                            results
                        )


                        # ------------------------------------------------
                        # 2. LIMIT CONTEXT
                        # ------------------------------------------------

                        MAX_CONTEXT_CHARS = 6000

                        if len(context) > MAX_CONTEXT_CHARS:

                            context = context[
                                :MAX_CONTEXT_CHARS
                            ]


                        # ------------------------------------------------
                        # 3. LIMIT CONVERSATION HISTORY
                        # ------------------------------------------------

                        MAX_MESSAGES = 6

                        recent_messages = (
                            st.session_state
                            .copilot_messages[
                                -MAX_MESSAGES:
                            ]
                        )


                        # ------------------------------------------------
                        # 4. CALL COPILOT
                        # ------------------------------------------------

                        answer = chat(
                            messages=recent_messages,
                            context=context,
                        )


                        # ------------------------------------------------
                        # 5. ADD REFERENCES
                        # ------------------------------------------------

                        references = (
                            format_references(
                                results
                            )
                        )

                        if references:

                            answer += (
                                "\n\n"
                                "### 📚 References used\n"
                                + "\n".join(
                                    references
                                )
                            )


                        # ------------------------------------------------
                        # 6. DISPLAY ANSWER
                        # ------------------------------------------------

                        st.markdown(
                            answer
                        )


                        # ------------------------------------------------
                        # 7. SAVE ANSWER
                        # ------------------------------------------------

                        st.session_state.copilot_messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )

                        st.session_state.last_copilot_answer = answer


                        # ------------------------------------------------
                        # 8. CLEAR INPUT
                        # ------------------------------------------------

                        st.session_state.voice_text = ""

                    except Exception as error:

                        st.markdown(
                            f"<div class='rs-disclaimer rs-red'>Copilot error: {error}</div>",
                            unsafe_allow_html=True,
                        )

# ============================================================
# PYTHON TEXT-TO-SPEECH
# ============================================================

if st.button(
    "🔊 Read Latest Response Aloud",
    use_container_width=True,
):

    answer = st.session_state.last_copilot_answer

    if not answer:

        st.markdown(
            "<div class='rs-disclaimer'>There is no Copilot response to read yet.</div>",
            unsafe_allow_html=True,
        )

    else:
        try:
            client = Groq(
                api_key=st.secrets["GROQ_API_KEY"]
            )

            # Groq Orpheus currently limits each TTS request
            # to 200 characters, so split longer responses.
            chunks = [
                answer[i:i + 200]
                for i in range(0, len(answer), 200)
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
                    format="audio/wav"
                )

            st.markdown(
                "<div class='rs-disclaimer' style='border-color:#3ce089;"
                "color:#c9f5df;'>🔊 Press ▶️ to hear the response.</div>",
                unsafe_allow_html=True,
            )

        except Exception as error:

            st.markdown(
                f"<div class='rs-disclaimer rs-red'>Groq TTS error: {error}</div>",
                unsafe_allow_html=True,
            )

# ============================================================
# KNOWLEDGE MANAGER
# ============================================================

st.divider()

rs_section("Knowledge Manager")

st.caption(
    "UPLOAD AND MANAGE AUTHORIZED EMS REFERENCE DOCUMENTS."
)


uploaded_file = st.file_uploader(
    "Upload a PDF or Word reference document",
    type=["pdf", "docx"],
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

    st.markdown(
        f"<div class='rs-disclaimer' style='border-color:#3ce089;"
        f"color:#c9f5df;'>✓ Uploaded: {uploaded_file.name}</div>",
        unsafe_allow_html=True,
    )


    registry = {}


    if REGISTRY_PATH.exists():

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8-sig",
        ) as file:

            registry = json.load(file)


    existing = registry.get(
        uploaded_file.name,
        {},
    )


    with st.expander(
        "Source Metadata",
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
            "Document type",
            [
                "educational_reference",
                "agency_protocol",
                "medical_director_order",
                "manufacturer_reference",
                "other",
            ],
        )


        effective_date = st.text_input(
            "Effective date",
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
            "Authority / issuing organization",
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
            "Requires review",
            value=existing.get(
                "review_required",
                True,
            ),
        )


        if st.button(
            "💾 Save Metadata"
        ):

            registry[
                uploaded_file.name
            ] = {

                "title": title,

                "jurisdiction": jurisdiction,

                "document_type": document_type,

                "effective_date": (
                    effective_date or None
                ),

                "version": (
                    version or None
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


            st.markdown(
                "<div class='rs-disclaimer' style='border-color:#3ce089;"
                "color:#c9f5df;'>✓ Metadata saved.</div>",
                unsafe_allow_html=True,
            )


        if st.button(
            "🔄 Ingest & Rebuild Knowledge Index"
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

                st.markdown(
                    "<div class='rs-disclaimer rs-red'>Document ingestion failed.</div>",
                    unsafe_allow_html=True,
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

                    st.markdown(
                        "<div class='rs-disclaimer rs-red'>Knowledge index rebuild failed.</div>",
                        unsafe_allow_html=True,
                    )

                    st.code(
                        index_result.stderr
                    )

                else:

                    st.markdown(
                        "<div class='rs-disclaimer' style='border-color:#3ce089;"
                        "color:#c9f5df;'>✓ Knowledge index rebuilt successfully.</div>",
                        unsafe_allow_html=True,
                    )

                    st.code(
                        index_result.stdout
                    )


# ============================================================
# REGISTERED SOURCES
# ============================================================

with st.expander(
    "📖 View Registered Sources"
):

    if REGISTRY_PATH.exists():

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8-sig",
        ) as file:

            registry = json.load(file)


        if registry:

            for filename, metadata in registry.items():

                st.markdown(
                    f"<div class='rs-panel'>"
                    f"<b style='color:#3ce089;'>📄 "
                    f"{metadata.get('title', filename)}</b><br>"
                    f"<span style='color:#7c8b93;'>FILE:</span> {filename}<br>"
                    f"<span style='color:#7c8b93;'>JURISDICTION:</span> "
                    f"{metadata.get('jurisdiction', 'UNSPECIFIED')}<br>"
                    f"<span style='color:#7c8b93;'>TYPE:</span> "
                    f"{metadata.get('document_type', 'UNKNOWN')}<br>"
                    f"<span style='color:#7c8b93;'>STATUS:</span> "
                    f"{metadata.get('status', 'UNKNOWN')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if metadata.get(
                    "review_required",
                    True,
                ):

                    st.markdown(
                        "<div class='rs-disclaimer'>⚠️ Review required</div>",
                        unsafe_allow_html=True,
                    )

                else:

                    st.markdown(
                        "<div class='rs-disclaimer' style='border-color:#3ce089;"
                        "color:#c9f5df;'>✓ Review complete</div>",
                        unsafe_allow_html=True,
                    )

        else:

            st.caption("NO REGISTERED SOURCES.")

    else:

        st.caption("NO KNOWLEDGE REGISTRY FOUND.")


# ============================================================
# SAFETY NOTICE
# ============================================================

st.divider()

st.markdown(
    """
    <div class="rs-disclaimer rs-red" style="text-align:center;
    font-weight:700; letter-spacing:0.08em;">
        ⚠ DEMO ONLY — NOT FOR CLINICAL DECISION MAKING
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="rs-footer">
        Always follow current local EMS protocols, medical direction,
        scope of practice, manufacturer instructions, and applicable
        regulations.<br>
        PARAMEDIC AI — companion tool, styled after
        <b style="color:#3ce089;">RUNSHEET</b> // built for EMT / paramedic students.
    </div>
    """,
    unsafe_allow_html=True,
)
