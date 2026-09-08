
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
    page_title="Paramedic AI | EMS Decision Support",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0B1F33, #0066CC);
        padding: 25px 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        color: white;
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }

    .main-header p {
        color: #E8F1F8;
        margin-top: 8px;
        font-size: 1.1rem;
    }

    .stButton > button[kind="primary"] {
    background-color: #0066CC;
    color: white;
    border: 1px solid #0066CC;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #0052A3;
        border-color: #0052A3;
        color: white;
    }


    /* EMS section headers */
    .ems-section {
        border-left: 5px solid #0066CC;
        padding: 10px 0 10px 16px;
        margin: 20px 0 15px 0;
        border-radius: 4px;
    }

    .ems-section h2,
    .ems-section h3 {
        margin: 0;
    }


    /* ---------------------------------------------------------
       Compact Knowledge Manager
    --------------------------------------------------------- */

    .knowledge-compact {
        border-left: 4px solid #0066CC;
        padding-left: 12px;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .knowledge-compact h2 {
        font-size: 1.25rem;
        margin: 0;
    }

    .knowledge-compact p {
        font-size: 0.82rem;
        opacity: 0.70;
        margin: 3px 0 0 0;
    }

    /* Make the uploader more compact */
    [data-testid="stFileUploader"] {
        margin-top: -8px;
        margin-bottom: -8px;
    }

    [data-testid="stFileUploader"] section {
        padding: 10px;
    }

    /* Smaller safety notice */
    .compact-safety {
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .compact-safety p {
        font-size: 0.78rem;
        line-height: 1.35;
    }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🚑 Paramedic AI</h1>
    <p>EMS education and decision-support demonstration</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — PATIENT ASSESSMENT
# ============================================================

st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #0B1F33, #0066CC);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 20px;
">
    <h2 style="
        color: white;
        margin: 0;
        font-size: 1.3rem;
    ">🩺 Patient Assessment</h2>
    <p style="
        color: #E8F1F8;
        margin: 5px 0 0 0;
        font-size: 0.85rem;
    ">Enter patient vital signs</p>
</div>
""", unsafe_allow_html=True)
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

st.markdown("## 🩺 Patient Vital Signs")

st.caption("Current patient assessment")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("❤️ Heart Rate", f"{heart_rate:.0f}", "bpm")

with col2:
    st.metric("🩸 Blood Pressure", f"{systolic_bp:.0f}/{diastolic_bp:.0f}", "mmHg")

with col3:
    st.metric("🫁 SpO₂", f"{spo2:.0f}", "%")

with col4:
    st.metric("🌡️ Temperature", f"{temperature:.1f}", "°F")

st.divider()


# ============================================================
# ML ASSESSMENT
# ============================================================

st.markdown(
    """
    <div class="ems-section">
        <h3>📊 ML Clinical Risk Assessment</h3>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Run a demonstration risk assessment using the current patient vitals."
)

if st.button(
    "🔵 Run ML Assessment",
    type="primary",
    use_container_width=True,
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

                st.error(
                    "HIGHER RISK"
                )

            elif category == "INTERMEDIATE RISK":

                st.warning(
                    "INTERMEDIATE RISK"
                )

            else:

                st.success(
                    "LOWER RISK"
                )

        with st.expander(
            "View patient data"
        ):

            st.dataframe(
                patient,
                use_container_width=True,
            )

        st.info(
            "This prediction is generated by a "
            "demonstration machine-learning model "
            "and is NOT a clinical diagnosis."
        )

    except Exception as error:

        st.error(
            f"ML prediction error: {error}"
        )


# ============================================================
# COPILOT
# ============================================================

st.divider()

st.markdown(
    """
    <div class="ems-section">
        <h2> Paramedic Copilot</h2>
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
    "🎙️ Speak your question, then review the transcription before sending."
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

    st.markdown("### 📝 Review your question")

    reviewed_question = st.text_area(
        "Edit the transcription if needed:",
        value=st.session_state.voice_text,
        height=120,
        key="reviewed_question",
    )

    col1, col2 = st.columns(2)

    with col1:

        send_question = st.button(
            " Send to Copilot",
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

            st.warning(
                "Please enter or speak a question first."
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

                        error_message = (
                            f"Copilot error: {error}"
                        )

                        st.error(
                            error_message
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

        st.warning(
            "There is no Copilot response to read yet."
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

            st.success(
                "🔊 Press ▶️ to hear the response."
            )

        except Exception as error:

            st.error(
                f"Groq TTS error: {error}"
            )
           
# ============================================================
# KNOWLEDGE MANAGER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="knowledge-compact">
        <h2>📚 Knowledge Manager</h2>
        <p>Upload and manage authorized EMS reference documents.</p>
    </div>
    """,
    unsafe_allow_html=True,
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


            st.success(
                "Metadata saved."
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
                    f"### 📄 {metadata.get('title', filename)}"
                )

                st.write(
                    f"**File:** {filename}"
                )

                st.write(
                    f"**Jurisdiction:** "
                    f"{metadata.get('jurisdiction', 'UNSPECIFIED')}"
                )

                st.write(
                    f"**Type:** "
                    f"{metadata.get('document_type', 'UNKNOWN')}"
                )

                st.write(
                    f"**Status:** "
                    f"{metadata.get('status', 'UNKNOWN')}"
                )


                if metadata.get(
                    "review_required",
                    True,
                ):

                    st.warning(
                        "⚠️ Review required"
                    )

                else:

                    st.success(
                        "✓ Review complete"
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

st.divider()

st.markdown(
'<div class="compact-safety">',
unsafe_allow_html=True,
)

st.warning(
"DEMO ONLY — NOT FOR CLINICAL DECISION MAKING."
)

st.caption(
"Always follow current local EMS protocols, "
"medical direction, scope of practice, "
"manufacturer instructions, and applicable regulations."
)

st.markdown(
"</div>",
unsafe_allow_html=True,
)
