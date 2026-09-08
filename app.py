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

PROJECT_ROOT = Path(**file**).resolve().parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

REGISTRY_PATH = (
PROJECT_ROOT
/ "data"
/ "knowledge"
/ "sources.json"
)

# ============================================================

# PAGE CONFIGURATION

# ============================================================

st.set_page_config(
page_title="Paramedic AI",
page_icon="🚑",
layout="wide",
initial_sidebar_state="expanded",
)

# ============================================================

# GLOBAL UI STYLING

# ============================================================

st.markdown(
"""

<style>

/* ---------------------------------------------------------
   General spacing
--------------------------------------------------------- */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ---------------------------------------------------------
   Main EMS header
--------------------------------------------------------- */

.paramedic-header {
    background: linear-gradient(
        135deg,
        #0B1F33 0%,
        #0066CC 100%
    );
    padding: 28px 32px;
    border-radius: 18px;
    margin-bottom: 24px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.12);
}

.paramedic-header h1 {
    color: white !important;
    margin: 0;
    font-size: 2.4rem;
    font-weight: 750;
}

.paramedic-header p {
    color: #E8F1F8 !important;
    margin: 8px 0 0 0;
    font-size: 1.05rem;
}


/* ---------------------------------------------------------
   Section headings
--------------------------------------------------------- */

.section-heading {
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 4px;
}

.section-subtitle {
    opacity: 0.72;
    margin-bottom: 18px;
}


/* ---------------------------------------------------------
   Sidebar
--------------------------------------------------------- */

.sidebar-header {
    background: linear-gradient(
        135deg,
        #0B1F33 0%,
        #0066CC 100%
    );
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 20px;
}

.sidebar-header h2 {
    color: white !important;
    margin: 0;
    font-size: 1.25rem;
}

.sidebar-header p {
    color: #E8F1F8 !important;
    margin: 5px 0 0 0;
    font-size: 0.84rem;
}


/* ---------------------------------------------------------
   Generic cards
   Uses Streamlit theme variables so light/dark mode works.
--------------------------------------------------------- */

.info-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.22);
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 14px;
}

.info-card h3,
.info-card h4 {
    margin-top: 0;
}


/* ---------------------------------------------------------
   Clinical status cards
--------------------------------------------------------- */

.patient-card {
    background: var(--secondary-background-color);
    border-left: 5px solid #0066CC;
    border-radius: 12px;
    padding: 16px 18px;
    min-height: 105px;
}

.safety-card {
    background: var(--secondary-background-color);
    border-left: 5px solid #D62828;
    border-radius: 12px;
    padding: 16px 18px;
    min-height: 105px;
}


/* ---------------------------------------------------------
   Copilot banner
--------------------------------------------------------- */

.copilot-banner {
    background: var(--secondary-background-color);
    border: 1px solid rgba(0, 102, 204, 0.30);
    border-left: 5px solid #0066CC;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 18px;
}


/* ---------------------------------------------------------
   Knowledge Manager banner
--------------------------------------------------------- */

.knowledge-banner {
    background: var(--secondary-background-color);
    border-left: 5px solid #008C95;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 18px;
}


/* ---------------------------------------------------------
   Buttons
--------------------------------------------------------- */

.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}


/* ---------------------------------------------------------
   Reduce excessive metric spacing
--------------------------------------------------------- */

[data-testid="stMetric"] {
    padding: 6px 2px;
}


/* ---------------------------------------------------------
   Mobile adjustments
--------------------------------------------------------- */

@media (max-width: 768px) {

    .paramedic-header {
        padding: 22px;
    }

    .paramedic-header h1 {
        font-size: 1.9rem;
    }

}

</style>

""",
unsafe_allow_html=True,
)

# ============================================================

# MAIN HEADER

# ============================================================

st.markdown(
"""

<div class="paramedic-header">
    <h1>🚑 Paramedic AI</h1>
    <p>
        EMS education and decision-support demonstration
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================

# SIDEBAR — PATIENT ASSESSMENT

# ============================================================

st.sidebar.markdown(
"""

<div class="sidebar-header">
    <h2>🩺 Patient Assessment</h2>
    <p>Enter current patient vital signs</p>
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

# ============================================================

# MAIN — VITAL SIGNS DASHBOARD

# ============================================================

st.markdown(
'<div class="section-heading">🩺 Patient Vital Signs</div>',
unsafe_allow_html=True,
)

st.markdown(
'<div class="section-subtitle">'
"Current patient assessment"
"</div>",
unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
st.metric(
"❤️ Heart Rate",
f"{heart_rate:.0f}",
"bpm",
)

with col2:
st.metric(
"🩸 Blood Pressure",
f"{systolic_bp:.0f}/{diastolic_bp:.0f}",
"mmHg",
)

with col3:
st.metric(
"🫁 SpO₂",
f"{spo2:.0f}",
"%",
)

with col4:
st.metric(
"🌡️ Temperature",
f"{temperature:.1f}",
"°F",
)

st.divider()

# ============================================================

# CLINICAL OVERVIEW

# ============================================================

st.markdown(
'<div class="section-heading">🚑 Clinical Overview</div>',
unsafe_allow_html=True,
)

st.markdown(
'<div class="section-subtitle">'
"Patient information and application safety information"
"</div>",
unsafe_allow_html=True,
)

overview_col1, overview_col2 = st.columns(2)

with overview_col1:

```
st.markdown(
    """
```

<div class="patient-card">
    <h4>👤 Patient</h4>
</div>
""",
        unsafe_allow_html=True,
    )

```
st.write(f"**Age:** {age} years")
st.write(
    f"**Respiratory Rate:** "
    f"{respiratory_rate:.0f} breaths/min"
)
```

with overview_col2:

```
st.markdown(
    """
```

<div class="safety-card">
    <h4>⚠️ Clinical Safety</h4>
</div>
""",
        unsafe_allow_html=True,
    )

```
st.warning(
    "This application is an educational and "
    "decision-support demonstration. It does not "
    "replace clinical judgment, medical protocols, "
    "medical direction, or local EMS requirements."
)
```

st.divider()

# ============================================================

# ML ASSESSMENT

# ============================================================

st.markdown(
'<div class="section-heading">📊 ML Risk Assessment</div>',
unsafe_allow_html=True,
)

st.markdown(
'<div class="section-subtitle">'
"Run the demonstration machine-learning model using "
"the current patient vital signs."
"</div>",
unsafe_allow_html=True,
)

if st.button(
"🔍 Run ML Assessment",
type="primary",
use_container_width=True,
):

```
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

    category = classify_risk(probability)

    risk_col1, risk_col2 = st.columns(2)


    with risk_col1:

        st.metric(
            "Estimated Risk Probability",
            f"{probability * 100:.1f}%",
        )


    with risk_col2:

        if category == "HIGHER RISK":

            st.error(
                "🔴 HIGHER RISK"
            )

        elif category == "INTERMEDIATE RISK":

            st.warning(
                "🟠 INTERMEDIATE RISK"
            )

        else:

            st.success(
                "🟢 LOWER RISK"
            )


    with st.expander(
        "📋 View Patient Data"
    ):

        st.dataframe(
            patient,
            use_container_width=True,
        )


    st.info(
        "This prediction is generated by a "
        "demonstration machine-learning model and "
        "is NOT a clinical diagnosis."
    )


except Exception as error:

    st.error(
        f"ML prediction error: {error}"
    )
```

# ============================================================

# COPILOT

# ============================================================

st.divider()

st.markdown(
'<div class="section-heading">🧠 Paramedic Copilot</div>',
unsafe_allow_html=True,
)

st.markdown(
"""

<div class="copilot-banner">
    <strong>Clinical Education Assistant</strong><br>
    Ask a question by typing or using your microphone.
    Review your question before sending it to Copilot.
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================

# SESSION STATE

# ============================================================

if "copilot_messages" not in st.session_state:

```
st.session_state.copilot_messages = []
```

if "voice_text" not in st.session_state:

```
st.session_state.voice_text = ""
```

if "last_copilot_answer" not in st.session_state:

```
st.session_state.last_copilot_answer = ""
```

# ============================================================

# DISPLAY CHAT HISTORY

# ============================================================

for message in st.session_state.copilot_messages:

```
with st.chat_message(
    message["role"]
):

    st.markdown(
        message["content"]
    )
```

# ============================================================

# VOICE INPUT

# ============================================================

st.caption(
"🎙️ Speak your question, then review the transcription "
"before sending."
)

voice_question = speech_to_text(
language="en",
start_prompt="🎙️ Start speaking",
stop_prompt="⏹️ Stop recording",
use_container_width=True,
key="voice_input",
)

if voice_question:

```
st.session_state.voice_text = voice_question
```

# ============================================================

# TEXT INPUT

# ============================================================

typed_question = st.chat_input(
"Type your clinical education question..."
)

if typed_question:

```
st.session_state.voice_text = typed_question
```

# ============================================================

# REVIEW TRANSCRIPTION

# ============================================================

if st.session_state.voice_text:

```
st.markdown(
    "### 📝 Review Your Question"
)

reviewed_question = st.text_area(
    "Edit the transcription if needed:",
    value=st.session_state.voice_text,
    height=120,
    key="reviewed_question",
)


col1, col2 = st.columns(2)


with col1:

    send_question = st.button(
        "🚀 Send to Copilot",
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

        # ------------------------------------------------
        # USER MESSAGE
        # ------------------------------------------------

        st.session_state.copilot_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )


        with st.chat_message("user"):

            st.markdown(question)


        # ------------------------------------------------
        # COPILOT RESPONSE
        # ------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching knowledge and thinking..."
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
                    # 3. LIMIT CONVERSATION HISTORY
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
                    # 5. ADD REFERENCES
                    # ----------------------------------------

                    references = (
                        format_references(
                            results
                        )
                    )


                    if references:

                        answer += (
                            "\n\n"
                            "### 📚 References Used\n"
                            + "\n".join(
                                references
                            )
                        )


                    # ----------------------------------------
                    # 6. DISPLAY ANSWER
                    # ----------------------------------------

                    st.markdown(
                        answer
                    )


                    # ----------------------------------------
                    # 7. SAVE ANSWER
                    # ----------------------------------------

                    st.session_state.copilot_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )


                    st.session_state.last_copilot_answer = answer


                    # ----------------------------------------
                    # 8. CLEAR INPUT
                    # ----------------------------------------

                    st.session_state.voice_text = ""


                except Exception as error:

                    error_message = (
                        f"Copilot error: {error}"
                    )

                    st.error(
                        error_message
                    )
```

# ============================================================

# TEXT-TO-SPEECH

# ============================================================

st.markdown(
"### 🔊 Response Audio"
)

if st.button(
"🔊 Read Latest Response Aloud",
use_container_width=True,
):

```
answer = (
    st.session_state.last_copilot_answer
)


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
            "🔊 Press ▶️ to hear the response."
        )


    except Exception as error:

        st.error(
            f"Groq TTS error: {error}"
        )
```

# ============================================================

# KNOWLEDGE MANAGER

# ============================================================

st.divider()

st.markdown(
'<div class="section-heading">📚 Knowledge Manager</div>',
unsafe_allow_html=True,
)

st.markdown(
"""

<div class="knowledge-banner">
    <strong>Authorized EMS References</strong><br>
    Upload, manage, and rebuild your application's
    reference knowledge base.
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
"Upload a PDF or Word reference document",
type=["pdf", "docx"],
)

if uploaded_file:

```
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
    "⚙️ Source Metadata",
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
```

# ============================================================

# REGISTERED SOURCES

# ============================================================

with st.expander(
"📖 View Registered Sources"
):

```
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
                f"### 📄 "
                f"{metadata.get('title', filename)}"
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
```

# ============================================================

# SAFETY NOTICE

# ============================================================

st.divider()

st.error(
"🚨 DEMO ONLY — NOT FOR CLINICAL DECISION MAKING."
)

st.caption(
"Always follow current local EMS protocols, "
"medical direction, scope of practice, "
"manufacturer instructions, and applicable regulations."
)
