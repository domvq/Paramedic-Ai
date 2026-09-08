
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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Paramedic AI",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — CHATGPT-STYLE UI
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background: #212121;
        color: #ececec;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 8rem;
    }

    /* Remove excessive Streamlit spacing */

    div[data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #171717;
        border-right: 1px solid #303030;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f5f5f5;
    }

    section[data-testid="stSidebar"] .stButton button {
        background: transparent;
        border: 1px solid #3a3a3a;
        color: #eeeeee;
        border-radius: 8px;
        text-align: left;
    }

    section[data-testid="stSidebar"] .stButton button:hover {
        background: #2a2a2a;
        border-color: #555;
    }

    /* --------------------------------------------------------
       TOP BRAND
    -------------------------------------------------------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0 22px 0;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(
            135deg,
            #19a974,
            #087f5b
        );
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }

    .brand-title {
        font-size: 22px;
        font-weight: 650;
        color: #f5f5f5;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #999;
        margin-top: 2px;
    }

    /* --------------------------------------------------------
       WELCOME SCREEN
    -------------------------------------------------------- */

    .welcome {
        text-align: center;
        padding: 80px 10px 30px 10px;
    }

    .welcome-icon {
        font-size: 52px;
        margin-bottom: 12px;
    }

    .welcome-title {
        font-size: 34px;
        font-weight: 650;
        color: #f5f5f5;
        margin-bottom: 8px;
    }

    .welcome-text {
        color: #a9a9a9;
        font-size: 15px;
    }

    /* --------------------------------------------------------
       PROMPT CARDS
    -------------------------------------------------------- */

    .prompt-card {
        background: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-radius: 12px;
        padding: 15px;
        min-height: 95px;
        transition: all 0.15s ease;
    }

    .prompt-card:hover {
        background: #303030;
        border-color: #505050;
    }

    .prompt-icon {
        font-size: 21px;
        margin-bottom: 8px;
    }

    .prompt-title {
        color: #f0f0f0;
        font-size: 14px;
        font-weight: 600;
    }

    .prompt-description {
        color: #999;
        font-size: 12px;
        margin-top: 4px;
    }

    /* --------------------------------------------------------
       CHAT MESSAGES
    -------------------------------------------------------- */

    div[data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 1.25rem 0;
    }

    div[data-testid="stChatMessage"] p {
        line-height: 1.65;
    }

    /* User message */

    div[data-testid="stChatMessage"]:has(
        div[data-testid="chatAvatarIcon-user"]
    ) {
        background: transparent;
    }

    /* --------------------------------------------------------
       CHAT INPUT
    -------------------------------------------------------- */

    div[data-testid="stChatInput"] {
        background: #2f2f2f;
        border: 1px solid #4a4a4a;
        border-radius: 18px;
        padding: 4px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.35);
    }

    div[data-testid="stChatInput"] textarea {
        background: transparent;
        color: #f5f5f5;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #888;
    }

    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton button {
        border-radius: 9px;
        border: 1px solid #444;
        background: #2b2b2b;
        color: #eee;
    }

    .stButton button:hover {
        border-color: #666;
        background: #353535;
    }

    /* --------------------------------------------------------
       INPUTS
    -------------------------------------------------------- */

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] {
        background: #2b2b2b;
        color: #eee;
        border-color: #444;
    }

    /* --------------------------------------------------------
       EXPANDERS
    -------------------------------------------------------- */

    details {
        background: #292929;
        border: 1px solid #3b3b3b;
        border-radius: 10px;
    }

    /* --------------------------------------------------------
       METRICS
    -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: #292929;
        border: 1px solid #3b3b3b;
        padding: 15px;
        border-radius: 10px;
    }

    /* --------------------------------------------------------
       ALERTS
    -------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* --------------------------------------------------------
       DIVIDERS
    -------------------------------------------------------- */

    hr {
        border-color: #363636;
    }

    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .medical-footer {
        text-align: center;
        color: #777;
        font-size: 11px;
        padding: 20px 0 5px 0;
    }

    .status-pill {
        display: inline-block;
        background: #163b2e;
        color: #6ee7b7;
        border: 1px solid #245b46;
        border-radius: 20px;
        padding: 4px 10px;
        font-size: 11px;
    }

    </style>
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

if "patient_assessment" not in st.session_state:
    st.session_state.patient_assessment = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🚑</div>
            <div>
                <div class="brand-title">Paramedic AI</div>
                <div class="brand-subtitle">
                    Prehospital clinical assistant
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "＋  New conversation",
        use_container_width=True,
    ):

        st.session_state.copilot_messages = []
        st.session_state.voice_text = ""
        st.session_state.last_copilot_answer = ""

        st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # PATIENT ASSESSMENT
    # --------------------------------------------------------

    with st.expander(
        "🩺 Patient assessment",
        expanded=True,
    ):

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=50,
        )

        heart_rate = st.number_input(
            "Heart rate",
            min_value=0.0,
            max_value=300.0,
            value=90.0,
        )

        systolic_bp = st.number_input(
            "Systolic BP",
            min_value=0.0,
            max_value=300.0,
            value=120.0,
        )

        diastolic_bp = st.number_input(
            "Diastolic BP",
            min_value=0.0,
            max_value=200.0,
            value=80.0,
        )

        respiratory_rate = st.number_input(
            "Respiratory rate",
            min_value=0.0,
            max_value=100.0,
            value=18.0,
        )

        spo2 = st.number_input(
            "SpO₂",
            min_value=0.0,
            max_value=100.0,
            value=98.0,
        )

        temperature = st.number_input(
            "Temperature °F",
            min_value=80.0,
            max_value=115.0,
            value=98.6,
        )

        if st.button(
            "Run risk assessment",
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

                st.session_state.patient_assessment = {
                    "patient": patient,
                    "probability": probability,
                    "category": category,
                }

            except Exception as error:

                st.error(
                    f"ML prediction error: {error}"
                )

    # --------------------------------------------------------
    # KNOWLEDGE MANAGER
    # --------------------------------------------------------

    with st.expander(
        "📚 Knowledge base",
        expanded=False,
    ):

        st.caption(
            "Manage authorized EMS reference documents."
        )

        uploaded_file = st.file_uploader(
            "Upload PDF or Word document",
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
                "Save metadata",
                use_container_width=True,
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
                "Rebuild knowledge index",
                use_container_width=True,
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

    # --------------------------------------------------------
    # REGISTERED SOURCES
    # --------------------------------------------------------

    with st.expander(
        "📖 Registered sources",
        expanded=False,
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
                        f"**{metadata.get('title', filename)}**"
                    )

                    st.caption(
                        f"{metadata.get('jurisdiction', 'UNSPECIFIED')} · "
                        f"{metadata.get('status', 'UNKNOWN')}"
                    )

                    if metadata.get(
                        "review_required",
                        True,
                    ):

                        st.warning(
                            "Review required"
                        )

            else:

                st.caption(
                    "No registered sources."
                )

        else:

            st.caption(
                "No knowledge registry found."
            )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            color:#777;
            font-size:11px;
            line-height:1.5;
        ">
        <strong>Demo / education only</strong><br>
        Not a substitute for clinical judgment,
        medical direction, or current local EMS protocols.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN HEADER
# ============================================================

col1, col2 = st.columns(
    [5, 1]
)

with col1:

    st.markdown(
        """
        <div style="
            font-size:14px;
            color:#999;
            margin-bottom:3px;
        ">
        PARAMEDIC AI
        </div>

        <div style="
            font-size:28px;
            font-weight:650;
            color:#f5f5f5;
        ">
        Prehospital clinical assistant
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        """
        <div style="
            text-align:right;
            padding-top:12px;
        ">
            <span class="status-pill">
                ● AI ONLINE
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("---")


# ============================================================
# RISK ASSESSMENT RESULT
# ============================================================

if st.session_state.patient_assessment:

    assessment = (
        st.session_state.patient_assessment
    )

    probability = assessment[
        "probability"
    ]

    category = assessment[
        "category"
    ]

    st.markdown(
        "### 🩺 Risk assessment"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Estimated risk",
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
            assessment["patient"],
            use_container_width=True,
        )

    st.info(
        "This is a demonstration machine-learning "
        "prediction and is not a clinical diagnosis."
    )


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.copilot_messages:

    st.markdown(
        """
        <div class="welcome">

            <div class="welcome-icon">🚑</div>

            <div class="welcome-title">
                How can I help with this call?
            </div>

            <div class="welcome-text">
                Ask about assessment, differential diagnosis,
                treatment considerations, or EMS references.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:12px;
            margin-bottom:25px;
        ">

            <div class="prompt-card">
                <div class="prompt-icon">🩺</div>
                <div class="prompt-title">
                    Patient assessment
                </div>
                <div class="prompt-description">
                    Help structure an initial assessment.
                </div>
            </div>

            <div class="prompt-card">
                <div class="prompt-icon">🧠</div>
                <div class="prompt-title">
                    Differential diagnosis
                </div>
                <div class="prompt-description">
                    Explore possible causes of a presentation.
                </div>
            </div>

            <div class="prompt-card">
                <div class="prompt-icon">🚑</div>
                <div class="prompt-title">
                    Treatment considerations
                </div>
                <div class="prompt-description">
                    Review potential prehospital considerations.
                </div>
            </div>

            <div class="prompt-card">
                <div class="prompt-icon">📚</div>
                <div class="prompt-title">
                    EMS references
                </div>
                <div class="prompt-description">
                    Search your authorized knowledge base.
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.copilot_messages:

    role = message["role"]

    avatar = (
        "👤"
        if role == "user"
        else "🚑"
    )

    with st.chat_message(
        role,
        avatar=avatar,
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# VOICE INPUT
# ============================================================

st.caption(
    "🎙️ Voice input is available below. Review the transcription before sending."
)

voice_question = speech_to_text(
    language="en",
    start_prompt="🎙️ Start speaking",
    stop_prompt="⏹️ Stop recording",
    use_container_width=True,
    key="voice_input",
)

if voice_question:

    st.session_state.voice_text = (
        voice_question
    )


# ============================================================
# CHAT INPUT
# ============================================================

typed_question = st.chat_input(
    "Message Paramedic AI..."
)

if typed_question:

    st.session_state.voice_text = (
        typed_question
    )


# ============================================================
# REVIEW QUESTION
# ============================================================

if st.session_state.voice_text:

    st.markdown(
        "#### 📝 Review question"
    )

    reviewed_question = st.text_area(
        "Edit before sending:",
        value=st.session_state.voice_text,
        height=100,
        label_visibility="collapsed",
        key="reviewed_question",
    )

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        send_question = st.button(
            "Send to Paramedic AI",
            type="primary",
            use_container_width=True,
        )

    with col2:

        clear_question = st.button(
            "Clear",
            use_container_width=True,
        )

    if clear_question:

        st.session_state.voice_text = ""

        st.rerun()

    if send_question:

        question = (
            reviewed_question.strip()
        )

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
                avatar="👤",
            ):

                st.markdown(
                    question
                )

            # ------------------------------------------------
            # AI RESPONSE
            # ------------------------------------------------

            with st.chat_message(
                "assistant",
                avatar="🚑",
            ):

                with st.spinner(
                    "Searching EMS references..."
                ):

                    try:

                        # ----------------------------------------
                        # KNOWLEDGE SEARCH
                        # ----------------------------------------

                        results = search_knowledge(
                            question,
                            top_k=3,
                        )

                        context = format_context(
                            results
                        )

                        MAX_CONTEXT_CHARS = 6000

                        if len(context) > MAX_CONTEXT_CHARS:

                            context = context[
                                :MAX_CONTEXT_CHARS
                            ]

                        # ----------------------------------------
                        # CONVERSATION HISTORY
                        # ----------------------------------------

                        MAX_MESSAGES = 6

                        recent_messages = (
                            st.session_state
                            .copilot_messages[
                                -MAX_MESSAGES:
                            ]
                        )

                        # ----------------------------------------
                        # COPILOT
                        # ----------------------------------------

                        answer = chat(
                            messages=recent_messages,
                            context=context,
                        )

                        # ----------------------------------------
                        # REFERENCES
                        # ----------------------------------------

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

                        # ----------------------------------------
                        # DISPLAY
                        # ----------------------------------------

                        st.markdown(
                            answer
                        )

                        # ----------------------------------------
                        # SAVE
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

                        st.session_state.voice_text = ""

                    except Exception as error:

                        error_message = (
                            f"Copilot error: {error}"
                        )

                        st.error(
                            error_message
                        )


# ============================================================
# RESPONSE TOOLS
# ============================================================

if st.session_state.last_copilot_answer:

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔊 Read latest response aloud",
            use_container_width=True,
        ):

            answer = (
                st.session_state.last_copilot_answer
            )

            try:

                client = Groq(
                    api_key=st.secrets[
                        "GROQ_API_KEY"
                    ]
                )

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
                        model=(
                            "canopylabs/"
                            "orpheus-v1-english"
                        ),
                        voice="troy",
                        input=chunk,
                        response_format="wav",
                    )

                    audio_bytes = (
                        response.read()
                    )

                    st.audio(
                        audio_bytes,
                        format="audio/wav",
                    )

            except Exception as error:

                st.error(
                    f"Groq TTS error: {error}"
                )


    with col2:

        if st.button(
            "🗑️ Clear conversation",
            use_container_width=True,
        ):

            st.session_state.copilot_messages = []
            st.session_state.voice_text = ""
            st.session_state.last_copilot_answer = ""

            st.rerun()


# ============================================================
# SAFETY NOTICE
# ============================================================

st.markdown(
    """
    <div class="medical-footer">
        <strong>Paramedic AI is an educational / demonstration tool.</strong><br>
        AI output may be inaccurate. Always verify information against
        current local EMS protocols, medical direction, scope of practice,
        manufacturer instructions, and applicable regulations.
    </div>
    """,
    unsafe_allow_html=True,
)

