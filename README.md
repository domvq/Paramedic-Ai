# 🚑 Paramedic AI

**Paramedic AI** is an educational EMS decision-support demonstration built with Python and Streamlit. It combines machine-learning risk assessment, an AI copilot, voice input, document-based knowledge retrieval, and optional C++ text-to-speech.

> ⚠️ **DEMO ONLY — NOT FOR CLINICAL DECISION MAKING**
>
> This application is intended for education, software demonstration, and development purposes. It must not be used as a substitute for current local EMS protocols, medical direction, scope of practice, manufacturer instructions, clinical guidelines, or applicable regulations.

---

## Features

### 🧠 ML Risk Assessment

The application provides a demonstration machine-learning risk assessment using basic patient information.

Potential inputs include:

* Age
* Heart rate
* Systolic blood pressure
* Diastolic blood pressure
* Respiratory rate
* SpO₂
* Temperature

The model can return:

* Estimated risk probability
* Risk category
* Demonstration prediction output

Example risk categories may include:

```text
LOWER RISK
INTERMEDIATE RISK
HIGHER RISK
```

The result is explicitly intended as a **demonstration prediction**, not a clinical diagnosis or validated medical risk score.

---

### 💬 Paramedic Copilot

The Paramedic Copilot provides an interactive question-and-answer interface for educational use.

Users can:

* Type questions
* Enter questions using voice input
* Review and edit transcriptions
* Submit questions to the Copilot
* Review previous conversation messages
* Receive supporting references from the knowledge system

The Copilot can use the project's knowledge-search functionality to retrieve relevant context before generating a response.

> AI-generated responses may be incorrect, incomplete, outdated, or inappropriate for a particular situation. Always independently verify information using authorized and current EMS resources.

---

### 🎙️ Voice Input

Voice questions can be recorded directly through the Streamlit interface using:

```text
streamlit-mic-recorder
```

The transcription can be reviewed and edited before being submitted to the Copilot.

This allows the application to demonstrate a more hands-free EMS-oriented interaction.

---

### 📚 Knowledge Manager

The Knowledge Manager allows reference documents to be uploaded and incorporated into the application's knowledge system.

Supported document types include:

* PDF
* Microsoft Word (`.docx`)

Uploaded documents are stored in:

```text
data/documents/
```

Metadata can include:

* Title
* Jurisdiction
* Document type
* Effective date
* Version
* Issuing organization
* Status
* Review-required flag

Knowledge-source metadata is stored in:

```text
data/knowledge/sources.json
```

The project also includes document ingestion and knowledge-index rebuilding functionality.

---

### 🔊 C++ Text-to-Speech

The project includes an optional C++ text-to-speech component.

The Streamlit application expects the compiled executable at:

```text
src/tts.exe
```

The latest Copilot response can be written to:

```text
data/tts_input.txt
```

The application can then launch the C++ executable to speak the response.

This component is primarily intended for Windows environments where the executable has been compiled and configured.

---

# Project Structure

A typical project structure is:

```text
Paramedic-Ai/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── backend/
│
├── data/
│   ├── documents/
│   └── knowledge/
│       └── sources.json
│
├── models/
│
├── src/
│   ├── ml_model.py
│   ├── copilot.py
│   ├── knowledge.py
│   ├── ingest.py
│   ├── build_index.py
│   └── tts.exe
│
├── tests/
│
├── build_dataset.py
├── build_training_data.py
├── deterioration_model.py
├── predict.py
├── train_model.py
│
└── launch_paramedic_ai.bat
```

The exact contents may vary depending on the current version of the repository.

---

# Requirements

The application requires:

* Python 3
* Streamlit
* Pandas
* `streamlit-mic-recorder`
* Additional packages listed in `requirements.txt`

Depending on the enabled features, additional dependencies may be required for:

* Machine learning
* AI/Copilot functionality
* Document processing
* Knowledge retrieval
* Text-to-speech

For Windows development, PowerShell can be used to install dependencies and launch the application.

---

# Installation

## 1. Clone the Repository

Clone the project from GitHub:

```powershell
git clone https://github.com/domvq/Paramedic-Ai.git
```

Move into the project directory:

```powershell
cd Paramedic-Ai
```

---

## 2. Create a Virtual Environment

Create a Python virtual environment:

```powershell
python -m venv .venv
```

---

## 3. Install Dependencies

Install the project's dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `streamlit-mic-recorder` is not included in the requirements file, install it separately:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit-mic-recorder
```

---

# Running the Application

From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit should provide a local address similar to:

```text
http://localhost:8501
```

Open the address in a web browser.

---

## PowerShell Execution Policy

On some Windows systems, activating the virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

may produce an error indicating that script execution is disabled.

You do **not** need to change the PowerShell execution policy.

Instead, run Streamlit directly through the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This also ensures that the correct Python environment is being used.

---

# Running the Knowledge Pipeline

The Knowledge Manager can be used to upload reference documents.

Documents are placed in:

```text
data/documents/
```

The knowledge pipeline generally consists of:

```text
Reference Document
        │
        ▼
data/documents/
        │
        ▼
src/ingest.py
        │
        ▼
Processed Knowledge
        │
        ▼
src/build_index.py
        │
        ▼
Knowledge Index
        │
        ▼
search_knowledge()
        │
        ▼
Paramedic Copilot
```

The Streamlit interface may provide an:

```text
🔄 Ingest & Rebuild Knowledge Index
```

button to execute the process.

If running the scripts manually, ingestion should be completed before rebuilding the knowledge index.

---

# Environment Variables

If the Copilot or other services require API credentials, configure them using the project's:

```text
.env.example
```

Create a local:

```text
.env
```

file as needed.

**Never commit API keys, passwords, access tokens, or other secrets to Git.**

A typical workflow is:

```text
.env.example
      │
      ▼
     .env
      │
      ▼
Local development configuration
```

The `.env` file should remain excluded from source control.

---

# Machine Learning

The ML portion of the application is organized around the project's model-loading and prediction functionality.

Important functions may include:

```python
load_model()
predict_risk()
classify_risk()
```

The ML implementation is contained in:

```text
src/ml_model.py
```

A typical workflow is:

```text
Training Data
     │
     ▼
train_model.py
     │
     ▼
Trained Model
     │
     ▼
src/ml_model.py
     │
     ▼
Streamlit Application
```

The exact model architecture, training procedure, datasets, and model format depend on the implementation included in the repository.

### Important

The ML output is a **software demonstration only**.

It should not be interpreted as:

* A diagnosis
* A validated clinical score
* A prediction of patient outcome
* A replacement for clinical judgment
* A replacement for EMS protocols or medical direction

---

# Testing

The project contains a:

```text
tests/
```

directory for automated tests.

If `pytest` is included in the project's dependencies, run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Before submitting changes, it is recommended to verify that:

* The application starts successfully
* The ML functionality loads correctly
* The Copilot interface works
* Document ingestion works
* Knowledge-index rebuilding works
* Voice input works when supported
* Optional TTS functionality works when configured

---

# Troubleshooting

## `ModuleNotFoundError`

For example:

```text
ModuleNotFoundError: No module named 'streamlit_mic_recorder'
```

Install the missing package into the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit-mic-recorder
```

Then restart Streamlit.

---

## Streamlit Command Not Found

Instead of:

```powershell
streamlit run app.py
```

use:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This guarantees that Streamlit is executed using the project's virtual environment.

---

## Missing `src` Modules

Errors such as:

```text
ModuleNotFoundError: No module named 'src.ml_model'
```

may indicate that the application is not being launched from the project root or that a required source file is missing.

Make sure you are inside:

```text
Paramedic-Ai/
```

before launching:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

## C++ TTS Error

If the application displays:

```text
C++ TTS error
```

check whether:

```text
src/tts.exe
```

exists.

The application expects a Windows executable named:

```text
tts.exe
```

If it has not been compiled or configured, the rest of the Streamlit application can still be used, but the optional text-to-speech functionality will not work.

---

## Knowledge Index Errors

If document ingestion fails, inspect the error displayed by the Knowledge Manager.

The normal workflow is:

```text
src/ingest.py
        │
        ▼
Successful ingestion
        │
        ▼
src/build_index.py
```

The index should be rebuilt after successfully processing new or updated reference documents.

---

# Development

The primary Streamlit entry point is:

```text
app.py
```

Core functionality is organized into the `src/` directory.

Important modules include:

### `src/ml_model.py`

Machine-learning model loading, prediction, and risk classification.

### `src/copilot.py`

AI Copilot functionality.

### `src/knowledge.py`

Knowledge retrieval, context formatting, and reference handling.

### `src/ingest.py`

Reference-document ingestion and processing.

### `src/build_index.py`

Knowledge-index construction.

---

# Contributing

Contributions are welcome for educational and software-development purposes.

When contributing:

1. Create a separate branch for your changes.
2. Keep changes focused and documented.
3. Add or update tests where appropriate.
4. Do not commit secrets or private data.
5. Do not commit patient-identifying information.
6. Verify that the application still starts successfully.
7. Clearly identify changes that affect the AI, ML, knowledge, or clinical-safety behavior.

Because this is an EMS-related demonstration, changes should be reviewed carefully before being presented as clinically meaningful.

---

# Security and Privacy

This application may process sensitive information if users enter it into the interface.

For development and testing:

* Do not use real patient-identifying information unless the system has been specifically designed and authorized to handle it.
* Do not commit patient data to Git.
* Do not commit API keys or credentials.
* Protect uploaded reference documents appropriately.
* Use authorized and current EMS reference material.
* Review AI-generated output before relying on it for educational purposes.
* Do not assume that locally stored documents or generated indexes are automatically secure.

If this project is deployed outside a local development environment, additional authentication, authorization, logging, encryption, and data-retention controls may be required.

---

# Clinical Safety Disclaimer

**Paramedic AI is a software demonstration and is NOT a medical device or clinical decision-making system.**

The machine-learning risk assessment and Copilot responses may be inaccurate, incomplete, outdated, or inappropriate for a particular patient or situation.

Users must independently verify information against:

* Current local EMS protocols
* Medical direction
* Scope of practice
* Applicable laws and regulations
* Current clinical guidelines
* Manufacturer instructions
* Authorized agency policies

**Never use this demonstration as the sole basis for patient-care decisions.**

---

# License

Add the project's applicable license here.

If the repository is not currently licensed, consider adding an appropriate open-source license before distributing the project publicly.

---

# Status

**Project:** Paramedic AI
**Repository:** `domvq/Paramedic-Ai`
**Type:** EMS education and decision-support demonstration
**Interface:** Streamlit
**Primary Language:** Python
**Optional TTS:** C++ / Windows executable

---

# Quick Start

For a new Windows installation:

```powershell
git clone https://github.com/domvq/Paramedic-Ai.git

cd Paramedic-Ai

python -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🚑 Paramedic AI

**Educational demonstration only.**

Not for clinical decision making.
