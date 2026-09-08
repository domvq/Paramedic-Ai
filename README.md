# 🚑 Paramedic AI

**Paramedic AI** is an educational EMS decision-support demonstration built with Python and Streamlit. It combines a demonstration machine-learning risk assessment, an AI copilot, voice input, a document-based knowledge system, and optional C++ text-to-speech.

> ⚠️ **DEMO ONLY — NOT FOR CLINICAL DECISION MAKING**
>
> This application is intended for education, software demonstration, and development purposes. It must not be used as a substitute for current local EMS protocols, medical direction, scope of practice, manufacturer instructions, or applicable regulations.

---

## Features

### 🧠 ML Risk Assessment

The application accepts basic patient information:

* Age
* Heart rate
* Systolic blood pressure
* Diastolic blood pressure
* Respiratory rate
* SpO₂
* Temperature

The machine-learning component returns:

* Estimated risk probability
* Risk category:

  * `LOWER RISK`
  * `INTERMEDIATE RISK`
  * `HIGHER RISK`

The application explicitly identifies the result as a demonstration prediction and not a clinical diagnosis.

---

### 💬 Paramedic Copilot

The Copilot provides an interactive question-and-answer interface.

Users can:

* Type questions
* Record questions using a microphone
* Review/edit speech transcription
* Submit questions to the Copilot
* View previous conversation messages
* Receive references from the knowledge system

The Copilot uses the project's knowledge-search functionality to provide relevant context before generating a response.

---

### 🎙️ Voice Input

Voice questions can be recorded directly through the Streamlit interface using:

```text
streamlit-mic-recorder
```

The application allows the user to review and edit the transcription before sending it to the Copilot.

---

### 📚 Knowledge Manager

The Knowledge Manager allows authorized reference documents to be uploaded.

Supported document types:

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

Metadata is stored in:

```text
data/knowledge/sources.json
```

The application can also run the document ingestion and knowledge-index rebuilding scripts.

---

### 🔊 C++ Text-to-Speech

The application includes an optional C++ text-to-speech component.

The Streamlit application expects the compiled executable at:

```text
src/tts.exe
```

The latest Copilot response is written to:

```text
data/tts_input.txt
```

The application then launches the C++ executable to speak the response.

This component is primarily intended for Windows environments where `tts.exe` has been compiled and configured.

---

# Project Structure

A typical project structure is:

```text
paramedic_ai/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
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

Some files may be optional depending on the current version of the project.

---

# Requirements

The application requires:

* Python
* Streamlit
* Pandas
* `streamlit-mic-recorder`
* Additional Python packages listed in `requirements.txt`

The Copilot, machine-learning model, knowledge system, and document ingestion components may have additional dependencies.

For Windows, PowerShell can be used to run the application.

---

# Installation

## 1. Open PowerShell

Navigate to the project directory:

```powershell
cd "C:\Users\paypa\paramedic_ai"
```

Replace the path if your project is located elsewhere.

---

## 2. Create a Virtual Environment

If a `.venv` does not already exist:

```powershell
python -m venv .venv
```

---

## 3. Install Dependencies

Use the project's requirements file:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `streamlit-mic-recorder` is missing, install it directly:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit-mic-recorder
```

---

# Running the Application

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit should provide a local address similar to:

```text
http://localhost:8501
```

Open that address in a web browser.

---

## PowerShell Execution Policy

On some Windows systems, activating the virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

may produce an error stating that script execution is disabled.

You do **not** need to change the PowerShell execution policy.

Instead, run Python directly from the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This also ensures that the correct project's Python environment is being used.

---

# Running the Knowledge Pipeline

After uploading a PDF or Word document through the Knowledge Manager, the application can run:

```text
src/ingest.py
```

followed by:

```text
src/build_index.py
```

The Streamlit interface provides an:

```text
🔄 Ingest & Rebuild Knowledge Index
```

button for this process.

The general pipeline is:

```text
Reference Document
        │
        ▼
data/documents/
        │
        ▼
ingest.py
        │
        ▼
Processed Knowledge
        │
        ▼
build_index.py
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

---

# Environment Variables

If the Copilot or other services require API credentials, configure them according to the project's `.env.example` file.

Do **not** commit API keys, passwords, tokens, or other secrets to source control.

For example:

```text
.env
```

should generally remain excluded from Git.

The repository includes:

```text
.env.example
```

as a template for required environment variables.

---

# Machine Learning

The ML portion of the application uses:

```python
load_model()
predict_risk()
classify_risk()
```

from:

```text
src/ml_model.py
```

The application expects the model implementation and any required model files to be available in the project.

A typical workflow may include:

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
ml_model.py
     │
     ▼
Streamlit Application
```

The exact model format and training procedure depend on the implementation contained in the project.

---

# Testing

The project contains a:

```text
tests/
```

directory.

Tests can be run using the project's configured test framework.

If `pytest` is included in the project's dependencies:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

---

# Troubleshooting

## `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'streamlit_mic_recorder'
```

Install the missing dependency into the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit-mic-recorder
```

Then restart Streamlit.

---

## Streamlit command not found

Instead of:

```powershell
streamlit run app.py
```

use:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This guarantees that Streamlit is run using the project's virtual environment.

---

## Missing `src` modules

Errors such as:

```text
ModuleNotFoundError: No module named 'src.ml_model'
```

usually indicate that the application is not being launched from the project root or that the required source file is missing.

Make sure you are in:

```text
paramedic_ai/
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

check that:

```text
src/tts.exe
```

exists.

The application currently expects a Windows executable named:

```text
tts.exe
```

If the executable has not been compiled, the rest of the Streamlit application can still be used, but the C++ text-to-speech feature will not work.

---

## Knowledge Index Errors

If document ingestion fails, inspect the error displayed by the Knowledge Manager.

The application runs:

```text
src/ingest.py
```

first and then:

```text
src/build_index.py
```

Only if ingestion succeeds does it attempt to rebuild the knowledge index.

---

# Development

When modifying the application, the main Streamlit entry point is:

```text
app.py
```

Core functionality is separated into the `src` directory.

Important modules include:

```text
src/ml_model.py
```

Machine-learning model loading and prediction.

```text
src/copilot.py
```

Copilot functionality.

```text
src/knowledge.py
```

Knowledge search, context formatting, and references.

```text
src/ingest.py
```

Document ingestion.

```text
src/build_index.py
```

Knowledge-index construction.

---

# Security and Privacy

This application may process sensitive or potentially identifiable patient information if users enter it.

For development and testing:

* Do not use real patient-identifying information unless the system has been specifically designed and authorized to handle it.
* Do not commit patient data to Git.
* Do not commit API keys or credentials.
* Protect uploaded reference documents appropriately.
* Use authorized and current EMS reference material.
* Review all AI-generated output before relying on it for educational purposes.

---

# Clinical Safety Disclaimer

**This application is a software demonstration and is NOT a medical device or clinical decision-making system.**

The machine-learning risk score and Copilot responses may be inaccurate, incomplete, outdated, or inappropriate for a particular patient.

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

If this project is not yet licensed, consider adding an appropriate license before distributing it publicly.

---

# Status

**Project:** Paramedic AI
**Type:** EMS education and decision-support demonstration
**Interface:** Streamlit
**Primary Language:** Python
**Optional TTS:** C++ / Windows executable

---

## Quick Start

For an existing Windows installation:

```powershell
cd "C:\Users\paypa\paramedic_ai"

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

🚑 **Paramedic AI — educational demonstration only.**
