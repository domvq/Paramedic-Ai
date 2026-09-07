@echo off
cd /d C:\Users\paypa\paramedic_ai
call .venv\Scripts\activate.bat
start "" python -m streamlit run app.py
