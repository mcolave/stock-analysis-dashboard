@echo off
cd /d "%~dp0"
echo Starting Stock Analysis Dashboard...
call ..\.venv\Scripts\activate.bat
python -m streamlit run stock_app.py
pause
