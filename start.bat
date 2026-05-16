@echo off
chcp 65001 >nul 2>&1
echo.
echo   ============================================
echo     MyLibrary RAG - Smart Document Q&amp;A System
echo   ============================================
echo.
cd /d "%~dp0"

set STREAMLIT_GLOBAL_DEVELOPMENTMODE=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo [*] Starting Streamlit...
echo.

D:\dev-tools\Python312\python.exe -m streamlit run app.py

pause
